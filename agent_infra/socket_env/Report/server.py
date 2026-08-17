"""Generic v2 protocol producer for any Gymnasium-compatible local env."""

from __future__ import annotations

import socket
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Callable, Optional, Protocol

import numpy as np
import gymnasium as gym

from ..Env.descriptor import EnvDescriptor
from ..Env.protocol import PROTOCOL_ID, PROTOCOL_VERSION, ProtocolError, command, error, recv_message, result, send_message


class RemoteReportableEnv(Protocol):
    """Minimum local contract required by the task-agnostic reporter."""

    observation_space: Any
    action_space: Any

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None): ...
    def step(self, action: Any): ...


class ReportableEnvAdapter(gym.Wrapper):
    """Attach immutable descriptor metadata to an otherwise standard Gym env."""

    def __init__(self, env: gym.Env, env_metadata: Mapping[str, Any]):
        super().__init__(env)
        self._env_metadata = dict(env_metadata)

    def get_env_metadata(self) -> dict[str, Any]:
        return dict(self._env_metadata)


class EnvReporter:
    """Serve one local Gym env without task-specific imports or preprocessing."""

    CAPABILITIES = ("describe", "reset", "step", "get_latest", "ping", "close")

    def __init__(self, env: RemoteReportableEnv, *, bind_host: str = "127.0.0.1", port: int = 0, request_cache_size: int = 128, max_message_bytes: int = 512 * 1024 * 1024, metadata_provider: Optional[Callable[[Any], Mapping[str, Any]]] = None):
        self.env = env
        self._metadata_provider = metadata_provider
        self.request_cache_size = int(request_cache_size)
        self.max_message_bytes = int(max_message_bytes)
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((bind_host, int(port)))
        self._server.listen(1)
        self._session_id = uuid.uuid4().hex
        metadata = self._metadata()
        meta_keys = metadata.get("meta_keys", {"obs": metadata.get("obs", {}), "action": metadata.get("action", {})})
        self.descriptor = EnvDescriptor.build(self._session_id, env.observation_space, env.action_space, metadata, meta_keys, self.CAPABILITIES)
        self._episode_id: Optional[str] = None
        self._transition_id: Optional[int] = None
        self._latest_packet: Optional[dict[str, Any]] = None
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._closed = False

    @property
    def address(self) -> tuple[str, int]:
        return self._server.getsockname()[:2]

    def _metadata(self) -> dict[str, Any]:
        if self._metadata_provider is not None:
            metadata = self._metadata_provider(self.env)
        else:
            get_meta = getattr(self.env, "get_env_metadata", None)
            metadata = get_meta() if callable(get_meta) else getattr(self.env, "metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("reportable env metadata must be a mapping")
        return dict(metadata)

    def _packet(self, kind: str, observation: Any, info: Mapping[str, Any], **transition: Any) -> dict[str, Any]:
        assert self._episode_id is not None and self._transition_id is not None
        return {"episode_id": self._episode_id, "transition_id": self._transition_id, "kind": kind, "descriptor_hash": self.descriptor.descriptor_hash, "observation": observation, "info": dict(info), **transition}

    def _cache_result(self, request_id: str, response: dict[str, Any]) -> dict[str, Any]:
        self._cache[request_id] = response
        self._cache.move_to_end(request_id)
        while len(self._cache) > self.request_cache_size:
            self._cache.popitem(last=False)
        return response

    def _require_session(self, message: Mapping[str, Any]) -> None:
        if message.get("session_id") != self._session_id:
            raise ProtocolError("session id mismatch")

    def _validate_command(self, message: Mapping[str, Any]) -> tuple[str, str, Mapping[str, Any]]:
        if message.get("kind") != "command" or message.get("protocol_id") != PROTOCOL_ID or int(message.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ProtocolError("unsupported command envelope")
        request_id, op, payload = message.get("request_id"), message.get("op"), message.get("payload", {})
        if not isinstance(request_id, str) or not request_id or not isinstance(op, str) or not isinstance(payload, Mapping):
            raise ProtocolError("invalid command fields")
        return request_id, op.upper(), payload

    def _dispatch(self, message: Mapping[str, Any]) -> dict[str, Any]:
        request_id = message.get("request_id") if isinstance(message.get("request_id"), str) else ""
        try:
            if message.get("protocol_id") != PROTOCOL_ID or int(message.get("protocol_version", -1)) != PROTOCOL_VERSION:
                response = error(request_id, self._session_id, "UNSUPPORTED_VERSION", "unsupported protocol id or version")
                return self._cache_result(request_id, response) if request_id else response
            request_id, op, payload = self._validate_command(message)
            if request_id in self._cache:
                return self._cache[request_id]
            if op == "DESCRIBE":
                response = result(request_id, self._session_id, {"descriptor": self.descriptor.to_wire()})
            else:
                self._require_session(message)
                if op == "RESET":
                    observation, info = self.env.reset(seed=payload.get("seed"), options=payload.get("options"))
                    if not isinstance(info, Mapping):
                        raise ProtocolError("env.reset info must be a mapping")
                    self._episode_id, self._transition_id = uuid.uuid4().hex, 0
                    self._latest_packet = self._packet("reset", observation, info)
                    response = result(request_id, self._session_id, {"packet": self._latest_packet})
                elif op == "STEP":
                    if self._latest_packet is None:
                        return self._cache_result(request_id, error(request_id, self._session_id, "NO_EPISODE", "RESET is required before STEP"))
                    if payload.get("episode_id") != self._episode_id or payload.get("transition_id") != self._transition_id:
                        return self._cache_result(request_id, error(request_id, self._session_id, "STALE_TRANSITION", "episode or transition identity mismatch"))
                    if payload.get("descriptor_hash") != self.descriptor.descriptor_hash:
                        return self._cache_result(request_id, error(request_id, self._session_id, "SCHEMA_MISMATCH", "descriptor hash mismatch"))
                    action = payload.get("action")
                    if not self.env.action_space.contains(action):
                        return self._cache_result(request_id, error(request_id, self._session_id, "INVALID_ACTION", "action is not a member of action_space"))
                    observation, reward, terminated, truncated, info = self.env.step(action)
                    if not isinstance(info, Mapping):
                        raise ProtocolError("env.step info must be a mapping")
                    self._transition_id += 1
                    self._latest_packet = self._packet("step", observation, info, reward=float(reward), terminated=bool(terminated), truncated=bool(truncated))
                    response = result(request_id, self._session_id, {"packet": self._latest_packet})
                elif op == "GET_LATEST":
                    if self._latest_packet is None:
                        response = error(request_id, self._session_id, "NO_EPISODE", "no reset packet exists")
                    else:
                        response = result(request_id, self._session_id, {"packet": self._latest_packet})
                elif op == "PING":
                    response = result(request_id, self._session_id, {"nonce": payload.get("nonce"), "state": "ACTIVE" if self._latest_packet else "DESCRIBED"})
                elif op == "CLOSE":
                    if not self._closed:
                        close = getattr(self.env, "close", None)
                        if callable(close): close()
                        self._closed = True
                    response = result(request_id, self._session_id, {"closed": True})
                else:
                    response = error(request_id, self._session_id, "UNSUPPORTED", f"unsupported operation {op}")
            return self._cache_result(request_id, response)
        except ProtocolError as exc:
            return self._cache_result(request_id, error(request_id, self._session_id, "INVALID_STATE", str(exc))) if request_id else error("", self._session_id, "INVALID_STATE", str(exc))
        except Exception as exc:
            return self._cache_result(request_id, error(request_id, self._session_id, "INTERNAL_ERROR", str(exc))) if request_id else error("", self._session_id, "INTERNAL_ERROR", str(exc))

    def serve_connection(self, peer: socket.socket) -> None:
        with peer:
            while True:
                try:
                    message = recv_message(peer, max_message_bytes=self.max_message_bytes)
                except ConnectionError:
                    return
                response = self._dispatch(message)
                send_message(peer, response)
                if response.get("result", {}).get("closed"):
                    return

    def serve_once(self, timeout_s: Optional[float] = None) -> None:
        self._server.settimeout(timeout_s)
        peer, _ = self._server.accept()
        self.serve_connection(peer)

    def close(self) -> None:
        if not self._closed:
            close = getattr(self.env, "close", None)
            if callable(close): close()
            self._closed = True
        self._server.close()
