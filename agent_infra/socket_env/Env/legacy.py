"""Protocol-driven TCP client for remote Gymnasium-style environments.

This module deliberately only deals with the versioned wire protocol and raw
array trees.  Model-facing preprocessing belongs in ``agent_factory``.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict as GymDict


class ProtocolError(RuntimeError):
    """A peer message does not satisfy the versioned transport contract."""


def encode_ndarray_tree(value: Any) -> Any:
    """Convert a nested ndarray tree to JSON-safe values without pickle."""
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): encode_ndarray_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_ndarray_tree(item) for item in value]
    return value


def decode_ndarray_tree(value: Any) -> Any:
    """Restore the raw nested ndarray tree emitted by :func:`encode_ndarray_tree`."""
    if isinstance(value, Mapping):
        payload = value
        marker = value.get("__ndarray__")
        # Accept the two JSON-safe discriminator forms used by independent
        # producers: a boolean marker with sibling fields, or a payload nested
        # under the marker.  No task identity participates in this decision.
        if isinstance(marker, Mapping):
            payload = marker
        if marker is True or isinstance(marker, Mapping):
            try:
                dtype = np.dtype(payload["dtype"])
                shape = tuple(int(dim) for dim in payload["shape"])
                raw = base64.b64decode(payload["data"], validate=True)
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolError("invalid ndarray payload") from exc
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if len(raw) != expected:
                raise ProtocolError("ndarray byte length does not match dtype and shape")
            return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
        return {key: decode_ndarray_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_ndarray_tree(item) for item in value]
    return value


def _space_from_value(value: Any) -> gym.Space:
    if isinstance(value, Mapping):
        return GymDict({key: _space_from_value(item) for key, item in value.items()})
    array = np.asarray(value)
    if array.dtype.kind in "iu":
        info = np.iinfo(array.dtype)
        return Box(info.min, info.max, shape=array.shape, dtype=array.dtype)
    if array.dtype.kind == "b":
        return Box(0, 1, shape=array.shape, dtype=array.dtype)
    return Box(-np.inf, np.inf, shape=array.shape, dtype=array.dtype)


def _vector_space(action_meta: Mapping[str, Any]) -> Tuple[Box, str, str]:
    """Read a generic external-vector schema; no task-specific field names."""
    spec = action_meta.get("external_vector", action_meta.get("vector", action_meta))
    if not isinstance(spec, Mapping):
        raise ProtocolError("env_meta action schema must describe an external vector")
    try:
        shape = tuple(int(dim) for dim in spec["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("action schema is missing vector shape") from exc
    if len(shape) != 1 or shape[0] < 0:
        raise ProtocolError("external action must be a one-dimensional vector")
    size = shape[0]
    low = np.broadcast_to(np.asarray(spec.get("low", -np.inf), dtype=np.float32), shape).copy()
    high = np.broadcast_to(np.asarray(spec.get("high", np.inf), dtype=np.float32), shape).copy()
    if np.any(low > high):
        raise ProtocolError("action schema has low > high")
    schema_id = str(spec.get("schema_id", action_meta.get("schema_id", "")))
    schema_version = str(spec.get("schema_version", action_meta.get("schema_version", "")))
    if not schema_id or not schema_version:
        raise ProtocolError("action schema id and version are required")
    return Box(low=low, high=high, shape=(size,), dtype=np.float32), schema_id, schema_version


class SocketEnv(gym.Env):
    """A raw remote environment client for the frozen length-prefixed JSON protocol."""

    metadata = {"render_modes": []}
    action_is_vector = True

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout: Optional[float] = 10.0,
        source_host: Optional[str] = None,
        source_port: Optional[int] = None,
        tcp_nodelay: bool = True,
        keepalive: bool = False,
        auto_connect: bool = False,
    ):
        super().__init__()
        self.host, self.port, self.timeout = host, int(port), timeout
        self.source_address = (source_host, int(source_port or 0)) if source_host else None
        self.tcp_nodelay, self.keepalive = bool(tcp_nodelay), bool(keepalive)
        self._socket: Optional[socket.socket] = None
        self._packet: Optional[Dict[str, Any]] = None
        self.env_meta: Dict[str, Any] = {}
        self.meta_keys: Dict[str, Any] = {"obs": {}, "action": {}}
        self.action_schema_id = ""
        self.action_schema_version = ""
        self.observation_space: gym.Space = GymDict({})
        self.action_space: gym.Space = Box(-np.inf, np.inf, shape=(0,), dtype=np.float32)
        if auto_connect:
            self.connect()

    def _send(self, message: Mapping[str, Any]) -> None:
        if self._socket is None:
            raise RuntimeError("SocketEnv is not connected")
        payload = json.dumps(encode_ndarray_tree(message), separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._socket.sendall(struct.pack("!I", len(payload)) + payload)

    def _recv_exact(self, size: int) -> bytes:
        if self._socket is None:
            raise RuntimeError("SocketEnv is not connected")
        parts = bytearray()
        while len(parts) < size:
            chunk = self._socket.recv(size - len(parts))
            if not chunk:
                raise ConnectionError("remote peer closed the connection")
            parts.extend(chunk)
        return bytes(parts)

    def _recv(self) -> Dict[str, Any]:
        size = struct.unpack("!I", self._recv_exact(4))[0]
        if size <= 0 or size > 512 * 1024 * 1024:
            raise ProtocolError("invalid frame length")
        try:
            message = decode_ndarray_tree(json.loads(self._recv_exact(size).decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("invalid JSON frame") from exc
        if not isinstance(message, dict):
            raise ProtocolError("protocol message must be an object")
        return message

    def _accept_observation(self, message: Mapping[str, Any], *, derive_spaces: bool = False) -> Dict[str, Any]:
        if message.get("kind") != "observation" or not isinstance(message.get("packet"), Mapping):
            raise ProtocolError("expected an observation packet")
        packet = dict(message["packet"])
        required = ("schema_id", "schema_version", "episode_id", "transition_id", "env_meta_hash", "observation")
        missing = [key for key in required if key not in packet]
        if missing:
            raise ProtocolError(f"observation packet missing fields: {missing}")
        if derive_spaces:
            meta = packet.get("env_meta")
            if not isinstance(meta, Mapping):
                raise ProtocolError("first observation must include env_meta")
            self.env_meta = dict(meta)
            self.observation_space = _space_from_value(packet["observation"])
            action_meta = meta.get("action_schema", meta.get("action"))
            if not isinstance(action_meta, Mapping):
                raise ProtocolError("env_meta is missing action schema")
            self.action_space, self.action_schema_id, self.action_schema_version = _vector_space(action_meta)
            meta_keys = meta.get("meta_keys", {"obs": meta.get("obs", {}), "action": meta.get("action", {})})
            if isinstance(meta_keys, Mapping):
                self.meta_keys = dict(meta_keys)
        self._packet = packet
        return packet

    def connect(self) -> "SocketEnv":
        self.close()
        peer = socket.create_connection((self.host, self.port), timeout=self.timeout, source_address=self.source_address)
        peer.settimeout(self.timeout)
        if self.tcp_nodelay:
            peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if self.keepalive:
            peer.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self._socket = peer
        self._accept_observation(self._recv(), derive_spaces=True)
        return self

    def reconnect(self) -> Tuple[Any, Dict[str, Any]]:
        """Explicitly reconnect and return the peer's retained latest packet.

        Reconnect never sends an action and therefore cannot create a transition.
        """
        self.connect()
        return self._packet_observation_and_info()

    def _packet_observation_and_info(self) -> Tuple[Any, Dict[str, Any]]:
        if self._packet is None:
            raise RuntimeError("no remote packet is available")
        context = dict(self._packet.get("transition_context") or {})
        public_info = context.get("info", context)
        if not isinstance(public_info, Mapping):
            raise ProtocolError("transition_context.info must be a mapping")
        return self._packet["observation"], dict(public_info)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._socket is None:
            self.connect()
        return self._packet_observation_and_info()

    def step(self, action: Any):
        if self._socket is None or self._packet is None:
            raise RuntimeError("call reset() before step()")
        vector = np.asarray(action, dtype=np.float32)
        if vector.shape != self.action_space.shape or not np.isfinite(vector).all() or not self.action_space.contains(vector):
            raise ValueError("action is not a finite member of the advertised external vector space")
        packet = self._packet
        self._send({"kind": "action", "reply": {
            "schema_id": packet["schema_id"], "schema_version": packet["schema_version"],
            "episode_id": packet["episode_id"], "transition_id": packet["transition_id"],
            "env_meta_hash": packet["env_meta_hash"], "action_schema_id": self.action_schema_id,
            "action_schema_version": self.action_schema_version, "action": vector.tolist(),
        }})
        next_packet = self._accept_observation(self._recv())
        if next_packet["episode_id"] != packet["episode_id"] or int(next_packet["transition_id"]) != int(packet["transition_id"]) + 1:
            raise ProtocolError("action reply did not produce the next transition")
        obs, _ = self._packet_observation_and_info()
        context = dict(self._packet.get("transition_context") or {})
        public_info = context.pop("info", context)
        if not isinstance(public_info, Mapping):
            raise ProtocolError("transition_context.info must be a mapping")
        return (
            obs,
            float(context.pop("reward", 0.0)),
            bool(context.pop("terminated", False)),
            bool(context.pop("truncated", False)),
            dict(public_info),
        )

    def get_env_metadata(self) -> Dict[str, Any]:
        return dict(self.env_meta)

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
