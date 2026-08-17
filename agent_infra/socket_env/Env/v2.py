"""Descriptor-first v2 remote Gymnasium client."""

from __future__ import annotations

import socket
import uuid
from collections.abc import Mapping
from typing import Any, Optional

import gymnasium as gym
from gymnasium.spaces import Box

from .descriptor import EnvDescriptor
from .protocol import PROTOCOL_ID, PROTOCOL_VERSION, ProtocolError, RemoteProtocolError, command, recv_message, send_message


class SocketEnvV2(gym.Env):
    metadata = {"render_modes": []}
    # A Box action descriptor is already the producer's external vector; the
    # model wrapper must not turn it into a synthetic component dictionary.
    action_is_vector = True

    def __init__(self, host: str, port: int, *, connect_timeout_s: float = 10.0, request_timeout_s: float = 10.0, tcp_nodelay: bool = True, keepalive: bool = False, source_host: Optional[str] = None, source_port: Optional[int] = None, auto_connect: bool = False):
        super().__init__()
        self.host, self.port = host, int(port)
        self.connect_timeout_s, self.request_timeout_s = float(connect_timeout_s), float(request_timeout_s)
        self.tcp_nodelay, self.keepalive = bool(tcp_nodelay), bool(keepalive)
        self.source_address = (source_host, int(source_port or 0)) if source_host else None
        self._socket = None
        self.descriptor: Optional[EnvDescriptor] = None
        self.session_id: Optional[str] = None
        self._packet: Optional[dict[str, Any]] = None
        self._close_sent = False
        if auto_connect: self.connect()

    def connect(self):
        self._close_transport()
        peer = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s, source_address=self.source_address)
        peer.settimeout(self.request_timeout_s)
        if self.tcp_nodelay: peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if self.keepalive: peer.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self._socket, self._close_sent = peer, False
        response = self._request("DESCRIBE", {}, session_id=None)
        descriptor = EnvDescriptor.from_wire(response["descriptor"])
        self.descriptor, self.session_id = descriptor, descriptor.session_id
        self.observation_space, self.action_space = descriptor.observation_space, descriptor.action_space
        self.env_meta, self.meta_keys = dict(descriptor.env_metadata), dict(descriptor.meta_keys)
        # A rank-1 Box is already one external vector; adapter code must not
        # turn it into a metadata dict merely to turn it back again.
        self.action_is_vector = isinstance(self.action_space, Box) and len(self.action_space.shape) == 1
        return self

    def get_env_metadata(self) -> dict[str, Any]:
        """Return immutable descriptor metadata for model-side adapters.

        This is intentionally a raw metadata accessor: ordering, image layout,
        and normalization are interpreted by ``agent_factory``, never here.
        """
        return dict(getattr(self, "env_meta", {}))

    def _request(self, op: str, payload: Mapping[str, Any], *, session_id: Optional[str] = None) -> Mapping[str, Any]:
        if self._socket is None: raise RuntimeError("SocketEnvV2 is not connected")
        request_id = uuid.uuid4().hex
        send_message(self._socket, command(request_id, self.session_id if session_id is None else session_id, op, payload))
        response = recv_message(self._socket)
        if response.get("request_id") != request_id or response.get("kind") != "result":
            raise ProtocolError("response request_id mismatch")
        if response.get("protocol_id") != PROTOCOL_ID or int(response.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ProtocolError("response protocol version mismatch")
        if response.get("session_id") != self.session_id and op != "DESCRIBE":
            raise ProtocolError("response session_id mismatch")
        if not response.get("ok"):
            failure = response.get("error", {})
            raise RemoteProtocolError(str(failure.get("code", "UNKNOWN")), str(failure.get("message", "remote failure")))
        value = response.get("result", {})
        if not isinstance(value, Mapping): raise ProtocolError("result payload must be a mapping")
        return value

    def _accept_packet(self, packet: Mapping[str, Any], expected_kind: Optional[str] = None, expected_successor: bool = False) -> tuple[Any, dict[str, Any]]:
        if self.descriptor is None: raise RuntimeError("DESCRIBE is required")
        required = ("episode_id", "transition_id", "kind", "descriptor_hash", "observation", "info")
        if any(key not in packet for key in required): raise ProtocolError("transition packet missing fields")
        if packet["descriptor_hash"] != self.descriptor.descriptor_hash: raise ProtocolError("transition descriptor hash mismatch")
        if expected_kind and packet["kind"] != expected_kind: raise ProtocolError("unexpected transition packet kind")
        if expected_successor and (self._packet is None or packet["episode_id"] != self._packet["episode_id"] or int(packet["transition_id"]) != int(self._packet["transition_id"]) + 1): raise ProtocolError("transition identity is not the expected successor")
        if not isinstance(packet["info"], Mapping): raise ProtocolError("transition info must be a mapping")
        self._packet = dict(packet)
        return packet["observation"], dict(packet["info"])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._socket is None: self.connect()
        payload = {"seed": seed, "options": options} if seed is not None or options is not None else {}
        response = self._request("RESET", payload)
        observation, info = self._accept_packet(response["packet"], expected_kind="reset")
        if int(self._packet["transition_id"]) != 0: raise ProtocolError("reset transition id must be zero")
        return observation, info

    def step(self, action: Any):
        if self._packet is None or self.descriptor is None: raise RuntimeError("call reset() before step()")
        if not self.action_space.contains(action): raise ValueError("action is not a member of the advertised action space")
        response = self._request("STEP", {"episode_id": self._packet["episode_id"], "transition_id": self._packet["transition_id"], "descriptor_hash": self.descriptor.descriptor_hash, "action": action})
        observation, info = self._accept_packet(response["packet"], expected_kind="step", expected_successor=True)
        packet = self._packet
        for key in ("reward", "terminated", "truncated"):
            if key not in packet: raise ProtocolError(f"step packet missing {key}")
        return observation, float(packet["reward"]), bool(packet["terminated"]), bool(packet["truncated"]), info

    def get_latest(self):
        response = self._request("GET_LATEST", {})
        return self._accept_packet(response["packet"])

    def ping(self, nonce: Any = None):
        return dict(self._request("PING", {"nonce": nonce}))

    def reconnect(self):
        self.connect()
        return self.get_latest()

    def _close_transport(self):
        if self._socket is not None:
            try: self._socket.close()
            finally: self._socket = None

    def close(self):
        if self._socket is not None and not self._close_sent:
            try:
                self._request("CLOSE", {})
            except (OSError, ConnectionError, ProtocolError):
                pass
            self._close_sent = True
        self._close_transport()
