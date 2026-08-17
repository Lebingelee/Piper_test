"""Versioned, task-agnostic wire primitives for remote Gym environments."""

from __future__ import annotations

import base64
import json
import socket
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np

PROTOCOL_ID = "remote-gym-env"
PROTOCOL_VERSION = 2
MAX_MESSAGE_BYTES = 512 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


class RemoteProtocolError(ProtocolError):
    def __init__(self, code: str, message: str):
        super().__init__(f"remote error {code}: {message}")
        self.code = code


def encode_ndarray_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {"__ndarray__": True, "dtype": array.dtype.str, "shape": list(array.shape), "data": base64.b64encode(array.tobytes()).decode("ascii")}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): encode_ndarray_tree(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode_ndarray_tree(child) for child in value]
    return value


def decode_ndarray_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        marker = value.get("__ndarray__")
        payload = marker if isinstance(marker, Mapping) else value
        if marker is True or isinstance(marker, Mapping):
            try:
                dtype = np.dtype(payload["dtype"])
                shape = tuple(int(dim) for dim in payload["shape"])
                data = base64.b64decode(payload["data"], validate=True)
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolError("invalid ndarray payload") from exc
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if len(data) != expected:
                raise ProtocolError("ndarray byte length does not match dtype and shape")
            return np.frombuffer(data, dtype=dtype).copy().reshape(shape)
        return {key: decode_ndarray_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [decode_ndarray_tree(child) for child in value]
    return value


def send_message(peer: socket.socket, message: Mapping[str, Any]) -> None:
    body = json.dumps(encode_ndarray_tree(message), separators=(",", ":"), allow_nan=False).encode("utf-8")
    peer.sendall(struct.pack("!I", len(body)) + body)


def _recv_exact(peer: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = peer.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("remote peer closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_message(peer: socket.socket, *, max_message_bytes: int = MAX_MESSAGE_BYTES) -> dict[str, Any]:
    size = struct.unpack("!I", _recv_exact(peer, 4))[0]
    if size <= 0 or size > max_message_bytes:
        raise ProtocolError("invalid frame length")
    try:
        message = decode_ndarray_tree(json.loads(_recv_exact(peer, size).decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON frame") from exc
    if not isinstance(message, dict):
        raise ProtocolError("protocol message must be an object")
    return message


def command(request_id: str, session_id: str | None, op: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"kind": "command", "protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION, "request_id": request_id, "session_id": session_id, "op": op, "payload": dict(payload or {})}


def result(request_id: str, session_id: str | None, value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"kind": "result", "protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION, "request_id": request_id, "session_id": session_id, "ok": True, "result": dict(value or {})}


def error(request_id: str, session_id: str | None, code: str, message: str) -> dict[str, Any]:
    return {"kind": "result", "protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION, "request_id": request_id, "session_id": session_id, "ok": False, "error": {"code": code, "message": message}}
