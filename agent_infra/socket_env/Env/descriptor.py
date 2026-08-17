"""Descriptor validation and stable hashing for remote Gym environments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from gymnasium import spaces

from .protocol import PROTOCOL_ID, PROTOCOL_VERSION, ProtocolError, encode_ndarray_tree
from .spaces import decode_space, encode_space


def descriptor_hash(value: Mapping[str, Any]) -> str:
    try:
        body = json.dumps(
            encode_ndarray_tree(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(
            "descriptor must contain only strict finite JSON metadata; "
            "use portable Gym spaces for non-finite bounds"
        ) from exc
    return hashlib.sha256(body).hexdigest()


@dataclass(frozen=True)
class EnvDescriptor:
    session_id: str
    descriptor_hash: str
    observation_space: spaces.Space
    action_space: spaces.Space
    env_metadata: Mapping[str, Any]
    meta_keys: Mapping[str, Any]
    capabilities: tuple[str, ...]

    @classmethod
    def build(cls, session_id: str, observation_space: spaces.Space, action_space: spaces.Space, env_metadata: Mapping[str, Any], meta_keys: Mapping[str, Any], capabilities: tuple[str, ...]):
        body = {"protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION, "session_id": session_id, "observation_space": encode_space(observation_space), "action_space": encode_space(action_space), "env_metadata": dict(env_metadata), "meta_keys": dict(meta_keys), "capabilities": list(capabilities)}
        return cls(session_id, descriptor_hash(body), observation_space, action_space, dict(env_metadata), dict(meta_keys), tuple(capabilities))

    def to_wire(self) -> dict[str, Any]:
        body = {"protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION, "session_id": self.session_id, "observation_space": encode_space(self.observation_space), "action_space": encode_space(self.action_space), "env_metadata": dict(self.env_metadata), "meta_keys": dict(self.meta_keys), "capabilities": list(self.capabilities)}
        return {**body, "descriptor_hash": self.descriptor_hash}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]):
        try:
            if data["protocol_id"] != PROTOCOL_ID or int(data["protocol_version"]) != PROTOCOL_VERSION:
                raise ProtocolError("unsupported remote Gym protocol version")
            canonical = {key: data[key] for key in ("protocol_id", "protocol_version", "session_id", "observation_space", "action_space", "env_metadata", "meta_keys", "capabilities")}
            if descriptor_hash(canonical) != data["descriptor_hash"]:
                raise ProtocolError("descriptor hash mismatch")
            return cls(str(data["session_id"]), str(data["descriptor_hash"]), decode_space(data["observation_space"]), decode_space(data["action_space"]), dict(data["env_metadata"]), dict(data["meta_keys"]), tuple(data["capabilities"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("descriptor missing required fields") from exc
