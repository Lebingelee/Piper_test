"""Portable, ordered JSON representation for Gymnasium spaces."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from .protocol import ProtocolError


def encode_space(space: spaces.Space) -> dict[str, Any]:
    if isinstance(space, spaces.Box):
        return {"type": "Box", "shape": list(space.shape), "dtype": np.dtype(space.dtype).str, "low": np.asarray(space.low), "high": np.asarray(space.high)}
    if isinstance(space, spaces.Dict):
        return {"type": "Dict", "keys": list(space.spaces.keys()), "spaces": {key: encode_space(child) for key, child in space.spaces.items()}}
    if isinstance(space, spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n), "start": int(space.start)}
    if isinstance(space, spaces.MultiDiscrete):
        return {"type": "MultiDiscrete", "nvec": np.asarray(space.nvec), "start": np.asarray(space.start)}
    if isinstance(space, spaces.MultiBinary):
        return {"type": "MultiBinary", "n": space.n}
    if isinstance(space, spaces.Tuple):
        return {"type": "Tuple", "spaces": [encode_space(child) for child in space.spaces]}
    raise TypeError(f"unsupported Gym space {type(space).__name__}")


def decode_space(data: Mapping[str, Any]) -> spaces.Space:
    try:
        kind = data["type"]
        if kind == "Box":
            dtype = np.dtype(data["dtype"])
            shape = tuple(int(dim) for dim in data["shape"])
            low, high = np.asarray(data["low"], dtype=dtype), np.asarray(data["high"], dtype=dtype)
            return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
        if kind == "Dict":
            keys, entries = list(data["keys"]), data["spaces"]
            if len(set(keys)) != len(keys) or set(keys) != set(entries):
                raise ProtocolError("Dict space keys must be explicit and complete")
            return spaces.Dict(OrderedDict((key, decode_space(entries[key])) for key in keys))
        if kind == "Discrete":
            return spaces.Discrete(int(data["n"]), start=int(data.get("start", 0)))
        if kind == "MultiDiscrete":
            return spaces.MultiDiscrete(np.asarray(data["nvec"]), start=np.asarray(data.get("start", 0)))
        if kind == "MultiBinary":
            return spaces.MultiBinary(data["n"])
        if kind == "Tuple":
            return spaces.Tuple(tuple(decode_space(child) for child in data["spaces"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("invalid portable Gym space") from exc
    raise ProtocolError(f"unsupported portable Gym space {kind!r}")
