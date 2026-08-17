"""Shared model-side observation transform for online raw trees and raw H5.

Transport and H5 remain raw.  This module is the single place that applies
declared leaf order, image layout, tensor conversion and RGB scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import h5py
import numpy as np
import torch


def _ordered(mapping: Mapping[str, Any], order: Any = None) -> tuple[str, ...]:
    keys = tuple(mapping.keys())
    if order is None:
        return tuple(sorted(keys))
    result = tuple(order)
    if len(set(result)) != len(result) or set(result) != set(keys):
        raise ValueError("explicit transform order must contain every leaf exactly once")
    return result


@dataclass(frozen=True)
class ModelTransformSpec:
    """Immutable model-facing interpretation of a raw observation tree."""

    group_order: tuple[str, ...]
    leaf_orders: Mapping[str, tuple[str, ...]]
    layouts: Mapping[tuple[str, str], str]
    rgb_leaves: frozenset[tuple[str, str]]

    @classmethod
    def from_env_meta(cls, env_meta: Mapping[str, Any], meta_keys: Optional[Mapping[str, Any]] = None):
        meta_keys = meta_keys or env_meta.get("meta_keys", {})
        obs = meta_keys.get("obs", env_meta.get("obs", {})) if isinstance(meta_keys, Mapping) else {}
        if not isinstance(obs, Mapping):
            raise TypeError("model transform metadata must expose observation groups")
        group_order = _ordered(obs, env_meta.get("observation_group_order", env_meta.get("obs_group_order")))
        order_map = env_meta.get("observation_order", env_meta.get("obs_order", {}))
        order_map = order_map if isinstance(order_map, Mapping) else {}
        leaf_orders = {group: _ordered(leaves, order_map.get(group)) for group, leaves in obs.items() if isinstance(leaves, Mapping)}
        layouts, rgb_leaves = {}, set()
        for root_name in ("observation_schema", "observation", "obs"):
            root = env_meta.get(root_name)
            if not isinstance(root, Mapping):
                continue
            groups = root.get("groups", root)
            if not isinstance(groups, Mapping):
                continue
            for group, group_spec in groups.items():
                leaves = group_spec.get("leaves", group_spec) if isinstance(group_spec, Mapping) else {}
                if not isinstance(leaves, Mapping):
                    continue
                for leaf, spec in leaves.items():
                    if not isinstance(spec, Mapping):
                        continue
                    key = (group, leaf)
                    if isinstance(spec.get("layout"), str):
                        layouts[key] = spec["layout"].upper()
                    if str(spec.get("semantic", "")).lower() in {"rgb", "image_rgb"}:
                        rgb_leaves.add(key)
        return cls(group_order, leaf_orders, layouts, frozenset(rgb_leaves))

    def leaf_order(self, group: str, values: Mapping[str, Any]) -> tuple[str, ...]:
        return self.leaf_orders.get(group, _ordered(values))

    def tensor(self, group: str, leaf: str, value: Any, *, device: str = "cpu") -> torch.Tensor:
        result = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))
        layout = self.layouts.get((group, leaf))
        if layout == "HWC" and result.ndim == 3:
            result = result.permute(2, 0, 1)
        elif layout == "NHWC" and result.ndim == 4:
            result = result.permute(0, 3, 1, 2)
        result = result.float()
        if (group, leaf) in self.rgb_leaves and result.numel() and result.max() > 1.01:
            result = result / 255.0
        return result.to(device)


def read_raw_h5_observation(node: h5py.Group, index: int) -> dict[str, Any]:
    """Read one raw H5 observation tree; no layout/normalization occurs here."""
    result = {}
    for key in node.keys():
        child = node[key]
        result[key] = read_raw_h5_observation(child, index) if isinstance(child, h5py.Group) else child[index]
    return result

