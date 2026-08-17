import os
import json
from pathlib import Path

import h5py
import yaml

from agent_factory.data.impl.vla_feature.feature_h5_dataset import (
    VLAFeatureDataset,
    VLAFeatureBlockSequenceDataset,
    VLAFeatureSequenceDataset,
    VLAFeatureTransitionDataset,
)
from agent_factory.data.registry import register_dataset_type


def _cfg_get(cfg, path: str, default):
    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _cfg_set_dataset_config(cfg, key: str, value):
    dataset_cfg = cfg.get("dataset") if isinstance(cfg, dict) else getattr(cfg, "dataset", None)
    if dataset_cfg is None:
        return
    if isinstance(dataset_cfg, dict):
        dataset_cfg.setdefault("config", {})[key] = value
        return
    config = getattr(dataset_cfg, "config", None)
    if config is None:
        setattr(dataset_cfg, "config", {key: value})
    else:
        config[key] = value


def _read_text_dataset(dataset):
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "shape") and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return value


def _iter_vla_feature_groups(h5_file):
    traj_keys = sorted(
        [
            key for key in h5_file.keys()
            if key.startswith("traj_") and isinstance(h5_file[key], h5py.Group)
        ]
    )
    if traj_keys:
        for key in traj_keys:
            yield h5_file[key]
    else:
        yield h5_file


def _infer_task_max_episode_steps_from_path(path: str, *, skip_bad_files: bool = False):
    root = Path(path)
    if not root.exists():
        return {}
    h5_paths = [root] if root.is_file() else sorted(root.rglob("*.h5"))
    mapping = {}
    for h5_path in h5_paths:
        try:
            with h5py.File(h5_path, "r") as h5_file:
                for group in _iter_vla_feature_groups(h5_file):
                    env_meta_node = group.get("meta/env_meta") or h5_file.get("meta/env_meta")
                    env_cfg_node = group.get("meta/env_cfg") or h5_file.get("meta/env_cfg")
                    if env_meta_node is None or env_cfg_node is None:
                        continue
                    env_meta = json.loads(_read_text_dataset(env_meta_node))
                    env_cfg = yaml.safe_load(_read_text_dataset(env_cfg_node)) or {}
                    task_name = str(env_meta.get("task_name", "") or "").strip()
                    max_episode_steps = env_cfg.get("max_episode_steps")
                    if not task_name or max_episode_steps is None:
                        continue
                    max_episode_steps = int(max_episode_steps)
                    previous = mapping.get(task_name)
                    if previous is not None and previous != max_episode_steps:
                        raise ValueError(
                            f"Conflicting max_episode_steps for task {task_name!r}: "
                            f"{previous} vs {max_episode_steps} in {h5_path}"
                        )
                    mapping[task_name] = max_episode_steps
        except OSError:
            if not skip_bad_files:
                raise
    return mapping


def _ensure_task_max_episode_steps(cfg, *, source_path: str = ""):
    existing = _cfg_get(cfg, "dataset.config.task_max_episode_steps", {}) or {}
    if hasattr(existing, "items") and dict(existing):
        return dict(existing)

    candidate_path = (
        str(_cfg_get(cfg, "dataset.config.max_episode_steps_source_path", "") or "")
        or str(_cfg_get(cfg, "dataset.config.replaybuffer_path", "") or "")
        or str(_cfg_get(cfg, "dataset.replaybuffer.folder_path", "") or "")
        or str(_cfg_get(cfg, "dataset.replaybuffer.replaybuffer_path", "") or "")
        or str(source_path or "")
    )
    if not candidate_path:
        return {}

    mapping = _infer_task_max_episode_steps_from_path(
        candidate_path,
        skip_bad_files=bool(_cfg_get(cfg, "dataset.config.skip_bad_files", False)),
    )
    if mapping:
        _cfg_set_dataset_config(cfg, "task_max_episode_steps", mapping)
    return mapping


def _build_dataset_for_path(
    cfg,
    path: str,
    required_keys=None,
    *,
    dataset_cls,
    source_role: str,
    split: str,
    num_traj=None,
):
    _ensure_task_max_episode_steps(cfg, source_path=path)
    kwargs = {
        "cfg": cfg,
        "required_keys": required_keys,
        "source_role": source_role,
        "split": split,
        "num_traj": num_traj,
    }
    if os.path.isdir(path):
        return dataset_cls(folder_path=path, **kwargs)
    if os.path.isfile(path):
        return dataset_cls(h5_path=path, **kwargs)
    raise FileNotFoundError(f"VLA feature dataset path not found: {path}")


def _build_vla_feature_expert_dataset(cfg, required_keys=None, expert_path=None, *, dataset_cls):
    path = str(_cfg_get(cfg, "dataset.config.expert_path", "") or expert_path or cfg.dataset.expert.demo_path)
    split = str(_cfg_get(cfg, "dataset.config.expert_split", _cfg_get(cfg, "dataset.expert.split", "train")))
    source_role = str(
        _cfg_get(cfg, "dataset.config.expert_source_role", _cfg_get(cfg, "dataset.expert.source_role", "demos"))
    )
    return _build_dataset_for_path(
        cfg,
        path,
        required_keys=required_keys,
        dataset_cls=dataset_cls,
        source_role=source_role,
        split=split,
        num_traj=getattr(cfg.dataset.expert, "num_traj", None),
    )


def _build_vla_feature_replaybuffer(cfg, required_keys=None, replaybuffer_path=None, *, dataset_cls):
    replaybuffer_path = str(_cfg_get(cfg, "dataset.config.replaybuffer_path", "") or replaybuffer_path or "")
    if not replaybuffer_path:
        raise ValueError("VLA feature replaybuffer requires a replaybuffer_path.")
    split = str(
        _cfg_get(cfg, "dataset.config.replaybuffer_split", _cfg_get(cfg, "dataset.replaybuffer.split", "train"))
    )
    source_role = str(
        _cfg_get(
            cfg,
            "dataset.config.replaybuffer_source_role",
            _cfg_get(cfg, "dataset.replaybuffer.source_role", "rollouts"),
        )
    )
    return _build_dataset_for_path(
        cfg,
        replaybuffer_path,
        required_keys=required_keys,
        dataset_cls=dataset_cls,
        source_role=source_role,
        split=split,
        num_traj=getattr(cfg.dataset.replaybuffer, "max_traj_num", None),
    )


def _build_vla_feature_rnn_expert_dataset(cfg, required_keys=None, expert_path=None):
    return _build_vla_feature_expert_dataset(
        cfg,
        required_keys=required_keys,
        expert_path=expert_path,
        dataset_cls=VLAFeatureSequenceDataset,
    )


def _build_vla_feature_rnn_replaybuffer(cfg, required_keys=None, replaybuffer_path=None):
    return _build_vla_feature_replaybuffer(
        cfg,
        required_keys=required_keys,
        replaybuffer_path=replaybuffer_path,
        dataset_cls=VLAFeatureSequenceDataset,
    )


def _build_vla_feature_mlp_expert_dataset(cfg, required_keys=None, expert_path=None):
    return _build_vla_feature_expert_dataset(
        cfg,
        required_keys=required_keys,
        expert_path=expert_path,
        dataset_cls=VLAFeatureTransitionDataset,
    )


def _build_vla_feature_mlp_replaybuffer(cfg, required_keys=None, replaybuffer_path=None):
    return _build_vla_feature_replaybuffer(
        cfg,
        required_keys=required_keys,
        replaybuffer_path=replaybuffer_path,
        dataset_cls=VLAFeatureTransitionDataset,
    )


def _build_vla_feature_block_rnn_expert_dataset(cfg, required_keys=None, expert_path=None):
    return _build_vla_feature_expert_dataset(
        cfg,
        required_keys=required_keys,
        expert_path=expert_path,
        dataset_cls=VLAFeatureBlockSequenceDataset,
    )


def _build_vla_feature_block_rnn_replaybuffer(cfg, required_keys=None, replaybuffer_path=None):
    return _build_vla_feature_replaybuffer(
        cfg,
        required_keys=required_keys,
        replaybuffer_path=replaybuffer_path,
        dataset_cls=VLAFeatureBlockSequenceDataset,
    )


register_dataset_type("vla_feature_tdqc_rnn", _build_vla_feature_rnn_expert_dataset, _build_vla_feature_rnn_replaybuffer)
register_dataset_type("vla_feature_tdqc_mlp", _build_vla_feature_mlp_expert_dataset, _build_vla_feature_mlp_replaybuffer)
register_dataset_type("vla_feature_rnn", _build_vla_feature_rnn_expert_dataset, _build_vla_feature_rnn_replaybuffer)
register_dataset_type("vla_feature_mlp", _build_vla_feature_mlp_expert_dataset, _build_vla_feature_mlp_replaybuffer)
register_dataset_type("vla_feature_block_rnn", _build_vla_feature_block_rnn_expert_dataset, _build_vla_feature_block_rnn_replaybuffer)
register_dataset_type("vla_feature_cpiql_rnn", _build_vla_feature_block_rnn_expert_dataset, _build_vla_feature_block_rnn_replaybuffer)
register_dataset_type("vla_feature_tdqc_block_rnn", _build_vla_feature_block_rnn_expert_dataset, _build_vla_feature_block_rnn_replaybuffer)

# Backward-compatible aliases. Prefer the explicit MLP/RNN names in new configs.
register_dataset_type("vla_feature", _build_vla_feature_rnn_expert_dataset, _build_vla_feature_rnn_replaybuffer)
register_dataset_type("vla_feature_tdqc", _build_vla_feature_rnn_expert_dataset, _build_vla_feature_rnn_replaybuffer)

__all__ = [
    "VLAFeatureDataset",
    "VLAFeatureBlockSequenceDataset",
    "VLAFeatureSequenceDataset",
    "VLAFeatureTransitionDataset",
]
