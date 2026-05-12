import argparse
import copy
import json
import os
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
from omegaconf import OmegaConf

from agent_factory.agents.registry import get_default_config, make_agent
from agent_factory.data.registry import build_training_bundle, infer_dataset_type_from_agent_type


TRAIN_CLI_KEYS = {
    "device",
    "train_object",
    "dataset_mode",
    "dataset_key",
    "critic_iters",
    "actor_iters",
    "batch_size",
    "num_workers",
    "save_interval",
    "save_root",
    "exp_name",
    "critic_ckpt_path",
}


def _read_json_dataset(dataset) -> Dict[str, Any]:
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return json.loads(value)


def _load_raw_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    loaded = OmegaConf.load(config_path)
    return OmegaConf.to_container(loaded, resolve=False)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ensure_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def _move_if_present(src: Dict[str, Any], src_key: str, dst: Dict[str, Any], dst_key: str) -> None:
    if src_key in src and dst_key not in dst:
        dst[dst_key] = src[src_key]


def _normalize_user_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert older flat convenience keys into the structured config shape.

    The resulting dict is suitable for OmegaConf.merge(default_cfg, user_cfg),
    so user-provided actor/critic/env/dataset parameters override defaults as a
    whole instead of being filtered through a hand-written allowlist.
    """
    model_cfg = copy.deepcopy(raw_cfg.get("model") if isinstance(raw_cfg.get("model"), dict) else raw_cfg)
    if not isinstance(model_cfg, dict):
        return {}

    train_cfg = _ensure_dict(model_cfg, "train")
    dataset_cfg = _ensure_dict(model_cfg, "dataset")
    env_cfg = _ensure_dict(model_cfg, "env")
    expert_cfg = _ensure_dict(dataset_cfg, "expert")
    replaybuffer_cfg = _ensure_dict(dataset_cfg, "replaybuffer")

    legacy_pipeline = _as_dict(raw_cfg.get("pipeline"))
    for key, value in legacy_pipeline.items():
        train_cfg.setdefault(key, value)

    for key in TRAIN_CLI_KEYS:
        _move_if_present(model_cfg, key, train_cfg, key)

    _move_if_present(model_cfg, "device", train_cfg, "device")
    _move_if_present(model_cfg, "dataset_type", dataset_cfg, "dataset_type")
    _move_if_present(model_cfg, "expert_dataset_path", expert_cfg, "demo_path")
    _move_if_present(model_cfg, "expert_num_traj", expert_cfg, "num_traj")
    _move_if_present(model_cfg, "expert_format", expert_cfg, "format")
    _move_if_present(model_cfg, "replaybuffer_path", replaybuffer_cfg, "replaybuffer_path")
    _move_if_present(model_cfg, "replaybuffer_path", replaybuffer_cfg, "folder_path")
    _move_if_present(model_cfg, "replay_max_traj_num", replaybuffer_cfg, "max_traj_num")
    _move_if_present(model_cfg, "include_rgb", dataset_cfg, "include_rgb")
    _move_if_present(model_cfg, "include_depth", dataset_cfg, "include_depth")

    for key in (
        "library",
        "env_id",
        "env_config_path",
        "env_control_mode",
        "controller_backend",
        "control_mode",
        "obs_mode",
        "max_episode_steps",
        "action_dim",
        "proprio_dim",
        "num_cameras",
    ):
        _move_if_present(model_cfg, key, env_cfg, key)

    for key in (
        "pipeline",
        "model",
        "device",
        "dataset_type",
        "expert_dataset_path",
        "expert_num_traj",
        "expert_format",
        "replaybuffer_path",
        "replay_max_traj_num",
        "include_rgb",
        "include_depth",
        "library",
        "env_id",
        "env_config_path",
        "env_control_mode",
        "controller_backend",
        "control_mode",
        "obs_mode",
        "max_episode_steps",
        "action_dim",
        "proprio_dim",
        "num_cameras",
    ):
        model_cfg.pop(key, None)
    for key in TRAIN_CLI_KEYS:
        model_cfg.pop(key, None)

    dataset_cfg.pop("replay", None)
    return model_cfg


def _infer_h5_stats(h5_path: str) -> Dict[str, int]:
    with h5py.File(h5_path, "r") as f:
        meta = _read_json_dataset(f["meta"]["env_meta"])

    obs_meta = meta.get("obs", {}) or {}
    state_meta = obs_meta.get("state", {}) or {}
    rgb_meta = obs_meta.get("rgb", {}) or {}
    action_meta = meta.get("action", {}) or {}

    return {
        "action_dim": int(sum(int(np.prod(shape)) for shape in action_meta.values())),
        "proprio_dim": int(sum(int(np.prod(shape)) for shape in state_meta.values())),
        "num_cameras": int(len(rgb_meta) or 1),
    }


def _has_path(cfg: Dict[str, Any], *path: str) -> bool:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def _fill_inferred_env_stats(cfg: Any, user_cfg: Dict[str, Any]) -> None:
    demo_path = str(cfg.dataset.expert.demo_path)
    if not demo_path or not os.path.exists(demo_path):
        return
    stats = _infer_h5_stats(demo_path)
    if not _has_path(user_cfg, "env", "action_dim"):
        cfg.env.action_dim = int(stats["action_dim"])
    if not _has_path(user_cfg, "env", "proprio_dim"):
        cfg.env.proprio_dim = int(stats["proprio_dim"])
    if not _has_path(user_cfg, "env", "num_cameras"):
        cfg.env.num_cameras = int(stats["num_cameras"])


def _sync_derived_fields(cfg: Any) -> None:
    cfg.device = str(getattr(cfg.train, "device", getattr(cfg, "device", "cpu")))
    cfg.actor.action_dim = int(cfg.env.action_dim)
    if getattr(cfg.actor, "encoder", None) is not None:
        cfg.actor.encoder.proprio_dim = int(cfg.env.proprio_dim)
    if getattr(cfg.critic, "encoder", None) is not None:
        cfg.critic.encoder.proprio_dim = int(cfg.env.proprio_dim)
    if not getattr(cfg.dataset, "dataset_type", ""):
        cfg.dataset.dataset_type = infer_dataset_type_from_agent_type(cfg.agent_type)


def _apply_train_cli_overrides(cfg: Any, overrides: Optional[Dict[str, Any]]) -> Any:
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in TRAIN_CLI_KEYS:
            raise ValueError(f"CLI override '{key}' is not allowed; only config.train fields may be overridden.")
        current = getattr(cfg.train, key)
        if isinstance(current, bool):
            value = bool(value)
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        else:
            value = str(value)
        setattr(cfg.train, key, value)
    _sync_derived_fields(cfg)
    return cfg


def load_training_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    raw_cfg = _load_raw_config(config_path)
    user_cfg = _normalize_user_config(raw_cfg)
    agent_type = str(user_cfg.get("agent_type") or "Diffusion_CPIQL_DAC")

    default_cfg = get_default_config(agent_type)
    cfg = OmegaConf.merge(default_cfg, OmegaConf.create(user_cfg))
    _fill_inferred_env_stats(cfg, user_cfg)
    _sync_derived_fields(cfg)
    cfg = _apply_train_cli_overrides(cfg, cli_overrides)

    runtime_spec = {
        "agent_type": str(cfg.agent_type),
        "user_cfg": user_cfg,
    }
    return cfg, runtime_spec


def _save_model_config_snapshot(cfg: Any, save_dir: str, filename: str = "model_config.yaml") -> None:
    os.makedirs(save_dir, exist_ok=True)
    snapshot = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(snapshot, dict):
        snapshot.pop("device", None)
        dataset_cfg = snapshot.get("dataset")
        if isinstance(dataset_cfg, dict):
            dataset_cfg.pop("replay", None)
    OmegaConf.save(OmegaConf.create(snapshot), os.path.join(save_dir, filename), resolve=True)


def train_universal(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    cfg, runtime_spec = load_training_config(config_path=config_path, cli_overrides=overrides)

    save_root = str(cfg.train.save_root)
    exp_name = str(cfg.train.exp_name or "piper_dual_merged_cpiql_dac")
    run_root = os.path.join(save_root, exp_name)
    critic_dir = os.path.join(run_root, "critic")
    actor_dir = os.path.join(run_root, "actor")
    os.makedirs(run_root, exist_ok=True)
    os.makedirs(critic_dir, exist_ok=True)
    os.makedirs(actor_dir, exist_ok=True)

    _save_model_config_snapshot(cfg, run_root, filename="model_config.yaml")
    _save_model_config_snapshot(cfg, critic_dir, filename="model_config.yaml")
    _save_model_config_snapshot(cfg, actor_dir, filename="model_config.yaml")

    if dry_run:
        return {
            "cfg": cfg,
            "runtime_spec": runtime_spec,
            "run_root": run_root,
            "critic_dir": critic_dir,
            "actor_dir": actor_dir,
        }

    agent = make_agent(runtime_spec["agent_type"], cfg)
    dataset_bundle = build_training_bundle(cfg, required_keys=getattr(agent, "required_keys", None))
    agent.start_train(dataset_bundle)

    return {
        "cfg": cfg,
        "agent": agent,
        "dataset_bundle": dataset_bundle,
        "runtime_spec": runtime_spec,
        "run_root": run_root,
        "critic_dir": critic_dir,
        "actor_dir": actor_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Universal training entrypoint.")
    parser.add_argument("--config", type=str, default=None, help="Path to a detailed model_config.yaml.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-object", type=str, default=None, choices=["critic", "actor", "critic_then_actor"], help="Which object to train.")
    parser.add_argument("--dataset-mode", type=str, default=None, choices=["expert_dataset", "expert_plus_replaybuffer", "replaybuffer"], help="Which dataset bundle to use.")
    parser.add_argument("--dataset-key", type=str, default=None, help="Which key to select from the dataset bundle.")
    parser.add_argument("--critic-iters", type=int, default=None)
    parser.add_argument("--actor-iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--critic-ckpt-path", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    overrides = {
        "device": args.device,
        "train_object": args.train_object,
        "dataset_mode": args.dataset_mode,
        "dataset_key": args.dataset_key,
        "critic_iters": args.critic_iters,
        "actor_iters": args.actor_iters,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "save_interval": args.save_interval,
        "save_root": args.save_root,
        "exp_name": args.exp_name,
        "critic_ckpt_path": args.critic_ckpt_path,
    }
    result = train_universal(config_path=args.config, overrides=overrides, dry_run=args.dry_run)
    if args.dry_run:
        print(f"[DryRun] Saved config snapshot to {result['run_root']}")


if __name__ == "__main__":
    main()
