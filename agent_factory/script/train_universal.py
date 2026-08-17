import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

from omegaconf import OmegaConf

if __package__ in {None, ""}:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from agent_factory.agents.registry import make_agent
from agent_factory.config.resolution import (
    RESOLVED_CONFIG_FILENAME,
    SIMPLE_CONFIG_FILENAME,
    TRAIN_OVERRIDE_KEYS,
    general_resolve,
    normalize_user_config,
    save_config_snapshots,
)
from agent_factory.data.registry import build_training_bundle


DEFAULT_FINETUNE_CONFIG_PATH = os.path.join(
    "run_results",
    "piper_dual_merged_cpiql_dac",
    "finetune_config.yaml",
)
DEFAULT_MODEL_CONFIG_PATH = os.path.join(
    "run_results",
    "piper_dual_merged_cpiql_dac",
    "model_config.yaml",
)


def _load_raw_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    loaded = OmegaConf.load(config_path)
    value = OmegaConf.to_container(loaded, resolve=False)
    return value if isinstance(value, dict) else {}


def _path_exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def _candidate_config_paths_from_checkpoint(ckpt_path: Optional[str]) -> Tuple[str, ...]:
    if not ckpt_path:
        return ()
    checkpoint_dir = os.path.dirname(os.path.abspath(ckpt_path))
    parent_dir = os.path.dirname(checkpoint_dir)
    candidates = []
    for base_dir in (checkpoint_dir, parent_dir):
        for filename in (RESOLVED_CONFIG_FILENAME, "finetune_config.yaml", "model_config.yaml"):
            candidate = os.path.join(base_dir, filename)
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def resolve_config_path(
    config_path: Optional[str],
    finetune: bool = False,
    ckpt_path: Optional[str] = None,
) -> Optional[str]:
    if config_path:
        return config_path
    if not finetune:
        return None

    for candidate in _candidate_config_paths_from_checkpoint(ckpt_path):
        if _path_exists(candidate):
            return candidate
    for candidate in (DEFAULT_FINETUNE_CONFIG_PATH, DEFAULT_MODEL_CONFIG_PATH):
        if _path_exists(candidate):
            return candidate
    raise ValueError(
        "Finetune mode requires a base config. Pass --config explicitly, or place "
        f"`finetune_config.yaml` / `model_config.yaml` under the checkpoint directory or `{os.path.dirname(DEFAULT_MODEL_CONFIG_PATH)}`."
    )


def _resolve_finetune_from_config(config_path: Optional[str]) -> Optional[bool]:
    raw_cfg = _load_raw_config(config_path)
    user_cfg = normalize_user_config(raw_cfg)
    train_cfg = user_cfg.get("train", {}) if isinstance(user_cfg, dict) else {}
    if not isinstance(train_cfg, dict) or "finetune" not in train_cfg:
        return None
    return bool(train_cfg.get("finetune"))


def load_training_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    finetune: Optional[bool] = None,
) -> Tuple[Any, Dict[str, Any]]:
    return general_resolve(
        override_config=cli_overrides or {},
        file_config=_load_raw_config(config_path),
        finetune=finetune,
    )


def train_universal(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    finetune: Optional[bool] = None,
) -> Dict[str, Any]:
    cfg, runtime_spec = load_training_config(
        config_path=config_path,
        cli_overrides=overrides,
        finetune=finetune,
    )

    save_root = str(cfg.train.save_root)
    exp_name = str(cfg.train.exp_name or "piper_dual_merged_cpiql_dac")
    run_root = os.path.join(save_root, exp_name)
    snapshot_paths = save_config_snapshots(cfg, runtime_spec["simple_cfg"], run_root)

    if dry_run:
        return {
            "cfg": cfg,
            "runtime_spec": runtime_spec,
            "run_root": run_root,
            "snapshot_paths": snapshot_paths,
            "simple_config_filename": SIMPLE_CONFIG_FILENAME,
            "resolved_config_filename": RESOLVED_CONFIG_FILENAME,
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
        "snapshot_paths": snapshot_paths,
        "simple_config_filename": SIMPLE_CONFIG_FILENAME,
        "resolved_config_filename": RESOLVED_CONFIG_FILENAME,
    }


def main():
    parser = argparse.ArgumentParser(description="Universal training entrypoint.")
    parser.add_argument("--config", type=str, default=None, help="Path to a model config YAML.")
    finetune_group = parser.add_mutually_exclusive_group()
    finetune_group.add_argument("--finetune", "--fintune", action="store_true", dest="finetune")
    finetune_group.add_argument("--no-finetune", action="store_false", dest="finetune")
    parser.set_defaults(finetune=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-object", "--train_object", type=str, default=None, choices=["critic", "actor", "critic_then_actor"], help="Which object to train.")
    parser.add_argument("--dataset-key", "--dataset_key", type=str, default=None, choices=["expert_dataset", "replaybuffer", "expert_dataset+replaybuffer"], help="Which dataset view to train on.")
    parser.add_argument("--critic-iters", "--critic_iters", type=int, default=None)
    parser.add_argument("--actor-iters", "--actor_iters", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=None)
    parser.add_argument("--save-interval", "--save_interval", type=int, default=None)
    parser.add_argument("--save-root", "--save_root", type=str, default=None)
    parser.add_argument("--exp-name", "--exp_name", type=str, default=None)
    parser.add_argument("--ckpt-path", "--ckpt_path", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_finetune = _resolve_finetune_from_config(args.config) if args.config else None
    effective_finetune = config_finetune if args.finetune is None else bool(args.finetune)

    config_path = resolve_config_path(
        config_path=args.config,
        finetune=bool(effective_finetune),
        ckpt_path=args.ckpt_path,
    )

    overrides = {
        "device": args.device,
        "train_object": args.train_object,
        "dataset_key": args.dataset_key,
        "critic_iters": args.critic_iters,
        "actor_iters": args.actor_iters,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "save_interval": args.save_interval,
        "save_root": args.save_root,
        "exp_name": args.exp_name,
        "ckpt_path": args.ckpt_path,
    }
    if args.finetune is not None:
        overrides["finetune"] = bool(args.finetune)
    overrides = {key: value for key, value in overrides.items() if value is not None}
    unknown = set(overrides) - TRAIN_OVERRIDE_KEYS
    if unknown:
        raise ValueError(f"Unsupported train override(s): {sorted(unknown)}")

    result = train_universal(
        config_path=config_path,
        overrides=overrides,
        dry_run=args.dry_run,
        finetune=effective_finetune,
    )
    if args.dry_run:
        print(f"[DryRun] Saved simple config snapshot to {result['snapshot_paths']['simple']}")
        print(f"[DryRun] Saved resolved config snapshot to {result['snapshot_paths']['resolved']}")


if __name__ == "__main__":
    main()
