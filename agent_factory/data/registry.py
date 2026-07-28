from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from torch.utils.data import ConcatDataset


@dataclass(frozen=True)
class DatasetTypeSpec:
    name: str
    build_expert: Callable[..., Any]
    build_replaybuffer: Optional[Callable[..., Any]] = None


_DATASET_REGISTRY: Dict[str, DatasetTypeSpec] = {}
_BUILTINS_LOADED = False


def _normalize_dataset_type(dataset_type: str) -> str:
    return str(dataset_type or "").strip().lower()


def register_dataset_type(
    dataset_type: str,
    build_expert: Callable[..., Any],
    build_replaybuffer: Optional[Callable[..., Any]] = None,
) -> DatasetTypeSpec:
    normalized = _normalize_dataset_type(dataset_type)
    if not normalized:
        raise ValueError("dataset_type must be a non-empty string.")
    spec = DatasetTypeSpec(
        name=normalized,
        build_expert=build_expert,
        build_replaybuffer=build_replaybuffer,
    )
    _DATASET_REGISTRY[normalized] = spec
    return spec


def ensure_builtin_dataset_types_loaded() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from agent_factory.data.impl import cpiql as _cpiql  # noqa: F401
    from agent_factory.data.impl import diffusion_itqc as _diffusion_itqc  # noqa: F401

    _BUILTINS_LOADED = True


def get_dataset_type_spec(dataset_type: str) -> DatasetTypeSpec:
    ensure_builtin_dataset_types_loaded()
    normalized = _normalize_dataset_type(dataset_type)
    if normalized not in _DATASET_REGISTRY:
        available = sorted(_DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset_type '{dataset_type}'. Available: {available}")
    return _DATASET_REGISTRY[normalized]


def infer_dataset_type_from_agent_type(agent_type: str) -> str:
    agent_name = str(agent_type or "").strip().lower()
    if "itqc" in agent_name:
        return "diffusion_itqc"
    if "cpiql" in agent_name:
        return "cpiql"
    return "cpiql"


def _resolve_replaybuffer_path(cfg: Any) -> str:
    replaybuffer_cfg = getattr(cfg.dataset, "replaybuffer", None)
    replay_cfg = getattr(cfg.dataset, "replay", None)
    for node in (replaybuffer_cfg, replay_cfg):
        if node is None:
            continue
        for attr in ("replaybuffer_path", "folder_path"):
            value = getattr(node, attr, "")
            if value:
                return str(value)
    return ""


def build_training_bundle(
    cfg: Any,
    required_keys: Optional[Sequence[str]] = None,
):
    ensure_builtin_dataset_types_loaded()

    dataset_type = getattr(cfg.dataset, "dataset_type", "") or infer_dataset_type_from_agent_type(
        getattr(cfg, "agent_type", "")
    )
    spec = get_dataset_type_spec(dataset_type)

    expert_path = str(getattr(cfg.dataset.expert, "demo_path", ""))
    if not expert_path:
        raise ValueError("cfg.dataset.expert.demo_path is required.")

    expert_dataset = spec.build_expert(
        cfg=cfg,
        required_keys=required_keys,
        expert_path=expert_path,
    )

    bundle = {
        "expert_dataset": expert_dataset,
        "offline": expert_dataset,
    }

    dataset_key = str(getattr(cfg.train, "dataset_key", "expert_dataset"))
    valid_dataset_keys = {"expert_dataset", "replaybuffer", "expert_dataset+replaybuffer"}
    if dataset_key not in valid_dataset_keys:
        raise ValueError(
            f"Unsupported train.dataset_key '{dataset_key}'. Expected one of {sorted(valid_dataset_keys)}."
        )

    replaybuffer_path = _resolve_replaybuffer_path(cfg)
    if dataset_key in {"replaybuffer", "expert_dataset+replaybuffer"} and not replaybuffer_path:
        raise ValueError(
            f"train.dataset_key='{dataset_key}' requires dataset.replaybuffer.folder_path or replaybuffer_path."
        )
    if not replaybuffer_path:
        return bundle

    if spec.build_replaybuffer is None:
        raise ValueError(f"dataset_type '{dataset_type}' does not provide a replaybuffer builder.")

    replay_dataset = spec.build_replaybuffer(
        cfg=cfg,
        required_keys=required_keys,
        replaybuffer_path=replaybuffer_path,
    )
    bundle["replaybuffer"] = replay_dataset
    bundle["online"] = replay_dataset

    if dataset_key == "expert_dataset+replaybuffer":
        bundle["expert_dataset+replaybuffer"] = ConcatDataset([expert_dataset, replay_dataset])

    return bundle


__all__ = [
    "DatasetTypeSpec",
    "build_training_bundle",
    "ensure_builtin_dataset_types_loaded",
    "get_dataset_type_spec",
    "infer_dataset_type_from_agent_type",
    "register_dataset_type",
]
