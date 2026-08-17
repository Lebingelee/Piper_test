import copy
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from omegaconf import OmegaConf

from agent_factory.modules.registry import get_module_resolve_rules


logger = logging.getLogger(__name__)

TRAIN_OVERRIDE_KEYS = {
    "device",
    "train_object",
    "dataset_key",
    "critic_iters",
    "actor_iters",
    "batch_size",
    "num_workers",
    "save_interval",
    "save_root",
    "exp_name",
    "ckpt_path",
    "finetune",
}
SIMPLE_CONFIG_FILENAME = "simple.yaml"
RESOLVED_CONFIG_FILENAME = "resolved.yaml"


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        out = OmegaConf.to_container(value, resolve=False)
        return out if isinstance(out, dict) else {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if hasattr(value, "__dict__"):
        return copy.deepcopy(vars(value))
    return {}


def _path_join(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else str(key)


def _select(cfg: Any, path: str, default: Any = None) -> Any:
    try:
        return OmegaConf.select(cfg, path, default=default)
    except Exception:
        return default


def _has_path(cfg: Any, path: str) -> bool:
    if cfg is None or not path:
        return False
    cur = OmegaConf.to_container(cfg, resolve=False) if OmegaConf.is_config(cfg) else cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _set_path(cfg: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for part in parts[:-1]:
        next_val = _select(cur, part, default=None)
        if next_val is None:
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _delete_path(cfg: Any, path: str) -> None:
    parts = path.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = _select(cur, part, default=None)
        if cur is None:
            return
    try:
        del cur[parts[-1]]
    except Exception:
        return


def _values_equal(left: Any, right: Any) -> bool:
    if OmegaConf.is_config(left):
        left = OmegaConf.to_container(left, resolve=True)
    if OmegaConf.is_config(right):
        right = OmegaConf.to_container(right, resolve=True)
    return left == right


def _iter_dict_nodes(node: Any, prefix: str = "") -> Iterable[Tuple[str, Dict[str, Any]]]:
    if not isinstance(node, dict):
        return
    yield prefix, node
    for key, value in list(node.items()):
        if isinstance(value, dict):
            yield from _iter_dict_nodes(value, _path_join(prefix, str(key)))
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    yield from _iter_dict_nodes(item, _path_join(prefix, str(idx)))


def resolve_module_config(
    module_cfg: Any,
    global_cfg: Any,
    *,
    module_path: str = "",
    explicit_global_cfg: Any = None,
    strict: bool = False,
) -> Any:
    """
    Materialize a single module config using the module's own RESOLVE_RULES.

    The resolver fills values from declared context sources when the user did
    not explicitly set the field. Explicit user values win by default; conflicts
    are warnings unless strict=True.
    """

    rules = get_module_resolve_rules(module_cfg)
    if not rules:
        return module_cfg

    out = OmegaConf.create(_to_plain_dict(module_cfg))
    for field_name, rule in rules.items():
        source = str(rule.get("source", "")).strip()
        required = bool(rule.get("required", False))
        has_default = "default" in rule
        context_value = _select(global_cfg, source, default=None) if source else None
        if context_value is None and has_default:
            context_value = copy.deepcopy(rule["default"])

        field_path = _path_join(module_path, str(field_name)) if module_path else str(field_name)
        user_set = _has_path(explicit_global_cfg, field_path)
        current_value = _select(out, str(field_name), default=None)

        if context_value is None:
            if required:
                raise ValueError(
                    f"Missing required resolve source '{source}' for module '{module_path or '<module>'}' field '{field_name}'."
                )
            continue

        if user_set:
            if not _values_equal(current_value, context_value):
                message = (
                    "[ConfigResolution] conflict "
                    f"module={module_path or '<module>'} field={field_name} "
                    f"user_value={current_value!r} source={source} "
                    f"context_value={context_value!r} selected_value={current_value!r}"
                )
                if strict:
                    raise ValueError(message)
                logger.warning(message)
            continue

        _set_path(out, str(field_name), context_value)

    return out


def materialize_global_config(
    cfg: Any,
    *,
    explicit_cfg: Any = None,
    strict: bool = False,
) -> Tuple[Any, List[str]]:
    """
    Resolve all registered module configs nested inside a global config.

    Traversal is registry/rule driven: any dict with a registered module type and
    RESOLVE_RULES is materialized in-place, without training-script type checks.
    """

    materialized = OmegaConf.create(_to_plain_dict(cfg))
    paths: List[str] = []

    # Iterate on a snapshot of paths because each resolved node is written back.
    plain = OmegaConf.to_container(materialized, resolve=False)
    for module_path, node in list(_iter_dict_nodes(plain)):
        if not isinstance(node, dict) or "type" not in node:
            continue
        rules = get_module_resolve_rules(node)
        if not rules:
            continue
        resolved = resolve_module_config(
            node,
            materialized,
            module_path=module_path,
            explicit_global_cfg=explicit_cfg,
            strict=strict,
        )
        if module_path:
            _set_path(materialized, module_path, resolved)
        else:
            materialized = resolved
        paths.append(module_path)

    return materialized, paths


def _rule_target_path(config_key: str, field_name: str, rule: Dict[str, Any]) -> str:
    target = str(rule.get("path") or field_name).strip()
    if not target:
        target = str(field_name)
    if target.startswith(f"{config_key}.") or target in {"env", "dataset", "train", "runner", "agent_sp"}:
        return target
    return _path_join(config_key, target)


def _iter_mixin_rules(agent_type: str, attr_name: str):
    from agent_factory.agents.registry import iter_agent_config_mixins

    seen = set()
    for mixin_cls in iter_agent_config_mixins(agent_type):
        config_key = getattr(mixin_cls, "CONFIG_KEY", "")
        rules = mixin_cls.__dict__.get(attr_name, None)
        if not isinstance(rules, dict):
            continue
        for field_name, rule in rules.items():
            if not isinstance(rule, dict):
                continue
            target_path = _rule_target_path(str(config_key), str(field_name), rule)
            rule_id = (attr_name, target_path, str(rule.get("source", "")))
            if rule_id in seen:
                continue
            seen.add(rule_id)
            yield mixin_cls, target_path, str(field_name), rule


def resolve_mixin_configs(
    cfg: Any,
    *,
    explicit_cfg: Any = None,
    agent_type: str,
    strict: bool = False,
) -> Tuple[Any, List[str]]:
    """
    Materialize mixin-declared derived fields for the registered agent.

    Mixin discovery is based on the registered agent class MRO, not on resolver
    guesses or agent-type hard-coding.
    """

    materialized = OmegaConf.create(_to_plain_dict(cfg))
    resolved_paths: List[str] = []

    for mixin_cls, target_path, _field_name, rule in _iter_mixin_rules(agent_type, "RESOLVE_RULES"):
        source = str(rule.get("source", "")).strip()
        required = bool(rule.get("required", False))
        has_default = "default" in rule
        context_value = _select(materialized, source, default=None) if source else None
        if context_value is None and has_default:
            context_value = copy.deepcopy(rule["default"])

        user_set = _has_path(explicit_cfg, target_path)
        current_value = _select(materialized, target_path, default=None)

        if context_value is None:
            if required:
                raise ValueError(
                    f"Missing required resolve source '{source}' for mixin '{mixin_cls.__name__}' field '{target_path}'."
                )
            continue

        if user_set:
            if not _values_equal(current_value, context_value):
                message = (
                    "[ConfigResolution] mixin conflict "
                    f"mixin={mixin_cls.__name__} field={target_path} "
                    f"user_value={current_value!r} source={source} "
                    f"context_value={context_value!r} selected_value={current_value!r}"
                )
                if strict:
                    raise ValueError(message)
                logger.warning(message)
            continue

        _set_path(materialized, target_path, context_value)
        resolved_paths.append(target_path)

    return materialized, resolved_paths


def validate_mixin_configs(
    cfg: Any,
    *,
    agent_type: str,
    strict: bool = False,
) -> List[str]:
    """Run mixin-declared protocol checks without changing config values."""

    warnings: List[str] = []
    for mixin_cls, target_path, _field_name, rule in _iter_mixin_rules(agent_type, "VALIDATION_RULES"):
        source = str(rule.get("source", "")).strip()
        context_value = _select(cfg, source, default=None) if source else None
        current_value = _select(cfg, target_path, default=None)
        if context_value is None or current_value is None or _values_equal(current_value, context_value):
            continue

        message = (
            "[ConfigResolution] mixin validation conflict "
            f"mixin={mixin_cls.__name__} field={target_path} "
            f"value={current_value!r} source={source} source_value={context_value!r}"
        )
        if strict or str(rule.get("on_conflict", "warn")).strip().lower() == "error":
            raise ValueError(message)
        logger.warning(message)
        warnings.append(message)

    return warnings


def validate_global_config(cfg: Any, *, strict: bool = False) -> List[str]:
    """
    Run cross-cutting config checks that are not owned by a single module/mixin.
    """

    warnings: List[str] = []
    mode = str(_select(cfg, "env.obs_mode", default="rgb") or "rgb").strip().lower()
    include_rgb = bool(_select(cfg, "dataset.include_rgb", default=True))
    include_depth = bool(_select(cfg, "dataset.include_depth", default=False))

    errors = []
    if mode == "rgb" and not include_rgb:
        errors.append("env.obs_mode='rgb' requires dataset.include_rgb=True")
    if mode == "depth" and not include_depth:
        errors.append("env.obs_mode='depth' requires dataset.include_depth=True")
    if mode == "rgbd" and not (include_rgb and include_depth):
        errors.append("env.obs_mode='rgbd' requires dataset.include_rgb=True and dataset.include_depth=True")

    if errors:
        message = "[ConfigResolution] global validation conflict: " + "; ".join(errors)
        raise ValueError(message)

    if mode == "state" and (include_rgb or include_depth):
        message = (
            "[ConfigResolution] global validation warning: "
            "env.obs_mode='state' but dataset includes visual keys."
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)
        warnings.append(message)

    return warnings


def simplify_global_config(materialized_cfg: Any, simple_cfg: Any) -> Any:
    """
    Remove resolver-derived module fields from a user-facing simple config.

    If a field covered by RESOLVE_RULES matches its context source, it is omitted
    from the simple snapshot. Conflicting explicit user values are preserved.
    """

    simple = OmegaConf.create(_to_plain_dict(simple_cfg))
    materialized_plain = OmegaConf.to_container(materialized_cfg, resolve=False)
    for module_path, node in list(_iter_dict_nodes(materialized_plain)):
        if not isinstance(node, dict) or "type" not in node:
            continue
        rules = get_module_resolve_rules(node)
        if not rules:
            continue
        for field_name, rule in rules.items():
            field_path = _path_join(module_path, str(field_name)) if module_path else str(field_name)
            if not _has_path(simple, field_path):
                continue
            source = str(rule.get("source", "")).strip()
            context_value = _select(materialized_cfg, source, default=None) if source else None
            if context_value is None and "default" in rule:
                context_value = copy.deepcopy(rule["default"])
            simple_value = _select(simple, field_path, default=None)
            if context_value is not None and _values_equal(simple_value, context_value):
                _delete_path(simple, field_path)

    return simple


def simplify_mixin_config(materialized_cfg: Any, simple_cfg: Any, *, agent_type: str) -> Any:
    """
    Remove mixin-derived fields from simple config when they match context.

    Explicit conflicts are preserved for human inspection.
    """

    simple = OmegaConf.create(_to_plain_dict(simple_cfg))
    for _mixin_cls, target_path, _field_name, rule in _iter_mixin_rules(agent_type, "RESOLVE_RULES"):
        if not _has_path(simple, target_path):
            continue
        source = str(rule.get("source", "")).strip()
        context_value = _select(materialized_cfg, source, default=None) if source else None
        if context_value is None and "default" in rule:
            context_value = copy.deepcopy(rule["default"])
        simple_value = _select(simple, target_path, default=None)
        if context_value is not None and _values_equal(simple_value, context_value):
            _delete_path(simple, target_path)
    return simple



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


def _dict_has_path(d: Dict[str, Any], *path: str) -> bool:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def normalize_user_config(raw_cfg: Any) -> Dict[str, Any]:
    """
    Convert older flat convenience keys into the structured config shape.

    This preserves file/user intent before defaults and resolver-derived fields
    are materialized.
    """

    raw_dict = _to_plain_dict(raw_cfg)
    model_cfg = copy.deepcopy(raw_dict.get("model") if isinstance(raw_dict.get("model"), dict) else raw_dict)
    if not isinstance(model_cfg, dict):
        return {}

    train_cfg = _ensure_dict(model_cfg, "train")
    dataset_cfg = _ensure_dict(model_cfg, "dataset")
    env_cfg = _ensure_dict(model_cfg, "env")
    expert_cfg = _ensure_dict(dataset_cfg, "expert")
    replaybuffer_cfg = _ensure_dict(dataset_cfg, "replaybuffer")

    if "critic_ckpt_path" in train_cfg and "ckpt_path" not in train_cfg:
        train_cfg["ckpt_path"] = train_cfg["critic_ckpt_path"]
    train_cfg.pop("critic_ckpt_path", None)
    train_cfg.pop("n_epochs", None)
    legacy_dataset_mode = train_cfg.pop("dataset_mode", None)
    if legacy_dataset_mode and "dataset_key" not in train_cfg:
        dataset_mode_to_key = {
            "expert_dataset": "expert_dataset",
            "replaybuffer": "replaybuffer",
            "expert_plus_replaybuffer": "expert_dataset+replaybuffer",
        }
        train_cfg["dataset_key"] = dataset_mode_to_key.get(str(legacy_dataset_mode), str(legacy_dataset_mode))

    legacy_pipeline = _as_dict(raw_dict.get("pipeline"))
    for key, value in legacy_pipeline.items():
        train_cfg.setdefault(key, value)

    for key in TRAIN_OVERRIDE_KEYS:
        _move_if_present(model_cfg, key, train_cfg, key)
    _move_if_present(model_cfg, "critic_ckpt_path", train_cfg, "ckpt_path")

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
        "critic_ckpt_path",
        "n_epochs",
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
    for key in TRAIN_OVERRIDE_KEYS:
        model_cfg.pop(key, None)

    dataset_cfg.pop("replay", None)
    return model_cfg


def normalize_train_overrides(override_config: Any) -> Dict[str, Any]:
    override_dict = _to_plain_dict(override_config)
    train_cfg = override_dict.get("train") if isinstance(override_dict.get("train"), dict) else override_dict
    out = {}
    for key, value in train_cfg.items():
        if value is None:
            continue
        if key not in TRAIN_OVERRIDE_KEYS:
            raise ValueError(f"Override '{key}' is not allowed; only config.train fields may be overridden.")
        out[key] = value
    return out


def _merge_train_overrides_into_simple_config(simple_cfg: Dict[str, Any], train_overrides: Dict[str, Any]) -> Dict[str, Any]:
    simple_cfg = copy.deepcopy(simple_cfg)
    train_cfg = _ensure_dict(simple_cfg, "train")
    for key, value in train_overrides.items():
        train_cfg[key] = value
    return simple_cfg


def _read_json_dataset(dataset) -> Dict[str, Any]:
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    else:
        try:
            if value.shape == ():
                value = value.item()
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
        except AttributeError:
            pass
    return json.loads(value)


def _infer_h5_stats(h5_path: str) -> Dict[str, int]:
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        if "meta" in f and "env_meta" in f["meta"]:
            meta = _read_json_dataset(f["meta"]["env_meta"])
        else:
            traj_keys = sorted(key for key in f.keys() if key.startswith("traj_"))
            if not traj_keys:
                raise KeyError(f"No root meta/env_meta or traj_*/meta/env_meta found in {h5_path}.")
            traj = f[traj_keys[0]]
            if "meta" not in traj or "env_meta" not in traj["meta"]:
                raise KeyError(f"Trajectory {traj_keys[0]} is missing meta/env_meta in {h5_path}.")
            meta = _read_json_dataset(traj["meta"]["env_meta"])

    obs_meta = meta.get("obs", {}) or {}
    state_meta = obs_meta.get("state", {}) or {}
    rgb_meta = obs_meta.get("rgb", {}) or {}
    action_meta = meta.get("action", {}) or {}

    return {
        "action_dim": int(sum(int(np.prod(shape)) for shape in action_meta.values())),
        "proprio_dim": int(sum(int(np.prod(shape)) for shape in state_meta.values())),
        "num_cameras": int(len(rgb_meta) or 1),
    }


def _fill_inferred_env_stats(cfg: Any, user_cfg: Dict[str, Any]) -> None:
    dataset_type = str(getattr(cfg.dataset, "dataset_type", "") or "").strip().lower()
    if dataset_type in {"tdqc", "tdqc_feature"}:
        return
    demo_path = str(cfg.dataset.expert.demo_path)
    if not demo_path or not os.path.exists(demo_path):
        return
    stats = _infer_h5_stats(demo_path)
    if not _dict_has_path(user_cfg, "env", "action_dim"):
        cfg.env.action_dim = int(stats["action_dim"])
    if not _dict_has_path(user_cfg, "env", "proprio_dim"):
        cfg.env.proprio_dim = int(stats["proprio_dim"])
    if not _dict_has_path(user_cfg, "env", "num_cameras"):
        cfg.env.num_cameras = int(stats["num_cameras"])


def _sync_derived_fields(cfg: Any) -> None:
    from agent_factory.agents.registry import infer_default_exp_name_from_agent_type
    from agent_factory.data.registry import infer_dataset_type_from_agent_type

    cfg.device = str(getattr(cfg.train, "device", getattr(cfg, "device", "cpu")))
    if getattr(cfg, "actor", None) is not None:
        cfg.actor.action_dim = int(cfg.env.action_dim)
    if not getattr(cfg.dataset, "dataset_type", ""):
        cfg.dataset.dataset_type = infer_dataset_type_from_agent_type(cfg.agent_type)
    if str(getattr(cfg.dataset, "dataset_type", "") or "").strip().lower() in {"tdqc", "tdqc_feature"}:
        if not str(getattr(cfg.train, "exp_name", "") or "").strip():
            cfg.train.exp_name = infer_default_exp_name_from_agent_type(cfg.agent_type) or "tdqc_mlp"


def _apply_train_overrides(cfg: Any, train_overrides: Dict[str, Any]) -> Any:
    if not train_overrides:
        return cfg
    for key, value in train_overrides.items():
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


def _apply_finetune_defaults(
    cfg: Any,
    user_cfg: Dict[str, Any],
    train_overrides: Dict[str, Any],
) -> Any:
    cfg.train.finetune = True

    if not _dict_has_path(user_cfg, "train", "dataset_key") and train_overrides.get("dataset_key") is None:
        cfg.train.dataset_key = "expert_dataset+replaybuffer"
    if not _dict_has_path(user_cfg, "train", "exp_name") and train_overrides.get("exp_name") is None:
        base_exp_name = str(getattr(cfg.train, "exp_name", "") or "piper_dual_merged_cpiql_dac")
        if not base_exp_name.endswith("_finetune"):
            cfg.train.exp_name = f"{base_exp_name}_finetune"

    _sync_derived_fields(cfg)
    return cfg


def _load_yaml_config_dict(config_path: str) -> Dict[str, Any]:
    loaded = OmegaConf.load(config_path)
    value = OmegaConf.to_container(loaded, resolve=False)
    return value if isinstance(value, dict) else {}


def _resolve_tdqc_encoder_config_path(cfg: Any, simple_cfg: Dict[str, Any]) -> None:
    agent_name = str(getattr(cfg, "agent_type", "") or "").strip().lower().replace("_", "-")
    if "tdqc" not in agent_name or getattr(cfg, "agent_sp", None) is None:
        return
    encoder_override = getattr(cfg.agent_sp, "encoder_override", None)
    if encoder_override is False:
        from agent_factory.config.structure import StateEncoderConfig

        cfg.agent_sp.encoder_override = OmegaConf.structured(StateEncoderConfig())
        return
    encoder_config_path = str(getattr(cfg.agent_sp, "encoder_config_path", "") or "").strip()
    if not encoder_config_path:
        return
    if not os.path.isabs(encoder_config_path):
        encoder_config_path = os.path.abspath(encoder_config_path)
        cfg.agent_sp.encoder_config_path = encoder_config_path
    if not os.path.exists(encoder_config_path):
        raise FileNotFoundError(f"TDQC encoder_config_path not found: {encoder_config_path}")
    if getattr(cfg.agent_sp, "encoder_override", None) is None:
        cfg.agent_sp.encoder_override = OmegaConf.create(_load_yaml_config_dict(encoder_config_path))

    simple_agent_sp = simple_cfg.get("agent_sp")
    if isinstance(simple_agent_sp, dict):
        simple_agent_sp["encoder_config_path"] = encoder_config_path


def general_resolve(
    override_config: Any = None,
    file_config: Any = None,
    *,
    finetune: Optional[bool] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build simple/materialized training configs with explicit precedence.

    Precedence is:
    config.train overrides from command line > file config > agent defaults.
    """

    from agent_factory.agents.registry import get_default_config
    from agent_factory.config.manager import ConfigManager

    normalized_file_cfg = normalize_user_config(file_config or {})
    train_overrides = normalize_train_overrides(override_config or {})
    if finetune is not None:
        train_overrides["finetune"] = bool(finetune)

    agent_type = str(normalized_file_cfg.get("agent_type") or "Diffusion_CPIQL_DAC")
    user_cfg = ConfigManager._resolve_runner_defaults(copy.deepcopy(normalized_file_cfg))
    user_cfg = ConfigManager._resolve_dataset_defaults(user_cfg)
    user_cfg = ConfigManager._resolve_env_defaults(user_cfg, normalized_file_cfg)
    simple_cfg = _merge_train_overrides_into_simple_config(normalized_file_cfg, train_overrides)
    simple_cfg.setdefault("agent_type", agent_type)
    simple_cfg["resolved"] = False

    default_cfg = get_default_config(agent_type)
    cfg = OmegaConf.merge(default_cfg, OmegaConf.create(user_cfg))
    _fill_inferred_env_stats(cfg, user_cfg)
    _sync_derived_fields(cfg)

    effective_finetune = bool(getattr(cfg.train, "finetune", False)) if finetune is None else bool(finetune)
    if effective_finetune:
        cfg = _apply_finetune_defaults(cfg, user_cfg, train_overrides)
    cfg = _apply_train_overrides(cfg, train_overrides)

    _resolve_tdqc_encoder_config_path(cfg, simple_cfg)
    cfg, resolved_module_paths = materialize_global_config(cfg, explicit_cfg=OmegaConf.create(simple_cfg))
    cfg, resolved_mixin_paths = resolve_mixin_configs(
        cfg,
        explicit_cfg=OmegaConf.create(simple_cfg),
        agent_type=agent_type,
    )
    mixin_validation_warnings = validate_mixin_configs(cfg, agent_type=agent_type)
    global_validation_warnings = validate_global_config(cfg)
    simple_cfg = OmegaConf.to_container(simplify_global_config(cfg, simple_cfg), resolve=False)
    simple_cfg = OmegaConf.to_container(simplify_mixin_config(cfg, simple_cfg, agent_type=agent_type), resolve=False)
    cfg.resolved = True

    runtime_spec = {
        "agent_type": str(cfg.agent_type),
        "user_cfg": user_cfg,
        "simple_cfg": simple_cfg,
        "resolved_module_paths": resolved_module_paths,
        "resolved_mixin_paths": resolved_mixin_paths,
        "mixin_validation_warnings": mixin_validation_warnings,
        "global_validation_warnings": global_validation_warnings,
        "train_overrides": train_overrides,
    }
    return cfg, runtime_spec


def save_config_snapshots(cfg: Any, simple_cfg: Dict[str, Any], save_dir: str) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)

    simple_path = os.path.join(save_dir, SIMPLE_CONFIG_FILENAME)
    resolved_path = os.path.join(save_dir, RESOLVED_CONFIG_FILENAME)

    resolved_snapshot = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(resolved_snapshot, dict):
        resolved_snapshot.pop("device", None)
        dataset_cfg = resolved_snapshot.get("dataset")
        if isinstance(dataset_cfg, dict):
            dataset_cfg.pop("replay", None)

    OmegaConf.save(OmegaConf.create(simple_cfg), simple_path, resolve=True)
    OmegaConf.save(OmegaConf.create(resolved_snapshot), resolved_path, resolve=True)

    return {
        "simple": simple_path,
        "resolved": resolved_path,
    }
