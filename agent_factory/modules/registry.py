from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict

from omegaconf import OmegaConf


_MODULE_REGISTRY: Dict[str, Callable[..., Any]] = {}
_BUILTINS_LOADED = False


def _normalize_module_type(module_type: str) -> str:
    return str(module_type or "").strip().lower()


def register_module(module_type: str):
    """
    Register a module builder under a config-facing type string.

    Builders receive the config object as their first positional argument and
    may accept contextual kwargs such as env_cfg or dataset_cfg.
    """

    def decorator(builder: Callable[..., Any]):
        normalized = _normalize_module_type(module_type)
        if not normalized:
            raise ValueError("module_type must be a non-empty string.")
        _MODULE_REGISTRY[normalized] = builder
        return builder

    return decorator


def ensure_builtin_modules_loaded() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from agent_factory.modules.encoders import state_encoder as _state_encoder  # noqa: F401
    from agent_factory.modules.encoders import feature_identity_encoder as _feature_identity_encoder  # noqa: F401

    _BUILTINS_LOADED = True


def module_config_to_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if OmegaConf.is_config(cfg):
        value = OmegaConf.to_container(cfg, resolve=True)
        return value if isinstance(value, dict) else {}
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    return {}


def module_cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, key, default=default)
    return getattr(cfg, key, default)


def make_module(module_cfg: Any, *args, module_type: str = None, **kwargs) -> Any:
    """
    Build a registered module from a config object.

    The selected type is `module_type` when provided, otherwise `module_cfg.type`.
    This intentionally mirrors the agent factory's config-driven construction
    pattern without forcing existing actor/critic modules to migrate at once.
    """

    ensure_builtin_modules_loaded()
    selected_type = module_type or module_cfg_get(module_cfg, "type", "")
    normalized = _normalize_module_type(selected_type)
    if normalized not in _MODULE_REGISTRY:
        available = sorted(_MODULE_REGISTRY.keys())
        raise ValueError(f"Unknown module type '{selected_type}'. Available: {available}")
    return _MODULE_REGISTRY[normalized](module_cfg, *args, **kwargs)


def get_module_builder(module_cfg: Any = None, module_type: str = None) -> Callable[..., Any]:
    ensure_builtin_modules_loaded()
    selected_type = module_type or module_cfg_get(module_cfg, "type", "")
    normalized = _normalize_module_type(selected_type)
    return _MODULE_REGISTRY.get(normalized)


def get_module_resolve_rules(module_cfg: Any = None, module_type: str = None) -> Dict[str, Any]:
    builder = get_module_builder(module_cfg=module_cfg, module_type=module_type)
    if builder is None:
        return {}
    rules = getattr(builder, "RESOLVE_RULES", None)
    return dict(rules) if isinstance(rules, dict) else {}


__all__ = [
    "ensure_builtin_modules_loaded",
    "get_module_builder",
    "get_module_resolve_rules",
    "make_module",
    "module_cfg_get",
    "module_config_to_dict",
    "register_module",
]
