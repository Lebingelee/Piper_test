from typing import Any

from omegaconf import OmegaConf


def runner_config_get(cfg_or_runner: Any, key: str, default: Any = None) -> Any:
    runner_cfg = getattr(cfg_or_runner, "runner", cfg_or_runner)
    extra_cfg = getattr(runner_cfg, "config", None)
    if extra_cfg is not None:
        try:
            value = OmegaConf.select(extra_cfg, key, default=None)
            if value is not None:
                return value
        except Exception:
            if hasattr(extra_cfg, "get"):
                value = extra_cfg.get(key, None)
                if value is not None:
                    return value
    return getattr(runner_cfg, key, default)
