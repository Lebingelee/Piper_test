from typing import Any

from agent_factory.modules.registry import make_module


class ModuleBuilderMixin:
    """Shared helpers for config-driven module construction in agent mixins."""

    def _build_module_from_config(self, module_cfg: Any):
        module = make_module(module_cfg)
        return module.to(self.device) if hasattr(module, "to") else module

    def _build_encoder_from_config(self, encoder_cfg: Any):
        return self._build_module_from_config(encoder_cfg)
