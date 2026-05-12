from typing import Any, Dict, Optional

import torch

from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.registry import register_agent


@register_agent("Identity")
class IdentityAgent(BaseAgent):
    """
    Minimal inference-only agent for runner hardware tests.

    In the Piper motion-test mode this agent emits a deterministic absolute
    joint chunk instead of loading a learned policy. The base vector is the
    rounded dual-arm initial pose, and dimensions 4 and 11 follow a smooth bump
    so both arms visibly move and then return.
    """

    uses_env_safe_action = False
    CHUNK_HORIZON = 64
    BUMP_AMPLITUDE_RAD = 0.4
    BASE_ACTION = (
        0.400, 1.000, -0.900, 0.099, 0.901, -0.199, 0.050,
        -0.399, 1.000, -0.899, 0.100, 0.900, -0.199, 0.049,
    )

    def _init_components(self):
        self.action_dim = int(
            getattr(
                getattr(self.cfg, "actor", None),
                "action_dim",
                getattr(self.cfg.env, "action_dim", 1),
            )
        )
        self.pred_horizon = self.CHUNK_HORIZON

    def _init_optimizers(self):
        return

    def _base_action_tensor(self) -> torch.Tensor:
        base = torch.tensor(self.BASE_ACTION, dtype=torch.float32, device=self.device)
        if self.action_dim == base.numel():
            return base
        if self.action_dim < base.numel():
            return base[:self.action_dim]
        pad = torch.zeros(self.action_dim - base.numel(), dtype=torch.float32, device=self.device)
        return torch.cat([base, pad], dim=0)

    def sample_action(self, obs: Dict[str, Any], initial_noise: Optional[torch.Tensor] = None):
        """
        Return a deterministic [B, 64, action_dim] absolute-joint action chunk.

        Dimensions are 1-indexed in the experiment note:
        - dim 4  -> index 3
        - dim 11 -> index 10
        """
        del initial_noise
        batch_size = 1
        if isinstance(obs, dict):
            for value in obs.values():
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    batch_size = int(value.shape[0])
                    break

        base = self._base_action_tensor()
        chunk = base.unsqueeze(0).repeat(self.pred_horizon, 1)
        bump = torch.sin(
            torch.linspace(0.0, torch.pi, self.pred_horizon, device=self.device)
        ) * self.BUMP_AMPLITUDE_RAD
        if self.action_dim > 3:
            chunk[:, 3] = base[3] + bump
        if self.action_dim > 10:
            chunk[:, 10] = base[10] + bump
        return chunk.unsqueeze(0).repeat(batch_size, 1, 1)

    def load(self, path: str):
        print(f"[IdentityAgent] Skip loading checkpoint for deterministic motion test agent: {path}")
        return {}

    def save(self, path: str, meta: Dict = None):
        print(f"[IdentityAgent] No parameters to save for deterministic motion test agent: {path}")

    def start_train(self, dataset, additional_args: Optional[Dict[str, Any]] = None):
        raise NotImplementedError("IdentityAgent is inference-only and cannot be trained.")
