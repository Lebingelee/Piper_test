import torch

from agent_factory.config.structure import Pi05ActorConfig
from agent_factory.modules.actors.pi05 import Pi05PolicyWrapper


class Pi05ActorMixin:
    """π₀.₅ actor surface used by prefix-feature export and later rollout."""

    CONFIG_CLASS = Pi05ActorConfig
    CONFIG_KEY = "actor"
    RESOLVE_RULES = {
        "state_dim": {"source": "env.proprio_dim", "kind": "derived", "required": True},
        "action_dim": {"source": "env.action_dim", "kind": "derived", "required": True},
        "pred_horizon": {"source": "env.pred_horizon", "kind": "derived", "required": True},
    }
    # LeRobot's own processor owns state/image normalization and prompt encoding.
    requires_prompt = True
    uses_raw_observations = True

    def _build_actor(self):
        cfg: Pi05ActorConfig = self.cfg.actor
        self.actor = Pi05PolicyWrapper(
            cfg=cfg,
            action_dim=int(cfg.action_dim),
            state_dim=int(cfg.state_dim),
        )
        self.actor.to(self.device)

    def update_actor(self, batch: dict) -> dict:
        del batch
        raise NotImplementedError(
            "π₀.₅ fine-tuning remains in LeRobot. agent_factory currently supports "
            "π₀.₅ prefix-feature export and the rollout interface only."
        )

    def sample_action(self, obs, initial_noise=None, **kwargs):
        del initial_noise, obs, kwargs
        raise NotImplementedError(
            "π₀.₅ rollout is intentionally deferred until server-side weight validation."
        )

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action
