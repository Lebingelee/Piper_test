import torch

from agent_factory.config.structure import SmolVLAActorConfig
from agent_factory.modules.actors.smolvla import (
    SMOLVLA_ACTION_KEY,
    SMOLVLA_STATE_KEY,
    SmolVLAPolicyWrapper,
    resolve_smolvla_normalization_source,
)

from ..module_builder import ModuleBuilderMixin
from ..normalization_mixins import ActionNormMixin, ObsNormMixin


class SmolVLAActorMixin(ModuleBuilderMixin, ActionNormMixin, ObsNormMixin):
    """
    Mixin for LeRobot SmolVLA fine-tuning and inference inside agent_factory.
    """

    CONFIG_CLASS = SmolVLAActorConfig
    CONFIG_KEY = "actor"
    RESOLVE_RULES = {
        "action_dim": {
            "source": "env.action_dim",
            "kind": "derived",
            "required": True,
        },
        "pred_horizon": {
            "source": "env.pred_horizon",
            "kind": "derived",
            "required": True,
        },
    }
    VALIDATION_RULES = {
        "obs_horizon": {
            "source": "env.obs_horizon",
            "kind": "protocol",
            "on_conflict": "warn",
        },
    }
    REQUIRED_KEYS = {
        "observation.state",
        "action",
        "task",
    }

    requires_prompt = True

    def _build_actor(self):
        cfg: SmolVLAActorConfig = self.cfg.actor
        resolve_smolvla_normalization_source(cfg)
        self._init_action_normalizer()
        self._init_obs_normalizer()

        pred_horizon = int(getattr(cfg, "pred_horizon", self.cfg.env.pred_horizon))
        if pred_horizon != 50:
            raise ValueError(
                "SmolVLA first-pass integration requires env.pred_horizon == 50. "
                "Changing it requires retraining/rebuilding the action expert."
            )

        self.actor = SmolVLAPolicyWrapper(
            cfg=cfg,
            action_dim=int(self.cfg.env.action_dim),
            pred_horizon=pred_horizon,
            state_dim=int(self.cfg.env.proprio_dim),
        )
        self.actor_optimizer = torch.optim.AdamW(
            [param for param in self.actor.parameters() if param.requires_grad],
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

    def _normalize_smolvla_training_batch(self, batch: dict) -> dict:
        out = dict(batch)
        state_key = str(getattr(self.cfg.actor, "state_key", SMOLVLA_STATE_KEY))
        source_state_key = state_key if state_key in out else SMOLVLA_STATE_KEY
        if source_state_key not in out:
            raise KeyError(f"SmolVLA training batch is missing state key '{source_state_key}'.")
        if SMOLVLA_ACTION_KEY not in out:
            raise KeyError(f"SmolVLA training batch is missing action key '{SMOLVLA_ACTION_KEY}'.")

        out[SMOLVLA_STATE_KEY] = self.normalize_obs_state(out[source_state_key].to(self.device).float())
        if source_state_key != SMOLVLA_STATE_KEY:
            out.pop(source_state_key, None)
        out[SMOLVLA_ACTION_KEY] = self.normalize_action(out[SMOLVLA_ACTION_KEY].to(self.device).float())
        return out

    def _normalize_smolvla_obs(self, obs: dict) -> dict:
        out = dict(obs)
        state_key = str(getattr(self.cfg.actor, "state_key", SMOLVLA_STATE_KEY))
        source_state_key = state_key if state_key in out else SMOLVLA_STATE_KEY if SMOLVLA_STATE_KEY in out else "state"
        if source_state_key not in out:
            raise KeyError("SmolVLA observations require 'observation.state' or 'state'.")
        out[source_state_key] = self.normalize_obs_state(torch.as_tensor(out[source_state_key], device=self.device).float())
        return out

    def update_actor(self, batch: dict) -> dict:
        loss, loss_dict = self.actor(self._normalize_smolvla_training_batch(batch))

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        out = {"loss_actor": float(loss.item())}
        out.update({str(key): float(value) for key, value in loss_dict.items() if isinstance(value, (int, float))})
        return out

    def sample_action(self, obs, initial_noise=None, **kwargs):
        del initial_noise
        prompt = kwargs.pop("prompt", None)
        was_training = self.actor.training
        self.actor.eval()
        try:
            with torch.no_grad():
                norm_action = self.actor.sample_action(self._normalize_smolvla_obs(obs), prompt=prompt, **kwargs)
                return self.denormalize_action(norm_action)
        finally:
            if was_training:
                self.actor.train()
