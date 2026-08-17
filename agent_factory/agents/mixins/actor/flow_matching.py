import torch

from agent_factory.config.structure import FlowMatchingActorConfig
from agent_factory.modules.actors.flow_matching import VanillaFlowMatchingPolicy

from ..module_builder import ModuleBuilderMixin
from ..normalization_mixins import ActionNormMixin


class FlowMatchingActorMixin(ModuleBuilderMixin, ActionNormMixin):
    """
    Mixin: provides vanilla flow matching actor construction, training, and inference.
    """

    CONFIG_CLASS = FlowMatchingActorConfig
    CONFIG_KEY = "actor"
    RESOLVE_RULES = {
        "action_dim": {
            "source": "env.action_dim",
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
        "pred_horizon": {
            "source": "env.pred_horizon",
            "kind": "protocol",
            "on_conflict": "warn",
        },
    }
    REQUIRED_KEYS = {"observations", "action"}

    def _build_actor(self):
        cfg: FlowMatchingActorConfig = self.cfg.actor

        self._init_action_normalizer()

        self.actor_encoder = self._build_encoder_from_config(cfg.encoder)

        self.actor = VanillaFlowMatchingPolicy(
            state_encoder=self.actor_encoder,
            action_dim=self.cfg.env.action_dim,
            pred_horizon=self.cfg.env.pred_horizon,
            obs_horizon=self.cfg.env.obs_horizon,
            unet_config=cfg.unet,
            num_inference_steps=cfg.num_inference_steps,
            time_beta_alpha=cfg.time_beta_alpha,
            time_beta_beta=cfg.time_beta_beta,
            time_eps=cfg.time_eps,
            time_embed_scale=cfg.time_embed_scale,
            clip_sample=cfg.clip_sample,
        )

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    def update_actor(self, batch: dict) -> dict:
        obs = self._preprocess_obs(batch["observations"])
        actions = self.normalize_action(batch["action"].to(self.device).float())

        loss = self.actor(obs, actions)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"loss_actor": loss.item()}

    def sample_action(self, obs, initial_noise=None, num_inference_steps=None):
        was_training = self.actor.training
        self.actor.eval()
        obs = self._preprocess_obs(obs)
        try:
            with torch.no_grad():
                if initial_noise is not None:
                    initial_noise = torch.as_tensor(
                        initial_noise,
                        dtype=torch.float32,
                        device=self.device,
                    )
                norm_action = self.actor.sample_action(
                    obs,
                    num_inference_steps=num_inference_steps,
                    initial_noise=initial_noise,
                )
                return self.denormalize_action(norm_action)
        finally:
            if was_training:
                self.actor.train()
