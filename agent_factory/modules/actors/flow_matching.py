from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_factory.modules.actors.diffusion_module.conditional_unet1d import ConditionalUnet1D
from agent_factory.modules.encoders.state_encoder import BaseStateEncoder


class VanillaFlowMatchingPolicy(nn.Module):
    """
    Observation-conditioned flow matching policy.

    The velocity field follows the SmolVLA action expert convention:
    x_t = t * noise + (1 - t) * action, target velocity = noise - action.
    """

    def __init__(
        self,
        state_encoder: BaseStateEncoder,
        action_dim: int,
        pred_horizon: int,
        obs_horizon: int,
        unet_config: Dict,
        num_inference_steps: int = 10,
        time_beta_alpha: float = 1.5,
        time_beta_beta: float = 1.0,
        time_eps: float = 1e-3,
        time_embed_scale: float = 100.0,
        clip_sample: bool = True,
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.num_inference_steps = int(num_inference_steps)
        self.time_beta_alpha = float(time_beta_alpha)
        self.time_beta_beta = float(time_beta_beta)
        self.time_eps = float(time_eps)
        self.time_embed_scale = float(time_embed_scale)
        self.clip_sample = bool(clip_sample)

        if state_encoder.view_fusion == "concat":
            self.global_cond_dim = state_encoder.out_dim * obs_horizon
        elif state_encoder.view_fusion == "mean":
            self.global_cond_dim = state_encoder.out_dim
        else:
            raise ValueError(f"Unknown view_fusion type: {state_encoder.view_fusion}")

        self.velocity_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=self.global_cond_dim,
            **unet_config,
        )

    def _get_global_cond(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        state_embed = self.state_encoder(obs_dict)
        if self.state_encoder.view_fusion == "mean":
            return state_embed.mean(dim=1)
        return state_embed.flatten(start_dim=1)

    def _sample_noise(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(actions)

    def _sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        beta = torch.distributions.Beta(self.time_beta_alpha, self.time_beta_beta)
        time = beta.sample((batch_size,)).to(device=device, dtype=torch.float32)
        eps = max(min(self.time_eps, 1.0), 0.0)
        return time * (1.0 - eps) + eps

    def compute_loss(self, actions: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        batch_size = actions.shape[0]
        noise = self._sample_noise(actions)
        time = self._sample_time(batch_size, actions.device)
        time_expanded = time.view(batch_size, 1, 1)

        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        target_velocity = noise - actions
        pred_velocity = self.velocity_net(
            sample=x_t,
            timestep=time * self.time_embed_scale,
            global_cond=global_cond,
        )
        return F.mse_loss(pred_velocity, target_velocity)

    def forward(self, obs_dict: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        global_cond = self._get_global_cond(obs_dict)
        return self.compute_loss(actions, global_cond)

    @torch.no_grad()
    def sample(
        self,
        global_cond: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = global_cond.shape[0]
        steps = int(num_inference_steps or self.num_inference_steps)
        if steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {steps}.")

        if initial_noise is not None:
            actions = initial_noise
        else:
            actions = torch.randn(
                (batch_size, self.pred_horizon, self.action_dim),
                device=global_cond.device,
            )

        dt = -1.0 / float(steps)
        for step_idx in range(steps):
            time_value = 1.0 + step_idx * dt
            time = torch.full(
                (batch_size,),
                time_value,
                dtype=torch.float32,
                device=global_cond.device,
            )
            velocity = self.velocity_net(
                sample=actions,
                timestep=time * self.time_embed_scale,
                global_cond=global_cond,
            )
            actions = actions + dt * velocity
            if self.clip_sample:
                actions = actions.clamp(-1.0, 1.0)

        return actions

    def sample_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        num_inference_steps: Optional[int] = None,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        global_cond = self._get_global_cond(obs_dict)
        return self.sample(
            global_cond=global_cond,
            num_inference_steps=num_inference_steps,
            initial_noise=initial_noise,
        )
