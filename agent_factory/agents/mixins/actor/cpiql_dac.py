import copy
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from agent_factory.config.structure import DiffusionActorConfig
from agent_factory.modules.actors.diffusion import ConditionalDiffusionPolicy

from .diffusion import DiffusionActorMixin


@dataclass
class CPIQLDACActorConfig(DiffusionActorConfig):
    type: str = "cpiql_dac_diffusion"

    dac_enabled: bool = True
    q_guidance_prob: float = 0.2
    q_guidance_lambda: float = 0.1

    eta_base: float = 1.0
    eta_min: float = 0.1
    eta_max: float = 10.0
    beta_delta: float = 1.0
    delta_max: float = 1.0
    delta_hard: float = 0.5

    grad_clip_norm: float = 1.0
    grad_raw_max: float = 10.0
    grad_norm_ema_decay: float = 0.99
    grad_norm_eps: float = 1e-6

    guide_timestep_min_frac: float = 0.2
    guide_timestep_max_frac: float = 0.8

    eta_auto_tune: bool = False
    eta_lr: float = 1e-3
    bc_loss_min: float = 0.2
    bc_loss_max: float = 1.0
    eta_low_weight: float = 0.5

    ema_actor_enabled: bool = True
    ema_actor_tau: float = 0.005
    use_ema_actor_for_inference: bool = True


class CPIQLDACActorMixin(DiffusionActorMixin):
    """
    Stage-wise Risk-Scaled CPIQL-DAC actor update.

    The critic is treated as frozen guidance. Q(o, noisy_action, k=0)
    provides the action-gradient, while Q(o, data_action, k=1)-Q(o,
    data_action, k=0) only scales or gates the guidance strength.
    """

    CONFIG_CLASS = CPIQLDACActorConfig
    CONFIG_KEY = "actor"
    REQUIRED_KEYS = {"observations", "action"}

    def _build_actor(self):
        super()._build_actor()
        self.register_buffer(
            "dac_grad_norm_ema",
            torch.tensor(float(getattr(self.cfg.actor, "grad_clip_norm", 1.0))),
        )
        self.register_buffer(
            "dac_log_eta_base",
            torch.tensor(float(getattr(self.cfg.actor, "eta_base", 1.0))).log(),
        )
        if bool(getattr(self.cfg.actor, "ema_actor_enabled", True)):
            self.ema_actor = copy.deepcopy(self.actor)
            self.ema_actor.requires_grad_(False)
        else:
            self.ema_actor = None

    def _freeze_cpiql_critic_for_actor(self):
        for name in ("q_net", "v_net", "target_q_net"):
            module = getattr(self, name, None)
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def _dac_global_cond(self, obs: Dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
        if isinstance(self.actor, ConditionalDiffusionPolicy):
            cond = batch.get("cond", None)
            if cond is None:
                raise ValueError("ConditionalDiffusionPolicy requires 'cond' in batch.")
            cond = cond.to(self.device).float()
            if cond.ndim == 1:
                cond = cond.unsqueeze(-1)

            if bool(getattr(self.cfg.actor, "use_cfg_loss", False)):
                drop_rate = float(getattr(self.cfg.actor, "cfg_drop_rate", 0.1))
                keep = (torch.rand((cond.shape[0], 1), device=self.device) > drop_rate).float()
                cond = cond * keep
            return self.actor._get_global_cond(obs, cond)

        state_embed = self.actor.state_encoder(obs)
        return state_embed.flatten(start_dim=1)

    def _dac_k(self, batch_size: int, value: float, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, 1), float(value), device=device, dtype=torch.float32)

    def _dac_min_q(self, obs: Dict[str, torch.Tensor], action: torch.Tensor, k_value: float) -> torch.Tensor:
        q_module = getattr(self, "target_q_net", None)
        if q_module is None:
            q_module = self.q_net
        k = self._dac_k(action.shape[0], k_value, action.device)
        q1, q2 = q_module(obs, action, k)
        return torch.min(q1, q2)

    def _dac_timestep_gate(self, timesteps: torch.Tensor) -> torch.Tensor:
        num_steps = int(self.actor.noise_scheduler.config.num_train_timesteps)
        lo = int(float(getattr(self.cfg.actor, "guide_timestep_min_frac", 0.2)) * num_steps)
        hi = int(float(getattr(self.cfg.actor, "guide_timestep_max_frac", 0.8)) * num_steps)
        lo = max(0, min(lo, num_steps - 1))
        hi = max(lo, min(hi, num_steps - 1))
        return ((timesteps >= lo) & (timesteps <= hi)).float().unsqueeze(-1)

    def _normalize_q_grad(self, grad: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.actor
        eps = float(getattr(cfg, "grad_norm_eps", 1e-6))
        flat = grad.flatten(start_dim=1)
        norm = flat.norm(dim=1, keepdim=True).clamp_min(eps)

        clip_norm = float(getattr(cfg, "grad_clip_norm", 1.0))
        if clip_norm > 0:
            coef = (clip_norm / norm).clamp(max=1.0)
            grad = grad * coef.view(-1, *([1] * (grad.ndim - 1)))

        with torch.no_grad():
            decay = float(getattr(cfg, "grad_norm_ema_decay", 0.99))
            batch_norm = norm.mean()
            self.dac_grad_norm_ema.mul_(decay).add_(batch_norm * (1.0 - decay))

        return grad / self.dac_grad_norm_ema.clamp_min(eps)

    def _update_eta_base(self, bc_loss: torch.Tensor):
        cfg = self.cfg.actor
        if not bool(getattr(cfg, "eta_auto_tune", False)):
            return

        with torch.no_grad():
            bc_value = bc_loss.detach()
            high = F.relu(bc_value - float(getattr(cfg, "bc_loss_max", 1.0)))
            low = F.relu(float(getattr(cfg, "bc_loss_min", 0.2)) - bc_value)
            delta = float(getattr(cfg, "eta_lr", 1e-3)) * (
                high - float(getattr(cfg, "eta_low_weight", 0.5)) * low
            )
            self.dac_log_eta_base.add_(delta)
            self.dac_log_eta_base.clamp_(
                min=torch.log(torch.tensor(float(getattr(cfg, "eta_min", 0.1)), device=self.device)),
                max=torch.log(torch.tensor(float(getattr(cfg, "eta_max", 10.0)), device=self.device)),
            )

    @torch.no_grad()
    def _update_ema_actor(self):
        if self.ema_actor is None:
            return
        tau = float(getattr(self.cfg.actor, "ema_actor_tau", 0.005))
        for ema_param, param in zip(self.ema_actor.parameters(), self.actor.parameters()):
            ema_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
        for ema_buffer, buffer in zip(self.ema_actor.buffers(), self.actor.buffers()):
            ema_buffer.data.copy_(buffer.data)

    def update_actor_cpiql_dac(self, batch: dict) -> dict:
        cfg = self.cfg.actor
        if not bool(getattr(cfg, "dac_enabled", True)):
            return super().update_actor(batch)
        if not hasattr(self, "q_net"):
            raise AttributeError("CPIQLDACActorMixin requires a CPIQL critic with q_net/target_q_net.")

        self._freeze_cpiql_critic_for_actor()

        obs = self._preprocess_obs(batch["observations"])
        raw_actions = batch["action"].to(self.device).float()
        actions = self.normalize_action(raw_actions)
        batch_size = actions.shape[0]

        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            int(self.actor.noise_scheduler.config.num_train_timesteps),
            (batch_size,),
            device=self.device,
        ).long()
        noisy_actions = self.actor.noise_scheduler.add_noise(actions, noise, timesteps)
        global_cond = self._dac_global_cond(obs, batch)
        noise_pred = self.actor.noise_pred_net(
            sample=noisy_actions,
            timestep=timesteps,
            global_cond=global_cond,
        )

        noisy_for_q = noisy_actions.detach().requires_grad_(True)
        raw_noisy_for_q = self.denormalize_action(noisy_for_q)
        q_noisy_0 = self._dac_min_q(obs, raw_noisy_for_q, k_value=0.0)
        q_grad = torch.autograd.grad(q_noisy_0.sum(), noisy_for_q, create_graph=False)[0].detach()

        with torch.no_grad():
            q_ref_0 = self._dac_min_q(obs, raw_actions, k_value=0.0)
            q_ref_1 = self._dac_min_q(obs, raw_actions, k_value=1.0)
            delta_q = torch.clamp(q_ref_1 - q_ref_0, min=0.0, max=float(cfg.delta_max))

            eta_base = self.dac_log_eta_base.exp()
            eta_eff = torch.clamp(
                eta_base * torch.exp(float(cfg.beta_delta) * delta_q),
                min=float(cfg.eta_min),
                max=float(cfg.eta_max),
            )

            raw_grad_norm = q_grad.flatten(start_dim=1).norm(dim=1, keepdim=True)
            m_q = (
                torch.rand((batch_size, 1), device=self.device) < float(cfg.q_guidance_prob)
            ).float()
            m_delta = (delta_q < float(cfg.delta_hard)).float()
            m_grad = (raw_grad_norm < float(cfg.grad_raw_max)).float()
            m_time = self._dac_timestep_gate(timesteps)
            m_safe = m_delta * m_grad * m_time

        grad_hat = self._normalize_q_grad(q_grad)
        alphas_cumprod = self.actor.noise_scheduler.alphas_cumprod.to(self.device)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alphas_cumprod[timesteps]).view(batch_size, 1, 1)
        lambda_eff = (
            m_q * m_safe * float(cfg.q_guidance_lambda) / eta_eff.clamp_min(float(cfg.eta_min))
        ).view(batch_size, 1, 1)
        epsilon_target = (noise - lambda_eff * sqrt_one_minus_alpha * grad_hat).detach()

        loss_actor = F.mse_loss(noise_pred, epsilon_target)
        bc_loss = F.mse_loss(noise_pred.detach(), noise)

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        self._update_eta_base(bc_loss)
        self._update_ema_actor()

        with torch.no_grad():
            target_shift = (epsilon_target - noise).flatten(start_dim=1).norm(dim=1)
            guided_mask = m_q * m_safe
            metrics = {
                "loss_actor": float(loss_actor.item()),
                "loss_actor_bc": float(bc_loss.item()),
                "dac_delta_q_mean": float(delta_q.mean().item()),
                "dac_delta_q_max": float(delta_q.max().item()),
                "dac_eta_base": float(self.dac_log_eta_base.exp().item()),
                "dac_eta_eff_mean": float(eta_eff.mean().item()),
                "dac_q_noisy_0_mean": float(q_noisy_0.detach().mean().item()),
                "dac_grad_norm_mean": float(raw_grad_norm.mean().item()),
                "dac_grad_norm_ema": float(self.dac_grad_norm_ema.item()),
                "dac_m_q_mean": float(m_q.mean().item()),
                "dac_m_safe_mean": float(m_safe.mean().item()),
                "dac_guided_ratio": float(guided_mask.mean().item()),
                "dac_target_shift_mean": float(target_shift.mean().item()),
            }
        return metrics

    def update_actor(self, batch: dict) -> dict:
        return self.update_actor_cpiql_dac(batch)

    def sample_action(self, obs, initial_noise=None):
        if not bool(getattr(self.cfg.actor, "use_ema_actor_for_inference", True)) or self.ema_actor is None:
            return super().sample_action(obs, initial_noise=initial_noise)

        online_actor = self.actor
        self.actor = self.ema_actor
        try:
            return super().sample_action(obs, initial_noise=initial_noise)
        finally:
            self.actor = online_actor
