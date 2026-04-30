from __future__ import annotations

from typing import Any, Dict

import torch
from torch.nn.utils import clip_grad_norm_

from agent_factory.config.structure import DSRLActorConfig
from .dsrl_adapter import GroupRLDiffusionPolicyAdapter
from agent_factory.modules.actors.dsrl_policy import DSRLNoisePolicy
from agent_factory.modules.utils.temperature import Temperature
from agent_factory.modules.encoders.state_encoder import BaseStateEncoder
from agent_factory.modules.encoders.visual_encoder import VisualEncoder


class DSRLActorMixin:
    CONFIG_CLASS = DSRLActorConfig
    CONFIG_KEY = "actor"
    REQUIRED_KEYS = {"observations", "discount"}
    _LOG_PROB_CLAMP = 1e6
    _ACTOR_GRAD_CLIP = 10.0
    _TEMP_GRAD_CLIP = 10.0
    _LOG_TEMP_MIN = -20.0
    _LOG_TEMP_MAX = 2.0
    _TEMP_DEBUG_WARMUP_STEPS = 10
    _TEMP_DEBUG_INTERVAL = 200

    def _build_actor(self):
        cfg: DSRLActorConfig = self.cfg.actor
        encoder_cfg = cfg.encoder
        use_visual = getattr(self.cfg.dataset, "include_rgb", True)

        self.noise_action_dim = int(cfg.action_dim)
        self.noise_pred_horizon = int(cfg.pred_horizon)
        self.train_action_dim = self.noise_action_dim
        if abs(float(cfg.action_limit) - 1.0) > 1e-6:
            print(
                f"[DSRL] Warning: actor.action_limit={cfg.action_limit} differs from 1.0. "
                "Please ensure base policy noise domain matches this scale."
            )

        visual_encoder = None
        if use_visual:
            visual_encoder = VisualEncoder(
                in_channels=encoder_cfg.visual.in_channels,
                out_dim=encoder_cfg.visual.out_dim,
                backbone_type=encoder_cfg.visual.backbone_type,
                pool_feature_map=encoder_cfg.visual.pool_feature_map,
                use_group_norm=encoder_cfg.visual.use_group_norm,
            )

        proprio_dim = encoder_cfg.proprio_dim or self.cfg.env.proprio_dim
        self.actor_encoder = BaseStateEncoder(
            visual_encoder=visual_encoder,
            proprio_dim=proprio_dim,
            out_dim=encoder_cfg.out_dim,
            num_cameras=self.cfg.env.num_cameras,
            view_fusion=encoder_cfg.view_fusion,
        )
        self.actor = DSRLNoisePolicy(
            state_encoder=self.actor_encoder,
            obs_horizon=cfg.obs_horizon,
            action_dim=self.noise_action_dim,
            pred_horizon=self.noise_pred_horizon,
            hidden_dims=cfg.hidden_dims,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
            action_limit=cfg.action_limit,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)

        self.temperature = Temperature(cfg.init_temperature)
        self.temp_optimizer = torch.optim.Adam(self.temperature.parameters(), lr=cfg.temp_lr)

        flat_action_dim = self.noise_action_dim * self.noise_pred_horizon
        self.flat_action_dim = int(flat_action_dim)
        self.entropy_log_prob_scale = 1.0 / max(float(self.flat_action_dim), 1.0)
        if isinstance(cfg.target_entropy, str) and cfg.target_entropy.lower() == "auto":
            # High-dimensional adaptation of SB3 auto target (-|A|): convert to per-dimension scale.
            self.target_entropy = -1.0
        else:
            self.target_entropy = float(cfg.target_entropy)

        self.base_policy = None
        # Base policy metadata lives in agent_sp for plugin decoupling.
        base_policy_cfg = getattr(self.cfg.agent_sp, "base_policy", None)
        if base_policy_cfg is not None:
            self.base_policy = GroupRLDiffusionPolicyAdapter(
                base_policy_cfg,
                self.device,
                expected_action_dim=self.cfg.env.action_dim,
                expected_pred_horizon=self.cfg.env.pred_horizon,
                expected_obs_horizon=self.cfg.env.obs_horizon,
            )

    def set_base_policy(self, policy: Any):
        self.base_policy = policy

    def ensure_base_policy(self):
        if self.base_policy is None:
            raise ValueError("DSRL requires an attached base diffusion policy.")
        if hasattr(self.base_policy, "load"):
            self.base_policy.load()
        return self.base_policy

    def _reshape_noise_action(self, flat_action: torch.Tensor) -> torch.Tensor:
        return flat_action.view(-1, self.noise_pred_horizon, self.noise_action_dim)

    def _entropy_log_prob(self, log_prob: torch.Tensor) -> torch.Tensor:
        return log_prob * float(self.entropy_log_prob_scale)

    def _policy_dist(self, obs: Dict[str, torch.Tensor]):
        processed = self._preprocess_obs(obs)
        # Fast fail for invalid inputs to avoid silently polluting parameters.
        for k, v in processed.items():
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                raise ValueError(f"[DSRL] Non-finite observation tensor detected at key '{k}'.")
        return self.actor(processed)

    def sample_noise_action(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            dist = self._policy_dist(obs)
            flat_action = dist.mode() if deterministic else dist.rsample()
            action = self._reshape_noise_action(flat_action)
        if was_training:
            self.actor.train()
        return action

    def _call_base_policy(self, obs: Dict[str, torch.Tensor], noise_action: torch.Tensor) -> torch.Tensor:
        policy = self.ensure_base_policy()
        if hasattr(policy, "infer"):
            action = policy.infer(obs, noise_action)
        elif callable(policy):
            action = policy(obs, noise_action)
        else:
            raise TypeError("Attached base diffusion policy must be callable or expose infer().")

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        return action.to(self.device, dtype=torch.float32)

    def sample_action(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        noise_action = self.sample_noise_action(obs, deterministic=deterministic)
        return self._call_base_policy(obs, noise_action)

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["observations"]
        dist = self._policy_dist(obs)
        flat_action = dist.rsample()
        noise_action = self._reshape_noise_action(flat_action)
        log_prob = dist.log_prob(flat_action)
        log_prob = torch.nan_to_num(
            log_prob,
            nan=0.0,
            posinf=self._LOG_PROB_CLAMP,
            neginf=-self._LOG_PROB_CLAMP,
        )
        entropy_log_prob = self._entropy_log_prob(log_prob)

        critic_requires_grad = [param.requires_grad for param in self.noise_q_net.parameters()]
        for param in self.noise_q_net.parameters():
            param.requires_grad_(False)

        q_values = self.noise_q_net(self._preprocess_obs(obs), noise_action)
        if not torch.isfinite(q_values).all():
            for param, requires_grad in zip(self.noise_q_net.parameters(), critic_requires_grad):
                param.requires_grad_(requires_grad)
            return {
                "loss_actor": float("nan"),
                "entropy": float("nan"),
                "log_prob_mean": float("nan"),
                "entropy_log_prob_mean": float("nan"),
                "q_pi": float("nan"),
                "temperature": float(self.temperature().item()),
                "mean_pi_avg": float("nan"),
                "std_pi_avg": float("nan"),
            }
        if self.cfg.critic.critic_reduction == "min":
            q_pi = q_values.min(dim=0).values
        elif self.cfg.critic.critic_reduction == "mean":
            q_pi = q_values.mean(dim=0)
        else:
            for param, requires_grad in zip(self.noise_q_net.parameters(), critic_requires_grad):
                param.requires_grad_(requires_grad)
            raise ValueError(f"Unsupported critic_reduction: {self.cfg.critic.critic_reduction}")

        temperature = self.temperature()
        actor_loss = (temperature.detach() * entropy_log_prob - q_pi).mean()
        if not torch.isfinite(actor_loss):
            for param, requires_grad in zip(self.noise_q_net.parameters(), critic_requires_grad):
                param.requires_grad_(requires_grad)
            return {
                "loss_actor": float("nan"),
                "entropy": float("nan"),
                "log_prob_mean": float("nan"),
                "entropy_log_prob_mean": float("nan"),
                "q_pi": float("nan"),
                "temperature": float(temperature.item()),
                "mean_pi_avg": float("nan"),
                "std_pi_avg": float("nan"),
            }

        # Save debug context for temperature update diagnostics.
        self._last_actor_temp_debug = {
            "flat_action_shape": tuple(flat_action.shape),
            "log_prob_shape": tuple(log_prob.shape),
            "q_pi_shape": tuple(q_pi.shape),
            "flat_action_dim_cfg": int(self.noise_action_dim * self.noise_pred_horizon),
            "log_prob_mean": float(log_prob.mean().item()),
            "entropy_log_prob_mean": float(entropy_log_prob.mean().item()),
            "log_prob_min": float(log_prob.min().item()),
            "log_prob_max": float(log_prob.max().item()),
            "entropy": float((-entropy_log_prob.mean()).item()),
            "target_entropy": float(self.target_entropy),
            "temperature": float(temperature.item()),
        }

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), max_norm=self._ACTOR_GRAD_CLIP)

        grad_finite = True
        for p in self.actor.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grad_finite = False
                break
        if grad_finite:
            self.actor_optimizer.step()
        else:
            self.actor_optimizer.zero_grad(set_to_none=True)

        for param, requires_grad in zip(self.noise_q_net.parameters(), critic_requires_grad):
            param.requires_grad_(requires_grad)

        base_dist = dist.base_dist.base_dist if hasattr(dist.base_dist, "base_dist") else dist.base_dist
        return {
            "loss_actor": actor_loss.item(),
            "entropy": -entropy_log_prob.mean().item(),
            "log_prob_mean": log_prob.mean().item(),
            "entropy_log_prob_mean": entropy_log_prob.mean().item(),
            "q_pi": q_pi.mean().item(),
            "temperature": temperature.item(),
            "mean_pi_avg": base_dist.loc.mean().item(),
            "std_pi_avg": base_dist.scale.mean().item(),
        }

    def update_temperature(self, entropy_log_prob_mean: float) -> Dict[str, float]:
        temperature = self.temperature()
        lp_plus_target = float(entropy_log_prob_mean + self.target_entropy)
        temp_loss = -(self.temperature.log_temperature * lp_plus_target)

        step = int(getattr(self, "step", 0))
        should_log = (
            step < self._TEMP_DEBUG_WARMUP_STEPS
            or (step % self._TEMP_DEBUG_INTERVAL == 0)
            or (abs(lp_plus_target) > 2.0)
            or (not torch.isfinite(temp_loss))
        )
        if should_log:
            actor_dbg = getattr(self, "_last_actor_temp_debug", {})
            print(
                "[DSRL Temp Debug] "
                f"step={step} "
                f"temp={float(temperature.item()):.6f} "
                f"entropy={actor_dbg.get('entropy', float('nan')):.6f} "
                f"log_prob_mean_raw={actor_dbg.get('log_prob_mean', float('nan')):.6f} "
                f"log_prob_mean_entropy={float(entropy_log_prob_mean):.6f} "
                f"target_entropy={float(self.target_entropy):.6f} "
                f"log_prob_plus_target={lp_plus_target:.6f} "
                f"temp_loss={float(temp_loss.item()) if torch.isfinite(temp_loss) else float('nan'):.6f} "
                f"log_prob_min={actor_dbg.get('log_prob_min', float('nan')):.6f} "
                f"log_prob_max={actor_dbg.get('log_prob_max', float('nan')):.6f} "
                f"shapes(flat={actor_dbg.get('flat_action_shape')},"
                f"log_prob={actor_dbg.get('log_prob_shape')},q_pi={actor_dbg.get('q_pi_shape')}) "
                f"flat_action_dim_cfg={actor_dbg.get('flat_action_dim_cfg', 'NA')}"
            )

        if not torch.isfinite(temp_loss):
            return {
                "temperature": temperature.item(),
                "loss_temperature": float("nan"),
            }

        self.temp_optimizer.zero_grad()
        temp_loss.backward()
        clip_grad_norm_(self.temperature.parameters(), max_norm=self._TEMP_GRAD_CLIP)

        grad_finite = True
        for p in self.temperature.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grad_finite = False
                break
        if grad_finite:
            self.temp_optimizer.step()
            with torch.no_grad():
                self.temperature.log_temperature.clamp_(self._LOG_TEMP_MIN, self._LOG_TEMP_MAX)
        else:
            self.temp_optimizer.zero_grad(set_to_none=True)

        return {
            "temperature": temperature.item(),
            "loss_temperature": temp_loss.item(),
        }
