import copy
import time
from typing import Dict, Optional

import torch

from agent_factory.data.normalization import get_action_normalizer
from agent_factory.config.structure import DSRLCriticConfig
from agent_factory.modules.critics.dsrl_critic import DSRLQEnsemble
from agent_factory.modules.encoders.state_encoder import BaseStateEncoder
from agent_factory.modules.encoders.visual_encoder import VisualEncoder


class DSRLCriticMixin:
    CONFIG_CLASS = DSRLCriticConfig
    CONFIG_KEY = "critic"
    REQUIRED_KEYS = {"observations", "next_observations", "action", "reward", "terminated", "discount"}
    _LOG_PROB_CLAMP = 1e6

    def _sync_cuda_for_timing(self):
        if torch.cuda.is_available() and getattr(self, "device", None) is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def _build_critic(self):
        cfg: DSRLCriticConfig = self.cfg.critic
        encoder_cfg = cfg.encoder
        use_visual = getattr(self.cfg.dataset, "include_rgb", True)

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

        def build_encoder():
            return BaseStateEncoder(
                visual_encoder=copy.deepcopy(visual_encoder) if visual_encoder is not None else None,
                proprio_dim=proprio_dim,
                out_dim=encoder_cfg.out_dim,
                num_cameras=self.cfg.env.num_cameras,
                view_fusion=encoder_cfg.view_fusion,
            )

        self.action_critic_encoder = build_encoder()
        self.noise_critic_encoder = build_encoder()

        self.action_q_net = DSRLQEnsemble(
            state_encoder=self.action_critic_encoder,
            obs_horizon=self.cfg.actor.obs_horizon,
            action_dim=self.cfg.env.action_dim,
            pred_horizon=self.cfg.env.pred_horizon,
            hidden_dims=cfg.hidden_dims,
            num_qs=cfg.num_qs,
        )
        self.target_action_q_net = copy.deepcopy(self.action_q_net)
        self.target_action_q_net.requires_grad_(False)

        self.noise_q_net = DSRLQEnsemble(
            state_encoder=self.noise_critic_encoder,
            obs_horizon=self.cfg.actor.obs_horizon,
            action_dim=self.cfg.actor.action_dim,
            pred_horizon=self.cfg.actor.pred_horizon,
            hidden_dims=cfg.hidden_dims,
            num_qs=cfg.num_qs,
        )
        self.target_noise_q_net = copy.deepcopy(self.noise_q_net)
        self.target_noise_q_net.requires_grad_(False)

        self.action_q_optimizer = torch.optim.Adam(self.action_q_net.parameters(), lr=cfg.lr)
        self.noise_q_optimizer = torch.optim.Adam(self.noise_q_net.parameters(), lr=cfg.lr)

        norm_cfg = getattr(cfg, "action_norm", None)
        norm_type = getattr(norm_cfg, "type", None) if norm_cfg else None
        norm_params = norm_cfg.params if norm_cfg and getattr(norm_cfg, "params", None) else {}
        self.action_normalizer = get_action_normalizer(
            norm_type=norm_type,
            action_dim=int(self.cfg.env.action_dim),
            **norm_params,
        ).to(self.device)

    def fit_action_normalizer(self, raw_actions: torch.Tensor):
        if raw_actions.ndim > 2:
            raw_actions = raw_actions.reshape(-1, raw_actions.shape[-1])
        self.action_normalizer.fit(raw_actions)

    def _normalize_env_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_normalizer.normalize(action.float())

    def _reduce_q(self, q_values: torch.Tensor, reduction: Optional[str] = None) -> torch.Tensor:
        reduction = reduction or self.cfg.critic.critic_reduction
        if reduction == "min":
            return q_values.min(dim=0).values
        if reduction == "mean":
            return q_values.mean(dim=0)
        raise ValueError(f"Unsupported critic reduction: {reduction}")

    def soft_update_target(self):
        tau = self.cfg.soft_update_tau
        for source, target in (
            (self.action_q_net, self.target_action_q_net),
            (self.noise_q_net, self.target_noise_q_net),
        ):
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - tau).add_(tau * param.data)

    def update_action_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.ensure_base_policy()

        self._sync_cuda_for_timing()
        t0 = time.perf_counter()

        obs_raw = batch["observations"]
        next_obs_raw = batch["next_observations"]
        obs = self._preprocess_obs(obs_raw)
        next_obs = self._preprocess_obs(next_obs_raw)
        actions = self._normalize_env_action(batch["action"])
        rewards_raw = batch["reward"].float().view(-1)
        reward_scale = float(getattr(self.cfg.env, "reward_scale", 1.0))
        if reward_scale <= 0:
            reward_scale = 1.0
        rewards = rewards_raw / reward_scale
        terminated = batch["terminated"].float().view(-1)
        discount = batch["discount"].float().view(-1)

        self._sync_cuda_for_timing()
        t1 = time.perf_counter()

        with torch.no_grad():
            self._sync_cuda_for_timing()
            t_pd0 = time.perf_counter()
            next_dist = self._policy_dist(next_obs_raw)
            next_noise_flat = next_dist.rsample()
            next_noise = self._reshape_noise_action(next_noise_flat)
            next_log_prob = next_dist.log_prob(next_noise_flat)
            next_log_prob = torch.nan_to_num(
                next_log_prob,
                nan=0.0,
                posinf=self._LOG_PROB_CLAMP,
                neginf=-self._LOG_PROB_CLAMP,
            )
            if hasattr(self, "_entropy_log_prob"):
                next_log_prob = self._entropy_log_prob(next_log_prob)
            self._sync_cuda_for_timing()
            t_pd1 = time.perf_counter()

            self._sync_cuda_for_timing()
            t_bp0 = time.perf_counter()
            next_env_action = self._call_base_policy(next_obs_raw, next_noise)
            next_env_action = self._normalize_env_action(next_env_action)
            self._sync_cuda_for_timing()
            t_bp1 = time.perf_counter()

            self._sync_cuda_for_timing()
            t_tq0 = time.perf_counter()
            next_qs = self.target_action_q_net(next_obs, next_env_action)
            next_q = self._reduce_q(next_qs, reduction=self.cfg.critic.critic_backup_combine_type)

            target_q = rewards + discount * (1.0 - terminated) * next_q
            if self.cfg.critic.backup_entropy:
                target_q = target_q - discount * (1.0 - terminated) * self.temperature() * next_log_prob
            self._sync_cuda_for_timing()
            t_tq1 = time.perf_counter()

        self._sync_cuda_for_timing()
        t_cq0 = time.perf_counter()
        current_qs = self.action_q_net(obs, actions)
        critic_loss = ((current_qs - target_q.unsqueeze(0)) ** 2).mean()

        self.action_q_optimizer.zero_grad()
        critic_loss.backward()
        self.action_q_optimizer.step()
        self._sync_cuda_for_timing()
        t_cq1 = time.perf_counter()

        return {
            "loss_action_critic": critic_loss.item(),
            "action_q_mean": current_qs.mean().item(),
            "action_target_q": target_q.mean().item(),
            "reward_raw_mean": rewards_raw.mean().item(),
            "reward_scaled_mean": rewards.mean().item(),
            "time_ms_action_preprocess": (t1 - t0) * 1000.0,
            "time_ms_action_policy_dist": (t_pd1 - t_pd0) * 1000.0,
            "time_ms_action_base_infer": (t_bp1 - t_bp0) * 1000.0,
            "time_ms_action_target_q": (t_tq1 - t_tq0) * 1000.0,
            "time_ms_action_q_fwb": (t_cq1 - t_cq0) * 1000.0,
            "time_ms_action_total": (t_cq1 - t0) * 1000.0,
        }

    def update_noise_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.ensure_base_policy()

        self._sync_cuda_for_timing()
        t0 = time.perf_counter()

        obs_raw = batch["observations"]
        obs = self._preprocess_obs(obs_raw)
        num_steps = max(int(self.cfg.critic.noise_critic_grad_steps), 1)

        self._sync_cuda_for_timing()
        t1 = time.perf_counter()

        total_loss = 0.0
        total_target = 0.0
        total_q = 0.0
        time_policy_dist_ms = 0.0
        time_base_infer_ms = 0.0
        time_target_q_ms = 0.0
        time_q_fwb_ms = 0.0
        for _ in range(num_steps):
            with torch.no_grad():
                self._sync_cuda_for_timing()
                t_pd0 = time.perf_counter()
                dist = self._policy_dist(obs_raw)
                noise_flat = dist.rsample()
                noise_action = self._reshape_noise_action(noise_flat)
                self._sync_cuda_for_timing()
                t_pd1 = time.perf_counter()

                self._sync_cuda_for_timing()
                t_bp0 = time.perf_counter()
                env_action = self._call_base_policy(obs_raw, noise_action)
                env_action = self._normalize_env_action(env_action)
                self._sync_cuda_for_timing()
                t_bp1 = time.perf_counter()

                self._sync_cuda_for_timing()
                t_tq0 = time.perf_counter()
                action_qs = self.target_action_q_net(obs, env_action)
                target_q = self._reduce_q(action_qs, reduction=self.cfg.critic.critic_backup_combine_type)
                self._sync_cuda_for_timing()
                t_tq1 = time.perf_counter()

            self._sync_cuda_for_timing()
            t_q0 = time.perf_counter()
            current_qs = self.noise_q_net(obs, noise_action)
            noise_critic_loss = ((current_qs - target_q.unsqueeze(0)) ** 2).mean()

            self.noise_q_optimizer.zero_grad()
            noise_critic_loss.backward()
            self.noise_q_optimizer.step()
            self._sync_cuda_for_timing()
            t_q1 = time.perf_counter()

            total_loss += float(noise_critic_loss.item())
            total_target += float(target_q.mean().item())
            total_q += float(current_qs.mean().item())
            time_policy_dist_ms += (t_pd1 - t_pd0) * 1000.0
            time_base_infer_ms += (t_bp1 - t_bp0) * 1000.0
            time_target_q_ms += (t_tq1 - t_tq0) * 1000.0
            time_q_fwb_ms += (t_q1 - t_q0) * 1000.0

        self._sync_cuda_for_timing()
        t2 = time.perf_counter()

        return {
            "loss_noise_critic": total_loss / num_steps,
            "noise_q_mean": total_q / num_steps,
            "noise_target_q": total_target / num_steps,
            "time_ms_noise_preprocess": (t1 - t0) * 1000.0,
            "time_ms_noise_policy_dist": time_policy_dist_ms,
            "time_ms_noise_base_infer": time_base_infer_ms,
            "time_ms_noise_target_q": time_target_q_ms,
            "time_ms_noise_q_fwb": time_q_fwb_ms,
            "time_ms_noise_loop_total": (t2 - t1) * 1000.0,
            "time_ms_noise_total": (t2 - t0) * 1000.0,
        }

    def update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        action_metrics = self.update_action_critic(batch)
        noise_metrics = self.update_noise_critic(batch)
        return {
            **action_metrics,
            **noise_metrics,
        }
