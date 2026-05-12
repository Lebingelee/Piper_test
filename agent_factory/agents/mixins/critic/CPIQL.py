import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from agent_factory.config.structure import StateEncoderConfig
from agent_factory.modules.critics.cpiql_critic import CPIQLQNet, CPIQLVNet
from agent_factory.modules.encoders.state_encoder import BaseStateEncoder
from agent_factory.modules.encoders.visual_encoder import VisualEncoder


@dataclass
class CPIQLCriticConfig:
    type: str = "cpiql"
    encoder: StateEncoderConfig = field(default_factory=StateEncoderConfig)
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    q_lr: float = 3e-4
    v_lr: float = 3e-4
    expectile: float = 0.7
    k_embed_dim: int = 16
    k_hidden_dims: List[int] = field(default_factory=lambda: [32])
    k_grid: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    low_k_grid: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4])
    gamma_floor: float = 1e-4
    alpha_progress: float = 1.0
    lambda_mono: float = 0.05
    lambda_smooth: float = 0.01
    intervention_terminal_reward: float = 0.2
    intervention_value_scale: float = 0.6
    intervention_reward_min: float = 0.05
    intervention_reward_max: float = 0.6
    intervention_reward_refresh_interval: int = 0
    intervention_progress_weight: float = 0.3
    success_progress_weight: float = 1.0
    anchor_intervention_pseudo: bool = False
    grad_clip_norm: float = 0.0
    target_update_interval: int = 1
    log_interval: int = 100


class CPIQLCriticMixin:
    """
    Failure-Conditioned Progress IQL critic.

    This mixin is intentionally independent from IQLCriticMixin. It keeps the
    IQL-style twin Q + V expectile structure, while conditioning both Q and V
    on a success-bias scalar k and consuming dataset-provided progress signals.
    """

    CONFIG_CLASS = CPIQLCriticConfig
    CONFIG_KEY = "critic"

    REQUIRED_KEYS = {
        "observations",
        "action",
        "next_observations",
        "reward",
        "terminated",
        "discount",
        "k",
        "progress_return",
        "progress_mask",
        "progress_weight",
    }

    def _build_critic(self):
        cfg = self.cfg.critic
        encoder_cfg = cfg.encoder
        use_visual = getattr(self.cfg.dataset, "include_rgb", True)

        vis_enc = None
        if use_visual:
            vis_enc = VisualEncoder(
                in_channels=encoder_cfg.visual.in_channels,
                out_dim=encoder_cfg.visual.out_dim,
                backbone_type=encoder_cfg.visual.backbone_type,
                pool_feature_map=encoder_cfg.visual.pool_feature_map,
                use_group_norm=encoder_cfg.visual.use_group_norm,
            )

        proprio_dim = encoder_cfg.proprio_dim or self.cfg.env.proprio_dim
        self.critic_encoder = BaseStateEncoder(
            visual_encoder=vis_enc,
            proprio_dim=proprio_dim,
            out_dim=encoder_cfg.out_dim,
            num_cameras=self.cfg.env.num_cameras,
            view_fusion=encoder_cfg.view_fusion,
        )

        common_args = dict(
            state_encoder=self.critic_encoder,
            obs_horizon=self.cfg.env.obs_horizon,
            hidden_dims=cfg.hidden_dims,
            k_embed_dim=cfg.k_embed_dim,
            k_hidden_dims=cfg.k_hidden_dims,
        )

        flat_action_dim = self.cfg.env.action_dim * self.cfg.env.pred_horizon
        self.v_net = CPIQLVNet(**common_args)
        self.q_net = CPIQLQNet(flat_action_dim=flat_action_dim, **common_args)
        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_q_net.requires_grad_(False)

        self.v_optimizer = torch.optim.AdamW(self.v_net.parameters(), lr=cfg.v_lr)
        self.q_optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=cfg.q_lr)

    def soft_update_target(self):
        tau = self.cfg.soft_update_tau
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _as_column(self, value: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        value = value.to(self.device, dtype=dtype)
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        return value

    def _prepare_critic_batch(self, batch: Dict[str, Any]):
        obs = self._preprocess_obs(batch["observations"])
        next_obs = self._preprocess_obs(batch["next_observations"])
        actions = batch["action"].to(self.device).float()
        k = self._as_column(batch["k"])
        reward = self._as_column(batch["reward"])
        discount = self._as_column(batch["discount"])
        terminated = self._as_column(batch["terminated"])
        progress_return = self._as_column(batch["progress_return"])
        progress_mask = self._as_column(batch["progress_mask"])
        progress_weight = self._as_column(batch["progress_weight"])
        optional = {}
        for key in (
            "intervention",
            "intervention_segment",
            "segment_end_is_intervention_boundary",
            "success",
            "truncated",
            "segment_type",
            "segment_terminal_reward",
        ):
            if key in batch:
                dtype = torch.long if key == "segment_type" else torch.float32
                optional[key] = self._as_column(batch[key], dtype=dtype)
        return obs, actions, next_obs, k, reward, discount, terminated, progress_return, progress_mask, progress_weight, optional

    def _expectile_loss(self, adv: torch.Tensor) -> torch.Tensor:
        expectile = float(self.cfg.critic.expectile)
        weight = torch.where(adv > 0, expectile, 1.0 - expectile)
        return (weight * adv.pow(2)).mean()

    def _progress_anchor_loss(
        self,
        v_pred: torch.Tensor,
        progress_return: torch.Tensor,
        progress_mask: torch.Tensor,
        progress_weight: torch.Tensor,
    ) -> torch.Tensor:
        weights = progress_mask * progress_weight
        denom = weights.sum().clamp_min(1.0)
        return (weights * (v_pred - progress_return).pow(2)).sum() / denom

    def _k_grid_tensor(self, batch_size: int, value: float, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, 1), float(value), device=device, dtype=torch.float32)

    def _v_k_regularization(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        zero = next(iter(obs.values())).new_tensor(0.0)
        return {"mono": zero, "smooth": zero}
        
        cfg = self.cfg.critic
        k_values = sorted(float(v) for v in cfg.k_grid)
        zero = next(iter(obs.values())).new_tensor(0.0)
        if len(k_values) < 2 or (cfg.lambda_mono <= 0 and cfg.lambda_smooth <= 0):
            return {"mono": zero, "smooth": zero}

        batch_size = next(iter(obs.values())).shape[0]
        values = [
            self.v_net(obs, self._k_grid_tensor(batch_size, k_value, next(iter(obs.values())).device))
            for k_value in k_values
        ]

        mono_terms = []
        smooth_terms = []
        for low_v, high_v in zip(values[:-1], values[1:]):
            mono_terms.append(F.relu(low_v - high_v).pow(2).mean())
            smooth_terms.append((high_v - low_v).pow(2).mean())

        mono = torch.stack(mono_terms).mean() if mono_terms else zero
        smooth = torch.stack(smooth_terms).mean() if smooth_terms else zero
        return {"mono": mono, "smooth": smooth}

    def update_critic(self, batch: Dict[str, Any]) -> Dict[str, float]:
        (
            obs,
            actions,
            next_obs,
            k,
            reward,
            discount,
            terminated,
            progress_return,
            progress_mask,
            progress_weight,
            optional,
        ) = self._prepare_critic_batch(batch)

        with torch.no_grad():
            q1_targ, q2_targ = self.target_q_net(obs, actions, k)
            q_target = torch.min(q1_targ, q2_targ)

        v_pred = self.v_net(obs, k)
        adv = q_target - v_pred
        loss_v_iql = self._expectile_loss(adv)
        loss_v_progress = self._progress_anchor_loss(v_pred, progress_return, progress_mask, progress_weight)
        reg = self._v_k_regularization(obs)
        loss_v = (
            loss_v_iql
            + float(self.cfg.critic.alpha_progress) * loss_v_progress
            + float(self.cfg.critic.lambda_mono) * reg["mono"]
            + float(self.cfg.critic.lambda_smooth) * reg["smooth"]
        )

        self.v_optimizer.zero_grad()
        loss_v.backward()
        if float(getattr(self.cfg.critic, "grad_clip_norm", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), float(self.cfg.critic.grad_clip_norm))
        self.v_optimizer.step()

        with torch.no_grad():
            next_v = self.v_net(next_obs, k)
            q_target_val = reward + discount * next_v * (1.0 - terminated)

        q1_pred, q2_pred = self.q_net(obs, actions, k)
        loss_q = F.mse_loss(q1_pred, q_target_val) + F.mse_loss(q2_pred, q_target_val)

        self.q_optimizer.zero_grad()
        loss_q.backward()
        if float(getattr(self.cfg.critic, "grad_clip_norm", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), float(self.cfg.critic.grad_clip_norm))
        self.q_optimizer.step()

        with torch.no_grad():
            gap = self.critic_gap(obs, preprocessed=True)

        metrics = {
            "loss_q": float(loss_q.item()),
            "loss_v": float(loss_v.item()),
            "loss_v_iql": float(loss_v_iql.item()),
            "loss_v_progress": float(loss_v_progress.item()),
            "loss_v_mono": float(reg["mono"].item()),
            "loss_v_smooth": float(reg["smooth"].item()),
            "adv_mean": float(adv.mean().item()),
            "q_target_mean": float(q_target_val.mean().item()),
            "discount_mean": float(discount.mean().item()),
            "progress_active_ratio": float((progress_mask > 0).float().mean().item()),
            "critic_gap_mean": float(gap.mean().item()),
        }
        if "intervention" in optional:
            metrics["intervention_ratio"] = float(optional["intervention"].float().mean().item())
        if "intervention_segment" in optional:
            metrics["intervention_segment_ratio"] = float(optional["intervention_segment"].float().mean().item())
        if "segment_end_is_intervention_boundary" in optional:
            metrics["intervention_boundary_ratio"] = float(optional["segment_end_is_intervention_boundary"].float().mean().item())
        if "success" in optional:
            metrics["success_ratio"] = float(optional["success"].float().mean().item())
        if "truncated" in optional:
            metrics["truncated_ratio"] = float(optional["truncated"].float().mean().item())
        if "segment_terminal_reward" in optional:
            metrics["segment_terminal_reward_mean"] = float(optional["segment_terminal_reward"].float().mean().item())
        return metrics

    @torch.no_grad()
    def refresh_replay_segment_rewards(
        self,
        dataset,
        segment_indices=None,
        segment_type: str = "",
        k: float = 0.0,
        value_scale: float = None,
        reward_min: float = None,
        reward_max: float = None,
        batch_size: int = 256,
        metric_prefix: str = "segment_reward",
    ) -> Dict[str, float]:
        """
        Refresh terminal rewards for selected replay sub-trajectories from V(o_end, k).
        """
        if not hasattr(dataset, "set_segment_terminal_rewards"):
            return {f"{metric_prefix}_refresh_count": 0.0}
        if segment_indices is None:
            if segment_type:
                if not hasattr(dataset, "segment_indices_by_type"):
                    return {f"{metric_prefix}_refresh_count": 0.0}
                segment_indices = dataset.segment_indices_by_type(segment_type)
            else:
                segment_indices = list(range(len(getattr(dataset, "segment_terminal_rewards", []))))
        else:
            segment_indices = list(segment_indices)
        if not segment_indices:
            return {f"{metric_prefix}_refresh_count": 0.0}

        was_training = self.training
        self.eval()
        rewards = []
        scale = float(value_scale if value_scale is not None else getattr(self.cfg.critic, "intervention_value_scale", 0.6))
        min_value = float(reward_min if reward_min is not None else getattr(self.cfg.critic, "intervention_reward_min", 0.05))
        max_value = float(reward_max if reward_max is not None else getattr(self.cfg.critic, "intervention_reward_max", 0.6))

        for start in range(0, len(segment_indices), int(batch_size)):
            chunk = segment_indices[start:start + int(batch_size)]
            obs_items = [
                dataset._get_obs_sequence(
                    segment_idx,
                    dataset.segment_terminal_indices[segment_idx],
                    dataset.obs_horizon,
                )
                for segment_idx in chunk
            ]
            obs_batch = {
                key: torch.stack([item[key] for item in obs_items], dim=0)
                for key in obs_items[0].keys()
            }
            obs_batch = self._preprocess_obs(obs_batch)
            values = self.predict_v(obs_batch, k, preprocessed=True)
            clipped = torch.clamp(scale * values, min=min_value, max=max_value)
            rewards.extend(clipped.squeeze(-1).detach().cpu().tolist())

        dataset.set_segment_terminal_rewards(rewards, segment_indices=segment_indices, refresh=True)
        if was_training:
            self.train()
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        return {
            f"{metric_prefix}_refresh_count": float(len(rewards)),
            f"{metric_prefix}_mean": float(rewards_tensor.mean().item()),
            f"{metric_prefix}_min": float(rewards_tensor.min().item()),
            f"{metric_prefix}_max": float(rewards_tensor.max().item()),
        }

    @torch.no_grad()
    def refresh_replay_intervention_rewards(self, dataset, batch_size: int = 256) -> Dict[str, float]:
        """
        Refresh pseudo terminal rewards for autonomous segments that end at an
        intervention boundary, using clip(lambda * V(o_end, k=0)).
        """
        if not hasattr(dataset, "intervention_boundary_segment_indices"):
            return {"intervention_reward_refresh_count": 0.0}
        segment_indices = dataset.intervention_boundary_segment_indices()
        return self.refresh_replay_segment_rewards(
            dataset,
            segment_indices=segment_indices,
            k=0.0,
            batch_size=batch_size,
            metric_prefix="intervention_reward",
        )

    def train_critic_step(self, batch: Dict[str, Any], update_target: bool = True) -> Dict[str, float]:
        metrics = self.update_critic(batch)
        self.step += 1
        interval = max(int(getattr(self.cfg.critic, "target_update_interval", 1)), 1)
        if update_target and self.step % interval == 0:
            self.soft_update_target()
        return metrics

    #---------------------------
    #非必要，Critic_mixin的训练循环可以由外部trainer控制，但提供一个默认实现以方便使用,并且集成了定期刷新干预奖励的功能
    #如果用户需要定制训练流程，可以重写该方法或者使用update_critic接口自行控制训练循环
    def train_critic_loop(self, dataloader, num_steps: int, save_dir: str = ""):
        import os
        from tqdm import tqdm

        self.train()
        log_interval = max(int(getattr(self.cfg.critic, "log_interval", 100)), 1)
        save_interval = max(int(getattr(self.cfg.train, "save_interval", num_steps // 4)), 1)

        def infinite_iterator(loader):
            while True:
                for batch in loader:
                    yield batch

        iterator = infinite_iterator(dataloader)
        running: Dict[str, float] = {}
        pbar = tqdm(range(num_steps), desc="Train CPIQL Critic", leave=True)
        for step_idx in pbar:
            refresh_interval = int(getattr(self.cfg.critic, "intervention_reward_refresh_interval", 0))
            if refresh_interval > 0 and step_idx % refresh_interval == 0:
                dataset = getattr(dataloader, "dataset", None)
                if dataset is not None:
                    refresh_metrics = self.refresh_replay_intervention_rewards(dataset)
                    for key, value in refresh_metrics.items():
                        running[key] = running.get(key, 0.0) + float(value)

            batch = self._batch_to_device_critic_only(next(iterator)) if hasattr(self, "_batch_to_device") else next(iterator)
            metrics = self.train_critic_step(batch)
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value)

            if (step_idx + 1) % log_interval == 0:
                shown = {
                    "q": running.get("loss_q", 0.0) / log_interval,
                    "v": running.get("loss_v", 0.0) / log_interval,
                    "gap": running.get("critic_gap_mean", 0.0) / log_interval,
                }
                if "intervention_ratio" in running:
                    shown["intervene"] = running["intervention_ratio"] / log_interval
                pbar.set_postfix(shown)
                running = {}

            if save_dir and (step_idx + 1) % save_interval == 0:
                os.makedirs(save_dir, exist_ok=True)
                self.save(os.path.join(save_dir, f"cpiql_critic_step_{step_idx + 1}.pth"), meta={"mode": "cpiql_critic"})
    def _batch_to_device_critic_only(self, batch: Any) -> Any:
        """
        递归地将 batch 中的所有 Tensor 移动到 self.device。
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._batch_to_device_critic_only(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._batch_to_device_critic_only(v) for v in batch]
        else:
            return batch
        
    #----------------------------





    def _prepare_inference_obs(self, obs: Dict[str, Any], preprocessed: bool = False) -> Dict[str, torch.Tensor]:
        return obs if preprocessed else self._preprocess_obs(obs)

    def _prepare_inference_k(self, k: Any, batch_size: int) -> torch.Tensor:
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, dtype=torch.float32, device=self.device)
        else:
            k = k.to(self.device).float()
        if k.ndim == 0:
            k = k.reshape(1, 1).repeat(batch_size, 1)
        elif k.ndim == 1:
            if k.numel() == 1:
                k = k.reshape(1, 1).repeat(batch_size, 1)
            else:
                k = k.reshape(batch_size, 1)
        return k

    @torch.no_grad()
    def predict_v(self, obs: Dict[str, Any], k: Any, preprocessed: bool = False) -> torch.Tensor:
        obs = self._prepare_inference_obs(obs, preprocessed=preprocessed)
        batch_size = next(iter(obs.values())).shape[0]
        k_tensor = self._prepare_inference_k(k, batch_size)
        return self.v_net(obs, k_tensor)

    @torch.no_grad()
    def predict_q(
        self,
        obs: Dict[str, Any],
        action: torch.Tensor,
        k: Any,
        preprocessed: bool = False,
    ) -> torch.Tensor:
        obs = self._prepare_inference_obs(obs, preprocessed=preprocessed)
        batch_size = next(iter(obs.values())).shape[0]
        action = action.to(self.device).float()
        k_tensor = self._prepare_inference_k(k, batch_size)
        q1, q2 = self.q_net(obs, action, k_tensor)
        return torch.min(q1, q2)

    @torch.no_grad()
    def critic_gap(self, obs: Dict[str, Any], preprocessed: bool = False) -> torch.Tensor:
        obs = self._prepare_inference_obs(obs, preprocessed=preprocessed)
        v_success = self.predict_v(obs, 1.0, preprocessed=True)
        v_realistic = self.predict_v(obs, 0.0, preprocessed=True)
        return v_success - v_realistic

    @torch.no_grad()
    def compute_advantage(
        self,
        obs: Dict[str, Any],
        action: torch.Tensor,
        k: Any,
        preprocessed: bool = False,
    ) -> torch.Tensor:
        obs = self._prepare_inference_obs(obs, preprocessed=preprocessed)
        q = self.predict_q(obs, action, k, preprocessed=True)
        v = self.predict_v(obs, k, preprocessed=True)
        return q - v
