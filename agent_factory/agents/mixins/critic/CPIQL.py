import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from agent_factory.agents.mixins.critic.base_eval import CriticEvalMixinBase
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
    gamma_floor: float = 1e-4
    alpha_progress: float = 1.0
    lambda_mono: float = 0.05
    lambda_smooth: float = 0.0
    intervention_terminal_reward: float = 0.2
    intervention_value_scale: float = 0.6
    intervention_reward_min: float = 0.05
    intervention_reward_max: float = 0.6
    intervention_reward_refresh_interval: int = 0
    intervention_progress_weight: float = 0.3
    success_progress_weight: float = 1.0
    failure_progress_weight: float = 1.0
    anchor_intervention_pseudo: bool = False
    use_per_k_backward: bool = False
    num_k_samples: Any = "all"
    grad_clip_norm: float = 0.0
    target_update_interval: int = 1
    log_interval: int = 100


class CPIQLCriticMixin(CriticEvalMixinBase):
    """
    Failure-Conditioned Progress IQL critic.

    This mixin is intentionally independent from IQLCriticMixin. It keeps the
    IQL-style twin Q + V expectile structure, while conditioning both Q and V
    on a success-bias scalar k and consuming dataset-provided progress signals.

    Weighting summary for different data types
    ------------------------------------------
    There are four multiplicative factors in critic training:

    1. distribution_mask
       - Decides whether the sample belongs to D_k.
       - success-side samples: always active for every k
       - failure-side samples: active only when failure_rank >= k
       - k=1: all failure-side samples are masked out

    2. class_balance_weight
       - Fixed once from the k=0 dataset composition.
       - Let A0 = number of active success-side slices at k=0
       - Let B0 = number of active failure-side slices at k=0
       - success_class_weight = (A0 + B0) / (2 * A0)
       - failure_class_weight = (A0 + B0) / (2 * B0)
       - This makes k=0 approximately success:failure = 1:1 in total loss mass.
       - For k>0, failure influence decays naturally because fewer failures stay active.

    3. progress_mask
       - Controls how much the sample participates in MC/progress supervision.
       - success-side samples: 1.0
       - failure-side samples: failure_progress_weight
       - intervention-boundary pseudo samples: optionally forced to 1.0 when
         anchor_intervention_pseudo=True

    4. progress_weight
       - Extra scale on top of progress_mask for MC/progress supervision.
       - default baseline: success_progress_weight
       - intervention-boundary pseudo samples: multiplied by intervention_progress_weight

    Final loss usage
    ----------------
    - V expectile / IQL loss:
        distribution_mask * class_balance_weight
    - Q TD loss:
        distribution_mask * class_balance_weight
    - V progress / MC anchor loss:
        distribution_mask * class_balance_weight * progress_mask * progress_weight

    Default interpretation with the current config
    ----------------------------------------------
    - normal success sample:
        progress_mask=1.0, progress_weight=1.0, class=success_class_weight
    - normal failure sample:
        progress_mask=failure_progress_weight, progress_weight=1.0, class=failure_class_weight
    - intervention-boundary pseudo sample:
        progress_mask=1.0 only if anchor_intervention_pseudo=True,
        progress_weight=success_progress_weight * intervention_progress_weight

    中文说明
    --------
    当前 critic 中，不同类型样本的训练权重由四层相乘得到：

    1. distribution_mask
       - 决定该样本在当前 k 下是否属于 D_k。
       - 成功侧样本：对所有 k 都有效。
       - 失败侧样本：只有当 failure_rank >= k 时才有效。
       - 当 k=1 时，所有失败侧样本都会被屏蔽掉。

    2. class_balance_weight
       - 按 k=0 时整个训练集的 success/failure slice 比例一次性统计。
       - 设 A0 为 k=0 时 success-side slice 数，B0 为 k=0 时 failure-side slice 数。
       - success_class_weight = (A0 + B0) / (2 * A0)
       - failure_class_weight = (A0 + B0) / (2 * B0)
       - 作用是让 k=0 时 success 和 failure 在总 loss 质量上近似 1:1。
       - 当 k 逐渐增大时，由于 active failure 数减少，failure 的总影响会自然衰减。

    3. progress_mask
       - 决定样本在 MC/progress supervision 中参与多少。
       - 成功侧样本：固定为 1.0。
       - 失败侧样本：固定为 failure_progress_weight。
       - intervention-boundary pseudo 样本：若 anchor_intervention_pseudo=True，
         则会被强制设为 1.0。

    4. progress_weight
       - 在 progress_mask 之外，再乘一个额外缩放系数。
       - 默认基础值为 success_progress_weight。
       - intervention-boundary pseudo 样本会再乘 intervention_progress_weight。

    最终进入各项 loss 的方式
    ------------------------
    - V 的 expectile / IQL loss:
        distribution_mask * class_balance_weight
    - Q 的 TD loss:
        distribution_mask * class_balance_weight
    - V 的 progress / MC anchor loss:
        distribution_mask * class_balance_weight * progress_mask * progress_weight

    按当前默认配置可直观理解为
    --------------------------
    - 普通成功样本：
        progress_mask=1.0, progress_weight=1.0, 类别权重为 success_class_weight
    - 普通失败样本：
        progress_mask=failure_progress_weight, progress_weight=1.0, 类别权重为 failure_class_weight
    - intervention-boundary pseudo 样本：
        只有 anchor_intervention_pseudo=True 时才参与 progress anchor，
        且其 progress_weight 会再乘 intervention_progress_weight
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
        "progress_return",
        "progress_mask",
        "progress_weight",
        "failure_rank",
        "is_success_segment",
        "is_failure_segment",
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
            visual_feature_dim=encoder_cfg.visual.out_dim if vis_enc is not None else None,
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
        self.use_per_k_backward = bool(getattr(cfg, "use_per_k_backward", False))
        self.success_class_weight = 1.0
        self.failure_class_weight = 1.0
        self.k0_success_count = 0
        self.k0_failure_count = 0

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
        reward = self._as_column(batch["reward"])
        discount = self._as_column(batch["discount"])
        terminated = self._as_column(batch["terminated"])
        progress_return = self._as_column(batch["progress_return"])
        progress_mask = self._as_column(batch["progress_mask"])
        progress_weight = self._as_column(batch["progress_weight"])
        failure_rank = self._as_column(batch["failure_rank"])
        is_success_segment = self._as_column(batch["is_success_segment"])
        is_failure_segment = self._as_column(batch["is_failure_segment"])
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
        return (
            obs,
            actions,
            next_obs,
            reward,
            discount,
            terminated,
            progress_return,
            progress_mask,
            progress_weight,
            failure_rank,
            is_success_segment,
            is_failure_segment,
            optional,
        )

    def _expectile_loss(self, adv: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        expectile = float(self.cfg.critic.expectile)
        expectile_weight = torch.where(adv > 0, expectile, 1.0 - expectile)
        weights = weights.to(device=adv.device, dtype=adv.dtype)
        denom = weights.sum().clamp_min(1.0)
        return (weights * expectile_weight * adv.pow(2)).sum() / denom

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(device=pred.device, dtype=pred.dtype)
        denom = weights.sum().clamp_min(1.0)
        return (weights * (pred - target).pow(2)).sum() / denom

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

    def _v_k_regularization(
        self,
        obs: Dict[str, torch.Tensor],
        k_values: Sequence[float] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg.critic
        k_values = sorted(float(v) for v in (k_values if k_values is not None else cfg.k_grid))
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
            if cfg.lambda_smooth > 0:
                smooth_terms.append((high_v - low_v).pow(2).mean())

        mono = torch.stack(mono_terms).mean() if mono_terms else zero
        smooth = torch.stack(smooth_terms).mean() if smooth_terms else zero
        return {"mono": mono, "smooth": smooth}

    def _k_values(self) -> List[float]:
        k_values = sorted(float(v) for v in getattr(self.cfg.critic, "k_grid", [0.0, 1.0]))
        return k_values if k_values else [0.0, 1.0]

    def _collect_k0_balance_counts(self, dataset: Any) -> Dict[str, int]:
        if dataset is None:
            return {"success": 0, "failure": 0}
        if hasattr(dataset, "datasets"):
            total_success = 0
            total_failure = 0
            for child in dataset.datasets:
                counts = self._collect_k0_balance_counts(child)
                total_success += int(counts["success"])
                total_failure += int(counts["failure"])
            return {"success": total_success, "failure": total_failure}

        if not hasattr(dataset, "slices_all"):
            return {"success": 0, "failure": 0}

        success_count = 0
        failure_count = 0
        for traj_idx, _start, _end, _global_idx in dataset.slices_all:
            if hasattr(dataset, "_is_success_side_segment") and dataset._is_success_side_segment(traj_idx):
                success_count += 1
            elif hasattr(dataset, "_is_failure_side_segment") and dataset._is_failure_side_segment(traj_idx):
                failure_count += 1
        return {"success": success_count, "failure": failure_count}

    def _configure_k0_balance_weights(self, dataset: Any) -> Dict[str, float]:
        counts = self._collect_k0_balance_counts(dataset)
        success_count = int(counts["success"])
        failure_count = int(counts["failure"])
        self.k0_success_count = success_count
        self.k0_failure_count = failure_count

        if success_count <= 0 or failure_count <= 0:
            self.success_class_weight = 1.0
            self.failure_class_weight = 1.0
        else:
            total = float(success_count + failure_count)
            self.success_class_weight = total / (2.0 * float(success_count))
            self.failure_class_weight = total / (2.0 * float(failure_count))

        return {
            "k0_success_count": float(self.k0_success_count),
            "k0_failure_count": float(self.k0_failure_count),
            "success_class_weight": float(self.success_class_weight),
            "failure_class_weight": float(self.failure_class_weight),
        }

    def _update_k_values(self) -> List[float]:
        k_values = self._k_values()
        raw_num_samples = getattr(self.cfg.critic, "num_k_samples", "all")

        if raw_num_samples is None:
            return k_values
        if isinstance(raw_num_samples, str):
            sample_text = raw_num_samples.strip().lower()
            if sample_text in ("", "all"):
                return k_values
            try:
                num_samples = int(sample_text)
            except ValueError:
                return k_values
        else:
            num_samples = int(raw_num_samples)

        if num_samples <= 0 or num_samples >= len(k_values):
            return k_values

        indices = torch.randperm(len(k_values))[:num_samples].tolist()
        return sorted(k_values[index] for index in indices)

    def _distribution_mask(
        self,
        k_value: float,
        failure_rank: torch.Tensor,
        is_success_segment: torch.Tensor,
        is_failure_segment: torch.Tensor,
    ) -> torch.Tensor:
        success_mask = (is_success_segment > 0).float()
        if float(k_value) >= 1.0:
            return success_mask
        failure_mask = ((is_failure_segment > 0) & (failure_rank >= float(k_value))).float()
        return torch.clamp(success_mask + failure_mask, max=1.0)

    def _class_balanced_weights(
        self,
        is_success_segment: torch.Tensor,
        is_failure_segment: torch.Tensor,
    ) -> torch.Tensor:
        success_mask = (is_success_segment > 0).float()
        failure_mask = (is_failure_segment > 0).float()
        weights = torch.ones_like(success_mask)
        weights = weights + success_mask * (float(self.success_class_weight) - 1.0)
        weights = weights + failure_mask * (float(self.failure_class_weight) - 1.0)
        return weights

    def update_critic(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if bool(getattr(self, "use_per_k_backward", getattr(self.cfg.critic, "use_per_k_backward", False))):
            return self._update_critic_per_k_backward(batch)
        return self._update_critic_all_k_graph(batch)

    def _backward_v_k_regularization_per_k(
        self,
        obs: Dict[str, torch.Tensor],
        k_values: Sequence[float],
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg.critic
        zero = next(iter(obs.values())).new_tensor(0.0)
        if len(k_values) < 2 or (cfg.lambda_mono <= 0 and cfg.lambda_smooth <= 0):
            return {"mono": zero, "smooth": zero}

        pair_count = max(len(k_values) - 1, 1)
        mono_total = zero
        smooth_total = zero
        batch_size = next(iter(obs.values())).shape[0]
        device = next(iter(obs.values())).device

        for low_k, high_k in zip(k_values[:-1], k_values[1:]):
            low_v = self.v_net(obs, self._k_grid_tensor(batch_size, low_k, device))
            high_v = self.v_net(obs, self._k_grid_tensor(batch_size, high_k, device))

            loss_terms = []
            mono = F.relu(low_v - high_v).pow(2).mean()
            mono_total = mono_total + mono.detach() / pair_count
            if cfg.lambda_mono > 0:
                loss_terms.append(float(cfg.lambda_mono) * mono / pair_count)

            if cfg.lambda_smooth > 0:
                smooth = (high_v - low_v).pow(2).mean()
                smooth_total = smooth_total + smooth.detach() / pair_count
                loss_terms.append(float(cfg.lambda_smooth) * smooth / pair_count)

            if loss_terms:
                torch.stack(loss_terms).sum().backward()

        return {"mono": mono_total, "smooth": smooth_total}

    def _update_critic_per_k_backward(self, batch: Dict[str, Any]) -> Dict[str, float]:
        (
            obs,
            actions,
            next_obs,
            reward,
            discount,
            terminated,
            progress_return,
            progress_mask,
            progress_weight,
            failure_rank,
            is_success_segment,
            is_failure_segment,
            optional,
        ) = self._prepare_critic_batch(batch)

        k_values = self._update_k_values()
        num_k = max(len(k_values), 1)
        batch_size = actions.shape[0]
        device = actions.device

        loss_v_iql_value = 0.0
        loss_v_progress_value = 0.0
        active_ratios = []
        adv_means = []
        class_balance = self._class_balanced_weights(is_success_segment, is_failure_segment)

        self.v_optimizer.zero_grad()
        for k_value in k_values:
            k_tensor = self._k_grid_tensor(batch_size, k_value, device)
            distribution_mask = self._distribution_mask(k_value, failure_rank, is_success_segment, is_failure_segment)
            weighted_distribution = distribution_mask * class_balance
            active_ratios.append(distribution_mask.detach().mean())

            with torch.no_grad():
                q1_targ, q2_targ = self.target_q_net(obs, actions, k_tensor)
                q_target = torch.min(q1_targ, q2_targ)

            v_pred = self.v_net(obs, k_tensor)
            adv = q_target - v_pred
            loss_v_iql_k = self._expectile_loss(adv, weighted_distribution)

            success_anchor_weight = weighted_distribution * progress_mask * progress_weight
            loss_v_progress_k = self._progress_anchor_loss(
                v_pred,
                progress_return,
                success_anchor_weight,
                torch.ones_like(success_anchor_weight),
            )
            loss_v_k = (loss_v_iql_k + float(self.cfg.critic.alpha_progress) * loss_v_progress_k) / float(num_k)
            loss_v_k.backward()

            loss_v_iql_value += float(loss_v_iql_k.detach().item()) / float(num_k)
            loss_v_progress_value += float(loss_v_progress_k.detach().item()) / float(num_k)
            adv_means.append(adv.detach().mean())

        reg = self._backward_v_k_regularization_per_k(obs, k_values)
        if float(getattr(self.cfg.critic, "grad_clip_norm", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), float(self.cfg.critic.grad_clip_norm))
        self.v_optimizer.step()

        loss_q_value = 0.0
        q_target_means = []
        self.q_optimizer.zero_grad()
        for k_value in k_values:
            k_tensor = self._k_grid_tensor(batch_size, k_value, device)
            distribution_mask = self._distribution_mask(k_value, failure_rank, is_success_segment, is_failure_segment)
            weighted_distribution = distribution_mask * class_balance
            with torch.no_grad():
                next_v = self.v_net(next_obs, k_tensor)
                q_target_val = reward + discount * next_v * (1.0 - terminated)
                q_target_means.append((distribution_mask * q_target_val).sum() / distribution_mask.sum().clamp_min(1.0))

            q1_pred, q2_pred = self.q_net(obs, actions, k_tensor)
            loss_q_k = (
                self._masked_mse(q1_pred, q_target_val, weighted_distribution)
                + self._masked_mse(q2_pred, q_target_val, weighted_distribution)
            )
            (loss_q_k / float(num_k)).backward()
            loss_q_value += float(loss_q_k.detach().item()) / float(num_k)

        if float(getattr(self.cfg.critic, "grad_clip_norm", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), float(self.cfg.critic.grad_clip_norm))
        self.q_optimizer.step()

        with torch.no_grad():
            gap = self.critic_gap(obs, preprocessed=True)
            penalty_k0 = self.v_net.decompose(obs, self._k_grid_tensor(batch_size, 0.0, device))
            penalty_k1 = self.v_net.decompose(obs, self._k_grid_tensor(batch_size, 1.0, device))
            effective_penalty_k0 = penalty_k0["k_gate"] * penalty_k0["penalty"]
            effective_penalty_k1 = penalty_k1["k_gate"] * penalty_k1["penalty"]

        loss_v_value = (
            loss_v_iql_value
            + float(self.cfg.critic.alpha_progress) * loss_v_progress_value
            + float(self.cfg.critic.lambda_mono) * float(reg["mono"].item())
            + float(self.cfg.critic.lambda_smooth) * float(reg["smooth"].item())
        )
        metrics = {
            "loss_q": float(loss_q_value),
            "loss_v": float(loss_v_value),
            "loss_v_iql": float(loss_v_iql_value),
            "loss_v_progress": float(loss_v_progress_value),
            "loss_v_success_anchor": float(loss_v_progress_value),
            "loss_v_iql_grid": float(loss_v_iql_value),
            "loss_q_grid": float(loss_q_value),
            "loss_v_mono": float(reg["mono"].item()),
            "loss_v_smooth": float(reg["smooth"].item()),
            "k_update_count": float(num_k),
            "k_grid_count": float(len(self._k_values())),
            "adv_mean": float(torch.stack(adv_means).mean().item()),
            "q_target_mean": float(torch.stack(q_target_means).mean().item()),
            "discount_mean": float(discount.mean().item()),
            "progress_active_ratio": float((progress_mask > 0).float().mean().item()),
            "rank_active_ratio": float(torch.stack(active_ratios).mean().item()),
            "success_class_weight": float(self.success_class_weight),
            "failure_class_weight": float(self.failure_class_weight),
            "k0_success_count": float(self.k0_success_count),
            "k0_failure_count": float(self.k0_failure_count),
            "penalty_mean_k0": float(effective_penalty_k0.mean().item()),
            "penalty_mean_k1": float(effective_penalty_k1.mean().item()),
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

    def _update_critic_all_k_graph(self, batch: Dict[str, Any]) -> Dict[str, float]:
        (
            obs,
            actions,
            next_obs,
            reward,
            discount,
            terminated,
            progress_return,
            progress_mask,
            progress_weight,
            failure_rank,
            is_success_segment,
            is_failure_segment,
            optional,
        ) = self._prepare_critic_batch(batch)

        k_values = self._update_k_values()
        batch_size = actions.shape[0]
        device = actions.device
        loss_v_iql_terms = []
        loss_v_progress_terms = []
        active_ratios = []
        class_balance = self._class_balanced_weights(is_success_segment, is_failure_segment)

        for k_value in k_values:
            k_tensor = self._k_grid_tensor(batch_size, k_value, device)
            distribution_mask = self._distribution_mask(k_value, failure_rank, is_success_segment, is_failure_segment)
            weighted_distribution = distribution_mask * class_balance
            active_ratios.append(distribution_mask.mean())

            with torch.no_grad():
                q1_targ, q2_targ = self.target_q_net(obs, actions, k_tensor)
                q_target = torch.min(q1_targ, q2_targ)

            v_pred = self.v_net(obs, k_tensor)
            adv = q_target - v_pred
            loss_v_iql_terms.append(self._expectile_loss(adv, weighted_distribution))

            success_anchor_weight = weighted_distribution * progress_mask * progress_weight
            loss_v_progress_terms.append(
                self._progress_anchor_loss(v_pred, progress_return, success_anchor_weight, torch.ones_like(success_anchor_weight))
            )

        loss_v_iql = torch.stack(loss_v_iql_terms).mean()
        loss_v_progress = torch.stack(loss_v_progress_terms).mean()
        reg = self._v_k_regularization(obs, k_values)
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

        loss_q_terms = []
        q_target_means = []
        for k_value in k_values:
            k_tensor = self._k_grid_tensor(batch_size, k_value, device)
            distribution_mask = self._distribution_mask(k_value, failure_rank, is_success_segment, is_failure_segment)
            weighted_distribution = distribution_mask * class_balance
            with torch.no_grad():
                next_v = self.v_net(next_obs, k_tensor)
                q_target_val = reward + discount * next_v * (1.0 - terminated)
                q_target_means.append((distribution_mask * q_target_val).sum() / distribution_mask.sum().clamp_min(1.0))

            q1_pred, q2_pred = self.q_net(obs, actions, k_tensor)
            loss_q_terms.append(
                self._masked_mse(q1_pred, q_target_val, weighted_distribution)
                + self._masked_mse(q2_pred, q_target_val, weighted_distribution)
            )
        loss_q = torch.stack(loss_q_terms).mean()

        self.q_optimizer.zero_grad()
        loss_q.backward()
        if float(getattr(self.cfg.critic, "grad_clip_norm", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), float(self.cfg.critic.grad_clip_norm))
        self.q_optimizer.step()

        with torch.no_grad():
            gap = self.critic_gap(obs, preprocessed=True)
            penalty_k0 = self.v_net.decompose(obs, self._k_grid_tensor(batch_size, 0.0, device))
            penalty_k1 = self.v_net.decompose(obs, self._k_grid_tensor(batch_size, 1.0, device))
            effective_penalty_k0 = penalty_k0["k_gate"] * penalty_k0["penalty"]
            effective_penalty_k1 = penalty_k1["k_gate"] * penalty_k1["penalty"]

        metrics = {
            "loss_q": float(loss_q.item()),
            "loss_v": float(loss_v.item()),
            "loss_v_iql": float(loss_v_iql.item()),
            "loss_v_progress": float(loss_v_progress.item()),
            "loss_v_success_anchor": float(loss_v_progress.item()),
            "loss_v_iql_grid": float(loss_v_iql.item()),
            "loss_q_grid": float(loss_q.item()),
            "loss_v_mono": float(reg["mono"].item()),
            "loss_v_smooth": float(reg["smooth"].item()),
            "k_update_count": float(len(k_values)),
            "k_grid_count": float(len(self._k_values())),
            "adv_mean": float(adv.detach().mean().item()),
            "q_target_mean": float(torch.stack(q_target_means).mean().item()),
            "discount_mean": float(discount.mean().item()),
            "progress_active_ratio": float((progress_mask > 0).float().mean().item()),
            "rank_active_ratio": float(torch.stack(active_ratios).mean().item()),
            "success_class_weight": float(self.success_class_weight),
            "failure_class_weight": float(self.failure_class_weight),
            "k0_success_count": float(self.k0_success_count),
            "k0_failure_count": float(self.k0_failure_count),
            "penalty_mean_k0": float(effective_penalty_k0.mean().item()),
            "penalty_mean_k1": float(effective_penalty_k1.mean().item()),
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
        if hasattr(dataloader, "dataset"):
            balance_metrics = self._configure_k0_balance_weights(dataloader.dataset)
            print(
                "[CPIQLCritic] k=0 class balance: "
                f"success_slices={int(balance_metrics['k0_success_count'])}, "
                f"failure_slices={int(balance_metrics['k0_failure_count'])}, "
                f"w_success={balance_metrics['success_class_weight']:.4f}, "
                f"w_failure={balance_metrics['failure_class_weight']:.4f}"
            )
        log_interval = max(int(getattr(self.cfg.critic, "log_interval", 100)), 1)
        save_interval = max(min(int(getattr(self.cfg.train, "save_interval", num_steps // 2)), num_steps // 4),1)

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
                    "vp": running.get("loss_v_progress", 0.0) / log_interval,
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

    @torch.no_grad()
    def eval_batch(
        self,
        batch: Dict[str, Any],
        only_obs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if "observations" not in batch:
            raise KeyError("CPIQL eval_batch requires 'observations' in batch.")
        if "action" not in batch:
            raise KeyError("CPIQL eval_batch requires 'action' in batch.")

        obs = self._preprocess_obs(batch["observations"])
        action = batch["action"].to(self.device).float()

        results: Dict[str, torch.Tensor] = {}
        if "frame" in batch:
            results["frame"] = batch["frame"].reshape(-1).detach().cpu().long()

        v_k0 = self.predict_v(obs, 0.0, preprocessed=True).reshape(-1).detach().cpu()
        v_k1 = self.predict_v(obs, 1.0, preprocessed=True).reshape(-1).detach().cpu()
        gap = (v_k1 - v_k0).detach().cpu()

        results["figure:value/V(k=0)"] = v_k0
        results["figure:value/V(k=1)"] = v_k1
        results["figure:value/critic_gap"] = gap

        if not only_obs:
            q_k0 = self.predict_q(obs, action, 0.0, preprocessed=True).reshape(-1).detach().cpu()
            q_k1 = self.predict_q(obs, action, 1.0, preprocessed=True).reshape(-1).detach().cpu()
            adv_k0 = self.compute_advantage(obs, action, 0.0, preprocessed=True).reshape(-1).detach().cpu()
            results["figure:action_value/Q(k=0)"] = q_k0
            results["figure:action_value/Q(k=1)"] = q_k1
            results["figure:action_value/adv(k=0)"] = adv_k0

        return results
