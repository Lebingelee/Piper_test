import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from agent_factory.agents.mixins.critic.base_eval import CriticEvalMixinBase
from agent_factory.agents.mixins.module_builder import ModuleBuilderMixin
from agent_factory.config.structure import StateEncoderConfig
from agent_factory.modules.critics.cpiql_rnn_critic import CPIQLRNNHistoryEncoder, CPIQLRNNQNet, CPIQLRNNVNet


@dataclass
class CPIQLRNNConfig:
    type: str = "cpiql_rnn"
    q_lr: float = 3e-4
    v_lr: float = 3e-4
    expectile: float = 0.7
    hidden_dims: List[int] = field(default_factory=lambda: [256])
    state_embed_dim: int = 512
    action_embed_dim: int = 128
    q_action_embed_dim: int = 128
    lstm_input_dim: int = 512
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.0
    alpha_progress: float = 0.1
    lambda_value_clip: float = 0.1
    target_update_interval: int = 1
    grad_clip_norm: float = 1.0
    log_interval: int = 100
    encoder: StateEncoderConfig = field(default_factory=StateEncoderConfig)


class CPIQLRNNMixin(ModuleBuilderMixin, CriticEvalMixinBase):
    CONFIG_CLASS = CPIQLRNNConfig
    CONFIG_KEY = "critic"
    REQUIRED_KEYS = {
        "observations", "next_observations", "history_actions", "history_action_valid_mask",
        "next_history_actions", "next_history_action_valid_mask", "valid_mask", "next_valid_mask",
        "action", "q_action_valid_mask", "reward", "discount", "terminated", "progress_return",
        "is_success_segment",
    }

    def _build_critic(self):
        cfg = self.cfg.critic
        self.critic_encoder = self._build_encoder_from_config(cfg.encoder)
        common = dict(
            obs_horizon=int(self.cfg.env.obs_horizon),
            action_dim=int(self.cfg.env.action_dim),
            state_embed_dim=int(cfg.state_embed_dim),
            action_embed_dim=int(cfg.action_embed_dim),
            lstm_input_dim=int(cfg.lstm_input_dim),
            hidden_dim=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            dropout=float(cfg.dropout),
        )
        self.v_net = CPIQLRNNVNet(
            CPIQLRNNHistoryEncoder(state_encoder=self.critic_encoder, **common),
            hidden_dims=list(cfg.hidden_dims),
        )
        self.q_net = CPIQLRNNQNet(
            state_encoder=self.critic_encoder,
            q_action_embed_dim=int(cfg.q_action_embed_dim),
            hidden_dims=list(cfg.hidden_dims),
            **common,
        )
        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_q_net.requires_grad_(False)
        self.v_optimizer = torch.optim.AdamW(self.v_net.parameters(), lr=float(cfg.v_lr))
        self.q_optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=float(cfg.q_lr))

    @staticmethod
    def _column(value: torch.Tensor, device: torch.device) -> torch.Tensor:
        return value.to(device=device, dtype=torch.float32).reshape(value.shape[0], -1)[:, :1]

    def _prepare_critic_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        prepared = {
            "observations": self._preprocess_obs(batch["observations"]),
            "next_observations": self._preprocess_obs(batch["next_observations"]),
        }
        for key in (
            "history_actions", "history_action_valid_mask", "next_history_actions",
            "next_history_action_valid_mask", "valid_mask", "next_valid_mask", "action",
            "q_action_valid_mask",
        ):
            prepared[key] = batch[key].to(self.device, dtype=torch.float32)
        for key in ("reward", "discount", "terminated", "progress_return", "is_success_segment"):
            prepared[key] = self._column(batch[key], self.device)
        return prepared

    def _v_outputs(self, prepared: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.v_net(
            prepared["observations"],
            prepared["history_actions"],
            prepared["valid_mask"],
            prepared["history_action_valid_mask"],
        )

    def _q_outputs(self, network, prepared: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return network(
            prepared["observations"], prepared["history_actions"], prepared["valid_mask"],
            prepared["history_action_valid_mask"], prepared["action"], prepared["q_action_valid_mask"],
        )

    def _expectile_loss(self, advantage: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        expectile = float(self.cfg.critic.expectile)
        asymmetric = torch.where(advantage > 0, expectile, 1.0 - expectile)
        return (weights * asymmetric * advantage.square()).sum() / weights.sum().clamp_min(1.0)

    @staticmethod
    def _weighted_mse(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (weights * (prediction - target).square()).sum() / weights.sum().clamp_min(1.0)

    def soft_update_target(self):
        tau = float(self.cfg.soft_update_tau)
        for online, target in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target.data.lerp_(online.data, tau)

    def update_critic(self, batch: Dict[str, Any]) -> Dict[str, float]:
        prepared = self._prepare_critic_batch(batch)
        success_weight = (prepared["is_success_segment"] > 0).float()
        all_weight = torch.ones_like(success_weight)

        with torch.no_grad():
            target_q = self._q_outputs(self.target_q_net, prepared)
            target_progress_q = torch.minimum(target_q["q1_progress"], target_q["q2_progress"])
            target_value_q = torch.minimum(target_q["q1_value"], target_q["q2_value"])

        v = self._v_outputs(prepared)
        loss_v_iql_progress = self._expectile_loss(target_progress_q - v["progress"], success_weight)
        loss_v_iql_value = self._expectile_loss(target_value_q - v["value"], all_weight)
        loss_v_anchor = self._weighted_mse(v["progress"], prepared["progress_return"], success_weight)
        loss_v_clip = (F.relu(-v["value"]).square() + F.relu(v["value"] - 1.0).square()).mean()
        loss_v = (
            loss_v_iql_progress + loss_v_iql_value
            + float(self.cfg.critic.alpha_progress) * loss_v_anchor
            + float(self.cfg.critic.lambda_value_clip) * loss_v_clip
        )
        self.v_optimizer.zero_grad()
        loss_v.backward()
        grad_clip = float(self.cfg.critic.grad_clip_norm)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), grad_clip)
        self.v_optimizer.step()

        next_prepared = {
            "observations": prepared["next_observations"],
            "history_actions": prepared["next_history_actions"],
            "valid_mask": prepared["next_valid_mask"],
            "history_action_valid_mask": prepared["next_history_action_valid_mask"],
        }
        with torch.no_grad():
            next_v = self._v_outputs(next_prepared)
            bootstrap = (1.0 - prepared["terminated"])
            q_target_progress = prepared["reward"] + prepared["discount"] * bootstrap * next_v["progress"]
            q_target_value = prepared["reward"] + prepared["discount"] * bootstrap * next_v["value"]

        q = self._q_outputs(self.q_net, prepared)
        loss_q_progress = self._weighted_mse(q["q1_progress"], q_target_progress, success_weight) + self._weighted_mse(q["q2_progress"], q_target_progress, success_weight)
        loss_q_value = self._weighted_mse(q["q1_value"], q_target_value, all_weight) + self._weighted_mse(q["q2_value"], q_target_value, all_weight)
        loss_q = loss_q_progress + loss_q_value
        self.q_optimizer.zero_grad()
        loss_q.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), grad_clip)
        self.q_optimizer.step()

        self.step += 1
        if self.step % max(int(self.cfg.critic.target_update_interval), 1) == 0:
            self.soft_update_target()
        with torch.no_grad():
            gap = v["progress"] - v["value"]
            failure_weight = 1.0 - success_weight
            gap_success = (gap * success_weight).sum() / success_weight.sum().clamp_min(1.0)
            gap_failure = (gap * failure_weight).sum() / failure_weight.sum().clamp_min(1.0)
        return {
            "loss_v": float(loss_v.detach().cpu()),
            "loss_q": float(loss_q.detach().cpu()),
            "loss_v_progress": float(loss_v_anchor.detach().cpu()),
            "loss_v_iql_progress": float(loss_v_iql_progress.detach().cpu()),
            "loss_v_iql_value": float(loss_v_iql_value.detach().cpu()),
            "loss_v_anchor": float(loss_v_anchor.detach().cpu()),
            "loss_v_clip": float(loss_v_clip.detach().cpu()),
            "loss_q_progress": float(loss_q_progress.detach().cpu()),
            "loss_q_value": float(loss_q_value.detach().cpu()),
            "critic_gap_success_mean": float(gap_success.detach().cpu()),
            "critic_gap_failure_mean": float(gap_failure.detach().cpu()),
        }

    def train_critic_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return self.update_critic(batch)

    def train_critic_loop(self, dataloader, num_steps: int, save_dir: str = ""):
        import os
        from tqdm import tqdm

        iterator = iter(dataloader)
        running: Dict[str, float] = {}
        interval = max(int(self.cfg.critic.log_interval), 1)
        save_interval = max(min(int(self.cfg.train.save_interval), max(num_steps // 4, 1)), 1)
        self.train()
        pbar = tqdm(range(num_steps), desc="Train CPIQL RNN", leave=True)
        for step_idx in pbar:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            metrics = self.update_critic(self._batch_to_device(batch))
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value
            if (step_idx + 1) % interval == 0:
                # Keep the same compact critic display as CPIQL-MLP: total Q,
                # progress-V supervision, and total V loss. tqdm formats these
                # floats with its standard three significant decimal places.
                pbar.set_postfix({
                    "q": running.get("loss_q", 0.0) / interval,
                    "vp": running.get("loss_v_progress", 0.0) / interval,
                    "v": running.get("loss_v", 0.0) / interval,
                    "gap_s": running.get("critic_gap_success_mean", 0.0) / interval,
                    "gap_f": running.get("critic_gap_failure_mean", 0.0) / interval,
                })
                running = {}
            if save_dir and (step_idx + 1) % save_interval == 0:
                self.save(os.path.join(save_dir, f"cpiql_rnn_step_{step_idx + 1}.pth"), meta={"mode": "cpiql_rnn"})

    @torch.no_grad()
    def eval_batch(self, batch: Dict[str, Any], only_obs: bool = True) -> Dict[str, Any]:
        prepared = self._prepare_critic_batch(batch)
        v = self._v_outputs(prepared)
        outputs = {
            "figure:value/V(k=0)": v["value"].reshape(-1),
            "figure:value/V(k=1)": v["progress"].reshape(-1),
            "figure:value/critic_gap": (v["progress"] - v["value"]).reshape(-1),
        }
        if not only_obs:
            q = self._q_outputs(self.q_net, prepared)
            q_progress = torch.minimum(q["q1_progress"], q["q2_progress"])
            q_value = torch.minimum(q["q1_value"], q["q2_value"])
            outputs.update({
                "figure:action_value/Q(k=0)": q_value.reshape(-1),
                "figure:action_value/Q(k=1)": q_progress.reshape(-1),
                "figure:action_value/adv(k=0)": (q_value - v["value"]).reshape(-1),
                "figure:action_value/adv(k=1)": (q_progress - v["progress"]).reshape(-1),
            })
        return outputs
