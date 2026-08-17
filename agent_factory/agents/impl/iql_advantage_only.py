import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.mixins.critic.IQL import IQLCriticMixin
from agent_factory.agents.registry import register_agent


@dataclass
class IQLAdvantageOnlyAgentSpecialConfig:
    pass


class MainMixin:
    CONFIG_CLASS = IQLAdvantageOnlyAgentSpecialConfig
    CONFIG_KEY = "agent_sp"


@register_agent("IQL_Advantage_Only")
class IQLAdvantageOnlyAgent(MainMixin, IQLCriticMixin, BaseAgent):
    """
    Critic-only IQL agent for advantage-based risk scoring.

    It intentionally avoids constructing an actor. The training target is the
    dataset return/value, so required dataset keys are limited to
    (s, a, s', gamma, return) as represented by observations/action/
    next_observations/discount/value in the current dataset contract.
    """

    @property
    def required_keys(self) -> list:
        return sorted(["observations", "action", "next_observations", "discount", "value"])

    def _init_components(self):
        self._build_critic()

    def _init_optimizers(self):
        pass

    def _resolve_save_dir(self) -> str:
        save_root = str(getattr(self.cfg.train, "save_root", "run_results"))
        exp_name = str(getattr(self.cfg.train, "exp_name", "") or "iql_advantage_only")
        return os.path.join(save_root, exp_name)

    def _select_dataset_by_key(self, dataset, dataset_key: str):
        if not isinstance(dataset, dict):
            return dataset
        if dataset_key in dataset:
            return dataset[dataset_key]
        if dataset_key == "expert_dataset" and "offline" in dataset:
            return dataset["offline"]
        if dataset_key == "replaybuffer" and "online" in dataset:
            return dataset["online"]
        if dataset_key == "expert_dataset+replaybuffer":
            expert = dataset.get("expert_dataset", dataset.get("offline"))
            replay = dataset.get("replaybuffer", dataset.get("online"))
            if expert is not None and replay is not None:
                from torch.utils.data import ConcatDataset

                return ConcatDataset([expert, replay])
            if expert is not None:
                return expert
            if replay is not None:
                return replay
        if "offline" in dataset:
            return dataset["offline"]
        return next(iter(dataset.values()))

    def _as_column(self, value: Any) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.to(self.device).float()
        return tensor.reshape(tensor.shape[0], -1)[:, :1]

    def _return_target(self, batch: Dict[str, Any]) -> torch.Tensor:
        for key in ("return", "value", "progress_return"):
            if key in batch:
                return self._as_column(batch[key])
        raise KeyError("IQL_Advantage_Only requires a return target key: value, progress_return, or return.")

    def update_critic(self, batch: Dict[str, Any]) -> Dict[str, float]:
        obs = self._preprocess_obs(batch["observations"])
        actions = batch["action"].to(self.device).float()
        return_target = self._return_target(batch)
        discount = self._as_column(batch["discount"]) if "discount" in batch else None

        with torch.no_grad():
            q1_targ, q2_targ = self.target_q_net(obs, actions)
            q_target = torch.min(q1_targ, q2_targ)

        v_pred = self.v_net(obs)
        adv = q_target - v_pred
        weight = torch.where(
            adv > 0,
            torch.as_tensor(float(self.cfg.critic.expectile), device=self.device),
            torch.as_tensor(float(1.0 - self.cfg.critic.expectile), device=self.device),
        )
        loss_v_expectile = (weight * (adv ** 2)).mean()
        loss_v_return = F.mse_loss(v_pred, return_target)
        return_mse_weight = float(getattr(self.cfg.critic, "return_mse_weight", 1.0))
        loss_v = loss_v_expectile + return_mse_weight * loss_v_return

        self.v_optimizer.zero_grad()
        loss_v.backward()
        self.v_optimizer.step()

        q1_pred, q2_pred = self.q_net(obs, actions)
        loss_q = F.mse_loss(q1_pred, return_target) + F.mse_loss(q2_pred, return_target)

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        metrics = {
            "loss_v": float(loss_v.item()),
            "loss_v_expectile": float(loss_v_expectile.item()),
            "loss_v_return": float(loss_v_return.item()),
            "loss_q": float(loss_q.item()),
            "adv_mean": float(adv.mean().item()),
            "return_mean": float(return_target.mean().item()),
        }
        if discount is not None:
            metrics["discount_mean"] = float(discount.mean().item())
        return metrics

    def train_critic_step(self, batch: Dict[str, Any], update_target: bool = True) -> Dict[str, float]:
        metrics = self.update_critic(batch)
        self.step += 1
        if update_target:
            self.soft_update_target()
        return metrics

    def train_critic_loop(self, dataloader, num_steps: int, save_dir: str = ""):
        self.train()
        log_interval = max(int(getattr(self.cfg.critic, "log_interval", 100)), 1)
        save_interval = max(min(int(getattr(self.cfg.train, "save_interval", num_steps // 2)), num_steps // 4), 1)

        def infinite_iterator(loader):
            while True:
                for batch in loader:
                    yield batch

        iterator = infinite_iterator(dataloader)
        running: Dict[str, float] = {}
        pbar = tqdm(range(num_steps), desc="Train IQL Advantage Critic", leave=True)
        for step_idx in pbar:
            batch = self._batch_to_device(next(iterator))
            metrics = self.train_critic_step(batch)
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value)

            if (step_idx + 1) % log_interval == 0:
                pbar.set_postfix(
                    {
                        "q": running.get("loss_q", 0.0) / log_interval,
                        "v": running.get("loss_v", 0.0) / log_interval,
                        "ret": running.get("loss_v_return", 0.0) / log_interval,
                        "adv": running.get("adv_mean", 0.0) / log_interval,
                    }
                )
                running = {}

            if save_dir and (step_idx + 1) % save_interval == 0:
                os.makedirs(save_dir, exist_ok=True)
                self.save(
                    os.path.join(save_dir, f"iql_advantage_critic_step_{step_idx + 1}.pth"),
                    meta={"mode": "iql_advantage_critic"},
                )

    def start_train(self, dataset, additional_args: Optional[dict] = None):
        del additional_args
        train_cfg = self.cfg.train
        dataset_key = str(getattr(train_cfg, "dataset_key", "expert_dataset+replaybuffer"))
        critic_iters = int(getattr(train_cfg, "critic_iters", 0))
        save_dir = self._resolve_save_dir()
        ckpt_path = str(getattr(train_cfg, "ckpt_path", "")).strip()

        os.makedirs(save_dir, exist_ok=True)
        selected_dataset = self._select_dataset_by_key(dataset, dataset_key)
        if ckpt_path:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"IQL_Advantage_Only checkpoint not found: {ckpt_path}")
            self.load(ckpt_path)

        loader = DataLoader(
            selected_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=(self.cfg.train.num_workers > 0),
        )
        print(f">>> Start IQL_Advantage_Only Critic Training ({critic_iters} steps) on dataset={dataset_key}")
        self.train_critic_loop(loader, critic_iters, save_dir=save_dir)
        self.save(
            os.path.join(save_dir, "critic_final.pth"),
            meta={"phase": "iql_advantage_only_critic_train_done", "dataset_key": dataset_key},
        )

    @torch.no_grad()
    def eval_batch(self, batch: Dict[str, Any], only_obs: bool = False) -> Dict[str, Any]:
        del only_obs
        obs = self._preprocess_obs(batch["observations"])
        actions = batch["action"].to(self.device).float()
        v_pred = self.v_net(obs)
        q1_pred, q2_pred = self.q_net(obs, actions)
        q_pred = torch.min(q1_pred, q2_pred)
        adv = q_pred - v_pred
        return {
            "figure:value/V": v_pred.reshape(-1),
            "figure:action_value/Q": q_pred.reshape(-1),
            "figure:action_value/adv": adv.reshape(-1),
            "figure:action_value/v_minus_q": (-adv).reshape(-1),
        }

    @torch.no_grad()
    def eval_risk_batch(self, batch: Dict[str, Any], only_obs: bool = False) -> Dict[str, Any]:
        outputs = self.eval_batch(batch, only_obs=only_obs)
        adv = outputs["figure:action_value/adv"]
        risk_score = -adv
        outputs = dict(outputs)
        outputs.update(
            {
                "risk_score": risk_score,
                "risk_signal/raw_v": outputs["figure:value/V"],
                "risk_signal/raw_q": outputs["figure:action_value/Q"],
                "risk_signal/raw_adv": adv,
                "risk_signal/raw_v_minus_q": risk_score,
                "risk_signal/score_formula": "v_minus_q",
                "risk_signal/method_id": "iql_v_minus_q",
                "risk_signal/display_name": "IQL V-Q",
            }
        )
        return outputs

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"[IQL_Advantage_Only] Warning: Path {path} not found.")
            return {}
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state = payload.get("model", payload)
        missing, unexpected = self.load_state_dict(state, strict=False)
        self.step = payload.get("step", 0) if isinstance(payload, dict) else 0
        if missing:
            print(f"[IQL_Advantage_Only] Missing keys while loading: {len(missing)}")
        if unexpected:
            print(f"[IQL_Advantage_Only] Ignored unexpected keys while loading: {len(unexpected)}")
        print(f"[IQL_Advantage_Only] Loaded critic weights from {path} (Step {self.step})")
        return payload.get("meta", {}) if isinstance(payload, dict) else {}
