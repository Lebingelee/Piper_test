import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.mixins.critic.CPIQL_RNN import CPIQLRNNMixin
from agent_factory.agents.registry import register_agent


def cpiql_rnn_collate_fn(items):
    """Collate block samples from sources with different local action horizons."""
    if not items:
        raise ValueError("cpiql_rnn_collate_fn received an empty batch.")

    history_horizon = max(int(item["history_actions"].shape[-2]) for item in items)
    q_horizon = max(int(item["action"].shape[-2]) for item in items)

    def pad_history(value):
        return F.pad(value, (0, 0, 0, history_horizon - value.shape[-2]))

    def pad_history_mask(value):
        return F.pad(value, (0, history_horizon - value.shape[-1]))

    def pad_q(value):
        return F.pad(value, (0, 0, 0, q_horizon - value.shape[-2]))

    def pad_q_mask(value):
        return F.pad(value, (0, q_horizon - value.shape[-1]))

    padded = []
    for item in items:
        item = dict(item)
        for key in ("history_actions", "next_history_actions"):
            item[key] = pad_history(item[key])
        for key in ("history_action_valid_mask", "next_history_action_valid_mask"):
            item[key] = pad_history_mask(item[key])
        item["action"] = pad_q(item["action"])
        item["q_action_valid_mask"] = pad_q_mask(item["q_action_valid_mask"])
        if "q_action" in item:
            item["q_action"] = pad_q(item["q_action"])
        padded.append(item)
    return default_collate(padded)


@dataclass
class CPIQLRNNAgentSpecialConfig:
    ratio_epsilon: float = 1e-6


class MainMixin:
    CONFIG_CLASS = CPIQLRNNAgentSpecialConfig
    CONFIG_KEY = "agent_sp"


@register_agent("cpiql-rnn")
class CPIQLRNNAgent(MainMixin, CPIQLRNNMixin, BaseAgent):
    def _init_components(self):
        self._build_critic()

    def _init_optimizers(self):
        pass

    def _select_dataset_by_key(self, dataset, dataset_key: str):
        if not isinstance(dataset, dict):
            return dataset
        if dataset_key in dataset:
            return dataset[dataset_key]
        if dataset_key == "expert_dataset+replaybuffer":
            expert = dataset.get("expert_dataset", dataset.get("offline"))
            replay = dataset.get("replaybuffer", dataset.get("online"))
            if expert is not None and replay is not None:
                from torch.utils.data import ConcatDataset
                return ConcatDataset([expert, replay])
            return expert if expert is not None else replay
        return dataset.get("offline", next(iter(dataset.values())))

    def _resolve_save_dir(self) -> str:
        return os.path.join(str(self.cfg.train.save_root), str(self.cfg.train.exp_name or "cpiql_rnn"))

    def start_train(self, dataset, additional_args: Optional[Dict[str, Any]] = None):
        del additional_args
        selected = self._select_dataset_by_key(dataset, str(self.cfg.train.dataset_key))
        if selected is None or len(selected) <= 0:
            raise ValueError("CPIQL-RNN selected dataset is empty.")
        save_dir = self._resolve_save_dir()
        os.makedirs(save_dir, exist_ok=True)
        if str(self.cfg.train.ckpt_path).strip():
            self.load(str(self.cfg.train.ckpt_path))
        loader = DataLoader(
            selected,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=int(self.cfg.train.num_workers),
            pin_memory=int(self.cfg.train.num_workers) > 0,
            collate_fn=cpiql_rnn_collate_fn,
        )
        self.train_critic_loop(loader, int(self.cfg.train.critic_iters), save_dir=save_dir)
        self.save(os.path.join(save_dir, "critic_final.pth"), meta={"mode": "cpiql_rnn"})

    @torch.no_grad()
    def eval_risk_batch(self, batch: Dict[str, Any], only_obs: bool = True) -> Dict[str, Any]:
        outputs = dict(self.eval_batch(batch, only_obs=only_obs))
        v_k0 = outputs["figure:value/V(k=0)"]
        v_k1 = outputs["figure:value/V(k=1)"]
        raw_ratio = (v_k1 - v_k0) / torch.clamp(v_k1, min=float(self.cfg.agent_sp.ratio_epsilon))
        outputs.update({
            "risk_score": torch.clamp(raw_ratio, min=0.0),
            "risk_signal/raw_v_k0": v_k0,
            "risk_signal/raw_v_k1": v_k1,
            "risk_signal/raw_delta_v": v_k1 - v_k0,
            "risk_signal/raw_cpiql_ratio": raw_ratio,
            "risk_signal/score_formula": "max_v_k1_minus_v_k0_over_v_k1_zero",
            "risk_signal/method_id": "cpiql_rnn_ratio",
            "risk_signal/display_name": "CPIQL RNN Ratio",
        })
        return outputs

    def load(self, path: str):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state = payload.get("model", payload)
        self.load_state_dict(state, strict=False)
        self.step = payload.get("step", 0) if isinstance(payload, dict) else 0
        return payload.get("meta", {}) if isinstance(payload, dict) else {}
