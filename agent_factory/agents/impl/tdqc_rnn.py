import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.mixins.critic.TDQC_RNN import TDQCRNNMixin
from agent_factory.agents.registry import register_agent
from agent_factory.data.impl.tdqc.feature_dataset import tdqc_collate_fn


@dataclass
class TDQCRNNAgentSpecialConfig:
    encoder_override: Any = None
    encoder_config_path: str = ""


class MainMixin:
    CONFIG_CLASS = TDQCRNNAgentSpecialConfig
    CONFIG_KEY = "agent_sp"


@register_agent("tdqc-rnn")
class TDQCRNNAgent(MainMixin, TDQCRNNMixin, BaseAgent):
    """
    Feature-based TDQC recurrent risk-value agent.

    The model trains on cached step_features and exports failure risk as
    1 - predicted success probability.
    """

    def _init_components(self):
        self._build_tdqc_predictor()

    def _init_optimizers(self):
        self._init_tdqc_optimizers()

    def _resolve_save_dir(self) -> str:
        save_root = str(getattr(self.cfg.train, "save_root", "run_results"))
        exp_name = str(getattr(self.cfg.train, "exp_name", "") or "tdqc_rnn")
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
        if "offline" in dataset:
            return dataset["offline"]
        return next(iter(dataset.values()))

    def start_train(self, dataset, additional_args: Optional[Dict[str, Any]] = None):
        del additional_args
        train_cfg = self.cfg.train
        dataset_key = str(getattr(train_cfg, "dataset_key", "expert_dataset"))
        tdqc_iters = int(getattr(train_cfg, "critic_iters", 0))
        save_dir = self._resolve_save_dir()
        ckpt_path = str(getattr(train_cfg, "ckpt_path", "")).strip()

        os.makedirs(save_dir, exist_ok=True)
        selected_dataset = self._select_dataset_by_key(dataset, dataset_key)
        if hasattr(selected_dataset, "__len__") and len(selected_dataset) <= 0:
            raise ValueError("TDQC-RNN selected dataset is empty.")

        if ckpt_path:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"TDQC RNN checkpoint not found: {ckpt_path}")
            self.load(ckpt_path)

        loader = DataLoader(
            selected_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=(self.cfg.train.num_workers > 0),
            collate_fn=tdqc_collate_fn,
        )
        print(f">>> Start TDQC-RNN Training ({tdqc_iters} steps) on dataset={dataset_key}")
        self.train_tdqc_loop(loader, tdqc_iters, save_dir=save_dir)
        self.save(
            os.path.join(save_dir, "tdqc_rnn_final.pth"),
            meta={"phase": "tdqc_rnn_train_done", "dataset_key": dataset_key},
        )

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"[TDQC_RNN] Warning: Path {path} not found.")
            return {}
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state = payload.get("model", payload)
        missing, unexpected = self.load_state_dict(state, strict=False)
        self.step = payload.get("step", 0) if isinstance(payload, dict) else 0
        if missing:
            print(f"[TDQC_RNN] Missing keys while loading: {len(missing)}")
        if unexpected:
            print(f"[TDQC_RNN] Ignored unexpected keys while loading: {len(unexpected)}")
        print(f"[TDQC_RNN] Loaded weights from {path} (Step {self.step})")
        return payload.get("meta", {}) if isinstance(payload, dict) else {}
