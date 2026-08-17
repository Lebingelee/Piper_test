import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.mixins.critic.CPIQL import CPIQLCriticMixin
from agent_factory.agents.registry import register_agent


@dataclass
class CPIQLOnlyAgentSpecialConfig:
    pass


class MainMixin:
    CONFIG_CLASS = CPIQLOnlyAgentSpecialConfig
    CONFIG_KEY = "agent_sp"


@register_agent("CPIQL_Only")
class CPIQLOnlyAgent(MainMixin, CPIQLCriticMixin, BaseAgent):
    """
    Critic-only CPIQL agent.

    This is intended for critic training and offline score export. It avoids
    constructing a diffusion actor while keeping the same CPIQL critic module
    names, so it can load critic weights from both critic-only and full
    Diffusion_CPIQL_DAC checkpoints.
    """

    def _init_components(self):
        self._build_critic()

    def _init_optimizers(self):
        pass

    def _resolve_save_dir(self) -> str:
        save_root = str(getattr(self.cfg.train, "save_root", "run_results"))
        exp_name = str(getattr(self.cfg.train, "exp_name", "") or "cpiql_only")
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

    def start_train(self, dataset, additional_args: Optional[dict] = None):
        train_cfg = self.cfg.train
        dataset_key = str(getattr(train_cfg, "dataset_key", "expert_dataset+replaybuffer"))
        critic_iters = int(getattr(train_cfg, "critic_iters", 0))
        save_dir = self._resolve_save_dir()
        ckpt_path = str(getattr(train_cfg, "ckpt_path", "")).strip()

        os.makedirs(save_dir, exist_ok=True)
        selected_dataset = self._select_dataset_by_key(dataset, dataset_key)

        if ckpt_path:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"CPIQL_Only checkpoint not found: {ckpt_path}")
            self.load(ckpt_path)

        loader = DataLoader(
            selected_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=(self.cfg.train.num_workers > 0),
        )
        print(f">>> Start CPIQL_Only Critic Training ({critic_iters} steps) on dataset={dataset_key}")
        self.train_critic_loop(loader, critic_iters, save_dir=save_dir)
        self.save(
            os.path.join(save_dir, "critic_final.pth"),
            meta={"phase": "cpiql_only_critic_train_done", "dataset_key": dataset_key},
        )

    @torch.no_grad()
    def eval_risk_batch(self, batch: Dict[str, Any], only_obs: bool = True) -> Dict[str, Any]:
        """
        Official CPIQL_Only phase-1 risk signal.

        Keep the current baseline semantics: DeltaV score is
        max(V(s, k=1) - V(s, k=0), 0). Future critic-only agent impls can
        override this method to export a different trained risk signal while
        leaving baseline JSONL/calibration code unchanged.
        """
        outputs = self.eval_batch(batch, only_obs=only_obs)
        delta_v = outputs["figure:value/critic_gap"]
        outputs = dict(outputs)
        outputs.update(
            {
                "risk_score": torch.clamp(delta_v, min=0.0),
                "risk_signal/raw_v_k0": outputs["figure:value/V(k=0)"],
                "risk_signal/raw_v_k1": outputs["figure:value/V(k=1)"],
                "risk_signal/raw_delta_v": delta_v,
                "risk_signal/score_formula": "max_v_k1_minus_v_k0",
                "risk_signal/method_id": "cpiql_deltav",
                "risk_signal/display_name": "CPIQL DeltaV",
            }
        )
        if "figure:action_value/Q(k=0)" in outputs:
            outputs["risk_signal/raw_q_k0"] = outputs["figure:action_value/Q(k=0)"]
        if "figure:action_value/Q(k=1)" in outputs:
            outputs["risk_signal/raw_q_k1"] = outputs["figure:action_value/Q(k=1)"]
        if "figure:action_value/adv(k=0)" in outputs:
            outputs["risk_signal/raw_adv_k0"] = outputs["figure:action_value/adv(k=0)"]
        return outputs

    def load(self, path: str):
        import torch

        if not os.path.exists(path):
            print(f"[CPIQL_Only] Warning: Path {path} not found.")
            return {}
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state = payload.get("model", payload)
        missing, unexpected = self.load_state_dict(state, strict=False)
        self.step = payload.get("step", 0) if isinstance(payload, dict) else 0
        if missing:
            print(f"[CPIQL_Only] Missing keys while loading: {len(missing)}")
        if unexpected:
            print(f"[CPIQL_Only] Ignored unexpected keys while loading: {len(unexpected)}")
        print(f"[CPIQL_Only] Loaded critic weights from {path} (Step {self.step})")
        return payload.get("meta", {}) if isinstance(payload, dict) else {}
