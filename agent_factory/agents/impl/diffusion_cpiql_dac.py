import os
from dataclasses import dataclass
from typing import Optional

from torch.utils.data import DataLoader
from tqdm import tqdm

from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.mixins.actor.cpiql_dac import CPIQLDACActorMixin
from agent_factory.agents.mixins.critic.CPIQL import CPIQLCriticMixin
from agent_factory.agents.registry import register_agent


@dataclass
class CPIQLDACAgentSpecialConfig:
    """
    CPIQL-DAC private extension slot.

    Universal training parameters live in cfg.train. Keep this structure
    available so downstream users can subclass the agent and add private knobs
    without changing the global config contract.
    """
    pass


class MainMixin:
    CONFIG_CLASS = CPIQLDACAgentSpecialConfig
    CONFIG_KEY = "agent_sp"


@register_agent("Diffusion_CPIQL_DAC")
class DiffusionCPIQLDACAgent(MainMixin, CPIQLDACActorMixin, CPIQLCriticMixin, BaseAgent):
    """
    Diffusion actor with frozen CPIQL critic guidance.

    Intended usage is stage-wise: train/load CPIQL critic first, then train the
    actor with update_actor_cpiql_dac().
    """

    def _init_components(self):
        self._build_critic()
        self._build_actor()

    def _init_optimizers(self):
        pass

    def update(self, batch: dict) -> dict:
        self.step += 1
        return self.update_actor(batch)

    def _make_infinite_iterator(self, loader):
        while True:
            for batch in loader:
                yield batch

    def _resolve_save_dir(self) -> str:
        save_root = str(getattr(self.cfg.train, "save_root", "run_results"))
        exp_name = str(getattr(self.cfg.train, "exp_name", "") or "piper_dual_merged_cpiql_dac")
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

    def train_actor_loop(self, dataloader, num_steps: int, save_dir: str = ""):
        self.train()
        run_save_interval = max(num_steps // 4, 1)
        iterator = self._make_infinite_iterator(dataloader)
        running = {}
        pbar = tqdm(range(num_steps), desc="Train CPIQL-DAC Actor", leave=True)
        for step_idx in pbar:
            batch = self._batch_to_device(next(iterator))
            metrics = self.update_actor(batch)
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value)

            if (step_idx + 1) % 100 == 0:
                denom = 100.0
                pbar.set_postfix({
                    "actor": running.get("loss_actor", 0.0) / denom,
                    "bc": running.get("loss_actor_bc", 0.0) / denom,
                    "guided": running.get("dac_guided_ratio", 0.0) / denom,
                    "dq": running.get("dac_delta_q_mean", 0.0) / denom,
                })
                running = {}

            if save_dir and (step_idx + 1) % run_save_interval == 0:
                self.save(
                    os.path.join(save_dir, f"actor_step_{step_idx + 1}.pth"),
                    meta={"step": step_idx + 1, "mode": "cpiql_dac_actor"},
                )
            self.step += 1

    def _select_actor_dataset(self, dataset):
        key = str(getattr(self.cfg.train, "dataset_key", "expert_dataset"))
        return self._select_dataset_by_key(dataset, key)

    def start_train(self, dataset, additional_args: Optional[dict] = None):
        train_cfg = self.cfg.train
        train_object = str(getattr(train_cfg, "train_object", "actor"))
        dataset_key = str(getattr(train_cfg, "dataset_key", "expert_dataset"))
        critic_iters = int(getattr(train_cfg, "critic_iters", 0))
        actor_iters = int(getattr(train_cfg, "actor_iters", 0))
        save_dir = self._resolve_save_dir()
        critic_ckpt_path = str(getattr(train_cfg, "critic_ckpt_path", ""))

        os.makedirs(save_dir, exist_ok=True)
        selected_dataset = self._select_dataset_by_key(dataset, dataset_key)

        def _make_loader(ds):
            return DataLoader(
                ds,
                batch_size=self.cfg.train.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.cfg.train.num_workers,
                pin_memory=(self.cfg.train.num_workers > 0),
            )

        if train_object in {"critic", "critic_then_actor"}:
            critic_loader = _make_loader(selected_dataset)
            print(f">>> Start CPIQL-DAC Critic Training ({critic_iters} steps) on dataset={dataset_key}")
            self.train_critic_loop(critic_loader, critic_iters, save_dir=save_dir)
            critic_ckpt_path = critic_ckpt_path or os.path.join(save_dir, "critic_final.pth")
            self.save(
                critic_ckpt_path,
                meta={"phase": "cpiql_dac_critic_train_done", "dataset_key": dataset_key},
            )
            if train_object == "critic":
                return

        if critic_ckpt_path:
            self.load(critic_ckpt_path)

        actor_dataset = selected_dataset
        self._fit_action_normalizer_from_dataset(actor_dataset)
        actor_loader = _make_loader(actor_dataset)
        print(f">>> Start CPIQL-DAC Actor Training ({actor_iters} steps) on dataset={dataset_key}")
        self.train_actor_loop(actor_loader, actor_iters, save_dir=save_dir)

        self.save(
            os.path.join(save_dir, "actor_final.pth"),
            meta={"phase": "cpiql_dac_actor_train_done", "dataset_key": dataset_key},
        )
