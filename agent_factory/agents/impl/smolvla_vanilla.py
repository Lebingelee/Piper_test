from dataclasses import dataclass
import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.mixins.actor.smolvla import SmolVLAActorMixin
from agent_factory.agents.registry import register_agent


@dataclass
class SmolVLAAgentSpecialConfig:
    save_dir: str = "run_results"
    exp_name: str = "smolvla"


class MainMixin:
    CONFIG_CLASS = SmolVLAAgentSpecialConfig
    CONFIG_KEY = "agent_sp"


@register_agent("SmolVLA")
@register_agent("SmolVLA_Vanilla")
class SmolVLAVanillaAgent(MainMixin, SmolVLAActorMixin, BaseAgent):
    """
    LeRobot SmolVLA policy trained through agent_factory.
    """

    def _init_components(self):
        self._build_actor()

    def _init_optimizers(self):
        pass

    def _resolve_save_dir(self):
        train_cfg = self.cfg.train
        save_root = str(getattr(train_cfg, "save_root", "") or "run_results")
        exp_name = str(getattr(train_cfg, "exp_name", "") or self.cfg.agent_type)
        return os.path.join(save_root, exp_name), exp_name

    def train_loop(self, dataloader, num_steps: int, save_dir: str = ""):
        self.train()
        num_steps = int(num_steps)
        start_step = int(getattr(self, "step", 0))
        if start_step >= num_steps:
            print(f"[SmolVLA] step={start_step} already reached target actor_iters={num_steps}; skip training loop.")
            return
        run_save_interval = max(num_steps // 4, 1)

        def infinite_iterator(loader):
            while True:
                for batch in loader:
                    yield batch

        iterator = infinite_iterator(dataloader)
        pbar = tqdm(range(start_step, num_steps), desc="Train SmolVLA", leave=True)
        total_loss_actor = 0.0

        for step_idx in pbar:
            batch = self._batch_to_device(next(iterator))
            loss_dict = self.update_actor(batch)
            total_loss_actor += float(loss_dict["loss_actor"])

            current_step = step_idx + 1
            if current_step % 10 == 0:
                pbar.set_postfix({"Loss_Actor": total_loss_actor / 10.0})
                total_loss_actor = 0.0

            self.step = current_step
            if save_dir and current_step % run_save_interval == 0:
                save_path = os.path.join(save_dir, f"step_{current_step}.pth")
                self.save(save_path, meta={"step": current_step, "mode": "smolvla_bc"})

    def start_train(self, dataset, additional_args=None):
        del additional_args
        cfg = self.cfg
        expert_dataset = dataset["offline"]

        checkpoint_dir, exp_name = self._resolve_save_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)

        self._fit_obs_normalizer_from_dataset(expert_dataset)
        self._fit_action_normalizer_from_dataset(expert_dataset)

        ckpt_path = str(getattr(cfg.train, "ckpt_path", "") or "").strip()
        if ckpt_path:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"SmolVLA checkpoint not found: {ckpt_path}")
            meta = self.load(ckpt_path) or {}
            if "step" in meta:
                self.step = max(int(self.step), int(meta["step"]))
            print(f">>> Loaded SmolVLA checkpoint from {ckpt_path} (step={self.step})")

        loader = DataLoader(
            expert_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.train.num_workers,
            pin_memory=(cfg.train.num_workers > 0),
        )

        actor_iters = int(getattr(cfg.train, "actor_iters", 0))
        print(f">>> Start SmolVLA Training ({actor_iters} steps)")
        self.train_loop(loader, actor_iters, save_dir=checkpoint_dir)

        ckpt_path = os.path.join(checkpoint_dir, f"{exp_name}_final.pth")
        self.save(ckpt_path, meta={"phase": "smolvla_train_done"})
