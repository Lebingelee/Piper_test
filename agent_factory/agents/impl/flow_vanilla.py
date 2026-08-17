from dataclasses import dataclass
import os

from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.mixins.actor.flow_matching import FlowMatchingActorMixin
from agent_factory.agents.registry import register_agent


@dataclass
class AgentSpecialConfig:
    """
    Vanilla flow matching policy config.
    """

    iters: int = 100000
    save_dir: str = "run_results"
    exp_name: str = ""


class MainMixin:
    CONFIG_CLASS = AgentSpecialConfig
    CONFIG_KEY = "agent_sp"


@register_agent("Flow_Vanilla")
class FlowVanillaAgent(MainMixin, FlowMatchingActorMixin, BaseAgent):
    """
    Vanilla Flow Matching Policy Agent (pure BC).
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

    def train_loop(self, dataloader, num_steps, save_dir=""):
        from tqdm import tqdm

        self.train()
        run_save_interval = max(num_steps // 4, 1)

        pbar = tqdm(range(num_steps), desc="Train FLOW MATCHING (Vanilla)", leave=True)

        def infinite_iterator(loader):
            while True:
                for batch in loader:
                    yield batch

        iterator = infinite_iterator(dataloader)
        total_loss_actor = 0

        for i in pbar:
            batch = next(iterator)
            batch = self._batch_to_device(batch)

            loss_dict = self.update_actor(batch)
            total_loss_actor += loss_dict["loss_actor"]

            if (i + 1) % 100 == 0:
                pbar.set_postfix({"Loss_Actor": total_loss_actor / 100})
                total_loss_actor = 0

            if save_dir and (i + 1) % run_save_interval == 0:
                current_step = i + 1
                save_path = os.path.join(save_dir, f"step_{current_step}.pth")
                self.save(save_path, meta={"step": current_step, "mode": "flow_matching_bc"})

            self.step += 1

    def start_train(self, dataset, additional_args=None):
        from torch.utils.data import DataLoader

        cfg = self.cfg
        expert_dataset = dataset["offline"]

        checkpoint_dir, exp_name = self._resolve_save_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)

        self._fit_action_normalizer_from_dataset(expert_dataset)

        loader = DataLoader(
            expert_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.train.num_workers,
            pin_memory=(cfg.train.num_workers > 0),
        )

        actor_iters = int(getattr(cfg.train, "actor_iters", 0))
        print(f">>> Start Vanilla Flow Matching Policy Training ({actor_iters} steps)")
        self.train_loop(loader, actor_iters, save_dir=checkpoint_dir)

        ckpt_path = os.path.join(checkpoint_dir, f"{exp_name}_final.pth")
        self.save(ckpt_path, meta={"phase": "flow_vanilla_train_done"})
