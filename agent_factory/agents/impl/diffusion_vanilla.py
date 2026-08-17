import os
from dataclasses import dataclass

import torch

from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.mixins.actor.diffusion import DiffusionActorMixin
from agent_factory.agents.registry import register_agent

@dataclass
class AgentSpecialConfig:
    """
    Vanilla Diffusion Policy 专用配置
    仅包含基础的训练步数和保存路径
    """
    iters: int = 100000
    save_dir: str = "run_results"
    exp_name: str = ""

class MainMixin:
    CONFIG_CLASS = AgentSpecialConfig
    CONFIG_KEY = "agent_sp"

@register_agent("Diffusion_Vanilla")
class DiffusionVanillaAgent(MainMixin, DiffusionActorMixin, BaseAgent):
    """
    标准的 Diffusion Policy Agent (纯行为克隆 BC)
    严格遵循 Observation -> Action 的扩散模型训练逻辑
    """

    def _init_components(self):
        # 仅初始化 Actor 组件
        self._build_actor()

    def _init_optimizers(self):
        # 优化器已在 Mixin 的 _build_actor 中初始化
        pass

    def _resolve_save_dir(self):
        train_cfg = self.cfg.train
        save_root = str(getattr(train_cfg, "save_root", "") or "run_results")
        exp_name = str(getattr(train_cfg, "exp_name", "") or self.cfg.agent_type)
        return os.path.join(save_root, exp_name), exp_name

    def train_loop(self, dataloader, num_steps, save_dir=""):
        from tqdm import tqdm
        self.train()
        
        # 设定保存频率：默认保存 4 次
        run_save_interval = max(num_steps // 4, 1)
        
        desc = "Train DIFFUSION (Vanilla)"
        pbar = tqdm(range(num_steps), desc=desc, leave=True)
        
        # 使用无限生成器复用子进程，避免 AssertionError 和启动开销
        def infinite_iterator(loader):
            while True:
                for batch in loader:
                    yield batch
        
        iterator = infinite_iterator(dataloader)
        total_loss_actor = 0

        for i in pbar:
            batch = next(iterator)
            
            # 使用 BaseAgent 提供的递归搬运工具
            batch = self._batch_to_device(batch)

            # 调用 DiffusionActorMixin 的更新逻辑
            loss_dict = self.update_actor(batch)
            total_loss_actor += loss_dict['loss_actor']

            if (i + 1) % 100 == 0:
                pbar.set_postfix({
                    "Loss_Actor": total_loss_actor / 100
                })
                total_loss_actor = 0

            if len(save_dir) > 0 and (i + 1) % run_save_interval == 0:
                current_step = i + 1
                save_path = os.path.join(save_dir, f"step_{current_step}.pth")
                self.save(save_path, meta={"step": current_step, "mode": "vanilla_bc"})
            
            self.step += 1

    def start_train(self, dataset, additional_args=None):
        from torch.utils.data import DataLoader

        cfg = self.cfg
        expert_dataset = dataset['offline']

        checkpoint_dir, exp_name = self._resolve_save_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 统一拟合动作归一化器（若配置启用）
        self._fit_action_normalizer_from_dataset(expert_dataset)
        
        # 构建数据加载器
        loader = DataLoader(
            expert_dataset, 
            batch_size=cfg.train.batch_size, 
            shuffle=True, 
            drop_last=True, 
            num_workers=cfg.train.num_workers,
            pin_memory=(cfg.train.num_workers > 0)
        )

        actor_iters = int(getattr(cfg.train, "actor_iters", 0))
        print(f">>> Start Vanilla Diffusion Policy Training ({actor_iters} steps)")
        self.train_loop(loader, actor_iters, save_dir=checkpoint_dir)

        # 保存最终模型
        ckpt_path = os.path.join(checkpoint_dir, f"{exp_name}_final.pth")
        self.save(ckpt_path, meta={"phase": "vanilla_train_done"})
