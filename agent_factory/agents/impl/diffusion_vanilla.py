import torch
from dataclasses import dataclass
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

    def train_loop(self, dataloader, num_steps, save_dir=""):
        from tqdm import tqdm
        import os
        self.train()
        
        # 设定保存频率：默认保存 4 次
        run_save_interval = num_steps // 4
        
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
        import os

        cfg = self.cfg
        cfg_sp = self.cfg.agent_sp
        expert_dataset = dataset['offline']

        if cfg_sp.exp_name == "":
            cfg_sp.exp_name = cfg.agent_type
           
        checkpoint_dir = os.path.join(cfg_sp.save_dir, cfg_sp.exp_name)

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

        print(f">>> Start Vanilla Diffusion Policy Training ({cfg_sp.iters} steps)")
        self.train_loop(loader, cfg_sp.iters, save_dir=checkpoint_dir)

        # 保存最终模型
        ckpt_path = os.path.join(checkpoint_dir, f"{cfg_sp.exp_name}_final.pth")
        self.save(ckpt_path, meta={"phase": "vanilla_train_done"})
