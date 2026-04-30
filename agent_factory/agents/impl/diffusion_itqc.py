import torch
from dataclasses import dataclass, field
from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.mixins.actor.conditional_diffusion import ConditionalDiffusionActorMixin
from agent_factory.agents.mixins.critic.ITQC import ITQCCriticMixin
from agent_factory.agents.registry import register_agent
from agent_factory.config.structure import DiffusionActorConfig



@dataclass
class AgentSpecialConfig:

    #offline parameters
    threshold_quantile_offline : float = 0.8
    offline_iters_critic: int = 12000
    offline_iters_actor: int = 80000
    save_dir: str = "run_results"
    exp_name: str = ""
    #online parameters
    threshold_quantile_online : float = 0.3
    online_iters_critic: int = 20000
    online_iters_actor: int = 10000
    only_suc: bool = True
 
    # 每个 agent 都有自己的配置
    

    
class MainMixin:
    CONFIG_CLASS = AgentSpecialConfig
    CONFIG_KEY = "agent_sp"


@register_agent("Diffusion_ITQC")
class DiffusionITQCAgent(MainMixin,ConditionalDiffusionActorMixin, ITQCCriticMixin, BaseAgent):
    """
    Diffusion + ITQC (Support Normal & Success Critic)
    """

    def _init_components(self):
        # 构建 Actor 和 Critic 组件
        self._build_critic() # ITQC Mixin 会初始化 Normal 和 Suc 两套网络
        self._build_actor()  

    
    def _init_optimizers(self):
        pass

    def train_loop(self, dataloader, num_steps, mode, save_dir=""):
        from tqdm import tqdm
        import os
        import gc
        self.train()
        if 'RL' in save_dir:
            run_save_interval = self.cfg.agent_sp.online_iters_actor
        else:
            run_save_interval = self.cfg.agent_sp.offline_iters_actor / 4
        
        desc = f"Train {mode.upper()}"
        pbar = tqdm(range(num_steps), desc=desc, leave=False)
        
        def infinite_iterator(loader):
            while True:
                for batch in loader:
                    yield batch
        
        # 🟢 增加对迭代器的显式管理
        it_obj = infinite_iterator(dataloader)
        
        total_loss_q = 0
        total_loss_v = 0
        total_loss_actor = 0

        try:
            for i in pbar:
                batch = next(it_obj)
                batch = self._batch_to_device(batch)

                if "critic" in mode:
                    critic_type = "Standard" if 'Standard' in mode else "Pretty"
                    loss_dict = self.update_critic(batch, critic_type=critic_type)
                    self.soft_update_target(critic_type=critic_type)
                    total_loss_q += loss_dict['loss_q']
                    total_loss_v += loss_dict['loss_v']
                elif mode == "actor":
                    loss_dict = self.update_actor(batch)
                    total_loss_actor += loss_dict['loss_actor']

                if (i + 1) % 100 == 0:
                    if "critic" in mode:
                        pbar.set_postfix({"Loss_Q": total_loss_q / 100, "Loss_V": total_loss_v / 100})
                        total_loss_q, total_loss_v = 0, 0
                    elif mode == "actor":
                        pbar.set_postfix({"Loss_Actor": total_loss_actor / 100})
                        total_loss_actor = 0

                if len(save_dir) > 0 and (i + 1) % run_save_interval == 0:
                    current_step = i + 1
                    self.save(os.path.join(save_dir, f"step_{current_step}.pth"), meta={"step": current_step, "mode": mode})
                self.step += 1
        finally:
            # 🟢 训练结束，强制销毁迭代器引用并清空 GPU 缓存
            del it_obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def start_train(self, dataset, additional_args=None):
        from torch.utils.data import DataLoader, ConcatDataset
        import os
        import torch
        import gc
        additional_args = additional_args or {}
        cfg = self.cfg
        cfg_sp = self.cfg.agent_sp

        expert_dataset = dataset['offline']
        replay_buffer = dataset.get('online', None)
        phase = additional_args.get("phase", "offline")
        checkpoint_dir = f"{cfg_sp.save_dir}/{cfg_sp.exp_name}"

        # 统一拟合动作归一化器（ConditionalDiffusionActorMixin）
        self._fit_action_normalizer_from_dataset(dataset)

        # 辅助函数：创建高性能且显存友好的 DataLoader
        def make_loader(ds):
            return DataLoader(
                ds, 
                batch_size=cfg.train.batch_size, 
                shuffle=True, drop_last=True, 
                num_workers=cfg.train.num_workers,
                pin_memory=True,
                persistent_workers=(cfg.train.num_workers > 0),
                prefetch_factor=2 if cfg.train.num_workers > 0 else None 
            )

        if phase == 'offline':
            print(">>> Training Pretty_Critic...")
            expert_dataset.switch('success')
            loader = make_loader(expert_dataset)
            self.train_loop(loader, cfg_sp.offline_iters_critic, mode="Pretty_critic")
            del loader
            gc.collect()

            print(">>> Training Standard_Critic...")
            expert_dataset.switch('all')
            loader = make_loader(expert_dataset)
            self.train_loop(loader, cfg_sp.offline_iters_critic, mode="Standard_critic")
            del loader
            gc.collect()
            
            self.save(f"{checkpoint_dir}/critic_only_ckpt.pth", meta={"phase": "critic_trained"})

            print(">>> Relabel Expert Data")
            expert_dataset.switch('all')
            self.relabel_data(expert_dataset, phase="pretrain")

            print(">>> Train Policy (Offline)")
            expert_dataset.switch('success')
            loader = make_loader(expert_dataset)
            self.train_loop(loader, cfg_sp.offline_iters_actor, mode="actor", save_dir=checkpoint_dir)
            self.save(f"{checkpoint_dir}/{cfg_sp.exp_name}_pretrain_final.pth", meta={"phase": "pretrain_done"})
            del loader
            gc.collect()
        
        elif phase == 'online':
            # ... (在线部分的优化同步应用)
            if len(replay_buffer) < additional_args.get("replay_minvol", 0):
                return
            
            print(">>> Training Standard_Critic (Online)...")
            replay_buffer.switch('all')
            combined_ds = ConcatDataset([expert_dataset, replay_buffer])
            loader = make_loader(combined_ds)
            self.train_loop(loader, cfg_sp.online_iters_critic, mode="Standard_critic")
            del loader
            gc.collect()

            print(">>> Training Pretty Critic (Online)...")
            replay_buffer.switch('success')
            combined_ds_suc = ConcatDataset([expert_dataset, replay_buffer])
            loader = make_loader(combined_ds_suc)
            self.train_loop(loader, cfg_sp.online_iters_critic, mode="Pretty_critic")
            del loader
            gc.collect()

            print(">>> Relabeling Online Data...")
            self.relabel_data(replay_buffer, phase="finetune")

            print(">>> Training Policy (Online)...")
            mode = 'success' if cfg_sp.only_suc else 'all'
            replay_buffer.switch(mode)
            combined_actor_ds = ConcatDataset([expert_dataset, replay_buffer])
            loader = make_loader(combined_actor_ds)
            self.train_loop(loader, cfg_sp.online_iters_actor, mode="actor", save_dir=f"{checkpoint_dir}/{additional_args.get('round_idx', 0)}_RL")
            del loader
            gc.collect()
