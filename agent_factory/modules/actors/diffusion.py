import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# 复用你项目中的现有模块
from agent_factory.modules.actors.diffusion_module.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from agent_factory.modules.encoders.state_encoder import BaseStateEncoder

class AbstractDiffusionPolicy(nn.Module):
    """
    Diffusion Policy 的通用基类。
    处理 Scheduler, 动作预测流程, 和 Loss 计算骨架。
    """
    def __init__(
        self,
        state_encoder: BaseStateEncoder,
        action_dim: int,
        pred_horizon: int,
        obs_horizon: int,
        unet_config: Dict,
        scheduler_config: Dict,
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon

        # Scheduler (DDPM) - 保持 clip_sample=True，新架构会确保输入在 [-1, 1]
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=scheduler_config.get("num_train_timesteps", 100),
            beta_schedule=scheduler_config.get("beta_schedule", "squaredcos_cap_v2"),
            clip_sample=True,
            prediction_type=scheduler_config.get("prediction_type", "epsilon"),
        )
        
        # UNet 将在子类中实例化
        self.noise_pred_net = None 

    def compute_loss(self, obs_dict: Dict, actions: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        """
        通用的 Diffusion Loss 计算逻辑。
        注意：此处 actions 应当是已经由 Mixin 归一化过的。
        """
        B = actions.shape[0]
        
        # 1. 采样噪声
        noise = torch.randn(actions.shape, device=actions.device)
        
        # 2. 采样时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=actions.device
        ).long()
        
        # 3. 加噪
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # 4. 预测噪声
        noise_pred = self.noise_pred_net(
            sample=noisy_actions, 
            timestep=timesteps, 
            global_cond=global_cond
        )
        
        # 5. MSE Loss
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, obs_dict: Dict, global_cond: torch.Tensor, num_inference_steps = 10, initial_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        通用的推理采样逻辑。
        返回的是 [-1, 1] 空间内的归一化动作。
        """
        B = global_cond.shape[0]
        
        if initial_noise is not None:
            noisy_actions = initial_noise
        else:
            noisy_actions = torch.randn(
                (B, self.pred_horizon, self.action_dim), 
                device=global_cond.device
            )
        self.noise_scheduler.set_timesteps(num_inference_steps = num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                sample=noisy_actions, 
                timestep=t, 
                global_cond=global_cond
            )
            
            noisy_actions = self.noise_scheduler.step(
                model_output=noise_pred, 
                timestep=t, 
                sample=noisy_actions
            ).prev_sample
            
        return noisy_actions


class VanillaDiffusionPolicy(AbstractDiffusionPolicy):
    """
    无条件 Diffusion Policy (p(a|s))
    Global Cond = State Embedding
    """
    def __init__(self, state_encoder, action_dim, pred_horizon, obs_horizon, unet_config, scheduler_config):
        super().__init__(state_encoder, action_dim, pred_horizon, obs_horizon, unet_config, scheduler_config)
        
        # 计算 UNet 的 Condition 维度 = 状态编码维度
        if state_encoder.view_fusion == 'concat':
            self.global_cond_dim = state_encoder.out_dim*obs_horizon
        elif state_encoder.view_fusion == 'mean':
            self.global_cond_dim = state_encoder.out_dim 
        else:
            raise ValueError(f"Unknown view_fusion type: {state_encoder.view_fusion}")
        
        # 实例化 UNet
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=self.global_cond_dim,
            **unet_config # down_dims, diffusion_step_embed_dim etc.
        )

    def forward(self, obs_dict: Dict, actions: torch.Tensor) -> torch.Tensor:
        # 1. 编码状态 [B, T, Embed] -> Flatten or Mean? 
        # 通常 Diffusion Policy 使用 Flatten 的 obs 特征作为 cond
        # BaseStateEncoder 输出 [B, T, D]，我们需要将其展平 [B, T*D]
        state_embed = self.state_encoder(obs_dict) 
        global_cond = state_embed.flatten(start_dim=1) 
        
        return super().compute_loss(obs_dict, actions, global_cond)

    def sample_action(self, obs_dict: Dict, num_inference_steps: int = 10, initial_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        state_embed = self.state_encoder(obs_dict)
        global_cond = state_embed.flatten(start_dim=1)
        return self.sample(obs_dict, global_cond, num_inference_steps=num_inference_steps, initial_noise=initial_noise)


class ConditionalDiffusionPolicy(AbstractDiffusionPolicy):
    """
    有条件 Diffusion Policy (p(a|s, c))
    Global Cond = Concat(State Embedding, Condition Embedding)
    支持 Classifier-Free Guidance (CFG)
    """
    def __init__(
        self, 
        state_encoder, 
        action_dim, 
        pred_horizon, 
        obs_horizon, 
        unet_config, 
        scheduler_config,
        cond_dim: int = 1, # 外部 Condition 维度 (例如 score 是 1)
        cond_embed_dim: int = 32, # Embedding 后的维度
        
    ):
        super().__init__(state_encoder, action_dim, pred_horizon, obs_horizon, unet_config, scheduler_config)
        
        # 1. Condition Embedding Network (MLP)
        self.cond_embed_net = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim)
        )
        
        # 2. UNet Condition = State Embed + Cond Embed
        if state_encoder.view_fusion == 'concat':
            self.global_cond_dim = state_encoder.out_dim*obs_horizon + cond_embed_dim
        elif state_encoder.view_fusion == 'mean':
            self.global_cond_dim = state_encoder.out_dim + cond_embed_dim 
        else:
            raise ValueError(f"Unknown view_fusion type: {state_encoder.view_fusion}")
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=self.global_cond_dim,
            **unet_config
        )

    def _get_global_cond(self, obs_dict, cond_val):
        """辅助函数：拼接状态和条件"""
        # State: [B, T, D] -> [B, T*D]
        state_embed = self.state_encoder(obs_dict).flatten(start_dim=1)
        
        # Condition: [B, 1] -> [B, D_cond]
        cond_embed = self.cond_embed_net(cond_val)
        
        # Concat
        return torch.cat([state_embed, cond_embed], dim=-1)

    def forward(self, obs_dict: Dict, actions: torch.Tensor, cond: torch.Tensor, use_cfg_loss: bool = False, cfg_drop_rate: float = 0.1) -> torch.Tensor:
        """
        Train Step with Optional CFG Dropout
        """
        B = actions.shape[0]
        
        # 处理 CFG Dropout
        if use_cfg_loss and cfg_drop_rate > 0:
            mask = (torch.rand((B, 1), device=actions.device) > cfg_drop_rate).float()
            train_cond = cond * mask
        else:
            train_cond = cond

        global_cond = self._get_global_cond(obs_dict, train_cond)
        
        return super().compute_loss(obs_dict, actions, global_cond)

    def sample_action(self, obs_dict: Dict, cond: torch.Tensor, guidance_scale: float = 1.0, num_inference_steps: int = 10, initial_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inference with CFG
        cond: [B, 1] 目标条件
        """
        B = next(iter(obs_dict.values())).shape[0]
        
        # 1. 分别构建条件和无条件下的 global_cond
        global_cond_pos = self._get_global_cond(obs_dict, cond)
        global_cond_neg = self._get_global_cond(obs_dict, torch.zeros_like(cond))
        
        # 2. 初始化噪声
        if initial_noise is not None:
            noisy_actions = initial_noise
        else:
            noisy_actions = torch.randn(
                (B, self.pred_horizon, self.action_dim), 
                device=cond.device
            )
        

        self.noise_scheduler.set_timesteps(num_inference_steps = num_inference_steps)
        # 3. 逐步去噪
        for t in self.noise_scheduler.timesteps:
            # 预测噪声 (包含 Classifier-Free Guidance)
            # 将 cond / uncond 分支按 batch 维拼接，单次前向并行计算。
            noisy_actions_cat = torch.cat([noisy_actions, noisy_actions], dim=0)
            global_cond_cat = torch.cat([global_cond_pos, global_cond_neg], dim=0)
            noise_pred_cat = self.noise_pred_net(noisy_actions_cat, t, global_cond=global_cond_cat)
            noise_pred_pos, noise_pred_neg = noise_pred_cat.chunk(2, dim=0)
            
            # CFG 公式: pred = uncond + scale * (cond - uncond)
            # 注意: uncond 对应 neg, cond 对应 pos
            noise_pred = noise_pred_neg + guidance_scale * (noise_pred_pos - noise_pred_neg)
            
            # 更新 action
            noisy_actions = self.noise_scheduler.step(noise_pred, t, noisy_actions).prev_sample
            
        # 4. 最终结果返回（由 Mixin 层负责反归一化）
        return noisy_actions
