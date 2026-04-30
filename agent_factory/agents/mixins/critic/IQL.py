import torch
import torch.nn.functional as F
import copy
import numpy as np
from tqdm import tqdm
from typing import Dict, Literal, Optional

# 引用组件
from agent_factory.modules.critics.iql_critic import IQLQNet, IQLVNet
from agent_factory.modules.critics.itqc_critic import MultiHeadQuantileNet
from agent_factory.modules.encoders.visual_encoder import VisualEncoder
from agent_factory.modules.encoders.state_encoder import BaseStateEncoder

from agent_factory.config.structure import IQLCriticConfig


class IQLCriticMixin:
    """
    IQL Critic 逻辑：
    - 维护 Q1, Q2, V, Target_Q1, Target_Q2 (注意：IQL 标准实现通常用 Target Q 计算 V Loss 的目标)
    - 计算 Expectile Loss (V) 和 MSE Loss (Q)
    """

    CONFIG_CLASS = IQLCriticConfig
    CONFIG_KEY = "critic"

    REQUIRED_KEYS = {
        "observations", "action", "next_observations", "reward", "terminated"
    }

    def _build_critic(self):
        cfg = self.cfg.critic
        encoder_cfg = cfg.encoder
        use_visual = getattr(self.cfg.dataset, "include_rgb", True)
        
        # 1. 初始化 Encoder
        vis_enc = None
        if use_visual:
            vis_enc = VisualEncoder(
                in_channels=encoder_cfg.visual.in_channels,
                out_dim=encoder_cfg.visual.out_dim,
                backbone_type=encoder_cfg.visual.backbone_type,
                pool_feature_map=encoder_cfg.visual.pool_feature_map,
                use_group_norm=encoder_cfg.visual.use_group_norm,
            )
        proprio_dim = encoder_cfg.proprio_dim or self.cfg.env.proprio_dim
        self.critic_encoder = BaseStateEncoder(
            visual_encoder=vis_enc,
            proprio_dim=proprio_dim,
            out_dim=encoder_cfg.out_dim,
            num_cameras=self.cfg.env.num_cameras,
            view_fusion=encoder_cfg.view_fusion,
        )
        
        # 2. 初始化网络
        common_args = dict(
            state_encoder=self.critic_encoder,
            obs_horizon=self.cfg.env.obs_horizon,
            hidden_dims=cfg.hidden_dims
        )
        
        # V Network
        self.v_net = IQLVNet(**common_args)
        
        # Q Network (Twin)
        # 假设 Action 需要 Flatten，维度由 env_config 提供
        act_dim_flat = self.cfg.env.action_dim * self.cfg.env.pred_horizon
        self.q_net = IQLQNet(action_dim=act_dim_flat, **common_args)
        
        # [修复] Target Q Networks (用于计算 V-loss 的目标: min(Q_targ_1, Q_targ_2) - V)
        # 虽然原始 IQL 论文可以直接用 Online Q，但为了稳定性通常维护 Target Q
        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_q_net.requires_grad_(False)
        
        # 3. Optimizers
        self.v_optimizer = torch.optim.AdamW(self.v_net.parameters(), lr=cfg.v_lr)
        self.q_optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=cfg.q_lr)

    def soft_update_target(self):
        tau = self.cfg.soft_update_tau
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update_critic(self, batch: Dict) -> Dict:
        obs = self._preprocess_obs(batch['observations'])
        actions = batch['action'].to(self.device)
        next_obs = self._preprocess_obs(batch['next_observations'])
        reward = batch['reward'].to(self.device).float()
        # IQL 不使用 terminated 来 mask target value? 
        # 通常: target = r + gamma * V(s') * (1-done)
        terminated = batch['terminated'].to(self.device).float()
        
        # --- 1. V Loss (Expectile) ---
        # L = |tau - I(min(Q_targ) - V < 0)| * (min(Q_targ) - V)^2
        with torch.no_grad():
            q1_targ, q2_targ = self.target_q_net(obs, actions)
            q_target = torch.min(q1_targ, q2_targ)
        
        v_pred = self.v_net(obs)
        adv = q_target - v_pred
        
        weight = torch.where(adv > 0, self.cfg.critic.expectile, (1 - self.cfg.critic.expectile))
        v_loss = (weight * (adv ** 2)).mean()
        
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        # --- 2. Q Loss (MSE) ---
        # Target = r + gamma * V(s')
        with torch.no_grad():
            next_v = self.v_net(next_obs)
            q_target_val = reward + self.cfg.env.gamma * next_v * (1.0 - terminated)
        
        q1_pred, q2_pred = self.q_net(obs, actions)
        q_loss = F.mse_loss(q1_pred, q_target_val) + F.mse_loss(q2_pred, q_target_val)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        return {"loss_v": v_loss.item(), "loss_q": q_loss.item(), "adv_mean": adv.mean().item()}

    def relabel_data(self, dataset, phase="pretrain"):
        """IQL Relabel Logic: Adv = min(TargetQ) - V"""
        print(f"[IQL Relabel] Phase: {phase}")
        self.eval()
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        
        all_advs = []
        with torch.no_grad():
            for batch in loader:
                obs = self._preprocess_obs(batch["observations"]) # 假设 Agent 有此辅助函数或 Mixin 提供
                for k in obs: obs[k] = obs[k].to(self.device)
                actions = batch["action"].to(self.device)
                
                v_pred = self.v_net(obs)
                q1, q2 = self.target_q_net(obs, actions)
                adv = torch.min(q1, q2) - v_pred
                all_advs.append(adv.cpu().numpy())
        
        self.train()
        all_advs = np.concatenate(all_advs)
        
        # Threshold logic
        cfg_sp = getattr(self.cfg, "agent_sp", None)
        quantile_pretrain = float(getattr(cfg_sp, "threshold_quantile_offline", 0.8))
        quantile_finetune = float(getattr(cfg_sp, "threshold_quantile_online", 0.3))
        quantile = quantile_pretrain if phase == "pretrain" else quantile_finetune
        threshold = np.percentile(all_advs, (1.0 - quantile) * 100)
        
        # Update
        is_positive = (all_advs.squeeze() > threshold)
        new_conds = torch.where(
            torch.from_numpy(is_positive).to(self.device),
            torch.tensor(1.0, device=self.device),
            torch.tensor(-1.0, device=self.device)
        )
        dataset.conds = new_conds.float().cpu() # Ensure it's on CPU for multiprocessing
        print(f"[IQL Relabel] Threshold: {threshold:.4f}, Positive Ratio: {is_positive.mean():.2%}")
