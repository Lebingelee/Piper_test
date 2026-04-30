import torch
import torch.nn as nn
from typing import Dict, Tuple
from agent_factory.modules.encoders.state_encoder import BaseStateEncoder
from agent_factory.modules.encoders.visual_encoder import make_mlp

class IQLQNet(nn.Module):
    """
    Twin Q-Network for IQL (Double Q-Learning)
    Input: State Embedding + Action
    Output: Q1, Q2 (Scalars)
    """      
    def __init__(self, state_encoder, action_dim, obs_horizon, hidden_dims=[256, 256]):
        super().__init__()
        self.encoder = state_encoder
        
        # Calculate Input Dim
        # Encoder Output: [B, T, D] -> Flatten -> [B, T*D]
        state_feat_dim = state_encoder.out_dim * obs_horizon
        input_dim = state_feat_dim + action_dim * obs_horizon # IQL 通常只用当前步，但在 Diffusion Policy 语境下可能是 History
        
        # 如果是标准 IQL (s,a)，通常只看当前帧。但为了兼容 Sequence 输入：
        # 这里假设 Critic 同时也看整个 History (s_{t-H:t}, a_{t-H:t}) 或者只看当前。
        # 原代码 IQL Q 输入包含 (obs_horizon * obs_dim) + (pred_horizon * act_dim)?
        # 让我们参考 recap_agent_IQL.py (未提供完整，但在 ITQC 中 MultiHeadQuantileNet 确实 Flatten 了所有 horizons)
        # 所以我们这里也 Flatten 所有 Time Steps。
        
        self.input_dim = state_feat_dim + action_dim # 这里假设 Action 只输入当前步或者 Flatten 后的 Action History
        # 修正：通常 Critic 的 Action 输入是单个动作 a_t (对于 IQL)。
        # 如果是 Horizon 长度的 Action 序列，维度应为 action_dim * pred_horizon。
        # 假设 IQL 此处针对的是单步决策或 Chunk 的第一步，通常输入 action_dim。
        # 但 Diffusion Policy 的 Critic 有时会评估整个 Trajectory。
        # 根据 recap_agent_ITQC.py，MultiHeadQuantileNet use_action 时输入了 pred_horizon * act_dim。
        # 我们这里保持灵活，由外部配置决定。
        
        self.q1 = make_mlp(input_dim, hidden_dims + [1], last_act=False)
        self.q2 = make_mlp(input_dim, hidden_dims + [1], last_act=False)

    def forward(self, obs_dict: Dict, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_dict: 状态字典
            actions: [B, A_Dim] or [B, T, A_Dim] -> 需 Flatten
        Returns:
            q1, q2: [B, 1]
        """
        # 1. State Encoding
        state_embed = self.encoder(obs_dict) # [B, T, D]
        state_flat = state_embed.flatten(start_dim=1) # [B, T*D]
        
        # 2. Action Handling
        act_flat = actions.flatten(start_dim=1) # [B, A_flat]
        
        x = torch.cat([state_flat, act_flat], dim=-1)
        
        return self.q1(x), self.q2(x)

class IQLVNet(nn.Module):
    """
    Value Network for IQL (V(s))
    Input: State Embedding
    Output: V (Scalar)
    """
    def __init__(self, state_encoder, obs_horizon, hidden_dims=[256, 256]):
        super().__init__()
        self.encoder = state_encoder
        
        state_feat_dim = state_encoder.out_dim * obs_horizon
        self.v = make_mlp(state_feat_dim, hidden_dims + [1], last_act=False)

    def forward(self, obs_dict: Dict) -> torch.Tensor:
        state_embed = self.encoder(obs_dict) # [B, T, D]
        state_flat = state_embed.flatten(start_dim=1)
        return self.v(state_flat)