import torch
import torch.nn as nn
from typing import Dict, Optional
from agent_factory.modules.encoders.state_encoder import BaseStateEncoder
from agent_factory.modules.encoders.visual_encoder import make_mlp

class MultiHeadQuantileNet(nn.Module):
    """
    ITQC 核心网络：支持 Q (s, a) 和 V (s) 两种模式。
    输出：[B, Num_Critics, Num_Quantiles]
    """
    def __init__(
        self,
        state_encoder: BaseStateEncoder,
        obs_horizon: int,
        action_dim: int = 0, # 如果为 0，则是 V Net；否则是 Q Net
        pred_horizon: int = 1, # 用于计算 Action Flatten 维度
        hidden_dims: list = [512, 512],
        num_critics: int = 2,
        num_quantiles: int = 5,
    ):
        super().__init__()
        self.encoder = state_encoder
        self.num_critics = num_critics
        self.num_quantiles = num_quantiles
        self.use_action = action_dim > 0
        
        # 注册固定的 Tau (0.1 ~ 0.9)
        Tau = [i*1.0/num_quantiles+0.5/num_quantiles for i in range(num_quantiles)]
        tau_values = torch.tensor(Tau, dtype=torch.float32)
        self.register_buffer('tau', tau_values)
        
        # 计算输入维度
        # State: [B, T, D] -> [B, T*D]
        input_dim = state_encoder.out_dim * obs_horizon
        
        if self.use_action:
            # Action: [B, T_pred, A] -> [B, T_pred*A]
            input_dim += pred_horizon * action_dim
            
        # 独立的 MLP Heads
        self.heads = nn.ModuleList()
        for _ in range(num_critics):
            head = make_mlp(
                in_channels=input_dim,
                mlp_channels=hidden_dims + [num_quantiles],
                last_act=False
            )
            self.heads.append(head)

    def forward(self, obs_dict: Dict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns: [B, Num_Critics, Num_Quantiles]
        """
        # 1. State Features
        state_embed = self.encoder(obs_dict) # [B, T, D]
        state_flat = state_embed.flatten(start_dim=1)
        
        features = [state_flat]
        
        # 2. Action Features (Optional)
        if self.use_action:
            if actions is None:
                raise ValueError("Action required for Q-Network")
            act_flat = actions.flatten(start_dim=1)
            features.append(act_flat)
            
        combined = torch.cat(features, dim=-1)
        
        # 3. Multi-Head Forward
        outputs = []
        for head in self.heads:
            outputs.append(head(combined))
            
        return torch.stack(outputs, dim=1) # [B, M, N]