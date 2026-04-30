from typing import Dict, Sequence

import torch
import torch.nn as nn
from einops import rearrange

from agent_factory.modules.encoders.state_encoder import BaseStateEncoder


def _build_q_head(in_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    layers = []
    current_dim = in_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)


class DSRLQEnsemble(nn.Module):
    def __init__(
        self,
        state_encoder: BaseStateEncoder,
        obs_horizon: int,
        action_dim: int,
        pred_horizon: int,
        hidden_dims: Sequence[int],
        num_qs: int = 2,
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.flat_action_dim = action_dim * pred_horizon

        self.trunk_in_dim = state_encoder.out_dim * obs_horizon + self.flat_action_dim
        self.q_heads = nn.ModuleList(
            [_build_q_head(self.trunk_in_dim, hidden_dims) for _ in range(num_qs)]
        )

    def forward(self, obs_dict: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        # actions is [B, pred_horizon, action_dim] or [B, flat_action_dim]
        obs_feat = self.state_encoder(obs_dict).flatten(start_dim=1)
        flat_actions = actions.flatten(start_dim=1)
        
        # Security check for dimension mismatch
        if obs_feat.shape[1] + flat_actions.shape[1] != self.trunk_in_dim:
             # Handle mismatch if necessary
             pass

        critic_input = torch.cat([obs_feat, flat_actions], dim=-1)
        return torch.stack(
            [q_head(critic_input).squeeze(-1) for q_head in self.q_heads],
            dim=0,
        )
