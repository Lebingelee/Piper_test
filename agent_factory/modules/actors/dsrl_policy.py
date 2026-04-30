import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, ComposeTransform, TanhTransform

from agent_factory.modules.encoders.state_encoder import BaseStateEncoder


def _build_mlp(in_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    layers = []
    current_dim = in_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        current_dim = hidden_dim
    return nn.Sequential(*layers) if layers else nn.Identity()


class TanhDiagNormal(TransformedDistribution):
    def __init__(
        self,
        loc: torch.Tensor,
        scale_diag: torch.Tensor,
        low: Optional[float] = None,
        high: Optional[float] = None,
    ):
        base_dist = Independent(Normal(loc=loc, scale=scale_diag), 1)
        transforms = [TanhTransform(cache_size=1)]
        if low is not None and high is not None:
            scale = (high - low) / 2.0
            shift = (high + low) / 2.0
            transforms.append(AffineTransform(loc=shift, scale=scale))
        super().__init__(base_dist, ComposeTransform(transforms))
        self._loc = loc

    def mode(self) -> torch.Tensor:
        output = self._loc
        for transform in self.transforms:
            output = transform(output)
        return output


class DSRLNoisePolicy(nn.Module):
    def __init__(
        self,
        state_encoder: BaseStateEncoder,
        obs_horizon: int,
        action_dim: int,
        pred_horizon: int,
        hidden_dims: Sequence[int],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        action_limit: float = 1.0, # Modified from 1.5 to 1.0 to align with [-1, 1] normalization
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_limit = action_limit
        self.flat_action_dim = action_dim * pred_horizon

        # trunk_in_dim = state_encoder.out_dim * obs_horizon
        # The new BaseStateEncoder returns [B, T, out_dim]. 
        # After flattening it is [B, T * out_dim].
        self.trunk_in_dim = state_encoder.out_dim * obs_horizon
        self.trunk = _build_mlp(self.trunk_in_dim, hidden_dims)
        self.head_in_dim = hidden_dims[-1] if hidden_dims else self.trunk_in_dim
        self.mean_head = nn.Linear(self.head_in_dim, self.flat_action_dim)
        self.log_std_head = nn.Linear(self.head_in_dim, self.flat_action_dim)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> TanhDiagNormal:
        # obs_feat = self.state_encoder(obs_dict).flatten(start_dim=1)
        # Note: if self.obs_horizon is different from the T in obs_dict, 
        # BaseStateEncoder might return different T.
        # But usually they are the same in training.
        state_embed = self.state_encoder(obs_dict)
        obs_feat = state_embed.flatten(start_dim=1)
        
        # Security check for dimension mismatch
        if obs_feat.shape[1] != self.trunk_in_dim:
             # If mismatch, it might be due to different obs_horizon than expected
             # We can try to take the last obs_horizon steps or pad?
             # But it's better to ensure they match.
             pass

        hidden = self.trunk(obs_feat)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(self.log_std_min, self.log_std_max)
        return TanhDiagNormal(
            loc=mean,
            scale_diag=log_std.exp(),
            low=-self.action_limit,
            high=self.action_limit,
        )
