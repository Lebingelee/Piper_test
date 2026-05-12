from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from agent_factory.modules.encoders.state_encoder import BaseStateEncoder
from agent_factory.modules.encoders.visual_encoder import make_mlp


class KConditionEncoder(nn.Module):
    """
    Small learned embedding for the success-bias condition k.
    """

    def __init__(self, embed_dim: int = 16, hidden_dims: Sequence[int] = (32,)):
        super().__init__()
        self.embed_dim = int(embed_dim)
        if self.embed_dim <= 0:
            self.net = nn.Identity()
            self.out_dim = 1
        else:
            self.net = make_mlp(1, list(hidden_dims) + [self.embed_dim], last_act=True)
            self.out_dim = self.embed_dim

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        if k.ndim == 0:
            k = k.reshape(1, 1)
        elif k.ndim == 1:
            k = k.unsqueeze(-1)
        else:
            k = k.reshape(k.shape[0], -1)
            if k.shape[-1] != 1:
                raise ValueError(f"CPIQL k must have one scalar per sample, got shape {tuple(k.shape)}")
        return self.net(k.float())


class CPIQLQNet(nn.Module):
    """
    Failure-conditioned twin Q network.

    Inputs:
        obs embedding: flattened [B, obs_horizon * encoder_out_dim]
        action chunk: flattened [B, pred_horizon * action_dim]
        k embedding: phi_k(k)
    Outputs:
        q1, q2: [B, 1]
    """

    def __init__(
        self,
        state_encoder: BaseStateEncoder,
        flat_action_dim: int,
        obs_horizon: int,
        hidden_dims: Sequence[int] = (256, 256),
        k_embed_dim: int = 16,
        k_hidden_dims: Sequence[int] = (32,),
    ):
        super().__init__()
        self.encoder = state_encoder
        self.obs_horizon = int(obs_horizon)
        self.flat_action_dim = int(flat_action_dim)
        self.k_encoder = KConditionEncoder(k_embed_dim, k_hidden_dims)

        state_feat_dim = state_encoder.out_dim * self.obs_horizon
        input_dim = state_feat_dim + self.flat_action_dim + self.k_encoder.out_dim
        self.q1 = make_mlp(input_dim, list(hidden_dims) + [1], last_act=False)
        self.q2 = make_mlp(input_dim, list(hidden_dims) + [1], last_act=False)

    def forward(self, obs_dict: Dict[str, torch.Tensor], actions: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_embed = self.encoder(obs_dict)
        state_flat = state_embed.flatten(start_dim=1)
        action_flat = actions.flatten(start_dim=1).float()
        k_embed = self.k_encoder(k.to(device=action_flat.device))

        if action_flat.shape[-1] != self.flat_action_dim:
            raise ValueError(
                f"CPIQLQNet expected flat action dim {self.flat_action_dim}, "
                f"got {action_flat.shape[-1]} from shape {tuple(actions.shape)}"
            )

        critic_input = torch.cat([state_flat, action_flat, k_embed], dim=-1)
        return self.q1(critic_input), self.q2(critic_input)


class CPIQLVNet(nn.Module):
    """
    Failure-conditioned value network V(o, k).
    """

    def __init__(
        self,
        state_encoder: BaseStateEncoder,
        obs_horizon: int,
        hidden_dims: Sequence[int] = (256, 256),
        k_embed_dim: int = 16,
        k_hidden_dims: Sequence[int] = (32,),
    ):
        super().__init__()
        self.encoder = state_encoder
        self.obs_horizon = int(obs_horizon)
        self.k_encoder = KConditionEncoder(k_embed_dim, k_hidden_dims)

        state_feat_dim = state_encoder.out_dim * self.obs_horizon
        input_dim = state_feat_dim + self.k_encoder.out_dim
        self.v = make_mlp(input_dim, list(hidden_dims) + [1], last_act=False)

    def forward(self, obs_dict: Dict[str, torch.Tensor], k: torch.Tensor) -> torch.Tensor:
        state_embed = self.encoder(obs_dict)
        state_flat = state_embed.flatten(start_dim=1)
        k_embed = self.k_encoder(k.to(device=state_flat.device))
        critic_input = torch.cat([state_flat, k_embed], dim=-1)
        return self.v(critic_input)
