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
    Failure-penalty twin Q network.

    Inputs:
        obs embedding: flattened [B, obs_horizon * encoder_out_dim]
        action chunk: flattened [B, pred_horizon * action_dim]
        k embedding: phi_k(k)
    Outputs:
        q1, q2: [B, 1], where Q(o, a, k) = Q_success(o, a) - (1-k) * P_fail(o, a, k).
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
        success_input_dim = state_feat_dim + self.flat_action_dim
        penalty_input_dim = success_input_dim + self.k_encoder.out_dim
        self.q_success_1 = make_mlp(success_input_dim, list(hidden_dims) + [1], last_act=False)
        self.q_success_2 = make_mlp(success_input_dim, list(hidden_dims) + [1], last_act=False)
        self.q_penalty_1 = make_mlp(penalty_input_dim, list(hidden_dims) + [1], last_act=False)
        self.q_penalty_2 = make_mlp(penalty_input_dim, list(hidden_dims) + [1], last_act=False)

    def _features(
        self,
        obs_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_embed = self.encoder(obs_dict)
        state_flat = state_embed.flatten(start_dim=1)
        action_flat = actions.flatten(start_dim=1).float()
        k_embed = self.k_encoder(k.to(device=action_flat.device))

        if action_flat.shape[-1] != self.flat_action_dim:
            raise ValueError(
                f"CPIQLQNet expected flat action dim {self.flat_action_dim}, "
                f"got {action_flat.shape[-1]} from shape {tuple(actions.shape)}"
            )

        success_input = torch.cat([state_flat, action_flat], dim=-1)
        penalty_input = torch.cat([success_input, k_embed], dim=-1)
        k_gate = (1.0 - k.to(device=action_flat.device).float().reshape(k_embed.shape[0], -1)[:, :1]).clamp(0.0, 1.0)
        return success_input, penalty_input, k_gate

    def decompose(
        self,
        obs_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        k: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        success_input, penalty_input, k_gate = self._features(obs_dict, actions, k)
        q_success_1 = self.q_success_1(success_input)
        q_success_2 = self.q_success_2(success_input)
        penalty_1 = torch.nn.functional.softplus(self.q_penalty_1(penalty_input))
        penalty_2 = torch.nn.functional.softplus(self.q_penalty_2(penalty_input))
        q1 = q_success_1 - k_gate * penalty_1
        q2 = q_success_2 - k_gate * penalty_2
        return {
            "q1": q1,
            "q2": q2,
            "q_success_1": q_success_1,
            "q_success_2": q_success_2,
            "penalty_1": penalty_1,
            "penalty_2": penalty_2,
            "k_gate": k_gate,
        }

    def forward(self, obs_dict: Dict[str, torch.Tensor], actions: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.decompose(obs_dict, actions, k)
        return outputs["q1"], outputs["q2"]


class CPIQLVNet(nn.Module):
    """
    Failure-penalty value network V(o, k) = V_success(o) - (1-k) * P_fail(o, k).
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
        penalty_input_dim = state_feat_dim + self.k_encoder.out_dim
        self.v_success = make_mlp(state_feat_dim, list(hidden_dims) + [1], last_act=False)
        self.v_penalty = make_mlp(penalty_input_dim, list(hidden_dims) + [1], last_act=False)

    def decompose(self, obs_dict: Dict[str, torch.Tensor], k: torch.Tensor) -> Dict[str, torch.Tensor]:
        state_embed = self.encoder(obs_dict)
        state_flat = state_embed.flatten(start_dim=1)
        k_embed = self.k_encoder(k.to(device=state_flat.device))
        penalty_input = torch.cat([state_flat, k_embed], dim=-1)
        k_gate = (1.0 - k.to(device=state_flat.device).float().reshape(k_embed.shape[0], -1)[:, :1]).clamp(0.0, 1.0)
        v_success = self.v_success(state_flat)
        penalty = torch.nn.functional.softplus(self.v_penalty(penalty_input))
        value = v_success - k_gate * penalty
        return {
            "value": value,
            "v_success": v_success,
            "penalty": penalty,
            "k_gate": k_gate,
        }

    def forward(self, obs_dict: Dict[str, torch.Tensor], k: torch.Tensor) -> torch.Tensor:
        return self.decompose(obs_dict, k)["value"]
