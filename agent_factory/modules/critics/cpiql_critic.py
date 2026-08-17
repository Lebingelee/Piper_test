from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from agent_factory.modules.encoders.state_encoder import BaseStateEncoder
from agent_factory.modules.encoders.visual_encoder import make_mlp


class _DiscreteKSelectorMixin:
    def _select_from_all(self, values_all: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if k.ndim == 0:
            k = k.reshape(1, 1)
        elif k.ndim == 1:
            k = k.unsqueeze(-1)
        else:
            k = k.reshape(k.shape[0], -1)
            if k.shape[-1] != 1:
                raise ValueError(f"CPIQL k must have one scalar per sample, got shape {tuple(k.shape)}")
        k = k.to(device=values_all.device, dtype=torch.float32)
        diff = (k - self._k_grid_buffer.reshape(1, -1)).abs()
        indices = diff.argmin(dim=-1)
        matched = diff.gather(dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)
        if torch.any(matched > 1e-6):
            bad_values = k.squeeze(-1)[matched > 1e-6][:5].detach().cpu().tolist()
            raise ValueError(
                f"CPIQL discrete-k critic only supports k_grid={self.k_grid}, got unsupported k values {bad_values}"
            )
        return values_all.gather(dim=-1, index=indices.unsqueeze(-1))


class CPIQLQNet(nn.Module, _DiscreteKSelectorMixin):
    """
    Discrete-k twin Q network with one success head A and a family of realistic
    heads B_k.

    Semantics:
        - k = 1.0 uses the success head A
        - k < 1.0 uses the corresponding family head B_k
    """

    def __init__(
        self,
        state_encoder: BaseStateEncoder,
        flat_action_dim: int,
        obs_horizon: int,
        hidden_dims: Sequence[int] = (256, 256),
        k_grid: Sequence[float] = (0.0, 1.0),
    ):
        super().__init__()
        self.encoder = state_encoder
        self.obs_horizon = int(obs_horizon)
        self.flat_action_dim = int(flat_action_dim)

        self.k_grid = [float(v) for v in k_grid]
        if not self.k_grid:
            raise ValueError("CPIQLQNet requires a non-empty k_grid.")
        self.register_buffer("_k_grid_buffer", torch.tensor(self.k_grid, dtype=torch.float32), persistent=False)
        self.family_head_indices = [idx for idx, value in enumerate(self.k_grid) if value < 1.0]
        self.success_head_indices = [idx for idx, value in enumerate(self.k_grid) if value >= 1.0]
        self.num_family_heads = len(self.family_head_indices)

        state_feat_dim = state_encoder.out_dim * self.obs_horizon
        input_dim = state_feat_dim + self.flat_action_dim

        self.q_success_1 = make_mlp(input_dim, list(hidden_dims) + [1], last_act=False)
        self.q_success_2 = make_mlp(input_dim, list(hidden_dims) + [1], last_act=False)

        family_out_dim = max(self.num_family_heads, 1)
        self.q_family_1 = make_mlp(input_dim, list(hidden_dims) + [family_out_dim], last_act=False)
        self.q_family_2 = make_mlp(input_dim, list(hidden_dims) + [family_out_dim], last_act=False)

    def _features(self, obs_dict: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        state_embed = self.encoder(obs_dict)
        state_flat = state_embed.flatten(start_dim=1)
        action_flat = actions.flatten(start_dim=1).float()
        if action_flat.shape[-1] != self.flat_action_dim:
            raise ValueError(
                f"CPIQLQNet expected flat action dim {self.flat_action_dim}, "
                f"got {action_flat.shape[-1]} from shape {tuple(actions.shape)}"
            )
        return torch.cat([state_flat, action_flat], dim=-1)

    def _empty_all(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, len(self.k_grid), device=device, dtype=torch.float32)

    def _scatter_family_heads(self, family_values: torch.Tensor) -> torch.Tensor:
        values_all = self._empty_all(family_values.shape[0], family_values.device)
        if self.num_family_heads > 0:
            values_all[:, self.family_head_indices] = family_values[:, :self.num_family_heads]
        return values_all

    def decompose_all(self, obs_dict: Dict[str, torch.Tensor], actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self._features(obs_dict, actions)
        q_success_1 = self.q_success_1(features)
        q_success_2 = self.q_success_2(features)
        q_family_1_heads = self.q_family_1(features)
        q_family_2_heads = self.q_family_2(features)

        q_family_1_all = self._scatter_family_heads(q_family_1_heads)
        q_family_2_all = self._scatter_family_heads(q_family_2_heads)
        q1_all = q_family_1_all.clone()
        q2_all = q_family_2_all.clone()
        if self.success_head_indices:
            success_cols = torch.tensor(self.success_head_indices, device=features.device, dtype=torch.long)
            q1_all[:, success_cols] = q_success_1.expand(-1, len(self.success_head_indices))
            q2_all[:, success_cols] = q_success_2.expand(-1, len(self.success_head_indices))

        return {
            "q1_all": q1_all,
            "q2_all": q2_all,
            "q_success_1": q_success_1,
            "q_success_2": q_success_2,
            "q_family_1_all": q_family_1_all,
            "q_family_2_all": q_family_2_all,
        }

    def decompose(
        self,
        obs_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        k: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.decompose_all(obs_dict, actions)
        return {
            "q1": self._select_from_all(outputs["q1_all"], k),
            "q2": self._select_from_all(outputs["q2_all"], k),
            "q_success_1": outputs["q_success_1"],
            "q_success_2": outputs["q_success_2"],
            "q_family_1": self._select_from_all(outputs["q_family_1_all"], k),
            "q_family_2": self._select_from_all(outputs["q_family_2_all"], k),
        }

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.decompose(obs_dict, actions, k)
        return outputs["q1"], outputs["q2"]


class CPIQLVNet(nn.Module, _DiscreteKSelectorMixin):
    """
    Discrete-k value network with one success head A and a family of realistic
    heads B_k.

    Semantics:
        - k = 1.0 uses the success head A
        - k < 1.0 uses the corresponding family head B_k
    """

    def __init__(
        self,
        state_encoder: BaseStateEncoder,
        obs_horizon: int,
        hidden_dims: Sequence[int] = (256, 256),
        k_grid: Sequence[float] = (0.0, 1.0),
    ):
        super().__init__()
        self.encoder = state_encoder
        self.obs_horizon = int(obs_horizon)

        self.k_grid = [float(v) for v in k_grid]
        if not self.k_grid:
            raise ValueError("CPIQLVNet requires a non-empty k_grid.")
        self.register_buffer("_k_grid_buffer", torch.tensor(self.k_grid, dtype=torch.float32), persistent=False)
        self.family_head_indices = [idx for idx, value in enumerate(self.k_grid) if value < 1.0]
        self.success_head_indices = [idx for idx, value in enumerate(self.k_grid) if value >= 1.0]
        self.num_family_heads = len(self.family_head_indices)

        state_feat_dim = state_encoder.out_dim * self.obs_horizon
        self.v_success = make_mlp(state_feat_dim, list(hidden_dims) + [1], last_act=False)
        family_out_dim = max(self.num_family_heads, 1)
        self.v_family = make_mlp(state_feat_dim, list(hidden_dims) + [family_out_dim], last_act=False)

    def _empty_all(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, len(self.k_grid), device=device, dtype=torch.float32)

    def _scatter_family_heads(self, family_values: torch.Tensor) -> torch.Tensor:
        values_all = self._empty_all(family_values.shape[0], family_values.device)
        if self.num_family_heads > 0:
            values_all[:, self.family_head_indices] = family_values[:, :self.num_family_heads]
        return values_all

    def decompose_all(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_embed = self.encoder(obs_dict)
        state_flat = state_embed.flatten(start_dim=1)
        success_value = self.v_success(state_flat)
        family_value_heads = self.v_family(state_flat)
        family_value_all = self._scatter_family_heads(family_value_heads)
        value_all = family_value_all.clone()
        if self.success_head_indices:
            success_cols = torch.tensor(self.success_head_indices, device=state_flat.device, dtype=torch.long)
            value_all[:, success_cols] = success_value.expand(-1, len(self.success_head_indices))
        return {
            "value_all": value_all,
            "success_value": success_value,
            "v_success": success_value,
            "family_value_all": family_value_all,
        }

    def decompose(self, obs_dict: Dict[str, torch.Tensor], k: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.decompose_all(obs_dict)
        return {
            "value": self._select_from_all(outputs["value_all"], k),
            "success_value": outputs["success_value"],
            "v_success": outputs["success_value"],
            "family_value": self._select_from_all(outputs["family_value_all"], k),
        }

    def forward(self, obs_dict: Dict[str, torch.Tensor], k: torch.Tensor) -> torch.Tensor:
        return self.decompose(obs_dict, k)["value"]
