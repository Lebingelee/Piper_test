import copy
from typing import Dict, Sequence

import torch
import torch.nn as nn

from agent_factory.modules.encoders.visual_encoder import make_mlp


class _MaskedActionEncoder(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(action_dim), int(embed_dim)),
            nn.ReLU(),
            nn.Linear(int(embed_dim), int(embed_dim)),
            nn.ReLU(),
        )

    def forward(self, actions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if actions.ndim not in {3, 4}:
            raise ValueError(f"Actions must have shape [B,H,A] or [B,T,H,A], got {tuple(actions.shape)}")
        if valid_mask.shape != actions.shape[:-1]:
            raise ValueError(
                f"Action mask shape {tuple(valid_mask.shape)} must match action prefix {tuple(actions.shape[:-1])}."
            )
        encoded = self.net(actions.float())
        weights = valid_mask.to(device=encoded.device, dtype=encoded.dtype).unsqueeze(-1)
        return (encoded * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


class CPIQLRNNHistoryEncoder(nn.Module):
    """Encode ``(x_0, A_0, ..., x_k)`` into the final valid LSTM state."""

    def __init__(
        self,
        state_encoder: nn.Module,
        obs_horizon: int,
        action_dim: int,
        state_embed_dim: int = 512,
        action_embed_dim: int = 128,
        lstm_input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.obs_horizon = int(obs_horizon)
        self.hidden_dim = int(hidden_dim)
        encoder_dim = int(getattr(state_encoder, "out_dim", 0))
        if encoder_dim <= 0:
            raise ValueError("CPIQL RNN state encoder requires a positive out_dim.")
        self.state_projection = nn.Sequential(
            nn.Linear(encoder_dim * self.obs_horizon, int(state_embed_dim)),
            nn.ReLU(),
        )
        self.history_action_encoder = _MaskedActionEncoder(action_dim, action_embed_dim)
        self.token_projection = nn.Sequential(
            nn.Linear(int(state_embed_dim) + int(action_embed_dim), int(lstm_input_dim)),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_size=int(lstm_input_dim),
            hidden_size=self.hidden_dim,
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )

    def _encode_state_blocks(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        feature = observations.get("feature")
        if feature is None or feature.ndim != 4:
            shape = None if feature is None else tuple(feature.shape)
            raise ValueError(
                "CPIQL RNN observations['feature'] must have shape [B,L,obs_horizon,D], "
                f"got {shape}."
            )
        batch_size, seq_len = feature.shape[:2]
        flattened = {}
        for key, value in observations.items():
            if not torch.is_tensor(value) or value.shape[:2] != (batch_size, seq_len):
                continue
            flattened[key] = value.reshape(batch_size * seq_len, *value.shape[2:])
        encoded = self.state_encoder(flattened)
        encoded = encoded.reshape(batch_size, seq_len, -1)
        expected_dim = int(getattr(self.state_encoder, "out_dim", 0)) * self.obs_horizon
        if encoded.shape[-1] != expected_dim:
            raise ValueError(
                f"CPIQL RNN expected flattened state encoder dim {expected_dim}, got {encoded.shape[-1]}."
            )
        return self.state_projection(encoded)

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        history_actions: torch.Tensor,
        valid_mask: torch.Tensor,
        history_action_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        state_tokens = self._encode_state_blocks(observations)
        batch_size, seq_len, _ = state_tokens.shape
        if valid_mask.shape != (batch_size, seq_len):
            raise ValueError(f"valid_mask must have shape [B,L]={batch_size, seq_len}, got {tuple(valid_mask.shape)}")
        if history_actions.shape[:2] != (batch_size, max(seq_len - 1, 0)):
            raise ValueError(
                "history_actions must align with all non-final state blocks, got "
                f"{tuple(history_actions.shape)} for state sequence {tuple(state_tokens.shape)}"
            )

        action_tokens = torch.zeros(
            batch_size, seq_len, self.history_action_encoder.net[0].out_features,
            device=state_tokens.device,
            dtype=state_tokens.dtype,
        )
        if seq_len > 1:
            action_tokens[:, :-1] = self.history_action_encoder(history_actions, history_action_valid_mask)
        tokens = self.token_projection(torch.cat([state_tokens, action_tokens], dim=-1))
        lengths = valid_mask.to(device=tokens.device).gt(0).long().sum(dim=1)
        if torch.any(lengths <= 0):
            raise ValueError("CPIQL RNN received a sample with no valid history blocks.")
        packed = nn.utils.rnn.pack_padded_sequence(
            tokens, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        return outputs[torch.arange(batch_size, device=tokens.device), lengths - 1]


class CPIQLRNNVNet(nn.Module):
    """Two value heads: progress (k=1) and all-data value (k=0)."""

    def __init__(self, history_encoder: CPIQLRNNHistoryEncoder, hidden_dims: Sequence[int] = (256,)):
        super().__init__()
        self.history_encoder = history_encoder
        self.progress_head = make_mlp(history_encoder.hidden_dim, list(hidden_dims) + [1], last_act=False)
        self.value_head = make_mlp(history_encoder.hidden_dim, list(hidden_dims) + [1], last_act=False)

    def forward(self, observations, history_actions, valid_mask, history_action_valid_mask) -> Dict[str, torch.Tensor]:
        hidden = self.history_encoder(observations, history_actions, valid_mask, history_action_valid_mask)
        return {"progress": self.progress_head(hidden), "value": self.value_head(hidden)}


class _CPIQLRNNQBranch(nn.Module):
    def __init__(
        self,
        history_encoder: CPIQLRNNHistoryEncoder,
        action_dim: int,
        q_action_embed_dim: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        self.history_encoder = history_encoder
        self.q_action_encoder = _MaskedActionEncoder(action_dim, q_action_embed_dim)
        head_input_dim = history_encoder.hidden_dim + int(q_action_embed_dim)
        self.progress_head = make_mlp(head_input_dim, list(hidden_dims) + [1], last_act=False)
        self.value_head = make_mlp(head_input_dim, list(hidden_dims) + [1], last_act=False)

    def forward(self, observations, history_actions, valid_mask, history_action_valid_mask, q_action, q_action_valid_mask):
        hidden = self.history_encoder(observations, history_actions, valid_mask, history_action_valid_mask)
        action_embed = self.q_action_encoder(q_action, q_action_valid_mask)
        features = torch.cat([hidden, action_embed], dim=-1)
        return {"progress": self.progress_head(features), "value": self.value_head(features)}


class CPIQLRNNQNet(nn.Module):
    """Twin Q networks; each branch emits progress and all-data value heads."""

    def __init__(
        self,
        state_encoder: nn.Module,
        obs_horizon: int,
        action_dim: int,
        state_embed_dim: int = 512,
        action_embed_dim: int = 128,
        q_action_embed_dim: int = 128,
        lstm_input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        hidden_dims: Sequence[int] = (256,),
    ):
        super().__init__()
        def make_history_encoder():
            return CPIQLRNNHistoryEncoder(
                state_encoder=copy.deepcopy(state_encoder),
                obs_horizon=obs_horizon,
                action_dim=action_dim,
                state_embed_dim=state_embed_dim,
                action_embed_dim=action_embed_dim,
                lstm_input_dim=lstm_input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

        self.q1 = _CPIQLRNNQBranch(make_history_encoder(), action_dim, q_action_embed_dim, hidden_dims)
        self.q2 = _CPIQLRNNQBranch(make_history_encoder(), action_dim, q_action_embed_dim, hidden_dims)

    def forward(self, observations, history_actions, valid_mask, history_action_valid_mask, q_action, q_action_valid_mask):
        q1 = self.q1(observations, history_actions, valid_mask, history_action_valid_mask, q_action, q_action_valid_mask)
        q2 = self.q2(observations, history_actions, valid_mask, history_action_valid_mask, q_action, q_action_valid_mask)
        return {
            "q1_progress": q1["progress"],
            "q1_value": q1["value"],
            "q2_progress": q2["progress"],
            "q2_value": q2["value"],
        }
