from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_factory.modules.encoders.visual_encoder import make_mlp


class TDQCMLP(nn.Module):
    """
    Independent per-step TDQC predictor.

    The model consumes fixed-width step features z_t and predicts a success
    logit per timestep. If input_dim <= 0, the first Linear layer is lazy so the
    feature width can be inferred from the first batch.
    """

    def __init__(
        self,
        input_dim: int = 0,
        hidden_dims: Sequence[int] = (256, 256),
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        hidden_dims = list(hidden_dims)
        if not hidden_dims:
            hidden_dims = [256]

        if self.input_dim > 0:
            self.net = make_mlp(self.input_dim, hidden_dims + [1], last_act=False)
        else:
            layers = [nn.LazyLinear(hidden_dims[0]), nn.ReLU()]
            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
                layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dims[-1], 1))
            self.net = nn.Sequential(*layers)

    def forward(self, step_features: torch.Tensor) -> torch.Tensor:
        if step_features.ndim != 3:
            raise ValueError(
                "TDQCMLP expects step_features with shape [B, T, D], "
                f"got {tuple(step_features.shape)}"
            )
        batch_size, seq_len, feat_dim = step_features.shape
        flat = step_features.reshape(batch_size * seq_len, feat_dim)
        logits = self.net(flat)
        return logits.reshape(batch_size, seq_len, 1)


class TDQCRNN(nn.Module):
    """
    Recurrent TDQC predictor over fixed-width step features.

    The RNN hidden state summarizes z_1:t and the head predicts one success
    logit per timestep. If input_dim <= 0, a lazy input projection infers the
    feature width from the first batch.
    """

    def __init__(
        self,
        input_dim: int = 0,
        rnn_type: str = "lstm",
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_dims: Sequence[int] = (256,),
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.rnn_type = str(rnn_type or "lstm").strip().lower()
        if self.rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"TDQCRNN supports rnn_type='lstm' or 'gru', got {rnn_type!r}.")
        if self.hidden_dim <= 0:
            raise ValueError("TDQCRNN hidden_dim must be positive.")
        if self.num_layers <= 0:
            raise ValueError("TDQCRNN num_layers must be positive.")

        self.input_projection = (
            nn.Linear(self.input_dim, self.hidden_dim)
            if self.input_dim > 0
            else nn.LazyLinear(self.hidden_dim)
        )
        rnn_dropout = float(dropout) if self.num_layers > 1 else 0.0
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        head_hidden_dims = list(head_hidden_dims)
        if head_hidden_dims:
            self.head = make_mlp(self.hidden_dim, head_hidden_dims + [1], last_act=False)
        else:
            self.head = nn.Linear(self.hidden_dim, 1)

    @staticmethod
    def _lengths_from_mask(valid_mask: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if valid_mask is None:
            return torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
        if valid_mask.ndim == 3 and valid_mask.shape[-1] == 1:
            valid_mask = valid_mask.squeeze(-1)
        if valid_mask.ndim != 2:
            raise ValueError(f"TDQCRNN valid_mask must have shape [B,T], got {tuple(valid_mask.shape)}")
        lengths = valid_mask.to(device=device).float().gt(0).long().sum(dim=1)
        if torch.any(lengths <= 0):
            raise ValueError("TDQCRNN received a trajectory with no valid timesteps.")
        return lengths

    def forward(self, step_features: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        if step_features.ndim != 3:
            raise ValueError(
                "TDQCRNN expects step_features with shape [B, T, D], "
                f"got {tuple(step_features.shape)}"
            )
        batch_size, seq_len, _feat_dim = step_features.shape
        projected = torch.relu(self.input_projection(step_features))
        lengths = self._lengths_from_mask(valid_mask, batch_size, seq_len, step_features.device)

        packed = nn.utils.rnn.pack_padded_sequence(
            projected,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _hidden = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=seq_len,
        )
        flat = rnn_out.reshape(batch_size * seq_len, self.hidden_dim)
        logits = self.head(flat)
        return logits.reshape(batch_size, seq_len, 1)


def _as_btd(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 3 and value.shape[-1] == 1:
        return value
    if value.ndim == 2:
        return value.unsqueeze(-1)
    if value.ndim == 1:
        return value.reshape(-1, 1, 1)
    raise ValueError(f"{name} must have shape [B,T], [B,T,1], or [B], got {tuple(value.shape)}")


def _last_valid_indices(valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.bool()
    lengths = valid.long().sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("TDQC batch contains a trajectory with no valid timesteps.")
    return lengths - 1


def masked_td0_loss(
    success_prob: torch.Tensor,
    target_success_prob: torch.Tensor,
    valid_mask: torch.Tensor,
    success_label: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    TD-0 loss over padded rollout batches.

    Non-terminal valid transitions regress p_t toward target p_{t+1}. The last
    valid timestep of each trajectory regresses toward the episode success label.
    """

    pred = _as_btd(success_prob, "success_prob").squeeze(-1)
    target = _as_btd(target_success_prob, "target_success_prob").squeeze(-1).detach()
    valid = _as_btd(valid_mask.float(), "valid_mask").squeeze(-1) > 0
    labels = success_label.float().to(pred.device).reshape(-1)
    if pred.shape != target.shape or pred.shape != valid.shape:
        raise ValueError(
            "TDQC TD-0 tensors must share [B,T] shape: "
            f"pred={tuple(pred.shape)}, target={tuple(target.shape)}, valid={tuple(valid.shape)}"
        )
    if labels.shape[0] != pred.shape[0]:
        raise ValueError(f"success_label batch size {labels.shape[0]} does not match predictions {pred.shape[0]}")

    if pred.shape[1] > 1:
        transition_mask = valid[:, :-1] & valid[:, 1:]
        transition_error = (pred[:, :-1] - target[:, 1:]).pow(2)
        transition_denom = transition_mask.float().sum().clamp_min(1.0)
        loss_transition = (transition_error * transition_mask.float()).sum() / transition_denom
    else:
        loss_transition = pred.sum() * 0.0
        transition_mask = valid[:, :0]

    batch_indices = torch.arange(pred.shape[0], device=pred.device)
    last_indices = _last_valid_indices(valid)
    terminal_pred = pred[batch_indices, last_indices]
    loss_terminal = (terminal_pred - labels).pow(2).mean()
    loss = loss_transition + loss_terminal

    return {
        "loss": loss,
        "loss_td_transition": loss_transition,
        "loss_td_terminal": loss_terminal,
        "td_transition_count": transition_mask.float().sum().detach(),
        "td_terminal_count": torch.tensor(float(pred.shape[0]), device=pred.device),
    }


def masked_bce_loss(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    success_label: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pred_logits = _as_btd(logits, "logits").squeeze(-1)
    valid = _as_btd(valid_mask.float(), "valid_mask").squeeze(-1)
    labels = success_label.float().to(pred_logits.device).reshape(-1, 1).expand_as(pred_logits)
    loss_map = F.binary_cross_entropy_with_logits(pred_logits, labels, reduction="none")
    denom = valid.sum().clamp_min(1.0)
    loss = (loss_map * valid).sum() / denom
    return {
        "loss": loss,
        "loss_bce": loss,
        "bce_valid_count": valid.sum().detach(),
    }
