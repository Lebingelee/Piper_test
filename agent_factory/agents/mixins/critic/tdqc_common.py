from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from agent_factory.modules.registry import make_module


@dataclass
class TDQCMLPConfig:
    type: str = "tdqc_mlp"
    input_dim: int = 0
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    lr: float = 3e-4
    loss_type: str = "td0"
    target_update_interval: int = 100
    grad_clip_norm: float = 0.0
    log_interval: int = 100


@dataclass
class TDQCRNNConfig:
    type: str = "tdqc_rnn"
    input_dim: int = 0
    rnn_type: str = "lstm"
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.0
    head_hidden_dims: List[int] = field(default_factory=lambda: [256])
    lr: float = 3e-4
    loss_type: str = "td0"
    target_update_interval: int = 100
    grad_clip_norm: float = 1.0
    log_interval: int = 100


class TDQCBatchMixin:
    REQUIRED_KEYS = {"step_features", "valid_mask", "success_label", "frame"}

    def _build_tdqc_encoder_override(self) -> None:
        agent_sp = getattr(self.cfg, "agent_sp", None)
        encoder_cfg = getattr(agent_sp, "encoder_override", None) if agent_sp is not None else None
        self.tdqc_encoder = make_module(encoder_cfg).to(self.device) if encoder_cfg not in (None, False) else None

    def _as_tdqc_tensor(self, batch: Dict[str, Any], key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"TDQC batch requires '{key}'.")
        return batch[key].to(self.device, dtype=dtype)

    def _prepare_tdqc_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        step_features = self.encode_batch(batch)
        if "valid_mask" in batch:
            valid_mask = self._as_tdqc_tensor(batch, "valid_mask")
        else:
            valid_mask = torch.ones(step_features.shape[:2], device=self.device, dtype=torch.float32)
        if "success_label" in batch:
            success_label = self._as_tdqc_tensor(batch, "success_label")
        elif "is_success_segment" in batch:
            success_label = self._as_tdqc_tensor(batch, "is_success_segment")
        else:
            raise KeyError("TDQC batch requires 'success_label' or CPIQL 'is_success_segment'.")
        if valid_mask.ndim == 3 and valid_mask.shape[-1] == 1:
            valid_mask = valid_mask.squeeze(-1)
        if success_label.ndim > 1:
            success_label = success_label.reshape(success_label.shape[0], -1)[:, 0]
        return {
            "step_features": step_features,
            "valid_mask": valid_mask,
            "success_label": success_label,
        }

    def encode_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        if "step_features" in batch:
            step_features = batch["step_features"].to(self.device, dtype=torch.float32)
            if step_features.ndim != 3:
                raise ValueError(
                    "TDQC step_features must have shape [B,T,D], "
                    f"got {tuple(step_features.shape)}"
                )
            return step_features
        if "processed_raw_state" in batch:
            step_features = batch["processed_raw_state"].to(self.device, dtype=torch.float32)
            if step_features.ndim == 2:
                step_features = step_features.unsqueeze(1)
            if step_features.ndim != 3:
                raise ValueError(
                    "TDQC processed_raw_state must have shape [B,D] or [B,T,D], "
                    f"got {tuple(step_features.shape)}"
                )
            return step_features
        if "observations" in batch:
            if getattr(self, "tdqc_encoder", None) is None:
                raise NotImplementedError(
                    "TDQC observation batches require agent_sp.encoder_override or "
                    "agent_sp.encoder_config_path. Cached 'step_features' batches do not need an encoder."
                )
            observations = batch["observations"]
            if hasattr(self, "_preprocess_obs"):
                observations = self._preprocess_obs(observations)
            embeddings = self.tdqc_encoder(observations)
            if embeddings.ndim != 3:
                raise ValueError(
                    "TDQC encoder_override must return step features with shape [B,T,D], "
                    f"got {tuple(embeddings.shape)}"
                )
            return embeddings
        raise KeyError("TDQC encode_batch requires either 'step_features' or 'observations'.")

    encoder_batch = encode_batch

    @staticmethod
    def _last_valid_indices(valid_mask: torch.Tensor) -> torch.Tensor:
        valid = valid_mask.bool()
        lengths = valid.long().sum(dim=1)
        if torch.any(lengths <= 0):
            raise ValueError("TDQC batch contains a trajectory with no valid timesteps.")
        return lengths - 1

    def _select_eval_timestep(self, values: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if values.ndim == 1:
            return values
        if values.ndim != 2:
            raise ValueError(f"Expected TDQC values with shape [B,T] or [B,T,1], got {tuple(values.shape)}")
        if valid_mask is None:
            return values[:, -1]
        mask = valid_mask.to(values.device)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        last_indices = self._last_valid_indices(mask)
        batch_indices = torch.arange(values.shape[0], device=values.device)
        return values[batch_indices, last_indices]
