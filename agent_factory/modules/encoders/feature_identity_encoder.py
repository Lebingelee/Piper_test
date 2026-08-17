from typing import Any, Dict

import torch
import torch.nn as nn

from agent_factory.modules.registry import module_cfg_get, register_module


class FeatureIdentityEncoder(nn.Module):
    """
    Pass through an already-extracted observation feature.

    VLA feature datasets expose observations["feature"] as [B, T, D]. This
    encoder keeps that contract intact so CPIQL/TDQC and risk score models can
    replace image/state encoders without adding model-specific adapters.
    """

    def __init__(self, obs_key: str = "feature", out_dim: int = 0):
        super().__init__()
        self.obs_key = str(obs_key or "feature")
        self.out_dim = int(out_dim or 0)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.obs_key not in obs_dict:
            available = sorted(obs_dict.keys())
            raise KeyError(
                f"FeatureIdentityEncoder expected observations[{self.obs_key!r}], "
                f"available keys: {available}"
            )
        feature = obs_dict[self.obs_key].float()
        if feature.ndim == 2:
            feature = feature.unsqueeze(1)
        if feature.ndim != 3:
            raise ValueError(
                f"FeatureIdentityEncoder expects [B,T,D] or [B,D], got {tuple(feature.shape)}"
            )
        if self.out_dim <= 0:
            self.out_dim = int(feature.shape[-1])
        elif int(feature.shape[-1]) != self.out_dim:
            raise ValueError(
                f"FeatureIdentityEncoder out_dim={self.out_dim} but input feature dim={feature.shape[-1]}"
            )
        return feature


def _build_feature_identity_encoder(encoder_cfg: Any, **_kwargs) -> FeatureIdentityEncoder:
    return FeatureIdentityEncoder(
        obs_key=str(module_cfg_get(encoder_cfg, "obs_key", "feature")),
        out_dim=int(module_cfg_get(encoder_cfg, "out_dim", 0) or 0),
    )


register_module("feature_identity_encoder")(_build_feature_identity_encoder)
register_module("identity_feature_encoder")(_build_feature_identity_encoder)
register_module("identity_encoder")(_build_feature_identity_encoder)

