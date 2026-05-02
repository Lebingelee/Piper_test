import torch
from .base import ActionNormalizer

class MinMaxNormalizer(ActionNormalizer):
    """
    Min-Max 动作归一化。
    将原始范围映射到 [-1, 1]。
    """
    def __init__(self, action_dim: int):
        super().__init__()
        self.register_buffer("min_val", torch.full((action_dim,), float('inf')))
        self.register_buffer("max_val", torch.full((action_dim,), float('-inf')))

    def fit(self, data: torch.Tensor):
        """ data: [N, D] """
        with torch.no_grad():
            self.min_val.copy_(data.min(dim=0).values)
            self.max_val.copy_(data.max(dim=0).values)
            self.initialized.fill_(True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [B, T, D] or [B, D] """
        # map to [-1, 1]
        norm_x = 2 * (x - self.min_val) / (self.max_val - self.min_val + 1e-8) - 1
        return norm_x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [B, T, D] or [B, D] """
        # map back to [min, max]
        return (x + 1) / 2 * (self.max_val - self.min_val) + self.min_val