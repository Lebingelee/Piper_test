import torch
from .base import ActionNormalizer

class MeanStdNormalizer(ActionNormalizer):
    """
    Mean-Std 动作归一化。
    将数据映射为均值为 0，标准差为 1 的分布。
    """
    def __init__(self, action_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros((action_dim,)))
        self.register_buffer("std", torch.ones((action_dim,)))

    def fit(self, data: torch.Tensor):
        """ data: [N, D] """
        with torch.no_grad():
            self.mean.copy_(data.mean(dim=0))
            self.std.copy_(data.std(dim=0))
            self.initialized.fill_(True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [B, T, D] or [B, D] """
        return (x - self.mean) / (self.std + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [B, T, D] or [B, D] """
        return x * self.std + self.mean