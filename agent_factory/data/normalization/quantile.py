import torch
from .base import ActionNormalizer

class QuantileNormalizer(ActionNormalizer):
    """
    分位数动作归一化 (Robust Scaling)。
    利用 [0.05, 0.95] 分位数将高密度数据映射到 [-1, 1]。
    """
    def __init__(self, action_dim: int, q_low: float = 0.05, q_high: float = 0.95):
        super().__init__()
        self.q_low = q_low
        self.q_high = q_high
        self.register_buffer("low_val", torch.full((action_dim,), -1.0))
        self.register_buffer("high_val", torch.full((action_dim,), 1.0))

    def fit(self, data: torch.Tensor):
        """ data: [N, D] """
        with torch.no_grad():
            # 计算各维度的分位数 [D]
            q_low = torch.quantile(data, self.q_low, dim=0)
            q_high = torch.quantile(data, self.q_high, dim=0)
            
            self.low_val.copy_(q_low)
            self.high_val.copy_(q_high)
            self.initialized.fill_(True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [B, T, D] or [B, D] """
        # map to [-1, 1]
        norm_x = 2 * (x - self.low_val) / (self.high_val - self.low_val + 1e-8) - 1
        return norm_x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [B, T, D] or [B, D] """
        # map back to original space
        return (x + 1) / 2 * (self.high_val - self.low_val) + self.low_val