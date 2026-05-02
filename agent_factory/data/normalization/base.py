import torch
import torch.nn as nn

class ActionNormalizer(nn.Module):
    """
    动作归一化基类 (Identity)。
    默认不执行任何操作，作为兜底和接口规范。
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("initialized", torch.tensor(False))

    def fit(self, data: torch.Tensor):
        """根据原始数据计算统计量"""
        # 基类不计算任何统计量，直接标记为已初始化
        self.initialized.fill_(True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """原始空间 -> 归一化空间 (Identity)"""
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """归一化空间 -> 原始空间 (Identity)"""
        return x

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return super().load_state_dict(state_dict, *args, **kwargs)