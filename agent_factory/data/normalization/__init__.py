from .base import ActionNormalizer
from .min_max import MinMaxNormalizer
from .mean_std import MeanStdNormalizer
from .quantile import QuantileNormalizer

def get_action_normalizer(norm_type: str, action_dim: int, **kwargs) -> ActionNormalizer:
    """
    归一化工厂方法。
    
    Args:
        norm_type: 'min_max', 'mean_std', 'quantile', or None
        action_dim: 动作维度
        **kwargs: 算法特定参数 (如 quantile 的 q_low/q_high)
    """
    if norm_type == "min_max":
        return MinMaxNormalizer(action_dim)
    elif norm_type == "mean_std":
        return MeanStdNormalizer(action_dim)
    elif norm_type == "quantile":
        return QuantileNormalizer(action_dim, **kwargs)
    else:
        # 兜底：Identity (None or unknown)
        if norm_type is not None:
            print(f"[Normalizer] Warning: Unknown norm_type '{norm_type}', using Identity.")
        return ActionNormalizer()