import torch
from ...data.normalization import get_action_normalizer

class ActionNormMixin:
    """
    动作归一化 Mixin。
    负责在 Mixin 层处理动作的归一化与反归一化，确保模型层只看到 [-1, 1] 范围内的数据。
    """
    def _init_action_normalizer(self):
        # 1. 从配置中获取归一化参数
        # 假设配置位于 cfg.actor.norm
        norm_cfg = getattr(self.cfg.actor, "norm", None)
        norm_type = getattr(norm_cfg, "type", None) if norm_cfg else None
        
        # 获取特定的算法参数 (例如 quantile 的 q_low/q_high)
        norm_params = {}
        if norm_cfg and hasattr(norm_cfg, "params") and norm_cfg.params:
            norm_params = norm_cfg.params
            
        action_dim = self.cfg.env.action_dim
        
        # 2. 通过工厂方法实例化 Normalizer
        self.action_normalizer = get_action_normalizer(
            norm_type=norm_type, 
            action_dim=action_dim, 
            **norm_params
        ).to(self.device)
        
        print(f"[ActionNormMixin] Initialized with type: {norm_type}")

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """训练阶段：原始动作 -> 归一化动作"""
        return self.action_normalizer.normalize(action)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """推理阶段：归一化动作 -> 原始动作"""
        return self.action_normalizer.denormalize(action)

    def fit_action_normalizer(self, raw_actions: torch.Tensor):
        """同步阶段：根据原始数据计算统计量"""
        self.action_normalizer.fit(raw_actions)
        print(f"[ActionNormMixin] Fit completed. Normalizer is now ready.")
