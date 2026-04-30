import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True,
             use_layernorm: bool = True # 新增：控制是否使用 LayerNorm
             ):
    """构建多层感知机 (Helper Function)"""
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            if use_layernorm:
                module_list.append(nn.LayerNorm(c_out))
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

class VisualEncoder(nn.Module):
    """
    模块化视觉编码器
    - 职责：仅负责将图像张量编码为特征向量
    - 输入：[B, C, H, W] (不处理时间维度 T，由 StateEncoder 处理)
    - 输出：[B, out_dim]
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 256,
        backbone_type: str = "plain", # 'plain' or 'resnet'
        pool_feature_map: bool = True,
        use_group_norm: bool = True, # 新增：控制是否使用 GroupNorm (Diffusion 常用)
    ):
        super().__init__()
        self.out_dim = out_dim
        self.backbone_type = backbone_type

        # --- Backbone 构建 ---
        if backbone_type == "plain":
            # 经典的 PlainConv (适合低分辨率 84x84 or 96x96)
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1),
                nn.GroupNorm(8, 16) if use_group_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # /2

                nn.Conv2d(16, 32, 3, padding=1),
                nn.GroupNorm(16, 32) if use_group_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # /4

                nn.Conv2d(32, 64, 3, padding=1),
                nn.GroupNorm(32, 64) if use_group_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # /8

                nn.Conv2d(64, 128, 3, padding=1),
                nn.GroupNorm(32, 128) if use_group_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # /16
                
                nn.Conv2d(128, 128, 1, padding=0),
                nn.GroupNorm(32, 128) if use_group_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            )
            # 计算全连接层输入维度
            # 假设输入 96x96 -> /16 -> 6x6
            # 如果使用 AdaptivePool，则固定为 1x1
            feature_map_size = 128 if pool_feature_map else 128 * 6 * 6 

        elif backbone_type == "resnet":
            # ResNet18 Backbone (适合更高分辨率)
            resnet = models.resnet18(pretrained=True)
            
            # 修改第一层以适应非3通道输入 (如 RGB-D)
            if in_channels != 3:
                original_conv1 = resnet.conv1
                resnet.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                # 可选：初始化新通道权重
                with torch.no_grad():
                    # 复制原有前3通道权重，其余初始化
                    resnet.conv1.weight[:, :3] = original_conv1.weight
            
            # 去掉最后两层 (AvgPool, FC)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            feature_map_size = 512 # ResNet18 output channels

        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # --- Head 构建 ---
        if pool_feature_map:
            self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
            self.flatten_dim = feature_map_size
        else:
            self.pool = nn.Identity()
            self.flatten_dim = feature_map_size # 需确保输入尺寸固定

        self.fc = nn.Linear(self.flatten_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim) # 输出加个 LN 通常有助于 RL 稳定

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        return: [B, out_dim]
        """
        x = self.backbone(x) # [B, Feat_C, H', W']
        x = self.pool(x)     # [B, Feat_C, 1, 1] or NoOp
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.ln(x)
        return x