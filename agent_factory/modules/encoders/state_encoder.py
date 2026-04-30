import torch
import torch.nn as nn
from einops import rearrange
from typing import Dict, Optional, Tuple

# 引用上面的 VisualEncoder 和 make_mlp
# 在实际项目中建议使用: from .visual_encoder import VisualEncoder, make_mlp
from agent_factory.modules.encoders.visual_encoder import VisualEncoder, make_mlp

class BaseStateEncoder(nn.Module):
    """
    统一状态编码器 (State Fusion Layer)
    职责：
    1. 处理输入的维度折叠 (Batch, Time, Multi-view-> Batch * Time * Multi-view)
    2. 调用 VisualEncoder 提取图像特征
    3. 融合 视觉特征 与 本体感觉(proprioception)
    4. 投影到统一的 Embedding 空间
    
    设计理念：
    Actor 和 Critic 均持有自己的 BaseStateEncoder 实例 (或共享其中一部分)。
    """
    def __init__(
        self,
        visual_encoder: Optional[VisualEncoder], # 允许传入共享的 Encoder 实例
        proprio_dim: int,
        out_dim: int = 256,
        hidden_dims: list = [256],
        visual_feature_dim: int = 256, # VisualEncoder 的输出维度
        num_cameras: int = 1,
        view_fusion: str = 'concat' # 'concat' or 'mean'
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.proprio_dim = proprio_dim
        self.out_dim = out_dim
        self.num_cameras = num_cameras
        self.view_fusion = view_fusion
        # 计算融合后的输入维度
        self.fusion_input_dim = 0
        
        if self.visual_encoder is not None:
            self.fusion_input_dim += visual_feature_dim*num_cameras if view_fusion == 'concat' else visual_feature_dim
            
        if self.proprio_dim > 0:
            self.fusion_input_dim += proprio_dim# out
            
        # 融合投影层 (MLP)
        # 作用：将 [Vis_Feat, Proprio] -> [Embedding]
        if self.fusion_input_dim > 0:
            self.projector = make_mlp(
                in_channels=self.fusion_input_dim,
                mlp_channels=hidden_dims + [out_dim],
                last_act=False # 输出 Embedding 通常不激活，或者是 Tanh/LayerNorm
            )
        else:
            self.projector = nn.Identity() # 极端情况：无输入

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        b, t = None, None
        features_list = []

        # 1. 处理视觉特征 (多视角拆分提取)
        if self.visual_encoder is not None and 'rgb' in obs_dict:
            img = obs_dict['rgb'] # [B, T, 3*k, H, W]
            b, t, c, h, w = img.shape
            
            # --- 核心优化：拆分多视角 ---
            # 将 3*k 通道拆开，把 k 压入 batch 维度处理
            # [B, T, 3*k, H, W] -> [(B * T * k), 3, H, W]
            img_multi_view = rearrange(img, 'b t (k c) h w -> (b t k) c h w', k=self.num_cameras, c=3)
            
            # 共享参数提取特征: [(B * T * k), vis_dim]
            vis_feat_raw = self.visual_encoder(img_multi_view)
            
            # 还原视角维度: [B*T, k, vis_dim]
            vis_feat_reshaped = rearrange(vis_feat_raw, '(bt k) d -> bt k d', k=self.num_cameras)
            
            # 视角融合 (View Fusion)
            if self.view_fusion == 'concat':
                # [B*T, k * vis_dim]
                vis_feat = rearrange(vis_feat_reshaped, 'bt k d -> bt (k d)')
            else:
                # [B*T, vis_dim]
                vis_feat = vis_feat_reshaped.mean(dim=1)
                
            features_list.append(vis_feat)

        # 2. 处理本体感觉
        if self.proprio_dim > 0 and 'state' in obs_dict:
            state = obs_dict['state']
            if b is None: b, t = state.shape[:2]
            features_list.append(rearrange(state, 'b t d -> (b t) d'))

        # 3. 拼接与投影
        if not features_list:
            raise ValueError("No valid features.")
             
        combined_feat = torch.cat(features_list, dim=-1) # [B*T, Total_Dim]
        embedding_flat = self.projector(combined_feat)
        
        # 恢复维度 [B, T, out_dim]
        return rearrange(embedding_flat, '(b t) d -> b t d', b=b, t=t)