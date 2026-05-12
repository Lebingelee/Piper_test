import numpy as np
import torch
from typing import Dict, Any

def compute_rl_signals(
    success_array: np.ndarray, 
    gamma: float, 
    penalty: float, 
    reward_mode: str = "sparse", 
    reward_type: str = "b", # 'a' for 0/1, 'b' for -1/0/penalty
    reward_shape: bool = False
):
    """
    统一计算 Reward 和 Value。
    Args:
        success_array: bool 数组，表示每一步是否达到成功状态
        gamma: 折扣因子
        penalty: 失败惩罚 (针对模式 b)
        reward_mode: 奖励模式，目前支持 "sparse"
        reward_type: 
            'a': 步骤奖励 0，成功瞬间 1 (Dense 0, Final 1)
            'b': 步骤奖励 -1，成功瞬间 0，最终失败惩罚 penalty
        reward_shape: 是否开启奖励塑形
    """
    T = len(success_array)
    rewards = np.zeros(T, dtype=np.float32)
    values = np.zeros(T, dtype=np.float32)
    
    # 1. 计算 Rewards
    if reward_mode == "sparse":
        if reward_type == 'a':
            # 模式 a: 0/1 (典型扩散策略或行为克隆常用)
            for t in range(T):
                rewards[t] = 1.0 if success_array[t] else 0.0
        else:
            # 模式 b: -1/0/penalty (典型强化学习 IQL/QL 常用)
            # 假设轨迹结束且未成功则为失败
            is_failed = not np.any(success_array)
            for t in range(T):
                if success_array[t]:
                    rewards[t] = 0.0
                elif t == T - 1 and is_failed:
                    rewards[t] = penalty
                else:
                    rewards[t] = -1.0
                    
    # 2. 计算 Values (从后往前迭代)
    running_return = 0.0
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        values[t] = running_return
        
        # 如果开启了 reward_shape 且该轨迹是失败的，可以对 Value 做特殊截断 (参考 IQL 原文)
        if reward_shape and not np.any(success_array):
            values[t] = min(values[t], penalty)
            
    return rewards, values


def compute_n_step_signals(rewards, start_idx, n, gamma):
    """
    计算 n-step 的折扣奖励总和及折扣因子。
    Args:
        rewards: 1-step 奖励数组 (Tensor or ndarray)
        start_idx: 起始索引
        n: 步数 (act_horizon)
        gamma: 折扣因子
    Returns:
        n_step_reward: 折扣和
        discount: 最终折扣因子 (gamma^n)
    """
    T = len(rewards)
    end_idx = min(start_idx + n, T)
    
    # 提取奖励片段
    chunk = rewards[start_idx:end_idx]
    if not isinstance(chunk, torch.Tensor):
        chunk = torch.from_numpy(chunk).float()
    
    # 计算折扣和
    if gamma < 1.0 - 1e-6:
        discounts = torch.pow(gamma, torch.arange(len(chunk)).float())
        n_step_reward = torch.sum(chunk * discounts)
        effective_discount = gamma ** n
    else:
        n_step_reward = torch.sum(chunk)
        effective_discount = 1.0
        
    return n_step_reward, effective_discount


def preprocess_obs(obs_dict: Dict[str, Any], device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
    """
    统一的观测预处理逻辑：
    1. 图像 (rgb/depth) 分别归一化。
    2. 类型转换 (float32)。
    3. 保留其他字段 (state, cond 等) 并搬运到指定设备。
    4. depth 保持独立模态，不隐式拼接进 rgb。
    """
    out = {}
    
    # 辅助函数：递归处理 Tensor 搬运和转换
    def to_torch(x):
        if isinstance(x, torch.Tensor):
            return x.float().to(device)
        return torch.from_numpy(np.array(x)).float().to(device)

    # 1. 处理 RGB
    if 'rgb' in obs_dict:
        rgb = obs_dict['rgb']
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.from_numpy(rgb)
        
        # 仅对 uint8 进行 0-1 归一化 (防止重复归一化)
        if rgb.dtype == torch.uint8:
            out['rgb'] = rgb.float().to(device) / 255.0
        else:
            out['rgb'] = rgb.float().to(device)

    # 2. 处理 Depth
    if 'depth' in obs_dict:
        depth = to_torch(obs_dict['depth'])
        # 归一化 (根据传感器量程，这里暂定 1024.0)
        # 如果已经是 float，假设已经归一化过
        if not isinstance(obs_dict['depth'], torch.Tensor) or obs_dict['depth'].dtype != torch.float32:
             depth = depth / 1024.0
        out['depth'] = depth
    
    # 3. 处理其他所有字段 (state, cond, action 等)
    for k, v in obs_dict.items():
        if k not in ['rgb', 'depth']:
            out[k] = to_torch(v)
                
    return out
