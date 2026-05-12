import torch
import numpy as np
import h5py
import json
import os
from typing import Optional, List, Dict, Any, Union
from torch.utils.data.dataset import Dataset

# ==================== 0. 基础工具逻辑 ====================

def compute_pad_vec(act_seq: torch.Tensor, is_abs_mode: bool, pad_action_arm: Optional[torch.Tensor] = None):
    """统一计算动作补帧向量"""
    if is_abs_mode:
        return act_seq[-1]
    else:
        # 增量模式：臂部补零，夹爪保持
        if pad_action_arm is not None:
            return torch.cat([pad_action_arm, act_seq[-1, -1:]])
        return act_seq[-1]

from .utils import compute_rl_signals, compute_n_step_signals

# ==================== 1. ExpertDataset (内存缓存加速版) ====================

class ExpertDataset(Dataset):
    """
    专家数据集类：支持全量数据载入内存以消除 IO 瓶颈。
    针对双臂多相机高维数据进行极致优化。
    """
    def __init__(self, 
                 cfg,
                 obs_space=None,  
                 device="cpu", 
                 required_keys=None):
        
        self.cfg = cfg
        self.obs_horizon = cfg.env.obs_horizon
        self.pred_horizon = cfg.env.pred_horizon
        self.act_horizon = cfg.env.act_horizon
        
        # 1. 探测文件
        self.demo_path = cfg.dataset.expert.demo_path
        if not os.path.exists(self.demo_path):
             raise FileNotFoundError(f"Expert dataset not found: {self.demo_path}")

        # 🟢 缓存占位
        self.rgb_data = []   # List of np.ndarray
        self.state_data = [] # List of np.ndarray
        self.actions = []
        self.terminated = []
        self.rewards = []
        self.values = []
        
        # 2. 一次性载入所有数据到内存
        print(f"[ExpertDataset] Loading ALL data from {self.demo_path} to RAM...")
        with h5py.File(self.demo_path, 'r') as f:
            # 获取轨迹 Key
            all_keys = sorted([k for k in f.keys() if k.startswith('traj_')], 
                             key=lambda x: int(x.split('_')[-1]) if '_' in x else x)
            self.traj_keys = all_keys[:cfg.dataset.expert.num_traj] if cfg.dataset.expert.num_traj else all_keys
            
            self.slices_all = []
            self.slices_success = []
            global_count = 0
            
            from tqdm import tqdm
            for i, tk in enumerate(tqdm(self.traj_keys, desc="Caching RAM")):
                g = f[tk]
                
                # 🟢 缓存视觉与状态 (这是 IO 密集的点，提前载入)
                self.rgb_data.append(g["obs"]["rgb"][()]) 
                self.state_data.append(g["obs"]["state"][()].astype(np.float32))
                
                # 动作与信号；新 schema 使用 action，兼容旧 actions 输入。
                action_key = "action" if "action" in g else "actions" if "actions" in g else None
                if action_key is None:
                    raise KeyError(f"Missing action/actions dataset in trajectory {tk}.")
                act = torch.from_numpy(g[action_key][()]).float()
                L = len(act)
                self.actions.append(act)
                
                if "terminated" in g:
                    term = g["terminated"][()]
                else:
                    term = np.zeros(L, dtype=bool)
                    if bool(g.attrs.get("success", False)) and L > 0:
                        term[-1] = True
                self.terminated.append(term)
                
                # 动态计算 RL 信号
                rew, val = compute_rl_signals(
                    success_array=term,
                    gamma=cfg.env.gamma,
                    penalty=cfg.env.penalty,
                    reward_mode=cfg.env.reward_mode,
                    reward_type='b',
                    reward_shape=cfg.env.reward_shape
                )
                self.rewards.append(rew)
                self.values.append(val)
                
                # 预计算切片
                is_suc = np.any(term)
                pad_before = self.obs_horizon - 1
                for start in range(-pad_before, L - self.act_horizon + 1):
                    sl = (i, start, start + self.pred_horizon, global_count)
                    self.slices_all.append(sl)
                    if is_suc: self.slices_success.append(sl)
                    global_count += 1

        self.slices = self.slices_all
        self.mode = 'all'
        
        # 3. 探测最终可用字段
        self.available_keys = {"observations", "action", "terminated", "reward", "value", "cond", "discount"}
        if required_keys is None:
            self.required_keys = self.available_keys | {"next_observations"}
        else:
            self.required_keys = set(required_keys)
            if "actions" in self.required_keys:
                self.required_keys.remove("actions")
                self.required_keys.add("action")
        
        # Cond 处理
        if "cond" in self.required_keys:
            self.conds = torch.zeros(len(self.slices_all), dtype=torch.float32)
        else:
            self.conds = None
        
        # 控制模式逻辑
        self.control_mode = cfg.env.control_mode
        self.is_abs_mode = ("pos" in self.control_mode) and ("delta" not in self.control_mode)
        self.pad_action_arm = torch.zeros((self.actions[0].shape[1] - 1,)) if not self.is_abs_mode else None
        
        print(f"[ExpertDataset] RAM Caching completed. Trajectories: {len(self.traj_keys)}, Slices: {len(self.slices_all)}")

    def switch(self, mode='all'):
        self.mode = mode
        self.slices = self.slices_all if mode == 'all' else self.slices_success

    def get_all_actions(self):
        return torch.cat(self.actions, dim=0)

    def _get_obs_sequence(self, traj_idx, start_idx, horizon):
        """
        🟢 极致优化：内存切片。不再有 HDF5 IO。
        """
        rgb_traj = self.rgb_data[traj_idx]
        state_traj = self.state_data[traj_idx]
        L_obs = len(rgb_traj)
        
        # 计算索引序列 (处理补帧)
        indices = [max(0, min(start_idx - (horizon - 1) + i, L_obs - 1)) for i in range(horizon)]
        
        # 直接通过内存高级索引获取
        # 对于连续切片，NumPy 优化极佳
        rgb_seq = rgb_traj[indices]
        state_seq = state_traj[indices]
            
        return {
            "rgb": torch.from_numpy(rgb_seq),
            "state": torch.from_numpy(state_seq)
        }

    def __getitem__(self, index):
        traj_idx, start, end, global_idx = self.slices[index]
        L = self.actions[traj_idx].shape[0]
        
        data = {}
        # 1. Observations
        if "observations" in self.required_keys:
            data["observations"] = self._get_obs_sequence(traj_idx, start, self.obs_horizon)
        
        # 2. Next Observations
        if "next_observations" in self.required_keys:
            data["next_observations"] = self._get_obs_sequence(traj_idx, min(start + self.act_horizon, L), self.obs_horizon)

        # 3. Action
        if "action" in self.required_keys:
            act_seq = self.actions[traj_idx][max(0, start) : end]
            if start < 0: act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
            if len(act_seq) < self.pred_horizon:
                pad_vec = compute_pad_vec(act_seq, self.is_abs_mode, self.pad_action_arm)
                act_seq = torch.cat([act_seq, pad_vec.unsqueeze(0).repeat(self.pred_horizon - len(act_seq), 1)], dim=0)
            data["action"] = act_seq

        # 4. RL Signals
        idx = min(max(0, start), L - 1)
        
        # 处理 n-step 信号 (用于 DSRL 等)
        n_step_reward, effective_discount = compute_n_step_signals(
            self.rewards[traj_idx], idx, self.act_horizon, self.cfg.env.gamma
        )

        if "reward" in self.required_keys:
            data["reward"] = n_step_reward.unsqueeze(0)
        if "terminated" in self.required_keys:
            # 取 n-step 后的终止状态
            term_idx = min(idx + self.act_horizon, L - 1)
            data["terminated"] = torch.tensor([self.terminated[traj_idx][term_idx]], dtype=torch.float32)
        if "value" in self.required_keys:
            data["value"] = torch.tensor([self.values[traj_idx][idx]], dtype=torch.float32)
        if "discount" in self.required_keys:
            data["discount"] = torch.tensor([effective_discount], dtype=torch.float32)
        if "cond" in self.required_keys and self.conds is not None:
            data["cond"] = self.conds[global_idx].reshape(1)

        return data

    def __len__(self): return len(self.slices)
