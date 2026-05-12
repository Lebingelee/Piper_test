import torch
import numpy as np
import h5py
import os
import glob
import json
from torch.utils.data.dataset import Dataset
from typing import Dict, List, Optional, Any
from agent_factory.control import (
    build_action_transform_meta,
    canonicalize_control_mode,
    extract_arm_pose_map_from_flat_state,
    inverse_transform_action,
    is_absolute_mode,
)
from agent_factory.data.base import BaseTrajectoryDataset
from agent_factory.data.utils import compute_rl_signals

# 引用基础工具
def compute_pad_vec(act_seq: torch.Tensor, is_abs_mode: bool, pad_action_arm: Optional[torch.Tensor] = None):
    if is_abs_mode: return act_seq[-1]
    if pad_action_arm is not None: return torch.cat([pad_action_arm, act_seq[-1, -1:]])
    return act_seq[-1]


def _action_dataset_key(h5_file: h5py.File) -> str:
    if "action" in h5_file and isinstance(h5_file["action"], h5py.Dataset):
        return "action"
    if "actions" in h5_file and isinstance(h5_file["actions"], h5py.Dataset):
        return "actions"
    raise KeyError("Replay H5 requires flat 'action' dataset.")


def _read_json_dataset(dataset) -> Dict[str, Any]:
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return json.loads(value)


def _load_env_meta(h5_file: h5py.File, traj_group: h5py.Group) -> Dict[str, Any]:
    if "meta" in h5_file and "env_meta" in h5_file["meta"]:
        return _read_json_dataset(h5_file["meta"]["env_meta"])
    if "meta" in traj_group and isinstance(traj_group["meta"], h5py.Group) and "env_meta" in traj_group["meta"]:
        return _read_json_dataset(traj_group["meta"]["env_meta"])
    return {}


def _normalize_env_meta(cfg, env_meta: Dict[str, Any]) -> Dict[str, Any]:
    env_meta = dict(env_meta or {})
    env_meta.setdefault("obs", {})
    env_meta.setdefault("action", {})
    env_meta["env_control_mode"] = canonicalize_control_mode(
        env_meta.get("env_control_mode") or getattr(cfg.env, "env_control_mode", getattr(cfg.env, "control_mode", "delta_pose"))
    )
    env_meta.setdefault("controller_backend", getattr(cfg.env, "controller_backend", ""))
    env_meta.setdefault("control", {})
    return env_meta

# ==================== 1. FileReplayBuffer (物理文件视图) ====================

class FileReplayBuffer(BaseTrajectoryDataset):
    """
    文件型回放缓冲区：将文件夹下的所有单轨迹 H5 视为一个逻辑整体。
    职责：
    1. 动态扫描文件夹。
    2. 基于 EnvConfig 动态计算 Reward 和 Value。
    """
    def __init__(self, cfg, folder_path, required_keys=None):
        self.cfg = cfg
        self.obs_horizon = cfg.env.obs_horizon
        self.pred_horizon = cfg.env.pred_horizon
        self.act_horizon = cfg.env.act_horizon
        super().__init__(folder_path=folder_path)
        
        # 1. 扫描并建立物理索引
        self.file_paths = sorted({ref.file_path for ref in self.trajectory_refs})
        self.trajs_info = []
        self.slices_all, self.slices_success = [], []
        
        self.rewards_all = []
        self.values_all = []
        self.terminated_all = []
        self.env_meta_all = []
        
        global_count = 0
        for i, ref in enumerate(self.trajectory_refs):
            with h5py.File(ref.file_path, 'r') as f:
                g = f if ref.traj_key is None else f[ref.traj_key]
                action_key = _action_dataset_key(g)
                L = g[action_key].shape[0]
                self.env_meta_all.append(_normalize_env_meta(cfg, _load_env_meta(f, g)))
                term = g["terminated"][()] if "terminated" in g else np.zeros(L, dtype=bool)
                is_suc = np.any(term)
                
                # 动态计算信号
                rew, val = compute_rl_signals(
                    success_array=term,
                    gamma=cfg.env.gamma,
                    penalty=cfg.env.penalty,
                    reward_mode=cfg.env.reward_mode,
                    reward_type='b',
                    reward_shape=cfg.env.reward_shape
                )
                self.trajs_info.append({"path": ref.file_path, "traj_key": ref.traj_key, "ref": ref, "len": L})
                self.terminated_all.append(term)
                self.rewards_all.append(rew)
                self.values_all.append(val)
                
                pad_before = self.obs_horizon - 1
                for t in range(-pad_before, L - self.act_horizon + 1):
                    sl = (i, t, global_count)
                    self.slices_all.append(sl)
                    if is_suc: self.slices_success.append(sl)
                    global_count += 1

        self.slices = self.slices_all
        self.mode = 'all'
        
        # 2. 确定输出字段
        self.available_keys = {"observations", "action", "terminated", "reward", "value", "next_observations", "cond"}
        self.required_keys = set(required_keys) if required_keys else self.available_keys
        if "actions" in self.required_keys:
            self.required_keys.remove("actions")
            self.required_keys.add("action")
        
        # 3. 动态配置
        self.agent_control_mode = canonicalize_control_mode(
            getattr(cfg, "agent_control_mode", getattr(cfg.env, "env_control_mode", getattr(cfg.env, "control_mode", "delta_pose")))
        )
        self.env_control_mode = canonicalize_control_mode(
            getattr(cfg.env, "env_control_mode", getattr(cfg.env, "control_mode", "delta_pose"))
        )
        self.is_abs_mode = is_absolute_mode(self.agent_control_mode)
        self.pad_action_arm = None 
        
        # Cond 占位
        if "cond" in self.required_keys:
            self.conds = torch.zeros(len(self.slices_all), dtype=torch.float32)
        else:
            self.conds = None

    def switch(self, mode='all'):
        self.mode = mode
        self.slices = self.slices_all if mode == 'all' else self.slices_success

    def _get_obs_seq(self, traj_idx, step_idx):
        traj_group = self.get_trajectory_group(self.trajs_info[traj_idx]["ref"])
        g = traj_group["obs"]
        L_obs = self.trajs_info[traj_idx]["len"] + 1
        seq = []
        for i in range(self.obs_horizon):
            idx = max(0, min(step_idx - (self.obs_horizon - 1) + i, L_obs - 1))
            seq.append({"rgb": torch.from_numpy(g["rgb"][idx]), "state": torch.from_numpy(g["state"][idx].astype(np.float32))})
        return {k: torch.stack([s[k] for s in seq]) for k in seq[0].keys()}

    def __getitem__(self, index):
        traj_idx, start, global_idx = self.slices[index]
        traj_group = self.get_trajectory_group(self.trajs_info[traj_idx]["ref"])
        L = self.trajs_info[traj_idx]["len"]
        
        data = {}
        if "observations" in self.required_keys: data["observations"] = self._get_obs_seq(traj_idx, start)
        if "next_observations" in self.required_keys: data["next_observations"] = self._get_obs_seq(traj_idx, min(start + self.act_horizon, L))
        
        if "action" in self.required_keys:
            act_all = traj_group[_action_dataset_key(traj_group)][()]
            if self.pad_action_arm is None and not self.is_abs_mode: self.pad_action_arm = torch.zeros((act_all.shape[1]-1,))
            act_seq = torch.from_numpy(act_all[max(0, start) : start + self.pred_horizon]).float()
            if start < 0:
                act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
            if len(act_seq) < self.pred_horizon:
                act_seq = torch.cat([act_seq, act_seq[-1].unsqueeze(0).repeat(self.pred_horizon - len(act_seq), 1)], dim=0)
            if self.agent_control_mode != self.env_meta_all[traj_idx]["env_control_mode"]:
                state_idx = min(max(0, start), self.trajs_info[traj_idx]["len"] - 1)
                current_state = traj_group["obs"]["state"][state_idx].astype(np.float32)
                current_poses = extract_arm_pose_map_from_flat_state(current_state, self.env_meta_all[traj_idx])
                meta = build_action_transform_meta(current_poses=current_poses, env_meta=self.env_meta_all[traj_idx])
                act_seq = torch.from_numpy(
                    inverse_transform_action(
                        obs=None,
                        next_obs=None,
                        env_action=act_seq.numpy(),
                        env_control_mode=self.env_meta_all[traj_idx]["env_control_mode"],
                        agent_control_mode=self.agent_control_mode,
                        meta=meta,
                    )
                ).float()
            data["action"] = act_seq

        idx = min(max(0, start), L-1)
        if "reward" in self.required_keys: data["reward"] = torch.tensor([self.rewards_all[traj_idx][idx]], dtype=torch.float32)
        if "terminated" in self.required_keys: data["terminated"] = torch.tensor([self.terminated_all[traj_idx][idx]], dtype=torch.float32)
        if "value" in self.required_keys: data["value"] = torch.tensor([self.values_all[traj_idx][idx]], dtype=torch.float32)
        if "cond" in self.required_keys and self.conds is not None: data["cond"] = self.conds[global_idx].reshape(1)
            
        return data

    def __len__(self): return len(self.slices)

# ==================== 2. ClassicReplayBuffer (内存队列) ====================

class ClassicReplayBuffer(Dataset):
    """
    经典回放缓冲区：支持内存中的轨迹入队/出队。
    """
    def __init__(self, cfg, max_traj_num=100, required_keys=None):
        self.cfg = cfg
        self.max_traj_num = max_traj_num
        self.obs_horizon = cfg.env.obs_horizon
        self.pred_horizon = cfg.env.pred_horizon
        self.act_horizon = cfg.env.act_horizon
        self.required_keys = set(required_keys) if required_keys else {"observations", "action", "terminated", "reward", "value", "next_observations", "cond"}
        if "actions" in self.required_keys:
            self.required_keys.remove("actions")
            self.required_keys.add("action")
        
        self.buffer = []
        self.slices_all, self.slices_success = [], []
        self.agent_control_mode = canonicalize_control_mode(
            getattr(cfg, "agent_control_mode", getattr(cfg.env, "env_control_mode", getattr(cfg.env, "control_mode", "delta_pose")))
        )
        self.env_control_mode = canonicalize_control_mode(
            getattr(cfg.env, "env_control_mode", getattr(cfg.env, "control_mode", "delta_pose"))
        )
        self.is_abs_mode = is_absolute_mode(self.agent_control_mode)
        self.pad_action_arm = None

    def push(self, trajectories: Dict[str, List]):
        action_key = "action" if "action" in trajectories else "actions"
        num_new = len(trajectories[action_key])
        for i in range(num_new):
            # 将 numpy 数据转换为内部存储
            traj = {
                "observations": trajectories["observations"][i],
                "action": trajectories[action_key][i],
                "terminated": trajectories["terminated"][i],
                "rewards": trajectories["rewards"][i],
                "env_meta": _normalize_env_meta(
                    self.cfg,
                    trajectories.get("env_meta", [{}] * num_new)[i]
                    if isinstance(trajectories.get("env_meta"), list)
                    else {},
                ),
            }
            
            # 动态重计算 RL 信号
            rew, val = compute_rl_signals(
                success_array=traj["terminated"],
                gamma=self.cfg.env.gamma,
                penalty=self.cfg.env.penalty,
                reward_mode=self.cfg.env.reward_mode,
                reward_type='b',
                reward_shape=self.cfg.env.reward_shape
            )
            traj["rewards_processed"] = rew
            traj["values_processed"] = val
            
            self.buffer.append(traj)
            if len(self.buffer) > self.max_traj_num: self.buffer.pop(0)
        self._rebuild_slices()

    def _rebuild_slices(self):
        self.slices_all, self.slices_success = [], []
        global_count = 0
        for i, traj in enumerate(self.buffer):
            L = len(traj["action"])
            is_suc = np.any(traj["terminated"])
            pad_before = self.obs_horizon - 1
            for t in range(-pad_before, L - self.act_horizon + 1):
                sl = (i, t, global_count)
                self.slices_all.append(sl)
                if is_suc: self.slices_success.append(sl)
                global_count += 1
        
        # 更新 conds
        if "cond" in self.required_keys:
            self.conds = torch.zeros(len(self.slices_all), dtype=torch.float32)
        else:
            self.conds = None
        self.slices = self.slices_all

    def _get_obs_seq(self, traj_idx, step_idx):
        obs_list = self.buffer[traj_idx]["observations"]
        L_obs = len(obs_list)
        seq = []
        for i in range(self.obs_horizon):
            idx = max(0, min(step_idx - (self.obs_horizon - 1) + i, L_obs - 1))
            seq.append(obs_list[idx])
        return {k: torch.stack([s[k] for s in seq]).float() for k in seq[0].keys()}

    def __getitem__(self, index):
        traj_idx, start, global_idx = self.slices[index]
        traj = self.buffer[traj_idx]
        L = len(traj["action"])
        
        data = {}
        if "observations" in self.required_keys: data["observations"] = self._get_obs_seq(traj_idx, start)
        if "next_observations" in self.required_keys: data["next_observations"] = self._get_obs_seq(traj_idx, min(start + self.act_horizon, L))
        
        if "action" in self.required_keys:
            act_traj = torch.from_numpy(traj["action"]).float() if isinstance(traj["action"], np.ndarray) else traj["action"]
            if self.pad_action_arm is None and not self.is_abs_mode: self.pad_action_arm = torch.zeros((act_traj.shape[1]-1,))
            act_seq = act_traj[max(0, start) : start + self.pred_horizon]
            if start < 0:
                act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
            if len(act_seq) < self.pred_horizon:
                act_seq = torch.cat([act_seq, act_seq[-1].unsqueeze(0).repeat(self.pred_horizon - len(act_seq), 1)], dim=0)
            if self.agent_control_mode != traj["env_meta"]["env_control_mode"]:
                state_idx = min(max(0, start), len(traj["observations"]) - 1)
                current_state = traj["observations"][state_idx]["state"]
                if isinstance(current_state, torch.Tensor):
                    current_state = current_state.detach().cpu().numpy()
                current_poses = extract_arm_pose_map_from_flat_state(current_state, traj["env_meta"])
                meta = build_action_transform_meta(current_poses=current_poses, env_meta=traj["env_meta"])
                act_seq = torch.from_numpy(
                    inverse_transform_action(
                        obs=None,
                        next_obs=None,
                        env_action=act_seq.detach().cpu().numpy(),
                        env_control_mode=traj["env_meta"]["env_control_mode"],
                        agent_control_mode=self.agent_control_mode,
                        meta=meta,
                    )
                ).float()
            data["action"] = act_seq

        idx = min(max(0, start), L-1)
        if "reward" in self.required_keys: data["reward"] = torch.tensor([traj["rewards_processed"][idx]], dtype=torch.float32)
        if "terminated" in self.required_keys: data["terminated"] = torch.tensor([traj["terminated"][idx]], dtype=torch.float32)
        if "value" in self.required_keys: data["value"] = torch.tensor([traj["values_processed"][idx]], dtype=torch.float32)
        if "cond" in self.required_keys and self.conds is not None: 
            data["cond"] = self.conds[global_idx].reshape(1)
        
        return data

    def __len__(self): return len(self.slices)
    
