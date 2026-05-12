import torch
import numpy as np
import h5py
import json
import os
from typing import Optional, List, Dict, Any, Union
from agent_factory.control import (
    build_action_transform_meta,
    canonicalize_control_mode,
    extract_arm_pose_map_from_flat_state,
    inverse_transform_action,
    is_absolute_mode,
)

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

from agent_factory.data.base import BaseTrajectoryDataset
from agent_factory.data.utils import compute_rl_signals, compute_n_step_signals

# ==================== 1. ExpertDataset (内存缓存加速版) ====================

def _read_json_dataset(dataset) -> Dict[str, Any]:
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return json.loads(value)


def _traj_sort_key(name: str):
    parts = name.split("_")
    for token in parts[1:]:
        try:
            return int(token)
        except ValueError:
            continue
    return name


def _is_dataset(group: h5py.Group, key: str) -> bool:
    return key in group and isinstance(group[key], h5py.Dataset)


def _is_group(group: h5py.Group, key: str) -> bool:
    return key in group and isinstance(group[key], h5py.Group)


def _looks_like_flat(traj_group: h5py.Group) -> bool:
    if "obs" not in traj_group or not isinstance(traj_group["obs"], h5py.Group):
        return False
    obs_group = traj_group["obs"]
    return (
        _is_dataset(obs_group, "rgb")
        and _is_dataset(obs_group, "state")
        and (_is_dataset(traj_group, "action") or _is_dataset(traj_group, "actions"))
    )


def _looks_like_structured(traj_group: h5py.Group) -> bool:
    if "obs" not in traj_group or not isinstance(traj_group["obs"], h5py.Group):
        return False
    obs_group = traj_group["obs"]
    return (
        _is_group(obs_group, "state")
        and _is_group(traj_group, "action")
    )


def _resolve_dataset_format(h5_file: h5py.File, requested_format: str) -> str:
    fmt = (requested_format or "flat").lower()
    aliases = {
        "flatten": "flat",
        "flattened": "flat",
        "merged": "structured",
        "raw": "structured",
    }
    fmt = aliases.get(fmt, fmt)
    if fmt not in {"flat", "structured", "auto"}:
        raise ValueError(
            "dataset.expert.format must be one of: flat, structured, auto "
            f"(got {requested_format!r})."
        )
    if fmt != "auto":
        return fmt

    traj_keys = [k for k in h5_file.keys() if k.startswith("traj_")]
    if not traj_keys:
        if _looks_like_flat(h5_file):
            return "flat"
        if _looks_like_structured(h5_file):
            return "structured"
        raise ValueError("Cannot infer dataset format from root H5 schema.")
    sample = h5_file[sorted(traj_keys, key=_traj_sort_key)[0]]
    if _looks_like_flat(sample):
        return "flat"
    if _looks_like_structured(sample):
        return "structured"
    raise ValueError(f"Cannot infer dataset format from trajectory schema: {sample.name}")

class ExpertDataset(BaseTrajectoryDataset):
    """
    专家数据集类：支持全量数据载入内存以消除 IO 瓶颈。
    针对双臂多相机高维数据进行极致优化。
    """
    def __init__(self, 
                 cfg,
                 obs_space=None,
                 device="cpu", 
                 required_keys=None,
                 dataset_format: Optional[str] = None):
        
        self.cfg = cfg
        self.obs_horizon = cfg.env.obs_horizon
        self.pred_horizon = cfg.env.pred_horizon
        self.act_horizon = cfg.env.act_horizon
        
        # 1. 探测文件
        self.demo_path = cfg.dataset.expert.demo_path
        if not os.path.exists(self.demo_path):
             raise FileNotFoundError(f"Expert dataset not found: {self.demo_path}")
        super().__init__(
            h5_path=self.demo_path,
            num_traj=cfg.dataset.expert.num_traj,
        )

        # 🟢 缓存占位
        self.rgb_data = []   # List of np.ndarray
        self.state_data = [] # List of np.ndarray
        self.action_data = []
        self.env_metas = []
        self.terminated = []
        self.rewards = []
        self.values = []
        
        # 2. 一次性载入所有数据到内存
        requested_format = dataset_format or getattr(cfg.dataset.expert, "format", "flat")
        print(f"[ExpertDataset] Loading ALL data from {self.demo_path} to RAM...")
        if not self.trajectory_refs:
            raise ValueError(f"No trajectories found in expert dataset: {self.demo_path}")

        with h5py.File(self.trajectory_refs[0].file_path, 'r') as f:
            self.dataset_format = _resolve_dataset_format(f, requested_format)
            print(f"[ExpertDataset] Dataset format: {self.dataset_format}")

        self.traj_keys = [ref.traj_key for ref in self.trajectory_refs]
        self.slices_all = []
        self.slices_success = []
        global_count = 0

        from tqdm import tqdm
        for i, ref in enumerate(tqdm(self.trajectory_refs, desc="Caching RAM")):
            with h5py.File(ref.file_path, 'r') as f:
                g = f if ref.traj_key is None else f[ref.traj_key]
                env_meta = self._load_env_meta(f, g) if self.dataset_format == "structured" else self._load_optional_env_meta(f, g)
                env_meta = self._normalize_env_meta(env_meta)
                traj = self._load_flat_trajectory(g) if self.dataset_format == "flat" else self._load_structured_trajectory(f, g)
                self.rgb_data.append(traj["rgb"])
                self.state_data.append(traj["state"])

                act = torch.from_numpy(traj["action"]).float()
                L = len(act)
                self.action_data.append(act)
                self.env_metas.append(env_meta)
                
                term = self._load_terminated(g, L)
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
        self.agent_control_mode = canonicalize_control_mode(
            getattr(cfg, "agent_control_mode", getattr(cfg.env, "env_control_mode", getattr(cfg.env, "control_mode", "delta_pose")))
        )
        self.env_control_mode = canonicalize_control_mode(
            getattr(cfg.env, "env_control_mode", getattr(cfg.env, "control_mode", "delta_pose"))
        )
        self.is_abs_mode = is_absolute_mode(self.agent_control_mode)
        self.pad_action_arm = torch.zeros((self.action_data[0].shape[1] - 1,)) if not self.is_abs_mode else None
        
        print(f"[ExpertDataset] RAM Caching completed. Trajectories: {len(self.traj_keys)}, Slices: {len(self.slices_all)}")

    def _load_flat_trajectory(self, traj_group: h5py.Group) -> Dict[str, np.ndarray]:
        if _is_dataset(traj_group, "action"):
            action_key = "action"
        elif _is_dataset(traj_group, "actions"):
            action_key = "actions"
        else:
            raise KeyError("Flat expert trajectory requires 'action' dataset.")
        return {
            "rgb": traj_group["obs"]["rgb"][()],
            "state": traj_group["obs"]["state"][()].astype(np.float32),
            "action": traj_group[action_key][()].astype(np.float32),
        }

    def _load_env_meta(self, h5_file: h5py.File, traj_group: h5py.Group) -> Dict[str, Any]:
        if "meta" in h5_file and "env_meta" in h5_file["meta"]:
            return _read_json_dataset(h5_file["meta"]["env_meta"])
        if "meta" in traj_group and "env_meta" in traj_group["meta"]:
            return _read_json_dataset(traj_group["meta"]["env_meta"])
        if "meta_keys" in traj_group:
            return _read_json_dataset(traj_group["meta_keys"])
        raise KeyError(
            "Structured expert dataset requires meta/env_meta or traj/meta/env_meta "
            "to determine flatten key order."
        )

    def _load_optional_env_meta(self, h5_file: h5py.File, traj_group: h5py.Group) -> Dict[str, Any]:
        try:
            return self._load_env_meta(h5_file, traj_group)
        except Exception:
            return {}

    def _normalize_env_meta(self, env_meta: Dict[str, Any]) -> Dict[str, Any]:
        env_meta = dict(env_meta or {})
        env_meta.setdefault("obs", {})
        env_meta.setdefault("action", {})
        env_meta["env_control_mode"] = canonicalize_control_mode(
            env_meta.get("env_control_mode") or getattr(self.cfg.env, "env_control_mode", getattr(self.cfg.env, "control_mode", "delta_pose"))
        )
        env_meta.setdefault("controller_backend", getattr(self.cfg.env, "controller_backend", ""))
        env_meta.setdefault("control", {})
        return env_meta

    def _load_structured_trajectory(self, h5_file: h5py.File, traj_group: h5py.Group) -> Dict[str, np.ndarray]:
        env_meta = self._load_env_meta(h5_file, traj_group)
        obs_group = traj_group["obs"]
        action_group = traj_group["action"]

        rgb_arrays = []
        for role in sorted(env_meta["obs"].get("rgb", {}).keys()):
            if "rgb" not in obs_group or role not in obs_group["rgb"]:
                raise KeyError(f"Missing structured obs/rgb/{role}")
            rgb_arrays.append(obs_group["rgb"][role][()])
        if rgb_arrays:
            rgb = np.concatenate(rgb_arrays, axis=1)
        else:
            action_len = self._first_leaf_length(action_group, sorted(env_meta["action"].keys()))
            rgb = np.zeros((action_len, 0, 1, 1), dtype=np.uint8)

        state_arrays = []
        for key in sorted(env_meta["obs"]["state"].keys()):
            if key not in obs_group["state"]:
                raise KeyError(f"Missing structured obs/state/{key}")
            data = obs_group["state"][key][()]
            state_arrays.append(data.reshape(data.shape[0], -1))
        state = np.concatenate(state_arrays, axis=-1).astype(np.float32)

        action_arrays = []
        for key in sorted(env_meta["action"].keys()):
            if key not in action_group:
                raise KeyError(f"Missing structured action/{key}")
            data = action_group[key][()]
            action_arrays.append(data.reshape(data.shape[0], -1))
        action = np.concatenate(action_arrays, axis=-1).astype(np.float32)

        return {
            "rgb": rgb,
            "state": state,
            "action": action,
        }

    @staticmethod
    def _first_leaf_length(group: h5py.Group, ordered_keys: List[str]) -> int:
        for key in ordered_keys:
            if key in group:
                return group[key].shape[0]
        raise KeyError("Unable to determine trajectory length from action group.")

    @staticmethod
    def _load_terminated(traj_group: h5py.Group, length: int) -> np.ndarray:
        if "terminated" in traj_group:
            return np.asarray(traj_group["terminated"][()], dtype=bool)
        if "done" in traj_group:
            return np.asarray(traj_group["done"][()], dtype=bool)

        term = np.zeros(length, dtype=bool)
        success_value = None
        if "success" in traj_group:
            success_arr = np.asarray(traj_group["success"][()])
            if success_arr.shape == ():
                success_value = bool(success_arr.item())
            elif len(success_arr) == length:
                return success_arr.astype(bool)
        if success_value is None and "success" in traj_group.attrs:
            success_value = bool(traj_group.attrs["success"])
        if success_value:
            term[-1] = True
        return term

    def switch(self, mode='all'):
        self.mode = mode
        self.slices = self.slices_all if mode == 'all' else self.slices_success

    def get_all_actions(self):
        return torch.cat(self.action_data, dim=0)

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

    def _slice_env_action_sequence(self, traj_idx: int, start: int, end: int) -> torch.Tensor:
        act_seq = self.action_data[traj_idx][max(0, start): end]
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if len(act_seq) < self.pred_horizon:
            act_seq = torch.cat([act_seq, act_seq[-1].unsqueeze(0).repeat(self.pred_horizon - len(act_seq), 1)], dim=0)
        return act_seq

    def _transform_action_sequence(self, traj_idx: int, start: int, act_seq: torch.Tensor) -> torch.Tensor:
        if self.agent_control_mode == self.env_metas[traj_idx]["env_control_mode"]:
            return act_seq
        state_idx = min(max(0, start), self.state_data[traj_idx].shape[0] - 1)
        current_poses = extract_arm_pose_map_from_flat_state(
            self.state_data[traj_idx][state_idx],
            self.env_metas[traj_idx],
        )
        meta = build_action_transform_meta(current_poses=current_poses, env_meta=self.env_metas[traj_idx])
        transformed = inverse_transform_action(
            obs=None,
            next_obs=None,
            env_action=act_seq.numpy(),
            env_control_mode=self.env_metas[traj_idx]["env_control_mode"],
            agent_control_mode=self.agent_control_mode,
            meta=meta,
        )
        return torch.from_numpy(transformed).float()

    def __getitem__(self, index):
        traj_idx, start, end, global_idx = self.slices[index]
        L = self.action_data[traj_idx].shape[0]
        
        data = {}
        # 1. Observations
        if "observations" in self.required_keys:
            data["observations"] = self._get_obs_sequence(traj_idx, start, self.obs_horizon)
        
        # 2. Next Observations
        if "next_observations" in self.required_keys:
            data["next_observations"] = self._get_obs_sequence(traj_idx, min(start + self.act_horizon, L), self.obs_horizon)

        # 3. Action
        if "action" in self.required_keys:
            act_seq = self._slice_env_action_sequence(traj_idx, start, end)
            act_seq = self._transform_action_sequence(traj_idx, start, act_seq)
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
