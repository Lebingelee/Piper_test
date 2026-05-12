import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import time
import h5py
import numpy as np
import torch

from agent_factory.control import (
    build_action_transform_meta,
    canonicalize_control_mode,
    extract_arm_pose_map_from_flat_state,
    inverse_transform_action,
    is_absolute_mode,
)
from agent_factory.data.base import BaseTrajectoryDataset, TrajectoryRef, traj_sort_key


SEGMENT_TYPE_TO_ID = {
    "failure": 0,
    "success": 1,
    "intervention": 2,
    "unknown": 3,
}


def cfg_get(cfg: Any, path: str, default: Any) -> Any:
    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def compute_pad_vec(act_seq: torch.Tensor, is_abs_mode: bool, pad_action_arm: Optional[torch.Tensor] = None):
    if is_abs_mode:
        return act_seq[-1]
    if pad_action_arm is not None:
        return torch.cat([pad_action_arm, act_seq[-1, -1:]])
    return act_seq[-1]


def compute_progress_gammas(
    length: int,
    max_episode_steps: int,
    gamma_floor: float = 1e-4,
    terminal_idx: Optional[int] = None,
) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    terminal_idx = length - 1 if terminal_idx is None else int(np.clip(terminal_idx, 0, length - 1))
    horizon = max(terminal_idx + 1, 1)
    max_steps = max(int(max_episode_steps), horizon)

    gammas = np.ones(length, dtype=np.float32)
    for t in range(length):
        if t >= terminal_idx:
            gammas[t] = 0.0
            continue
        numerator = max_steps - horizon + t
        denom = numerator + 1
        gamma = numerator / denom if denom > 0 else 0.0
        gammas[t] = max(float(gamma), float(gamma_floor))
    return gammas


def compute_progress_returns(rewards: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = float(rewards[t]) + float(gammas[t]) * running
        returns[t] = running
    return returns


def compute_n_step_progress_signals(
    rewards: np.ndarray,
    gammas: np.ndarray,
    start_idx: int,
    n: int,
) -> Tuple[torch.Tensor, float]:
    length = len(rewards)
    start_idx = int(np.clip(start_idx, 0, max(length - 1, 0)))
    end_idx = min(start_idx + int(n), length)
    running_discount = 1.0
    n_step_reward = 0.0
    for idx in range(start_idx, end_idx):
        n_step_reward += running_discount * float(rewards[idx])
        running_discount *= float(gammas[idx])
    return torch.tensor(n_step_reward, dtype=torch.float32), float(running_discount)


def _read_json_dataset(dataset) -> Dict[str, Any]:
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return json.loads(value)


def _ordered_group_keys(group: h5py.Group, meta: Optional[Dict[str, Any]], section: str, subtype: Optional[str] = None) -> List[str]:
    if meta:
        try:
            node = meta[section] if subtype is None else meta[section][subtype]
            if isinstance(node, dict):
                keys = [key for key in sorted(node.keys()) if key in group]
                if keys:
                    return keys
        except Exception:
            pass
    return sorted(group.keys())


def _concat_group_leaves(group: h5py.Group, keys: Sequence[str]) -> np.ndarray:
    arrays = []
    for key in keys:
        data = group[key][()]
        arrays.append(data.reshape(data.shape[0], -1) if data.ndim > 2 else data)
    if not arrays:
        raise KeyError(f"No datasets found under group {group.name}")
    return np.concatenate(arrays, axis=-1).astype(np.float32)


def _concat_rgb_group(group: h5py.Group, keys: Sequence[str]) -> np.ndarray:
    arrays = [group[key][()] for key in keys]
    if not arrays:
        raise KeyError(f"No rgb datasets found under group {group.name}")
    return np.concatenate(arrays, axis=1)


def _read_meta(h5_file: h5py.File, traj_group: h5py.Group) -> Dict[str, Any]:
    if "meta" in h5_file and "env_meta" in h5_file["meta"]:
        return _read_json_dataset(h5_file["meta"]["env_meta"])
    if "meta" in traj_group and isinstance(traj_group["meta"], h5py.Group) and "env_meta" in traj_group["meta"]:
        return _read_json_dataset(traj_group["meta"]["env_meta"])
    if "meta_keys" in traj_group:
        return _read_json_dataset(traj_group["meta_keys"])
    return {}


def load_flatten_trajectory(h5_file: h5py.File, traj_group: h5py.Group, include_rgb: bool = True) -> Dict[str, np.ndarray]:
    if "obs" not in traj_group or "action" not in traj_group:
        raise KeyError(f"CPIQL trajectory {traj_group.name} requires 'obs' and 'action' keys.")
    for key in ("success", "terminated", "truncated"):
        if key not in traj_group:
            raise KeyError(f"CPIQL trajectory {traj_group.name} requires '{key}' key.")

    meta = _read_meta(h5_file, traj_group)
    obs_group = traj_group["obs"]
    action_node = traj_group["action"]

    if isinstance(action_node, h5py.Dataset):
        action = action_node[()].astype(np.float32)
    else:
        action_keys = _ordered_group_keys(action_node, meta, "action")
        action = _concat_group_leaves(action_node, action_keys)

    obs: Dict[str, np.ndarray] = {}
    if include_rgb and "rgb" in obs_group:
        if isinstance(obs_group["rgb"], h5py.Dataset):
            obs["rgb"] = obs_group["rgb"][()]
        else:
            rgb_keys = _ordered_group_keys(obs_group["rgb"], meta, "obs", "rgb")
            obs["rgb"] = _concat_rgb_group(obs_group["rgb"], rgb_keys)
    if "state" in obs_group:
        if isinstance(obs_group["state"], h5py.Dataset):
            obs["state"] = obs_group["state"][()].astype(np.float32)
        else:
            state_keys = _ordered_group_keys(obs_group["state"], meta, "obs", "state")
            obs["state"] = _concat_group_leaves(obs_group["state"], state_keys)
    if not obs:
        raise KeyError(f"CPIQL trajectory {traj_group.name} has no usable obs/rgb or obs/state data.")

    length = action.shape[0]
    return {
        "obs": obs,
        "action": action,
        "env_meta": meta,
        "success": np.asarray(traj_group["success"][()], dtype=bool).reshape(-1)[:length],
        "terminated": np.asarray(traj_group["terminated"][()], dtype=bool).reshape(-1)[:length],
        "truncated": np.asarray(traj_group["truncated"][()], dtype=bool).reshape(-1)[:length],
    }


class CPIQLTrajectoryDataset(BaseTrajectoryDataset):
    """
    Shared CPIQL trajectory slicing and progress-signal construction.
    """

    def __init__(
        self,
        cfg,
        h5_path: Optional[str] = None,
        folder_path: Optional[str] = None,
        num_traj: Optional[int] = None,
        required_keys: Optional[Sequence[str]] = None,
        require_intervention: bool = False,
    ):
        start_time = time.time()
        self.cfg = cfg
        self.obs_horizon = int(cfg.env.obs_horizon)
        self.pred_horizon = int(cfg.env.pred_horizon)
        self.act_horizon = int(cfg.env.act_horizon)
        self.include_rgb = bool(cfg_get(cfg, "dataset.include_rgb", True))
        self.require_intervention = require_intervention

        super().__init__(h5_path=h5_path, folder_path=folder_path, num_traj=num_traj)
        if not self.trajectory_refs:
            raise ValueError("No CPIQL trajectories found.")

        self.k_grid = [float(v) for v in cfg_get(cfg, "critic.k_grid", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])]
        self.low_k_grid = [float(v) for v in cfg_get(cfg, "critic.low_k_grid", [0.0, 0.2, 0.4])]
        if not self.low_k_grid:
            self.low_k_grid = [min(self.k_grid)]
        self.gamma_floor = float(cfg_get(cfg, "critic.gamma_floor", 1e-4))
        self.success_progress_weight = float(cfg_get(cfg, "critic.success_progress_weight", 1.0))
        self.intervention_progress_weight = float(cfg_get(cfg, "critic.intervention_progress_weight", 0.3))
        self.intervention_terminal_reward = float(cfg_get(cfg, "critic.intervention_terminal_reward", 0.2))

        # 下面所有按轨迹存储的 list，索引单位都是“拆分后的子轨迹”，不是原始 H5 轨迹。
        # 例如一条原始轨迹 intervention=[0,0,0,1,1,0,0,0,1,0,0,0]，
        # 会被拆成 5 条子轨迹：[0:2], [3:4], [5:7], [8:8], [9:11]。
        self.obs_data: List[Dict[str, np.ndarray]] = []  # 每条子轨迹的观测字典，如 {"rgb": [T,C,H,W], "state": [T,D]}。
        self.action_data: List[torch.Tensor] = []  # 每条子轨迹的动作序列，形状通常为 [T, action_dim]。
        self.success: List[np.ndarray] = []  # 每条子轨迹逐帧成功标记；专家接管后直接成功时，专家段末帧可为 True。
        self.terminated: List[np.ndarray] = []  # 每条子轨迹逐帧自然终止标记，用于关闭 bootstrap。
        self.truncated: List[np.ndarray] = []  # 每条子轨迹逐帧截断标记，如超时、人工停止或失败截断。
        self.intervention: List[np.ndarray] = []  # 每条子轨迹逐帧专家接管标记；专家段通常全 True，自主段通常全 False。
        self.rewards: List[np.ndarray] = []  # 每条子轨迹逐帧 reward；通常只有末帧非零。
        self.gammas: List[np.ndarray] = []  # 每条子轨迹逐帧动态折扣 gamma，用于形成近似线性的 progress return。
        self.progress_returns: List[np.ndarray] = []  # 每条子轨迹逐帧进度回报；成功段末帧为 1.0，早期逐步变小。
        self.progress_masks: List[np.ndarray] = []  # 是否启用 progress anchor loss；成功段默认 1，失败段默认 0。
        self.progress_weights: List[np.ndarray] = []  # progress anchor loss 权重；可降低介入边界 pseudo reward 的监督强度。
        self.segment_type_ids: List[int] = []  # 子轨迹类型的整数编码，对应 SEGMENT_TYPE_TO_ID。
        self.segment_type_names: List[str] = []  # 子轨迹类型名，如 "failure"、"intervention"、"success"。
        self.segment_terminal_rewards: List[float] = []  # 子轨迹末帧 reward；如介入边界自主段 0.2，专家成功段 1.0。
        self.segment_terminal_indices: List[int] = []  # 子轨迹终点在子轨迹内部的索引，当前通常为 len(segment)-1。
        self.segment_source_refs: List[Tuple[str, Optional[str], int, int]] = []  # 子轨迹来源：(文件路径, H5轨迹key, 原始起点, 原始终点)。
        self.segment_end_is_intervention_boundary: List[bool] = []  # 自主段是否结束于下一帧专家接管；例如 [0:2] 后接 [3:4] 专家段则为 True。
        self.slices_all: List[Tuple[int, int, int, int]] = []  # 所有训练样本切片：(子轨迹idx, start, action_end, 全局样本idx)。
        self.slices_success: List[Tuple[int, int, int, int]] = []  # 只来自成功子轨迹的训练样本切片，用于按需切换 success-only 训练。
        self.env_metas: List[Dict[str, Any]] = []
        cache_start_time = time.time()

        self._cache_trajectories()
        self.slices = self.slices_all
        self.mode = "all"

        self.available_keys = {
            "observations",
            "next_observations",
            "action",
            "reward",
            "discount",
            "terminated",
            "truncated",
            "success",
            "progress_return",
            "progress_mask",
            "progress_weight",
            "segment_type",
            "segment_terminal_reward",
            "segment_end_is_intervention_boundary",
            "k",
        }
        if self.require_intervention:
            self.available_keys.add("intervention")
            self.available_keys.add("intervention_segment")
        self.required_keys = set(required_keys) if required_keys is not None else set(self.available_keys)
        if "actions" in self.required_keys:
            self.required_keys.remove("actions")
            self.required_keys.add("action")

        self.agent_control_mode = canonicalize_control_mode(
            getattr(cfg, "agent_control_mode", getattr(cfg.env, "env_control_mode", getattr(cfg.env, "control_mode", "delta_pose")))
        )
        self.env_control_mode = canonicalize_control_mode(
            getattr(cfg.env, "env_control_mode", getattr(cfg.env, "control_mode", "delta_pose"))
        )
        self.is_abs_mode = is_absolute_mode(self.agent_control_mode)
        self.pad_action_arm = torch.zeros((self.action_data[0].shape[1] - 1,)) if not self.is_abs_mode else None
        end_time = time.time()
        print(f"time: loaded and processed {len(self.slices_all)} CPIQL trajectory slices \n from {len(self.trajectory_refs)} trajectories in {end_time - start_time:.2f} seconds (caching took {end_time - cache_start_time:.2f} seconds), with agent_control_mode={self.agent_control_mode}, env_control_mode={self.env_control_mode}, is_abs_mode={self.is_abs_mode}.")


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

    def _collect_h5_files(self) -> List[str]:
        if not self.folder_path:
            return super()._collect_h5_files()
        folder = Path(self.folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"H5 folder not found: {self.folder_path}")
        traj_files = sorted(folder.glob("traj_*.h5"), key=lambda p: traj_sort_key(p.stem))
        if not traj_files:
            traj_files = sorted(folder.glob("*.h5"))
        return [str(path) for path in traj_files]

    def _cache_trajectories(self):
        global_count = 0
        for ref in self.trajectory_refs:
            with h5py.File(ref.file_path, "r") as h5_file:
                group = h5_file if ref.traj_key is None else h5_file[ref.traj_key]
                traj = load_flatten_trajectory(h5_file, group, include_rgb=self.include_rgb)
                intervention = self._load_intervention(group, len(traj["action"]))

            for start_idx, end_idx, segment_type in self._execution_segments(traj, intervention):
                traj_idx = len(self.action_data)
                end_is_intervention_boundary = (
                    segment_type != "intervention"
                    and end_idx + 1 < len(intervention)
                    and bool(intervention[end_idx + 1])
                )
                self._append_segment(
                    ref,
                    traj,
                    intervention,
                    start_idx,
                    end_idx,
                    segment_type,
                    end_is_intervention_boundary=end_is_intervention_boundary,
                )

                length = len(self.action_data[traj_idx])
                is_success = self.segment_type_names[traj_idx] == "success"
                pad_before = self.obs_horizon - 1
                for start in range(-pad_before, length - self.act_horizon + 1):
                    sl = (traj_idx, start, start + self.pred_horizon, global_count)
                    self.slices_all.append(sl)
                    if is_success:
                        self.slices_success.append(sl)
                    global_count += 1

    def _append_segment(
        self,
        ref: TrajectoryRef,
        traj: Dict[str, np.ndarray],
        intervention: np.ndarray,
        start_idx: int,
        end_idx: int,
        segment_type: str,
        end_is_intervention_boundary: bool = False,
    ):
        obs = {key: value[start_idx:end_idx + 1] for key, value in traj["obs"].items()}
        action = traj["action"][start_idx:end_idx + 1]
        success = traj["success"][start_idx:end_idx + 1].copy()
        terminated = traj["terminated"][start_idx:end_idx + 1].copy()
        truncated = traj["truncated"][start_idx:end_idx + 1].copy()
        local_intervention = intervention[start_idx:end_idx + 1].copy()
        env_meta = self._normalize_env_meta(traj.get("env_meta", {}))

        if end_is_intervention_boundary:
            success[:] = False
            terminated[:] = False
            truncated[:] = False
            terminated[-1] = True
            terminal_reward = self.intervention_terminal_reward
        elif segment_type == "intervention":
            terminal_reward = 1.0 if np.any(success) else 0.0
            if terminal_reward > 0.0:
                success[-1] = True
                terminated[-1] = True
            elif not np.any(terminated | truncated):
                truncated[-1] = True
        elif segment_type == "success":
            success[-1] = True
            terminated[-1] = True
            terminal_reward = 1.0
        else:
            success[:] = False
            if not np.any(terminated | truncated):
                truncated[-1] = True
            terminal_reward = 0.0

        self.obs_data.append(obs)
        self.action_data.append(torch.from_numpy(action).float())
        self.success.append(success)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.intervention.append(local_intervention)
        self.segment_type_names.append(segment_type)
        self.segment_type_ids.append(SEGMENT_TYPE_TO_ID[segment_type])
        self.segment_terminal_rewards.append(float(terminal_reward))
        self.segment_terminal_indices.append(len(action) - 1)
        self.segment_source_refs.append((ref.file_path, ref.traj_key, int(start_idx), int(end_idx)))
        self.segment_end_is_intervention_boundary.append(bool(end_is_intervention_boundary))
        self.env_metas.append(env_meta)
        self._build_progress_signals(len(self.action_data) - 1)

    def _execution_segments(self, traj: Dict[str, np.ndarray], intervention: np.ndarray) -> List[Tuple[int, int, str]]:
        length = len(traj["action"])
        if length <= 0:
            return []
        if not self.require_intervention or not np.any(intervention):
            segment_type = "success" if np.any(traj["success"]) else "failure"
            return [(0, length - 1, segment_type)]

        segments: List[Tuple[int, int, str]] = []
        start_idx = 0
        current_flag = bool(intervention[0])
        for idx in range(1, length):
            flag = bool(intervention[idx])
            if flag == current_flag:
                continue
            segment_type = self._segment_type_for_range(traj, intervention, start_idx, idx - 1)
            segments.append((start_idx, idx - 1, segment_type))
            start_idx = idx
            current_flag = flag
        segment_type = self._segment_type_for_range(traj, intervention, start_idx, length - 1)
        segments.append((start_idx, length - 1, segment_type))
        return segments

    @staticmethod
    def _segment_type_for_range(
        traj: Dict[str, np.ndarray],
        intervention: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> str:
        if bool(intervention[start_idx]):
            return "intervention"
        if np.any(traj["success"][start_idx:end_idx + 1]):
            return "success"
        return "failure"

    def _load_intervention(self, traj_group: h5py.Group, length: int) -> np.ndarray:
        if "intervention" not in traj_group:
            if self.require_intervention:
                raise KeyError(f"CPIQL replay trajectory {traj_group.name} requires 'intervention' key.")
            return np.zeros(length, dtype=bool)
        return np.asarray(traj_group["intervention"][()], dtype=bool).reshape(-1)[:length]

    def _terminal_index(self, traj_idx: int) -> int:
        boundary = self.success[traj_idx] | self.terminated[traj_idx] | self.truncated[traj_idx]
        indices = np.flatnonzero(boundary)
        return int(indices[0]) if len(indices) else len(self.action_data[traj_idx]) - 1

    def _build_progress_signals(self, traj_idx: int):
        length = len(self.action_data[traj_idx])
        terminal_idx = self.segment_terminal_indices[traj_idx]
        segment_type = self.segment_type_names[traj_idx]

        rewards = np.zeros(length, dtype=np.float32)
        rewards[terminal_idx] = float(self.segment_terminal_rewards[traj_idx])
        gammas = compute_progress_gammas(
            length=length,
            max_episode_steps=self.cfg.env.max_episode_steps,
            gamma_floor=self.gamma_floor,
            terminal_idx=terminal_idx,
        )
        returns = compute_progress_returns(rewards, gammas)
        progress_mask = np.ones(length, dtype=np.float32) if segment_type == "success" else np.zeros(length, dtype=np.float32)
        if self.segment_end_is_intervention_boundary[traj_idx] and bool(cfg_get(self.cfg, "critic.anchor_intervention_pseudo", False)):
            progress_mask[:] = 1.0
        progress_weight = progress_mask * self.success_progress_weight
        if self.segment_end_is_intervention_boundary[traj_idx]:
            progress_weight *= self.intervention_progress_weight

        if traj_idx < len(self.rewards):
            self.rewards[traj_idx] = rewards
            self.gammas[traj_idx] = gammas
            self.progress_returns[traj_idx] = returns
            self.progress_masks[traj_idx] = progress_mask
            self.progress_weights[traj_idx] = progress_weight.astype(np.float32)
        else:
            self.rewards.append(rewards)
            self.gammas.append(gammas)
            self.progress_returns.append(returns)
            self.progress_masks.append(progress_mask)
            self.progress_weights.append(progress_weight.astype(np.float32))

    def refresh_progress_signals(self, segment_indices: Optional[Sequence[int]] = None):
        if segment_indices is None:
            segment_indices = list(range(len(self.action_data)))
        for segment_idx in segment_indices:
            segment_idx = int(segment_idx)
            self._build_progress_signals(segment_idx)

    def set_segment_terminal_rewards(
        self,
        rewards: Sequence[float],
        segment_indices: Optional[Sequence[int]] = None,
        refresh: bool = True,
    ):
        if segment_indices is None:
            segment_indices = list(range(len(self.segment_terminal_rewards)))
        else:
            segment_indices = list(segment_indices)
        if len(rewards) != len(segment_indices):
            raise ValueError(f"Expected {len(segment_indices)} terminal rewards, got {len(rewards)}.")
        for segment_idx, reward in zip(segment_indices, rewards):
            self.segment_terminal_rewards[int(segment_idx)] = float(reward)
        if refresh:
            self.refresh_progress_signals(segment_indices)

    def segment_indices_by_type(self, segment_type: str) -> List[int]:
        return [
            idx for idx, current_type in enumerate(self.segment_type_names)
            if current_type == segment_type
        ]

    def intervention_boundary_segment_indices(self) -> List[int]:
        return [
            idx for idx, is_boundary in enumerate(self.segment_end_is_intervention_boundary)
            if is_boundary
        ]

    def set_terminal_rewards_for_type(
        self,
        segment_type: str,
        rewards: Sequence[float],
        refresh: bool = True,
    ):
        segment_indices = self.segment_indices_by_type(segment_type)
        self.set_segment_terminal_rewards(rewards, segment_indices, refresh=refresh)

    def set_intervention_terminal_rewards(
        self,
        rewards: Sequence[float],
        segment_indices: Optional[Sequence[int]] = None,
        refresh: bool = True,
    ):
        if segment_indices is None:
            segment_indices = self.intervention_boundary_segment_indices()
        else:
            segment_indices = list(segment_indices)
        invalid = [idx for idx in segment_indices if not self.segment_end_is_intervention_boundary[int(idx)]]
        if invalid:
            raise ValueError(f"Segments do not end at intervention boundaries: {invalid}")
        self.set_segment_terminal_rewards(rewards, segment_indices, refresh=refresh)

    def switch(self, mode: str = "all"):
        self.mode = mode
        self.slices = self.slices_success if mode == "success" else self.slices_all

    def get_all_actions(self):
        return torch.cat(self.action_data, dim=0)

    def _get_obs_sequence(self, traj_idx: int, start_idx: int, horizon: int) -> Dict[str, torch.Tensor]:
        obs_traj = self.obs_data[traj_idx]
        length = len(next(iter(obs_traj.values())))
        indices = [max(0, min(start_idx - (horizon - 1) + i, length - 1)) for i in range(horizon)]
        return {key: torch.from_numpy(value[indices]) for key, value in obs_traj.items()}

    def _sample_k(self, traj_idx: int, step_idx: int) -> float:
        del step_idx
        if self.segment_type_names[traj_idx] in {"failure", "intervention"} or self.segment_end_is_intervention_boundary[traj_idx]:
            grid = self.low_k_grid
        else:
            grid = self.k_grid
        return float(grid[np.random.randint(0, len(grid))])

    def _transform_action_sequence(self, traj_idx: int, start: int, act_seq: torch.Tensor) -> torch.Tensor:
        if self.agent_control_mode == self.env_metas[traj_idx]["env_control_mode"]:
            return act_seq
        state_idx = min(max(0, start), len(self.obs_data[traj_idx]["state"]) - 1)
        current_poses = extract_arm_pose_map_from_flat_state(
            self.obs_data[traj_idx]["state"][state_idx],
            self.env_metas[traj_idx],
        )
        meta = build_action_transform_meta(current_poses=current_poses, env_meta=self.env_metas[traj_idx])
        transformed = inverse_transform_action(
            obs=None,
            next_obs=None,
            env_action=act_seq.detach().cpu().numpy(),
            env_control_mode=self.env_metas[traj_idx]["env_control_mode"],
            agent_control_mode=self.agent_control_mode,
            meta=meta,
        )
        return torch.from_numpy(transformed).float()

    def __getitem__(self, index):
        traj_idx, start, end, _global_idx = self.slices[index]
        length = len(self.action_data[traj_idx])
        idx = min(max(0, start), length - 1)
        next_idx = min(idx + self.act_horizon, length - 1)

        data: Dict[str, Any] = {}
        if "observations" in self.required_keys:
            data["observations"] = self._get_obs_sequence(traj_idx, start, self.obs_horizon)
        if "next_observations" in self.required_keys:
            data["next_observations"] = self._get_obs_sequence(traj_idx, min(start + self.act_horizon, length), self.obs_horizon)
        if "action" in self.required_keys:
            act_seq = self.action_data[traj_idx][max(0, start):end]
            if start < 0:
                act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
            if len(act_seq) < self.pred_horizon:
                act_seq = torch.cat([act_seq, act_seq[-1].unsqueeze(0).repeat(self.pred_horizon - len(act_seq), 1)], dim=0)
            if self.agent_control_mode != self.env_metas[traj_idx]["env_control_mode"]:
                act_seq = self._transform_action_sequence(traj_idx, start, act_seq)
            data["action"] = act_seq

        reward, discount = compute_n_step_progress_signals(self.rewards[traj_idx], self.gammas[traj_idx], idx, self.act_horizon)
        boundary_window = slice(idx, min(idx + self.act_horizon + 1, length))
        terminated = bool(np.any(self.success[traj_idx][boundary_window] | self.terminated[traj_idx][boundary_window] | self.truncated[traj_idx][boundary_window]))

        if "reward" in self.required_keys:
            data["reward"] = reward.reshape(1)
        if "discount" in self.required_keys:
            data["discount"] = torch.tensor([discount], dtype=torch.float32)
        if "terminated" in self.required_keys:
            data["terminated"] = torch.tensor([terminated], dtype=torch.float32)
        if "truncated" in self.required_keys:
            data["truncated"] = torch.tensor([self.truncated[traj_idx][next_idx]], dtype=torch.float32)
        if "success" in self.required_keys:
            data["success"] = torch.tensor([self.success[traj_idx][next_idx]], dtype=torch.float32)
        if "progress_return" in self.required_keys:
            data["progress_return"] = torch.tensor([self.progress_returns[traj_idx][idx]], dtype=torch.float32)
        if "progress_mask" in self.required_keys:
            data["progress_mask"] = torch.tensor([self.progress_masks[traj_idx][idx]], dtype=torch.float32)
        if "progress_weight" in self.required_keys:
            data["progress_weight"] = torch.tensor([self.progress_weights[traj_idx][idx]], dtype=torch.float32)
        if "segment_type" in self.required_keys:
            data["segment_type"] = torch.tensor([self.segment_type_ids[traj_idx]], dtype=torch.long)
        if "segment_terminal_reward" in self.required_keys:
            data["segment_terminal_reward"] = torch.tensor([self.segment_terminal_rewards[traj_idx]], dtype=torch.float32)
        if "segment_end_is_intervention_boundary" in self.required_keys:
            data["segment_end_is_intervention_boundary"] = torch.tensor([self.segment_end_is_intervention_boundary[traj_idx]], dtype=torch.float32)
        if "k" in self.required_keys:
            data["k"] = torch.tensor([self._sample_k(traj_idx, idx)], dtype=torch.float32)
        if "intervention" in self.required_keys:
            data["intervention"] = torch.tensor([self.intervention[traj_idx][idx]], dtype=torch.float32)
        if "intervention_segment" in self.required_keys:
            data["intervention_segment"] = torch.tensor([self.segment_type_names[traj_idx] == "intervention"], dtype=torch.float32)
        return data

    def __len__(self):
        return len(self.slices)
