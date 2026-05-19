from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from agent_factory.data.base import traj_sort_key
from agent_factory.data.impl.cpiql.common import load_flatten_trajectory


def _resolve_single_trajectory_group(
    h5_file: h5py.File,
    traj_idx: int = 0,
):
    traj_keys = sorted(
        [
            key for key in h5_file.keys()
            if key.startswith("traj_") and isinstance(h5_file[key], h5py.Group)
        ],
        key=traj_sort_key,
    )
    if traj_keys:
        if traj_idx < 0 or traj_idx >= len(traj_keys):
            raise IndexError(
                f"traj_idx={traj_idx} is out of range for {len(traj_keys)} trajectories in {h5_file.filename}."
            )
        traj_key = traj_keys[traj_idx]
        return h5_file[traj_key], traj_key
    if traj_idx != 0:
        raise IndexError(
            f"traj_idx={traj_idx} is invalid because {h5_file.filename} stores only one root trajectory."
        )
    return h5_file, None


def load_replay_eval_trajectory(
    h5_path: str,
    include_rgb: bool = True,
    traj_idx: int = 0,
) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as h5_file:
        traj_group, traj_key = _resolve_single_trajectory_group(h5_file, traj_idx=traj_idx)
        trajectory = load_flatten_trajectory(h5_file, traj_group, include_rgb=include_rgb)

    trajectory["file_path"] = h5_path
    trajectory["traj_key"] = traj_key
    trajectory["traj_idx"] = int(traj_idx)
    return trajectory


def _aligned_obs_sequence(obs_array: np.ndarray, action_length: int) -> np.ndarray:
    if obs_array.shape[0] < action_length:
        raise ValueError(
            f"Observation length {obs_array.shape[0]} is shorter than action length {action_length}."
        )
    # Current replay rollouts typically store T+1 observations for T actions.
    # Eval aligns windows to action timesteps and therefore uses obs[:T].
    return obs_array[:action_length]


def _window_indices(anchor_t: int, horizon: int, max_length: int) -> List[int]:
    start = anchor_t - (horizon - 1)
    return [max(0, min(start + i, max_length - 1)) for i in range(horizon)]


def _action_chunk(action_array: np.ndarray, anchor_t: int, pred_horizon: int) -> np.ndarray:
    chunk = action_array[anchor_t:anchor_t + pred_horizon]
    if chunk.shape[0] == 0:
        raise ValueError(f"Anchor timestep {anchor_t} produced an empty action chunk.")
    if chunk.shape[0] < pred_horizon:
        pad = np.repeat(chunk[-1:], pred_horizon - chunk.shape[0], axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


def build_eval_window_batch(
    trajectory: Dict[str, Any],
    obs_horizon: int,
    pred_horizon: int,
    anchor_t: int,
) -> Dict[str, Any]:
    action = np.asarray(trajectory["action"], dtype=np.float32)
    length = action.shape[0]
    if anchor_t < 0 or anchor_t >= length:
        raise IndexError(f"anchor_t={anchor_t} is out of range for action length {length}.")

    obs_batch: Dict[str, torch.Tensor] = {}
    obs_indices = _window_indices(anchor_t, obs_horizon, length)
    for key, value in trajectory["obs"].items():
        aligned = _aligned_obs_sequence(np.asarray(value), length)
        obs_batch[key] = torch.from_numpy(aligned[obs_indices])

    batch = {
        "frame": torch.tensor(anchor_t, dtype=torch.long),
        "observations": obs_batch,
        "action": torch.from_numpy(_action_chunk(action, anchor_t, pred_horizon)),
    }
    return batch


def build_eval_window_batches(
    trajectory: Dict[str, Any],
    obs_horizon: int,
    pred_horizon: int,
) -> List[Dict[str, Any]]:
    action = np.asarray(trajectory["action"], dtype=np.float32)
    return [
        build_eval_window_batch(
            trajectory=trajectory,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            anchor_t=anchor_t,
        )
        for anchor_t in range(action.shape[0])
    ]


@dataclass
class ReplayEvalWindowDataset(Dataset):
    trajectory: Dict[str, Any]
    obs_horizon: int
    pred_horizon: int

    def __post_init__(self):
        self.length = int(np.asarray(self.trajectory["action"]).shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return build_eval_window_batch(
            trajectory=self.trajectory,
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            anchor_t=int(index),
        )
