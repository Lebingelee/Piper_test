from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from agent_factory.data.impl.lerobot.h5_utils import (
    _first_leaf_length,
    _flatten_group_step,
    _read_rgb_frame,
    get_trajectory_group,
    list_h5_trajectories,
    load_env_meta,
    read_prompt,
)
from agent_factory.modules.actors.smolvla import (
    SMOLVLA_ACTION_KEY,
    SMOLVLA_IMAGE_PREFIX,
    SMOLVLA_STATE_KEY,
    SMOLVLA_TASK_KEY,
)


@dataclass(frozen=True)
class SmolVLASampleRef:
    task: str
    traj_key: str
    start_idx: int


def _ordered_keys(mapping: Dict[str, Any]) -> List[str]:
    return list(mapping.keys())


def _flatten_group_all(group: h5py.Group, keys: Sequence[str]) -> np.ndarray:
    arrays = []
    for key in keys:
        data = np.asarray(group[key][()])
        arrays.append(data.reshape(data.shape[0], -1) if data.ndim > 2 else data)
    if not arrays:
        raise KeyError(f"No datasets found under H5 group {group.name}.")
    return np.concatenate(arrays, axis=-1).astype(np.float32, copy=False)


def _flatten_group_range(
    group: h5py.Group,
    keys: Sequence[str],
    start_idx: int,
    chunk_size: int,
    traj_len: int,
) -> np.ndarray:
    end_idx = min(int(start_idx) + int(chunk_size), int(traj_len))
    arrays = []
    for key in keys:
        data = np.asarray(group[key][start_idx:end_idx])
        arrays.append(data.reshape(data.shape[0], -1) if data.ndim > 2 else data)
    if not arrays:
        raise KeyError(f"No datasets found under H5 group {group.name}.")
    chunk = np.concatenate(arrays, axis=-1).astype(np.float32, copy=False)
    if chunk.shape[0] < chunk_size:
        pad_len = int(chunk_size) - int(chunk.shape[0])
        pad_value = chunk[-1:] if chunk.shape[0] else np.zeros((1, 0), dtype=np.float32)
        chunk = np.concatenate([chunk, np.repeat(pad_value, pad_len, axis=0)], axis=0)
    return chunk


def _task_label(traj_group: h5py.Group, prompt: str) -> str:
    if "input_label" in traj_group.attrs:
        value = traj_group.attrs["input_label"]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)
    lowered = prompt.lower()
    for token in ("can", "square", "stack"):
        if token in lowered:
            return token
    return prompt.strip() or "unknown"


class SmolVLAMultitaskH5Dataset(Dataset):
    """
    Direct Robosuite H5 dataset for agent_factory SmolVLA fine-tuning.

    Each item is one observation frame plus a chunk of future actions:
        observation.state: [state_dim]
        observation.images.<camera>: [3,H,W] float32 in [0,1]
        task: prompt string
        action: [chunk_size, action_dim]
        action_is_pad: [chunk_size]

    The index space is task-balanced by construction. If one task has fewer
    valid start frames, it is deterministically oversampled to match the largest
    task bucket.
    """

    def __init__(
        self,
        cfg: Any,
        h5_path: str,
        required_keys: Optional[Sequence[str]] = None,
    ):
        del required_keys
        self.cfg = cfg
        self.h5_path = str(h5_path)
        self.chunk_size = int(getattr(cfg.env, "pred_horizon", 50))
        if self.chunk_size != 50:
            raise ValueError(
                "SmolVLA H5 dataset requires env.pred_horizon == 50 for the first implementation."
            )

        actor_cfg = getattr(cfg, "actor", None)
        self.image_keys = list(getattr(actor_cfg, "image_keys", []) or [])
        self._h5: Optional[h5py.File] = None

        self.traj_infos: Dict[str, Dict[str, Any]] = {}
        self.task_to_samples: Dict[str, List[SmolVLASampleRef]] = {}
        self.samples: List[SmolVLASampleRef] = []
        self._all_states: Optional[torch.Tensor] = None
        self._all_actions: Optional[torch.Tensor] = None

        self._index_file()

    def _open_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    def __del__(self):
        h5_file = getattr(self, "_h5", None)
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception:
                pass

    def _index_file(self) -> None:
        state_chunks = []
        action_chunks = []
        with h5py.File(self.h5_path, "r") as h5_file:
            traj_keys = list_h5_trajectories(h5_file)
            if traj_keys == [None]:
                raise ValueError(f"SmolVLA requires traj_* groups in {self.h5_path}.")
            max_traj = getattr(self.cfg.dataset.expert, "num_traj", None)
            if max_traj is not None:
                traj_keys = traj_keys[: int(max_traj)]

            for traj_key in traj_keys:
                traj_group = get_trajectory_group(h5_file, traj_key)
                env_meta = load_env_meta(h5_file, traj_group)
                prompt = read_prompt(traj_group)
                task = _task_label(traj_group, prompt)

                obs_group = traj_group["obs"]
                action_group = traj_group["action"]
                state_keys = _ordered_keys(env_meta["obs"].get("state", {}))
                action_keys = _ordered_keys(env_meta.get("action", {}))
                camera_keys = self.image_keys or _ordered_keys(env_meta["obs"].get("rgb", {}))
                if not camera_keys:
                    raise ValueError(f"Trajectory {traj_key} has no RGB cameras for SmolVLA.")
                traj_len = _first_leaf_length(action_group, action_keys)
                if traj_len <= 0:
                    continue

                traj_key_str = str(traj_key)
                self.traj_infos[traj_key_str] = {
                    "prompt": prompt,
                    "task": task,
                    "state_keys": state_keys,
                    "action_keys": action_keys,
                    "camera_keys": camera_keys,
                    "length": traj_len,
                }
                bucket = self.task_to_samples.setdefault(task, [])
                for start_idx in range(traj_len):
                    bucket.append(SmolVLASampleRef(task=task, traj_key=traj_key_str, start_idx=start_idx))

                state_arr = _flatten_group_all(obs_group["state"], state_keys)
                action_arr = _flatten_group_all(action_group, action_keys)
                state_chunks.append(torch.from_numpy(state_arr.astype(np.float32, copy=False)))
                action_chunks.append(torch.from_numpy(action_arr.astype(np.float32, copy=False)))

        if not self.task_to_samples:
            raise ValueError(f"No SmolVLA samples found in {self.h5_path}.")

        task_names = sorted(self.task_to_samples.keys())
        max_count = max(len(self.task_to_samples[task]) for task in task_names)
        for sample_idx in range(max_count):
            for task in task_names:
                bucket = self.task_to_samples[task]
                self.samples.append(bucket[sample_idx % len(bucket)])

        self._all_states = torch.cat(state_chunks, dim=0) if state_chunks else None
        self._all_actions = torch.cat(action_chunks, dim=0) if action_chunks else None

    def __len__(self) -> int:
        return len(self.samples)

    def _action_chunk(self, action_group: h5py.Group, action_keys: Sequence[str], start_idx: int, traj_len: int):
        action = _flatten_group_range(action_group, action_keys, start_idx, self.chunk_size, traj_len)
        valid_len = max(0, min(int(traj_len) - int(start_idx), self.chunk_size))
        is_pad = np.arange(self.chunk_size) >= valid_len
        return (
            torch.from_numpy(action.astype(np.float32, copy=False)),
            torch.from_numpy(is_pad.astype(np.bool_, copy=False)),
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ref = self.samples[int(index)]
        h5_file = self._open_h5()
        traj_group = get_trajectory_group(h5_file, ref.traj_key)
        info = self.traj_infos[ref.traj_key]
        obs_group = traj_group["obs"]
        action_group = traj_group["action"]

        state = _flatten_group_step(obs_group["state"], info["state_keys"], ref.start_idx)
        action, action_is_pad = self._action_chunk(
            action_group,
            info["action_keys"],
            ref.start_idx,
            int(info["length"]),
        )

        item: Dict[str, Any] = {
            SMOLVLA_STATE_KEY: torch.from_numpy(state.astype(np.float32, copy=False)),
            SMOLVLA_ACTION_KEY: action,
            "action_is_pad": action_is_pad,
            SMOLVLA_TASK_KEY: str(info["prompt"]),
            "task_label": str(info["task"]),
        }
        for camera_name in info["camera_keys"]:
            image = _read_rgb_frame(obs_group["rgb"][camera_name], ref.start_idx)
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32, copy=False)
            item[f"{SMOLVLA_IMAGE_PREFIX}{camera_name}"] = torch.from_numpy(image)
        return item

    def get_all_states(self) -> Optional[torch.Tensor]:
        return self._all_states

    def get_all_actions(self) -> Optional[torch.Tensor]:
        return self._all_actions

    def task_counts(self) -> Dict[str, int]:
        return {task: len(samples) for task, samples in self.task_to_samples.items()}

    def balanced_task_counts(self) -> Dict[str, int]:
        counts = {task: 0 for task in self.task_to_samples}
        for sample in self.samples:
            counts[sample.task] += 1
        return counts
