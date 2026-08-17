from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch

from agent_factory.data.base import BaseTrajectoryDataset


class TDQCFeatureDataset(BaseTrajectoryDataset):
    """
    H5-backed TDQC feature dataset.

    Expected schema:
        traj_xxx/step_features: [T, D]
        traj_xxx/success_label: scalar or [1]

    Optional:
        traj_xxx/valid_mask: [T]
        traj_xxx/frame: [T]

    The dataset returns one unpadded rollout per item. Use tdqc_collate_fn to
    batch variable-length rollouts with padding and valid_mask.
    """

    def __init__(
        self,
        cfg,
        h5_path: Optional[str] = None,
        folder_path: Optional[str] = None,
        required_keys: Optional[Sequence[str]] = None,
        num_traj: Optional[int] = None,
    ):
        self.cfg = cfg
        super().__init__(
            h5_path=h5_path,
            folder_path=folder_path,
            num_traj=num_traj,
        )
        if not self.trajectory_refs:
            raise ValueError("No TDQC feature trajectories found.")

        self.available_keys = {"step_features", "valid_mask", "success_label", "frame", "traj_id"}
        self.required_keys = set(required_keys) if required_keys is not None else {
            "step_features",
            "valid_mask",
            "success_label",
            "frame",
        }
        unknown = self.required_keys - self.available_keys
        if unknown:
            raise ValueError(f"Unsupported TDQCFeatureDataset keys: {sorted(unknown)}")

        self.feature_dim = self._infer_feature_dim()

    @staticmethod
    def _read_scalar(group: h5py.Group, key: str) -> float:
        if key in group:
            value = group[key][()]
        elif key in group.attrs:
            value = group.attrs[key]
        else:
            raise KeyError(f"TDQC trajectory {group.name} requires '{key}' dataset or attribute.")
        array = np.asarray(value)
        if array.shape == ():
            return float(array.item())
        return float(array.reshape(-1)[0])

    def _infer_feature_dim(self) -> int:
        group = self.get_trajectory_group(self.trajectory_refs[0])
        if "step_features" not in group:
            raise KeyError(f"TDQC trajectory {group.name} requires 'step_features'.")
        shape = group["step_features"].shape
        if len(shape) != 2:
            raise ValueError(f"TDQC step_features must have shape [T,D], got {shape} in {group.name}")
        return int(shape[-1])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ref = self.trajectory_refs[index]
        group = self.get_trajectory_group(ref)
        if "step_features" not in group:
            raise KeyError(f"TDQC trajectory {group.name} requires 'step_features'.")

        features = np.asarray(group["step_features"][()], dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"TDQC step_features must have shape [T,D], got {features.shape} in {group.name}")
        seq_len = features.shape[0]

        data: Dict[str, Any] = {}
        if "step_features" in self.required_keys:
            data["step_features"] = torch.from_numpy(features)
        if "valid_mask" in self.required_keys:
            if "valid_mask" in group:
                valid_mask = np.asarray(group["valid_mask"][()], dtype=np.float32).reshape(-1)
                if valid_mask.shape[0] != seq_len:
                    raise ValueError(
                        f"valid_mask length {valid_mask.shape[0]} does not match step_features length {seq_len} "
                        f"in {group.name}"
                    )
            else:
                valid_mask = np.ones(seq_len, dtype=np.float32)
            data["valid_mask"] = torch.from_numpy(valid_mask)
        if "success_label" in self.required_keys:
            data["success_label"] = torch.tensor(self._read_scalar(group, "success_label"), dtype=torch.float32)
        if "frame" in self.required_keys:
            if "frame" in group:
                frame = np.asarray(group["frame"][()], dtype=np.int64).reshape(-1)
                if frame.shape[0] != seq_len:
                    raise ValueError(
                        f"frame length {frame.shape[0]} does not match step_features length {seq_len} in {group.name}"
                    )
            else:
                frame = np.arange(seq_len, dtype=np.int64)
            data["frame"] = torch.from_numpy(frame)
        if "traj_id" in self.required_keys:
            data["traj_id"] = str(ref.traj_key if ref.traj_key is not None else ref.file_path)
        return data


def tdqc_collate_fn(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        raise ValueError("tdqc_collate_fn received an empty batch.")

    batch: Dict[str, Any] = {}
    if "step_features" in items[0]:
        feature_dim = int(items[0]["step_features"].shape[-1])
        max_len = max(int(item["step_features"].shape[0]) for item in items)
        padded = torch.zeros((len(items), max_len, feature_dim), dtype=torch.float32)
        default_mask = torch.zeros((len(items), max_len), dtype=torch.float32)
        for idx, item in enumerate(items):
            features = item["step_features"].float()
            seq_len = int(features.shape[0])
            padded[idx, :seq_len] = features
            default_mask[idx, :seq_len] = 1.0
        batch["step_features"] = padded
    else:
        max_len = max(int(item["valid_mask"].shape[0]) for item in items if "valid_mask" in item)
        default_mask = torch.zeros((len(items), max_len), dtype=torch.float32)

    if "valid_mask" in items[0]:
        valid_mask = torch.zeros((len(items), max_len), dtype=torch.float32)
        for idx, item in enumerate(items):
            mask = item["valid_mask"].float().reshape(-1)
            valid_mask[idx, : mask.shape[0]] = mask
        batch["valid_mask"] = valid_mask
    else:
        batch["valid_mask"] = default_mask

    if "success_label" in items[0]:
        batch["success_label"] = torch.stack([item["success_label"].float().reshape(()) for item in items], dim=0)

    if "frame" in items[0]:
        frame = torch.full((len(items), max_len), -1, dtype=torch.long)
        for idx, item in enumerate(items):
            values = item["frame"].long().reshape(-1)
            frame[idx, : values.shape[0]] = values
        batch["frame"] = frame

    if "traj_id" in items[0]:
        batch["traj_id"] = [item["traj_id"] for item in items]

    if "task_name" in items[0]:
        batch["task_name"] = [item["task_name"] for item in items]

    if "task_label" in items[0]:
        batch["task_label"] = torch.stack([item["task_label"].long().reshape(()) for item in items], dim=0)

    if "source_path" in items[0]:
        batch["source_path"] = [item["source_path"] for item in items]

    if "source_traj_key" in items[0]:
        batch["source_traj_key"] = [item["source_traj_key"] for item in items]

    if "feature_length" in items[0]:
        batch["feature_length"] = torch.stack([item["feature_length"].long().reshape(()) for item in items], dim=0)

    return batch
