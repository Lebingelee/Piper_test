import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
import yaml

from agent_factory.data.base import BaseTrajectoryDataset, TrajectoryRef, traj_sort_key
from agent_factory.data.impl.cpiql.common import (
    compute_n_step_progress_signals,
    compute_pad_vec,
    compute_progress_gammas,
    compute_progress_returns,
)


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


def _read_json_dataset(dataset: h5py.Dataset) -> Dict[str, Any]:
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return json.loads(value)


class VLAFeatureDataset(BaseTrajectoryDataset):
    """
    H5-backed VLA prefix-feature dataset for feature-based risk models.

    Expected schema:
        traj_xxx/ or root trajectory/
          obs/feature/state_token: [T, D]
          obs/feature/prefix_valid_mask: [T, L]
          obs/feature/state_token_index: [T]
          meta/env_meta: JSON with task_name
          success: [T'] or scalar

    Base items follow the CPIQL transition/window contract. Observations are
    explicit VLA feature observations:
        observations["feature"]: [obs_horizon, D]
        next_observations["feature"]: [obs_horizon, D]
    """

    _DEFAULT_REQUIRED_KEYS = {
        "observations",
        "next_observations",
        "action",
        "reward",
        "discount",
        "value",
        "terminated",
        "success",
        "progress_return",
        "progress_mask",
        "progress_weight",
        "failure_rank",
        "is_success_segment",
        "is_failure_segment",
        "segment_end_is_intervention_boundary",
        "step_features",
        "valid_mask",
        "success_label",
        "frame",
        "task_name",
        "task_label",
        "traj_id",
    }

    def __init__(
        self,
        cfg,
        h5_path: Optional[str] = None,
        folder_path: Optional[str] = None,
        required_keys: Optional[Sequence[str]] = None,
        num_traj: Optional[int] = None,
        split: str = "all",
        source_role: str = "rollouts",
        pooling: Optional[str] = None,
    ):
        self.cfg = cfg
        self.obs_horizon = int(cfg_get(cfg, "env.obs_horizon", 1))
        self.pred_horizon = int(cfg_get(cfg, "env.pred_horizon", 1))
        self.act_horizon = int(cfg_get(cfg, "env.act_horizon", 1))
        self.split = str(split or "all").strip().lower()
        self.source_role = str(source_role or "rollouts").strip().lower()
        self.pooling = str(pooling or cfg_get(cfg, "dataset.config.pooling", "state_token")).strip().lower()
        self.skip_bad_files = bool(cfg_get(cfg, "dataset.config.skip_bad_files", False))
        self.gamma_floor = float(cfg_get(cfg, "critic.gamma_floor", 1e-4))
        self.gamma_default_margin = float(cfg_get(cfg, "critic.gamma_default_margin", 2.0))
        if self.pooling != "state_token":
            raise ValueError(
                f"Unsupported VLA feature pooling={self.pooling!r}. "
                "Only 'state_token' is available for the current state-only feature H5 files."
            )

        super().__init__(h5_path=h5_path, folder_path=folder_path, num_traj=num_traj)
        if not self.trajectory_refs:
            raise ValueError("No VLA feature trajectories found.")

        self.available_keys = set(self._DEFAULT_REQUIRED_KEYS) | {
            "source_path",
            "source_traj_key",
            "feature_length",
            "truncated",
            "segment_terminal_reward",
            "segment_type",
        }
        self.required_keys = set(required_keys) if required_keys is not None else set(self._DEFAULT_REQUIRED_KEYS)
        unknown = self.required_keys - self.available_keys
        if unknown:
            raise ValueError(f"Unsupported VLAFeatureDataset keys: {sorted(unknown)}")

        self.trajectory_infos: List[Dict[str, Any]] = []
        self.task_name_list: List[str] = []
        self.task_to_label: Dict[str, int] = {}
        self.feature_dim = 0
        self._cache_trajectory_infos()
        self._split_indices = self._select_split_indices()
        self.slices_all = self._build_slices(self._split_indices)
        self.slices_success = [
            sl for sl in self.slices_all
            if self._is_success_side_segment(sl[0])
        ]
        self.slices = self.slices_all
        self.mode = "all"
        self._visible_tasks: Optional[set[str]] = None
        self._active_indices = list(range(len(self.slices)))

    def _collect_h5_files(self) -> List[str]:
        if not self.folder_path:
            return super()._collect_h5_files()
        folder = Path(self.folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"H5 folder not found: {self.folder_path}")
        paths = sorted(
            folder.rglob("*.h5"),
            key=lambda p: (str(p.parent), traj_sort_key(p.stem), p.name),
        )
        return [str(path) for path in paths]

    def _discover_trajectories(self) -> List[TrajectoryRef]:
        refs: List[TrajectoryRef] = []
        for file_path in self._collect_h5_files():
            try:
                with h5py.File(file_path, "r") as h5_file:
                    traj_keys = sorted(
                        [
                            key for key in h5_file.keys()
                            if key.startswith("traj_") and isinstance(h5_file[key], h5py.Group)
                        ],
                        key=traj_sort_key,
                    )
                    if traj_keys:
                        refs.extend(TrajectoryRef(file_path, key) for key in traj_keys)
                    else:
                        refs.append(TrajectoryRef(file_path, None))
            except OSError as exc:
                if not self.skip_bad_files:
                    raise OSError(f"Failed to open VLA feature H5 file {file_path}: {exc}") from exc
                warnings.warn(f"Skipping unreadable VLA feature H5 file {file_path}: {exc}")

        if self.num_traj is not None:
            refs = refs[: self.num_traj]
        return refs

    @staticmethod
    def _read_env_meta(h5_file: h5py.File, group: h5py.Group) -> Dict[str, Any]:
        if "meta" in group and isinstance(group["meta"], h5py.Group) and "env_meta" in group["meta"]:
            return _read_json_dataset(group["meta"]["env_meta"])
        if "meta" in h5_file and isinstance(h5_file["meta"], h5py.Group) and "env_meta" in h5_file["meta"]:
            return _read_json_dataset(h5_file["meta"]["env_meta"])
        return {}

    @staticmethod
    def _read_env_cfg(h5_file: h5py.File, group: h5py.Group) -> Dict[str, Any]:
        candidates = []
        if "meta" in group and isinstance(group["meta"], h5py.Group) and "env_cfg" in group["meta"]:
            candidates.append(group["meta"]["env_cfg"])
        if "meta" in h5_file and isinstance(h5_file["meta"], h5py.Group) and "env_cfg" in h5_file["meta"]:
            candidates.append(h5_file["meta"]["env_cfg"])
        for dataset in candidates:
            value = dataset[()]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
            if isinstance(value, str):
                loaded = yaml.safe_load(value) or {}
                return loaded if isinstance(loaded, dict) else {}
            if isinstance(value, dict):
                return value
        return {}

    @staticmethod
    def _read_final_success(group: h5py.Group) -> float:
        if "success" not in group:
            raise KeyError(f"VLA feature trajectory {group.name} requires 'success'.")
        values = np.asarray(group["success"][()])
        if values.shape == ():
            return float(bool(values.item()))
        flat = values.reshape(-1)
        if flat.size <= 0:
            raise ValueError(f"VLA feature trajectory {group.name} has an empty 'success' dataset.")
        return float(bool(flat[-1]))

    @staticmethod
    def _feature_group(group: h5py.Group) -> h5py.Group:
        if "obs" not in group or "feature" not in group["obs"]:
            raise KeyError(f"VLA feature trajectory {group.name} requires 'obs/feature'.")
        feature_group = group["obs"]["feature"]
        if "state_token" not in feature_group:
            raise KeyError(f"VLA feature trajectory {group.name} requires 'obs/feature/state_token'.")
        return feature_group

    @staticmethod
    def _ordered_action_keys(action_group: h5py.Group, env_meta: Dict[str, Any]) -> List[str]:
        action_meta = env_meta.get("action", {}) if isinstance(env_meta, dict) else {}
        if isinstance(action_meta, dict):
            keys = [key for key in sorted(action_meta.keys()) if key in action_group]
            if keys:
                return keys
        return sorted(action_group.keys())

    def _read_action_array(self, group: h5py.Group, env_meta: Dict[str, Any]) -> np.ndarray:
        if "action" not in group:
            raise KeyError(f"VLA feature trajectory {group.name} requires 'action'.")
        action_node = group["action"]
        if isinstance(action_node, h5py.Dataset):
            action = np.asarray(action_node[()], dtype=np.float32)
        else:
            arrays = []
            for key in self._ordered_action_keys(action_node, env_meta):
                data = np.asarray(action_node[key][()])
                arrays.append(data.reshape(data.shape[0], -1) if data.ndim > 2 else data)
            if not arrays:
                raise KeyError(f"No action datasets found under {action_node.name}.")
            action = np.concatenate(arrays, axis=-1).astype(np.float32)
        if action.ndim != 2:
            raise ValueError(f"VLA action must have shape [T,A], got {action.shape} in {group.name}")
        return action

    @staticmethod
    def _read_bool_array(group: h5py.Group, key: str, length: int, default: bool = False) -> np.ndarray:
        if key not in group:
            return np.full(length, bool(default), dtype=bool)
        values = np.asarray(group[key][()], dtype=bool).reshape(-1)
        if values.size < length:
            pad = np.full(length - values.size, bool(values[-1]) if values.size else bool(default), dtype=bool)
            values = np.concatenate([values, pad], axis=0)
        return values[:length]

    def _configured_task_max_episode_steps(self, task_name: str) -> Optional[int]:
        mapping = cfg_get(self.cfg, "dataset.config.task_max_episode_steps", {}) or {}
        if not hasattr(mapping, "items") and hasattr(mapping, "__dict__"):
            mapping = vars(mapping)
        if not hasattr(mapping, "items"):
            return None
        task_name = str(task_name or "").strip()
        normalized = task_name.lower()
        for key, value in mapping.items():
            if str(key).strip() == task_name or str(key).strip().lower() == normalized:
                return int(value)
        return None

    def _max_episode_steps(
        self,
        h5_file: h5py.File,
        group: h5py.Group,
        env_meta: Dict[str, Any],
        task_name: str,
    ) -> int:
        env_cfg = self._read_env_cfg(h5_file, group)
        candidates = [
            env_cfg.get("max_episode_steps") if isinstance(env_cfg, dict) else None,
            env_meta.get("env_cfg", {}).get("max_episode_steps")
            if isinstance(env_meta.get("env_cfg", {}), dict) else None,
            self._configured_task_max_episode_steps(task_name),
        ]
        for value in candidates:
            if value is not None:
                return int(value)
        raise KeyError(
            f"VLA feature trajectory {group.name} requires meta/env_cfg.max_episode_steps "
            "or dataset.config.task_max_episode_steps[task_name] to compute per-task discount."
        )

    def _step_gamma_from_max_steps(self, max_episode_steps: int) -> float:
        max_steps = max(float(max_episode_steps), 1.0)
        gamma = 1.0 - self.gamma_default_margin / max_steps
        return max(float(self.gamma_floor), min(1.0, float(gamma)))

    def _trajectory_horizon(
        self,
        h5_file: h5py.File,
        group: h5py.Group,
        env_meta: Dict[str, Any],
        key: str,
        default: int,
    ) -> int:
        """Read a trajectory-local planning horizon, falling back to the run config."""
        env_cfg = self._read_env_cfg(h5_file, group)
        meta_env_cfg = env_meta.get("env_cfg", {}) if isinstance(env_meta, dict) else {}
        candidates = [
            env_cfg.get(key) if isinstance(env_cfg, dict) else None,
            meta_env_cfg.get(key) if isinstance(meta_env_cfg, dict) else None,
            default,
        ]
        for value in candidates:
            if value is not None and int(value) > 0:
                return int(value)
        raise ValueError(f"VLA feature trajectory {group.name} requires a positive {key}.")

    @staticmethod
    def _traj_id(ref: TrajectoryRef) -> str:
        if ref.traj_key is not None:
            return f"{Path(ref.file_path).name}:{ref.traj_key}"
        return Path(ref.file_path).as_posix()

    def _cache_trajectory_infos(self) -> None:
        task_names = set()
        feature_dim: Optional[int] = None
        for ref in self.trajectory_refs:
            with h5py.File(ref.file_path, "r") as h5_file:
                group = h5_file if ref.traj_key is None else h5_file[ref.traj_key]
                feature_group = self._feature_group(group)
                shape = feature_group["state_token"].shape
                if len(shape) != 2:
                    raise ValueError(
                        f"VLA state_token must have shape [T,D], got {shape} in {group.name}"
                    )
                if feature_dim is None:
                    feature_dim = int(shape[-1])
                elif int(shape[-1]) != feature_dim:
                    raise ValueError(
                        f"VLA feature dim mismatch: expected {feature_dim}, got {shape[-1]} in {group.name}"
                    )

                env_meta = self._read_env_meta(h5_file, group)
                task_name = str(env_meta.get("task_name", "") or "").strip()
                if not task_name:
                    raise KeyError(f"VLA feature trajectory {group.name} requires meta/env_meta.task_name.")
                action = self._read_action_array(group, env_meta)
                action_length = int(action.shape[0])
                if action_length <= 0:
                    raise ValueError(f"VLA feature trajectory {group.name} has empty action data.")
                if int(shape[0]) < action_length:
                    raise ValueError(
                        f"VLA feature length must be >= action length, got feature={shape[0]}, "
                        f"action={action_length} in {group.name}"
                    )
                success = self._read_bool_array(group, "success", action_length, default=False)
                terminated = self._read_bool_array(group, "terminated", action_length, default=False)
                truncated = self._read_bool_array(group, "truncated", action_length, default=False)
                is_success = bool(success[-1])
                if is_success:
                    success[-1] = True
                    terminated[-1] = True
                    terminal_reward = 1.0
                    segment_type = "success"
                else:
                    terminal_reward = 0.0
                    if not np.any(terminated | truncated):
                        truncated[-1] = True
                    segment_type = "failure"
                max_episode_steps = self._max_episode_steps(h5_file, group, env_meta, task_name)
                traj_act_horizon = self._trajectory_horizon(
                    h5_file, group, env_meta, "act_horizon", self.act_horizon
                )
                traj_pred_horizon = self._trajectory_horizon(
                    h5_file, group, env_meta, "pred_horizon", self.pred_horizon
                )
                gammas = compute_progress_gammas(
                    length=action_length,
                    max_episode_steps=max_episode_steps,
                    gamma_floor=self.gamma_floor,
                    terminal_idx=action_length - 1,
                    gamma=self._step_gamma_from_max_steps(max_episode_steps),
                    mode="constant",
                )
                rewards = np.zeros(action_length, dtype=np.float32)
                rewards[-1] = float(terminal_reward)
                progress_returns = compute_progress_returns(rewards, gammas)
                task_names.add(task_name)
                self.trajectory_infos.append(
                    {
                        "ref": ref,
                        "traj_id": self._traj_id(ref),
                        "task_name": task_name,
                        "success_label": float(is_success),
                        "feature_length": int(shape[0]),
                        "action_length": action_length,
                        "action_dim": int(action.shape[-1]),
                        "max_episode_steps": max_episode_steps,
                        "act_horizon": traj_act_horizon,
                        "pred_horizon": traj_pred_horizon,
                        "success": success,
                        "terminated": terminated,
                        "truncated": truncated,
                        "rewards": rewards,
                        "gammas": gammas,
                        "progress_returns": progress_returns,
                        "progress_masks": np.ones(action_length, dtype=np.float32),
                        "progress_weights": np.ones(action_length, dtype=np.float32),
                        "terminal_reward": float(terminal_reward),
                        "segment_type": segment_type,
                    }
                )

        self.feature_dim = int(feature_dim or 0)
        self.task_name_list = sorted(task_names)
        self.task_to_label = {task: idx for idx, task in enumerate(self.task_name_list)}

    def _build_slices(self, trajectory_indices: Sequence[int]) -> List[tuple[int, int, int, int]]:
        slices: List[tuple[int, int, int, int]] = []
        global_idx = 0
        for traj_index in trajectory_indices:
            length = int(self.trajectory_infos[traj_index]["action_length"])
            pad_before = self.obs_horizon - 1
            for start in range(-pad_before, length):
                slices.append((traj_index, start, start + self.pred_horizon, global_idx))
                global_idx += 1
        return slices

    def _select_split_indices(self) -> List[int]:
        split = self.split
        if split in {"all", "*"}:
            return list(range(len(self.trajectory_infos)))
        if split in {"calib", "calibration"}:
            split = "calibration"
        if split not in {"train", "calibration", "test"}:
            raise ValueError("VLA feature split must be one of train, calibration, test, or all.")

        if self.source_role in {"demo", "demos", "expert", "expert_dataset"}:
            return list(range(len(self.trajectory_infos))) if split == "train" else []

        selected: List[int] = []
        for task_name in self.task_name_list:
            task_indices = [
                idx for idx, info in enumerate(self.trajectory_infos)
                if info["task_name"] == task_name
            ]
            train_indices, calibration_indices, test_indices = np.array_split(np.asarray(task_indices), 3)
            if split == "train":
                selected.extend(int(idx) for idx in train_indices.tolist())
            elif split == "calibration":
                selected.extend(int(idx) for idx in calibration_indices.tolist())
            else:
                selected.extend(int(idx) for idx in test_indices.tolist())
        return selected

    def see_task(self, task_name_list: Sequence[str]):
        """
        Restrict visible samples to trajectories whose task_name is in task_name_list.
        """
        visible = {str(task).strip() for task in task_name_list if str(task).strip()}
        self._visible_tasks = visible
        self._active_indices = [
            idx for idx, sl in enumerate(self.slices)
            if self.trajectory_infos[sl[0]]["task_name"] in visible
        ]
        return self

    def reset_seen_tasks(self):
        self._visible_tasks = None
        self._active_indices = list(range(len(self.slices)))
        return self

    def switch(self, mode: str = "all"):
        mode = str(mode or "all").strip().lower()
        if mode not in {"all", "success"}:
            raise ValueError("VLAFeatureDataset.switch mode must be 'all' or 'success'.")
        self.mode = mode
        self.slices = self.slices_all if mode == "all" else self.slices_success
        if self._visible_tasks is None:
            self._active_indices = list(range(len(self.slices)))
        else:
            self.see_task(sorted(self._visible_tasks))

    def _is_success_side_segment(self, traj_idx: int) -> bool:
        return bool(self.trajectory_infos[traj_idx]["success_label"])

    def _is_failure_side_segment(self, traj_idx: int) -> bool:
        return not self._is_success_side_segment(traj_idx)

    def get_all_actions(self):
        actions = []
        for info in self.trajectory_infos:
            ref = info["ref"]
            with h5py.File(ref.file_path, "r") as h5_file:
                group = h5_file if ref.traj_key is None else h5_file[ref.traj_key]
                env_meta = self._read_env_meta(h5_file, group)
                actions.append(torch.from_numpy(self._read_action_array(group, env_meta)).float())
        return torch.cat(actions, dim=0) if actions else None

    def _feature_window(self, features: np.ndarray, start_idx: int, horizon: int) -> torch.Tensor:
        length = int(features.shape[0])
        indices = [max(0, min(start_idx - (horizon - 1) + i, length - 1)) for i in range(horizon)]
        return torch.from_numpy(features[indices]).float()

    def _load_features_and_action(self, info: Dict[str, Any]):
        ref = info["ref"]
        group = self.get_trajectory_group(ref)
        feature_group = self._feature_group(group)
        features = np.asarray(feature_group["state_token"][()], dtype=np.float32)
        env_meta = self._read_env_meta(self._get_handle(ref.file_path), group)
        action = torch.from_numpy(self._read_action_array(group, env_meta)).float()
        return features, action, group

    def __len__(self):
        return len(self._active_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        traj_idx, start, end, global_idx = self.slices[self._active_indices[index]]
        info = self.trajectory_infos[traj_idx]
        features, action_data, group = self._load_features_and_action(info)
        if features.ndim != 2:
            raise ValueError(f"VLA state_token must have shape [T,D], got {features.shape} in {group.name}")

        length = int(info["action_length"])
        idx = min(max(0, start), length - 1)
        next_idx = min(idx + self.act_horizon, length - 1)
        terminal_flags = info["success"] | info["terminated"] | info["truncated"]
        reward, discount, terminated = compute_n_step_progress_signals(
            info["rewards"],
            info["gammas"],
            terminal_flags,
            idx,
            self.act_horizon,
        )
        data: Dict[str, Any] = {}
        if "observations" in self.required_keys:
            data["observations"] = {"feature": self._feature_window(features, start, self.obs_horizon)}
        if "next_observations" in self.required_keys:
            data["next_observations"] = {
                "feature": self._feature_window(features, min(start + self.act_horizon, length), self.obs_horizon)
            }
        if "action" in self.required_keys:
            act_seq = action_data[max(0, start):end]
            if start < 0:
                act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
            if len(act_seq) < self.pred_horizon:
                pad_vec = compute_pad_vec(act_seq, is_abs_mode=False, pad_action_arm=None)
                act_seq = torch.cat(
                    [act_seq, pad_vec.unsqueeze(0).repeat(self.pred_horizon - len(act_seq), 1)],
                    dim=0,
                )
            data["action"] = act_seq
        if "reward" in self.required_keys:
            data["reward"] = reward.reshape(1)
        if "discount" in self.required_keys:
            data["discount"] = torch.tensor([discount], dtype=torch.float32)
        if "value" in self.required_keys:
            data["value"] = torch.tensor([info["progress_returns"][idx]], dtype=torch.float32)
        if "terminated" in self.required_keys:
            data["terminated"] = torch.tensor([terminated], dtype=torch.float32)
        if "truncated" in self.required_keys:
            data["truncated"] = torch.tensor([info["truncated"][next_idx]], dtype=torch.float32)
        if "success" in self.required_keys:
            data["success"] = torch.tensor([info["success"][next_idx]], dtype=torch.float32)
        if "progress_return" in self.required_keys:
            data["progress_return"] = torch.tensor([info["progress_returns"][idx]], dtype=torch.float32)
        if "progress_mask" in self.required_keys:
            data["progress_mask"] = torch.tensor([info["progress_masks"][idx]], dtype=torch.float32)
        if "progress_weight" in self.required_keys:
            data["progress_weight"] = torch.tensor([info["progress_weights"][idx]], dtype=torch.float32)
        if "failure_rank" in self.required_keys:
            data["failure_rank"] = torch.tensor([1.0], dtype=torch.float32)
        if "is_success_segment" in self.required_keys:
            data["is_success_segment"] = torch.tensor([self._is_success_side_segment(traj_idx)], dtype=torch.float32)
        if "is_failure_segment" in self.required_keys:
            data["is_failure_segment"] = torch.tensor([self._is_failure_side_segment(traj_idx)], dtype=torch.float32)
        if "segment_end_is_intervention_boundary" in self.required_keys:
            data["segment_end_is_intervention_boundary"] = torch.zeros(1, dtype=torch.float32)
        if "segment_terminal_reward" in self.required_keys:
            data["segment_terminal_reward"] = torch.tensor([info["terminal_reward"]], dtype=torch.float32)
        if "segment_type" in self.required_keys:
            data["segment_type"] = torch.tensor([1 if info["segment_type"] == "success" else 0], dtype=torch.long)
        if "step_features" in self.required_keys:
            data["step_features"] = torch.from_numpy(features[idx:idx + 1]).float()
        if "valid_mask" in self.required_keys:
            data["valid_mask"] = torch.ones(1, dtype=torch.float32)
        if "success_label" in self.required_keys:
            data["success_label"] = torch.tensor(float(info["success_label"]), dtype=torch.float32)
        if "frame" in self.required_keys:
            data["frame"] = torch.tensor([idx], dtype=torch.long)
        if "task_name" in self.required_keys:
            data["task_name"] = info["task_name"]
        if "task_label" in self.required_keys:
            data["task_label"] = torch.tensor(self.task_to_label[info["task_name"]], dtype=torch.long)
        if "traj_id" in self.required_keys:
            data["traj_id"] = info["traj_id"]
        if "source_path" in self.required_keys:
            data["source_path"] = info["ref"].file_path
        if "source_traj_key" in self.required_keys:
            data["source_traj_key"] = info["ref"].traj_key or ""
        if "feature_length" in self.required_keys:
            data["feature_length"] = torch.tensor(int(info["feature_length"]), dtype=torch.long)
        return data


class VLAFeatureSequenceDataset(VLAFeatureDataset):
    """
    RNN-oriented VLA feature pipeline.

    Each item is one full trajectory sequence with shape [T, D].
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_trajectory_indices = list(self._split_indices)

    def see_task(self, task_name_list: Sequence[str]):
        visible = {str(task).strip() for task in task_name_list if str(task).strip()}
        self._visible_tasks = visible
        self._active_trajectory_indices = [
            idx for idx in self._split_indices
            if self.trajectory_infos[idx]["task_name"] in visible
        ]
        return self

    def reset_seen_tasks(self):
        self._visible_tasks = None
        self._active_trajectory_indices = list(self._split_indices)
        return self

    def switch(self, mode: str = "all"):
        if str(mode or "all").strip().lower() not in {"all", "success"}:
            raise ValueError("VLAFeatureSequenceDataset.switch mode must be 'all' or 'success'.")
        self.mode = str(mode or "all").strip().lower()
        if self.mode == "success":
            self._split_indices = [
                idx for idx in self._select_split_indices()
                if self._is_success_side_segment(idx)
            ]
        else:
            self._split_indices = self._select_split_indices()
        return self.reset_seen_tasks()

    def __len__(self):
        return len(self._active_trajectory_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        traj_idx = self._active_trajectory_indices[index]
        info = self.trajectory_infos[traj_idx]
        features, _action_data, _group = self._load_features_and_action(info)
        seq_len = int(info["feature_length"])

        data: Dict[str, Any] = {}
        if "step_features" in self.required_keys:
            data["step_features"] = torch.from_numpy(features).float()
        if "valid_mask" in self.required_keys:
            data["valid_mask"] = torch.ones(seq_len, dtype=torch.float32)
        if "success_label" in self.required_keys:
            data["success_label"] = torch.tensor(float(info["success_label"]), dtype=torch.float32)
        if "frame" in self.required_keys:
            data["frame"] = torch.arange(seq_len, dtype=torch.long)
        if "task_name" in self.required_keys:
            data["task_name"] = info["task_name"]
        if "task_label" in self.required_keys:
            data["task_label"] = torch.tensor(self.task_to_label[info["task_name"]], dtype=torch.long)
        if "traj_id" in self.required_keys:
            data["traj_id"] = info["traj_id"]
        if "source_path" in self.required_keys:
            data["source_path"] = info["ref"].file_path
        if "source_traj_key" in self.required_keys:
            data["source_traj_key"] = info["ref"].traj_key or ""
        if "feature_length" in self.required_keys:
            data["feature_length"] = torch.tensor(seq_len, dtype=torch.long)
        return data


class VLAFeatureTransitionDataset(VLAFeatureDataset):
    """
    MLP-oriented VLA feature pipeline.

    Each item is one visible transition/timestep with shape [1, D]. The label is
    still the trajectory-level final-frame success, per the current contract.
    """
    pass


class VLAFeatureBlockSequenceDataset(VLAFeatureDataset):
    """
    Block-aligned recurrent VLA feature dataset.

    Each item is anchored at a raw timestep ``t`` and represents
    ``(x_{t-(L-1)H}, A_hist, ..., x_t, A_q_t, x_{t+H})``. ``H`` is the
    trajectory-local act_horizon. Historical actions have length ``H`` while
    the Q-conditioning action has length ``pred_horizon``. Both are padded to
    the largest horizon present in this dataset and carry explicit masks.

    ``observations`` and ``next_observations`` contain state histories with
    shape [max_seq_len, obs_horizon, feature_dim]. Valid states are stored as a
    prefix so pack_padded_sequence can consume valid_mask directly.
    """

    _BLOCK_KEYS = {
        "history_actions",
        "history_action_valid_mask",
        "q_action",
        "q_action_valid_mask",
        "next_history_actions",
        "next_history_action_valid_mask",
        "next_valid_mask",
        "history_frame",
        "bootstrap_frame",
        "block_act_horizon",
        "block_pred_horizon",
    }
    _DEFAULT_REQUIRED_KEYS = VLAFeatureDataset._DEFAULT_REQUIRED_KEYS | _BLOCK_KEYS

    def __init__(self, *args, **kwargs):
        self.max_seq_len = int(cfg_get(kwargs.get("cfg", args[0] if args else None), "dataset.config.rnn_max_seq_len", 10))
        if self.max_seq_len <= 0:
            raise ValueError("dataset.config.rnn_max_seq_len must be positive.")
        super().__init__(*args, **kwargs)
        # Expert and rollout datasets are constructed independently and then
        # concatenated by the registry. Keep their dense action shapes stable
        # by treating the run-level horizons as lower bounds for padding.
        self.max_act_horizon = max(
            self.act_horizon,
            max(int(info["act_horizon"]) for info in self.trajectory_infos),
        )
        self.max_pred_horizon = max(
            self.pred_horizon,
            max(int(info["pred_horizon"]) for info in self.trajectory_infos),
        )

    def _build_slices(self, trajectory_indices: Sequence[int]) -> List[tuple[int, int]]:
        # Every raw anchor is valid. Adjacent anchors retain the same block
        # spacing but start at a different phase, which preserves frame-level
        # score export compatibility.
        return [
            (traj_idx, anchor)
            for traj_idx in trajectory_indices
            for anchor in range(int(self.trajectory_infos[traj_idx]["action_length"]))
        ]

    def _history_indices(self, anchor: int, act_horizon: int) -> List[int]:
        count = min(self.max_seq_len, anchor // act_horizon + 1)
        start = anchor - (count - 1) * act_horizon
        return [start + offset * act_horizon for offset in range(count)]

    @staticmethod
    def _padded_action_block(
        actions: torch.Tensor,
        start: int,
        requested_horizon: int,
        padded_horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if requested_horizon > padded_horizon:
            raise ValueError("requested_horizon cannot exceed padded_horizon.")
        action_dim = int(actions.shape[-1])
        block = torch.zeros((padded_horizon, action_dim), dtype=torch.float32)
        valid_mask = torch.zeros(padded_horizon, dtype=torch.float32)
        available = max(min(int(actions.shape[0]) - int(start), int(requested_horizon)), 0)
        if available <= 0:
            return block, valid_mask
        values = actions[start:start + available].float()
        block[:available] = values
        valid_mask[:available] = 1.0
        if available < requested_horizon:
            block[available:requested_horizon] = values[-1]
        return block, valid_mask

    def _build_block_context(
        self,
        features: np.ndarray,
        actions: torch.Tensor,
        anchor: int,
        act_horizon: int,
    ) -> Dict[str, torch.Tensor]:
        state_indices = self._history_indices(anchor, act_horizon)
        feature_shape = (self.max_seq_len, self.obs_horizon, int(features.shape[-1]))
        state_history = torch.zeros(feature_shape, dtype=torch.float32)
        state_mask = torch.zeros(self.max_seq_len, dtype=torch.float32)
        frame_history = torch.full((self.max_seq_len,), -1, dtype=torch.long)
        action_history = torch.zeros(
            (max(self.max_seq_len - 1, 0), self.max_act_horizon, int(actions.shape[-1])),
            dtype=torch.float32,
        )
        action_mask = torch.zeros(
            (max(self.max_seq_len - 1, 0), self.max_act_horizon), dtype=torch.float32
        )

        for pos, state_idx in enumerate(state_indices):
            state_history[pos] = self._feature_window(features, state_idx, self.obs_horizon)
            state_mask[pos] = 1.0
            frame_history[pos] = int(state_idx)
            if pos < len(state_indices) - 1:
                block, block_mask = self._padded_action_block(
                    actions, state_idx, act_horizon, self.max_act_horizon
                )
                action_history[pos] = block
                action_mask[pos] = block_mask

        return {
            "feature": state_history,
            "valid_mask": state_mask,
            "history_actions": action_history,
            "history_action_valid_mask": action_mask,
            "history_frame": frame_history,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        traj_idx, anchor = self.slices[self._active_indices[index]]
        info = self.trajectory_infos[traj_idx]
        features, action_data, group = self._load_features_and_action(info)
        if features.ndim != 2:
            raise ValueError(f"VLA state_token must have shape [T,D], got {features.shape} in {group.name}")

        length = int(info["action_length"])
        act_horizon = int(info["act_horizon"])
        pred_horizon = int(info["pred_horizon"])
        terminal_flags = info["success"] | info["terminated"] | info["truncated"]
        reward, discount, terminated = compute_n_step_progress_signals(
            info["rewards"], info["gammas"], terminal_flags, anchor, act_horizon
        )
        bootstrap_anchor = min(anchor + act_horizon, length - 1)
        context = self._build_block_context(features, action_data, anchor, act_horizon)
        bootstrap_context = self._build_block_context(
            features, action_data, bootstrap_anchor, act_horizon
        )
        q_action, q_action_mask = self._padded_action_block(
            action_data, anchor, pred_horizon, self.max_pred_horizon
        )
        next_idx = min(anchor + act_horizon, length - 1)

        data: Dict[str, Any] = {}
        if "observations" in self.required_keys:
            data["observations"] = {"feature": context["feature"]}
        if "next_observations" in self.required_keys:
            data["next_observations"] = {"feature": bootstrap_context["feature"]}
        if "history_actions" in self.required_keys:
            data["history_actions"] = context["history_actions"]
        if "history_action_valid_mask" in self.required_keys:
            data["history_action_valid_mask"] = context["history_action_valid_mask"]
        if "q_action" in self.required_keys:
            data["q_action"] = q_action
        if "q_action_valid_mask" in self.required_keys:
            data["q_action_valid_mask"] = q_action_mask
        if "next_history_actions" in self.required_keys:
            data["next_history_actions"] = bootstrap_context["history_actions"]
        if "next_history_action_valid_mask" in self.required_keys:
            data["next_history_action_valid_mask"] = bootstrap_context["history_action_valid_mask"]
        if "next_valid_mask" in self.required_keys:
            data["next_valid_mask"] = bootstrap_context["valid_mask"]
        if "action" in self.required_keys:
            data["action"] = q_action
        if "reward" in self.required_keys:
            data["reward"] = reward.reshape(1)
        if "discount" in self.required_keys:
            data["discount"] = torch.tensor([discount], dtype=torch.float32)
        if "value" in self.required_keys:
            data["value"] = torch.tensor([info["progress_returns"][anchor]], dtype=torch.float32)
        if "terminated" in self.required_keys:
            data["terminated"] = torch.tensor([terminated], dtype=torch.float32)
        if "truncated" in self.required_keys:
            data["truncated"] = torch.tensor([info["truncated"][next_idx]], dtype=torch.float32)
        if "success" in self.required_keys:
            data["success"] = torch.tensor([info["success"][next_idx]], dtype=torch.float32)
        if "progress_return" in self.required_keys:
            data["progress_return"] = torch.tensor([info["progress_returns"][anchor]], dtype=torch.float32)
        if "progress_mask" in self.required_keys:
            data["progress_mask"] = torch.tensor([info["progress_masks"][anchor]], dtype=torch.float32)
        if "progress_weight" in self.required_keys:
            data["progress_weight"] = torch.tensor([info["progress_weights"][anchor]], dtype=torch.float32)
        if "failure_rank" in self.required_keys:
            data["failure_rank"] = torch.tensor([1.0], dtype=torch.float32)
        if "is_success_segment" in self.required_keys:
            data["is_success_segment"] = torch.tensor([self._is_success_side_segment(traj_idx)], dtype=torch.float32)
        if "is_failure_segment" in self.required_keys:
            data["is_failure_segment"] = torch.tensor([self._is_failure_side_segment(traj_idx)], dtype=torch.float32)
        if "segment_end_is_intervention_boundary" in self.required_keys:
            data["segment_end_is_intervention_boundary"] = torch.zeros(1, dtype=torch.float32)
        if "step_features" in self.required_keys:
            data["step_features"] = context["feature"].reshape(self.max_seq_len, -1)
        if "valid_mask" in self.required_keys:
            data["valid_mask"] = context["valid_mask"]
        if "success_label" in self.required_keys:
            data["success_label"] = torch.tensor(float(info["success_label"]), dtype=torch.float32)
        if "frame" in self.required_keys:
            data["frame"] = context["history_frame"]
        if "history_frame" in self.required_keys:
            data["history_frame"] = context["history_frame"]
        if "bootstrap_frame" in self.required_keys:
            data["bootstrap_frame"] = torch.tensor(bootstrap_anchor, dtype=torch.long)
        if "block_act_horizon" in self.required_keys:
            data["block_act_horizon"] = torch.tensor(act_horizon, dtype=torch.long)
        if "block_pred_horizon" in self.required_keys:
            data["block_pred_horizon"] = torch.tensor(pred_horizon, dtype=torch.long)
        if "task_name" in self.required_keys:
            data["task_name"] = info["task_name"]
        if "task_label" in self.required_keys:
            data["task_label"] = torch.tensor(self.task_to_label[info["task_name"]], dtype=torch.long)
        if "traj_id" in self.required_keys:
            data["traj_id"] = info["traj_id"]
        if "source_path" in self.required_keys:
            data["source_path"] = info["ref"].file_path
        if "source_traj_key" in self.required_keys:
            data["source_traj_key"] = info["ref"].traj_key or ""
        if "feature_length" in self.required_keys:
            data["feature_length"] = torch.tensor(int(info["feature_length"]), dtype=torch.long)
        return data
