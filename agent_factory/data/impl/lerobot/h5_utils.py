"""Shared structured-H5 and LeRobotDataset conversion helpers."""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch


def _traj_sort_key(name: str):
    parts = name.split("_")
    for token in parts[1:]:
        try:
            return int(token)
        except ValueError:
            continue
    return name


def list_h5_trajectories(h5_file: h5py.File) -> List[Optional[str]]:
    traj_keys = [
        key
        for key in h5_file.keys()
        if key.startswith("traj_") and isinstance(h5_file[key], h5py.Group)
    ]
    return sorted(traj_keys, key=_traj_sort_key) or [None]


def get_trajectory_group(h5_file: h5py.File, traj_key: Optional[str]) -> h5py.Group:
    return h5_file if traj_key is None else h5_file[traj_key]


def _read_dataset_value(dataset: h5py.Dataset):
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_json_dataset(dataset: h5py.Dataset) -> Any:
    return json.loads(_read_dataset_value(dataset))


def load_env_meta(h5_file: h5py.File, traj_group: h5py.Group) -> Dict[str, Any]:
    if "meta" in traj_group and isinstance(traj_group["meta"], h5py.Group) and "env_meta" in traj_group["meta"]:
        return _read_json_dataset(traj_group["meta"]["env_meta"])
    if "meta" in h5_file and isinstance(h5_file["meta"], h5py.Group) and "env_meta" in h5_file["meta"]:
        return _read_json_dataset(h5_file["meta"]["env_meta"])
    raise KeyError(f"Trajectory {traj_group.name} does not contain traj/meta/env_meta or root meta/env_meta.")


def read_prompt(traj_group: h5py.Group, default_prompt: Optional[str] = None) -> str:
    if "prompt" in traj_group and isinstance(traj_group["prompt"], h5py.Dataset):
        value = _read_dataset_value(traj_group["prompt"])
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default_prompt or ""
            first = value.reshape(-1)[0]
            if isinstance(first, bytes):
                return first.decode("utf-8")
            return str(first)
        return str(value)
    if default_prompt is not None:
        return str(default_prompt)
    raise KeyError(f"Trajectory {traj_group.name} is missing prompt. Run merge_h5_converter first.")


def _ordered_keys(mapping: Dict[str, Any]) -> List[str]:
    return list(mapping.keys())


def _flatten_group_step(group: h5py.Group, keys: List[str], step_idx: int) -> np.ndarray:
    arrays = []
    for key in keys:
        if key not in group:
            raise KeyError(f"Missing key '{key}' under H5 group {group.name}.")
        arrays.append(np.asarray(group[key][step_idx]).reshape(-1))
    if not arrays:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32)


def _read_rgb_frame(dataset: h5py.Dataset, step_idx: int) -> np.ndarray:
    image = np.asarray(dataset[step_idx])
    if image.ndim != 3:
        raise ValueError(f"RGB dataset {dataset.name} must be rank-3 per frame, got shape {image.shape}.")
    if image.shape[0] == 3:
        return image
    if image.shape[-1] == 3:
        return np.transpose(image, (2, 0, 1))
    raise ValueError(
        f"RGB dataset {dataset.name} must be CHW or HWC with 3 channels, got per-frame shape {image.shape}."
    )


def _first_leaf_length(group: h5py.Group, keys: List[str]) -> int:
    for key in keys:
        if key in group and isinstance(group[key], h5py.Dataset):
            return int(group[key].shape[0])
    raise KeyError(f"No datasets found under {group.name} for keys {keys}.")


def _trajectory_success(traj_group: h5py.Group) -> bool:
    if "success" in traj_group and isinstance(traj_group["success"], h5py.Dataset):
        raw = np.asarray(traj_group["success"][()], dtype=np.bool_)
        if raw.shape == ():
            return bool(raw.item())
        return bool(raw.reshape(-1).any())
    if "success" in traj_group.attrs:
        return bool(traj_group.attrs["success"])
    return False


def _flat_feature_names(prefix: str, dim: int) -> List[str]:
    return [f"{prefix}_{idx}" for idx in range(int(dim))]


def _image_axis_names(shape) -> List[str]:
    shape = tuple(shape)
    if len(shape) != 3:
        return [f"dim_{idx}" for idx in range(len(shape))]
    if shape[0] in (1, 3, 4):
        return ["channels", "height", "width"]
    if shape[-1] in (1, 3, 4):
        return ["height", "width", "channels"]
    return ["dim_0", "dim_1", "dim_2"]


def build_lerobot_features(env_meta: Dict[str, Any], *, use_videos: bool = True) -> Dict[str, Dict[str, Any]]:
    state_dim = sum(int(np.prod(shape)) for shape in env_meta["obs"].get("state", {}).values())
    action_dim = sum(int(np.prod(shape)) for shape in env_meta.get("action", {}).values())
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": _flat_feature_names("state", state_dim),
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": _flat_feature_names("action", action_dim),
        },
    }

    image_dtype = "video" if use_videos else "image"
    for camera_name, shape in env_meta["obs"].get("rgb", {}).items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": image_dtype,
            "shape": tuple(shape),
            "names": _image_axis_names(shape),
        }
    return features


def _resolve_output_dir(output_dir: str, overwrite: bool) -> Path:
    path = Path(output_dir)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "LeRobot is required for H5 -> LeRobotDataset conversion. "
            "Use the Python 3.12 LeRobot environment from requirements-py312.txt."
        ) from exc
    return LeRobotDataset


def _make_rgb_encoder(video_codec: Optional[str], video_backend: Optional[str]):
    if not video_codec:
        return None
    try:
        from lerobot.configs.video import RGBEncoderConfig
    except ImportError as exc:
        raise ImportError("LeRobot RGBEncoderConfig is required when video_codec is set.") from exc
    kwargs: Dict[str, Any] = {"vcodec": video_codec}
    if video_backend:
        kwargs["video_backend"] = video_backend
    return RGBEncoderConfig(**kwargs)


def infer_lerobot_repo_id(root: str) -> str:
    root_path = Path(root)
    manifest_path = root_path / "h5_conversion_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        repo_id = manifest.get("repo_id")
        if repo_id:
            return str(repo_id)
    return root_path.name


def _set_local_hf_cache(root: str) -> None:
    cache_root = Path(root) / ".hf_cache"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))


def load_lerobot_policy_dataset(
    root: str,
    *,
    repo_id: Optional[str] = None,
    video_backend: Optional[str] = None,
    return_uint8: bool = True,
    **kwargs,
):
    _set_local_hf_cache(root)
    LeRobotDataset = _import_lerobot_dataset()
    return LeRobotDataset(
        repo_id=repo_id or infer_lerobot_repo_id(root),
        root=root,
        video_backend=video_backend,
        return_uint8=return_uint8,
        **kwargs,
    )


def convert_h5_to_lerobot_dataset(
    input_h5: str,
    output_dir: str,
    *,
    repo_id: Optional[str] = None,
    fps: int = 30,
    default_prompt: Optional[str] = None,
    use_videos: bool = True,
    video_backend: Optional[str] = None,
    video_codec: Optional[str] = None,
    max_episodes: Optional[int] = None,
    max_frames_per_episode: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    LeRobotDataset = _import_lerobot_dataset()
    input_path = Path(input_h5)
    if not input_path.exists():
        raise FileNotFoundError(f"Input H5 does not exist: {input_h5}")

    output_path = _resolve_output_dir(output_dir, overwrite=overwrite)
    repo_id = repo_id or output_path.name

    episode_info: List[Dict[str, Any]] = []
    env_meta_signatures: Dict[str, str] = {}
    reference_features: Optional[Dict[str, Dict[str, Any]]] = None

    with h5py.File(input_path, "r") as h5_in:
        traj_keys = list_h5_trajectories(h5_in)
        if max_episodes is not None:
            traj_keys = traj_keys[: int(max_episodes)]
        if not traj_keys:
            raise ValueError(f"No trajectories found in {input_h5}.")
        first_group = get_trajectory_group(h5_in, traj_keys[0])
        first_env_meta = load_env_meta(h5_in, first_group)
        reference_features = build_lerobot_features(first_env_meta, use_videos=use_videos)

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(fps),
            root=str(output_path),
            features=reference_features,
            use_videos=bool(use_videos),
            video_backend=video_backend,
            rgb_encoder=_make_rgb_encoder(video_codec, video_backend) if use_videos else None,
        )

        for episode_idx, traj_key in enumerate(traj_keys):
            traj_group = get_trajectory_group(h5_in, traj_key)
            env_meta = load_env_meta(h5_in, traj_group)
            features = build_lerobot_features(env_meta, use_videos=use_videos)
            if features != reference_features:
                raise ValueError(
                    "All trajectories must share LeRobot feature schema for one dataset. "
                    f"Mismatch at trajectory {traj_key or '<root>'}."
                )

            env_meta_signatures[str(traj_key or "root")] = json.dumps(env_meta, sort_keys=True)
            prompt = read_prompt(traj_group, default_prompt=default_prompt)
            obs_group = traj_group["obs"]
            action_group = traj_group["action"]
            state_keys = _ordered_keys(env_meta["obs"].get("state", {}))
            action_keys = _ordered_keys(env_meta.get("action", {}))
            camera_keys = _ordered_keys(env_meta["obs"].get("rgb", {}))
            traj_len = _first_leaf_length(action_group, action_keys)
            write_len = traj_len
            if max_frames_per_episode is not None:
                write_len = min(write_len, int(max_frames_per_episode))
            if write_len <= 0:
                raise ValueError(f"Trajectory {traj_key or '<root>'} has no frames to write.")

            for step_idx in range(write_len):
                frame = {
                    "observation.state": torch.from_numpy(
                        _flatten_group_step(obs_group["state"], state_keys, step_idx)
                    ),
                    "action": torch.from_numpy(
                        _flatten_group_step(action_group, action_keys, step_idx)
                    ),
                    "task": prompt,
                }
                for camera_name in camera_keys:
                    image = _read_rgb_frame(obs_group["rgb"][camera_name], step_idx)
                    frame[f"observation.images.{camera_name}"] = torch.from_numpy(image)
                dataset.add_frame(frame)

            dataset.save_episode()
            episode_info.append(
                {
                    "episode_id": episode_idx,
                    "source_traj": traj_key,
                    "prompt": prompt,
                    "success": _trajectory_success(traj_group),
                    "length": traj_len,
                    "written_length": write_len,
                    "env_meta_signature": env_meta_signatures[str(traj_key or "root")],
                }
            )

        if hasattr(dataset, "finalize"):
            dataset.finalize()

    manifest = {
        "input_h5": str(input_path),
        "output_dir": str(output_path),
        "repo_id": repo_id,
        "fps": int(fps),
        "use_videos": bool(use_videos),
        "video_backend": video_backend,
        "video_codec": video_codec,
        "max_episodes": max_episodes,
        "max_frames_per_episode": max_frames_per_episode,
        "features": reference_features,
        "episodes": episode_info,
    }
    (output_path / "h5_conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "build_lerobot_features",
    "convert_h5_to_lerobot_dataset",
    "get_trajectory_group",
    "infer_lerobot_repo_id",
    "list_h5_trajectories",
    "load_lerobot_policy_dataset",
    "load_env_meta",
    "read_prompt",
]
