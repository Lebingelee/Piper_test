import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import h5py
import numpy as np
import torch

from agent_infra.Piper_Env.Record.h5_utils import (
    DEFAULT_JPEG_QUALITY,
    IMAGE_CODECS,
    copy_h5_dataset,
    create_h5_dataset,
    read_h5_dataset,
)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False


def _read_json_dataset(dataset) -> Dict[str, Any]:
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return json.loads(value)


def _load_env_meta_from_h5(h5_file: h5py.File) -> Dict[str, Any]:
    if "meta" not in h5_file or "env_meta" not in h5_file["meta"]:
        raise KeyError("Input H5 does not contain meta/env_meta.")
    return _read_json_dataset(h5_file["meta"]["env_meta"])


def _write_json_dataset(group: h5py.Group, key: str, value: Dict[str, Any]):
    group.create_dataset(key, data=json.dumps(value))


def _collect_h5_files(input_paths: List[str]) -> List[str]:
    h5_files: List[str] = []
    for path_str in input_paths:
        path = Path(path_str)
        if path.is_dir():
            h5_files.extend(str(p) for p in sorted(path.glob("*.h5")))
        elif path.is_file() and path.suffix == ".h5":
            h5_files.append(str(path))
    return h5_files


def _list_traj_names(h5_file: h5py.File) -> List[Optional[str]]:
    traj_names = [
        name for name in sorted(h5_file.keys())
        if name.startswith("traj_") and isinstance(h5_file[name], h5py.Group)
    ]
    return traj_names or [None]


def _get_traj_group(h5_file: h5py.File, traj_name: Optional[str]):
    return h5_file if traj_name is None else h5_file[traj_name]


def _copy_group_recursive(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    image_codec: str = "jpeg",
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
):
    for key in src_group.keys():
        src_item = src_group[key]
        if isinstance(src_item, h5py.Group):
            _copy_group_recursive(
                src_item,
                dst_group.create_group(key),
                image_codec=image_codec,
                jpeg_quality=jpeg_quality,
            )
        else:
            copy_h5_dataset(
                src_item,
                dst_group,
                key,
                image_codec=image_codec,
                jpeg_quality=jpeg_quality,
            )


def _action_to_state_key(action_key: str, control_mode: str) -> str:
    if action_key.endswith("gripper"):
        return f"{action_key[:-7]}gripper_pos"
    if action_key.endswith("arm"):
        state_suffix = "joint_pos" if control_mode == "joint" else "ee_pose"
        return f"{action_key[:-3]}{state_suffix}"
    raise KeyError(f"Unsupported action key for mode conversion: {action_key}")


def _copy_or_convert_action_group(
    src_obs_group: h5py.Group,
    src_action_group: h5py.Group,
    dst_action_group: h5py.Group,
    env_meta: Dict[str, Any],
    target_control_mode: Optional[str],
):
    if target_control_mode is None:
        _copy_group_recursive(src_action_group, dst_action_group)
        return

    action_keys = list(env_meta["action"].keys())
    action_len = _get_first_leaf_length(src_action_group, action_keys)

    for action_key in action_keys:
        if action_key.endswith("arm") and target_control_mode == "delta_pose":
            state_key = _action_to_state_key(action_key, target_control_mode)
            if state_key not in src_obs_group["state"]:
                raise KeyError(
                    f"Missing obs/state/{state_key}; cannot convert control mode."
                )
            poses = src_obs_group["state"][state_key][:action_len]
            data = np.zeros_like(poses)
            if action_len > 1:
                data[:-1] = poses[1:] - poses[:-1]
        elif action_key.endswith("arm"):
            state_key = _action_to_state_key(action_key, target_control_mode)
            if state_key not in src_obs_group["state"]:
                raise KeyError(
                    f"Missing obs/state/{state_key}; cannot convert control mode."
                )
            data = src_obs_group["state"][state_key][:action_len]
        elif action_key in src_action_group:
            data = src_action_group[action_key][()]
        else:
            state_key = _action_to_state_key(action_key, target_control_mode)
            if state_key not in src_obs_group["state"]:
                raise KeyError(
                    f"Missing action/{action_key} and obs/state/{state_key}; cannot convert."
                )
            data = src_obs_group["state"][state_key][:action_len]

        create_h5_dataset(dst_action_group, action_key, data)


def _env_meta_signature(env_meta: Dict[str, Any]) -> str:
    return json.dumps(env_meta, sort_keys=True)


def _get_first_leaf_length(group: h5py.Group, ordered_keys: List[str]) -> int:
    for key in ordered_keys:
        if key in group:
            return group[key].shape[0]
    raise KeyError("Unable to determine trajectory length from action group.")


def _read_bool_signal(src_traj: h5py.Group, key: str, length: int) -> Optional[np.ndarray]:
    if key not in src_traj:
        return None

    raw = np.asarray(src_traj[key][()], dtype=np.bool_)
    if raw.shape == ():
        values = np.zeros(length, dtype=np.bool_)
        if length > 0:
            values[-1] = bool(raw.item())
        return values

    values = raw.reshape(raw.shape[0], -1)
    if values.shape[0] != length:
        raise ValueError(
            f"Signal '{key}' length mismatch: expected {length}, got {values.shape[0]}."
        )
    if values.shape[1] == 1:
        return values[:, 0].astype(np.bool_)
    return np.any(values, axis=1).astype(np.bool_)


def _build_episode_signals(
    src_traj: h5py.Group,
    length: int,
    success: bool,
) -> Dict[str, np.ndarray]:
    success_signal = _read_bool_signal(src_traj, "success", length)
    if success_signal is None:
        success_signal = np.zeros(length, dtype=np.bool_)
        if length > 0:
            success_signal[-1] = bool(success)

    terminated_signal = _read_bool_signal(src_traj, "terminated", length)
    if terminated_signal is None:
        terminated_signal = _read_bool_signal(src_traj, "done", length)
    if terminated_signal is None:
        terminated_signal = np.zeros(length, dtype=np.bool_)
    if length > 0:
        terminated_signal[-1] = True

    truncated_signal = _read_bool_signal(src_traj, "truncated", length)
    if truncated_signal is None:
        truncated_signal = np.zeros(length, dtype=np.bool_)

    return {
        "success": success_signal,
        "terminated": terminated_signal,
        "truncated": truncated_signal,
    }


def _write_episode_signals(dst_traj: h5py.Group, signals: Dict[str, np.ndarray]):
    for key in ("success", "terminated", "truncated"):
        create_h5_dataset(
            dst_traj,
            key,
            np.asarray(signals[key], dtype=np.bool_),
        )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _build_lerobot_features(env_meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    state_dim = sum(shape[0] for shape in env_meta["obs"]["state"].values())
    action_dim = sum(shape[0] for shape in env_meta["action"].values())
    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,)},
        "action": {"dtype": "float32", "shape": (action_dim,)},
    }

    if "under_control" in env_meta["obs"]:
        for name in env_meta["obs"]["under_control"].keys():
            features[f"observation.under_control.{name}"] = {
                "dtype": "bool",
                "shape": (1,),
            }

    if "rgb" in env_meta["obs"]:
        for role, shape in env_meta["obs"]["rgb"].items():
            features[f"observation.images.{role}"] = {
                "dtype": "video",
                "shape": shape,
            }

    return features


def _normalize_lerobot_input_dir(input_dir: str) -> Path:
    path = Path(input_dir)
    if not path.exists() and "agent_agent_infra" in str(path):
        typo_fixed = Path(str(path).replace("agent_agent_infra", "agent_infra"))
        if typo_fixed.exists():
            print(f"[Postprocess] Input path typo fixed: {path} -> {typo_fixed}")
            path = typo_fixed

    if path.name == "meta" and (path / "info.json").exists():
        print(f"[Postprocess] Input points to meta dir; using LeRobot root: {path.parent}")
        return path.parent

    if (path / "meta" / "info.json").exists():
        return path

    if (path / "lerobot" / "meta" / "info.json").exists():
        print(f"[Postprocess] Input points to task dir; using LeRobot root: {path / 'lerobot'}")
        return path / "lerobot"

    raise FileNotFoundError(
        "Unable to locate a LeRobot dataset root. Expected a directory "
        "containing meta/info.json, or a task directory containing lerobot/meta/info.json. "
        f"Got: {input_dir}"
    )


def _shape_from_feature(feature: Dict[str, Any]) -> tuple:
    return tuple(int(x) for x in feature.get("shape", ()))


def _infer_env_meta_from_lerobot_info(input_dir: Path) -> Dict[str, Any]:
    info_path = input_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"LeRobot info file not found: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    state_shape = _shape_from_feature(features.get("observation.state", {}))
    action_shape = _shape_from_feature(features.get("action", {}))
    if not state_shape or not action_shape:
        raise FileNotFoundError(
            "env_meta.json is missing and LeRobot meta/info.json does not "
            "contain observation.state/action feature shapes."
        )
    state_dim = state_shape[0]
    action_dim = action_shape[0]

    under_control_keys = [
        key.rsplit(".", 1)[-1]
        for key in features.keys()
        if key.startswith("observation.under_control.")
    ]
    if under_control_keys:
        arm_names = under_control_keys
    elif state_dim == 38 and action_dim == 14:
        arm_names = ["left", "right"]
    elif state_dim == 19 and action_dim == 7:
        arm_names = ["single"]
    else:
        raise FileNotFoundError(
            "env_meta.json is missing and env_meta cannot be inferred from "
            f"state/action dims: state={state_dim}, action={action_dim}. "
            "Please pass --env-meta path/to/env_meta.json."
        )

    obs_state = {}
    action = {}
    for name in arm_names:
        prefix = f"{name}_" if len(arm_names) > 1 else ""
        obs_state.update(
            {
                f"{prefix}joint_pos": (6,),
                f"{prefix}joint_vel": (6,),
                f"{prefix}ee_pose": (6,),
                f"{prefix}gripper_pos": (1,),
            }
        )
        action.update(
            {
                f"{prefix}arm": (6,),
                f"{prefix}gripper": (1,),
            }
        )

    env_meta = {
        "obs": {
            "state": obs_state,
        },
        "action": action,
    }

    if under_control_keys:
        env_meta["obs"]["under_control"] = {
            name: (1,) for name in under_control_keys
        }

    rgb = {}
    for key, feature in features.items():
        if key.startswith("observation.images."):
            role = key.replace("observation.images.", "", 1)
            rgb[role] = _shape_from_feature(feature)
    if rgb:
        env_meta["obs"]["rgb"] = rgb

    print(
        "[Postprocess] env_meta.json not found; inferred Piper env_meta from "
        f"LeRobot features (arms={arm_names})."
    )
    return env_meta


def _load_lerobot_env_meta(input_dir: Path, env_meta_path: Optional[str]) -> Dict[str, Any]:
    candidates = []
    if env_meta_path:
        candidates.append(Path(env_meta_path))
    candidates.extend(
        [
            input_dir / "env_meta.json",
            input_dir / "meta" / "env_meta.json",
            input_dir.parent / "env_meta.json",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))

    return _infer_env_meta_from_lerobot_info(input_dir)


def merge_h5_trajectories(
    input_paths: List[str],
    output_path: str,
    target_control_mode: Optional[str] = None,
    image_codec: str = "jpeg",
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
):
    h5_files = _collect_h5_files(input_paths)
    if not h5_files:
        raise FileNotFoundError("No input .h5 files found for merging.")

    if target_control_mode is not None and target_control_mode not in {"joint", "pose", "delta_pose"}:
        raise ValueError("target_control_mode must be one of: joint, pose, delta_pose")
    if image_codec not in IMAGE_CODECS:
        raise ValueError(f"image_codec must be one of: {', '.join(IMAGE_CODECS)}")
    if not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_meta: Optional[Dict[str, Any]] = None
    env_signature: Optional[str] = None
    merged_count = 0
    source_entries = []

    with h5py.File(output_path, "w") as h5_out:
        meta_group = h5_out.create_group("meta")

        for file_path in h5_files:
            with h5py.File(file_path, "r") as h5_in:
                current_env_meta = _load_env_meta_from_h5(h5_in)
                current_signature = _env_meta_signature(current_env_meta)

                if env_meta is None:
                    env_meta = current_env_meta
                    env_signature = current_signature
                    _write_json_dataset(meta_group, "env_meta", env_meta)
                elif current_signature != env_signature:
                    raise ValueError(
                        f"Incompatible env_meta found in {file_path}; cannot merge."
                    )

                for traj_name in _list_traj_names(h5_in):
                    src_traj = _get_traj_group(h5_in, traj_name)
                    dst_traj = h5_out.create_group(f"traj_{merged_count}")
                    _copy_group_recursive(
                        src_traj["obs"],
                        dst_traj.create_group("obs"),
                        image_codec=image_codec,
                        jpeg_quality=jpeg_quality,
                    )
                    _copy_or_convert_action_group(
                        src_traj["obs"],
                        src_traj["action"],
                        dst_traj.create_group("action"),
                        current_env_meta,
                        target_control_mode,
                    )

                    success = bool(src_traj.attrs.get("success", h5_in.attrs.get("success", True)))
                    action_len = _get_first_leaf_length(
                        dst_traj["action"],
                        list(current_env_meta["action"].keys()),
                    )
                    signals = _build_episode_signals(src_traj, action_len, success)
                    _write_episode_signals(dst_traj, signals)

                    dst_traj.attrs["success"] = success
                    dst_traj.attrs["source_file"] = file_path
                    if traj_name is not None:
                        dst_traj.attrs["source_traj"] = traj_name

                    source_entries.append({
                        "merged_traj": f"traj_{merged_count}",
                        "source_file": file_path,
                        "source_traj": traj_name,
                        "success": success,
                        "terminated": bool(signals["terminated"][-1]) if action_len > 0 else False,
                        "truncated": bool(signals["truncated"][-1]) if action_len > 0 else False,
                    })
                    merged_count += 1

        _write_json_dataset(
            meta_group,
            "merge_info",
            {
                "sources": source_entries,
                "target_control_mode": target_control_mode or "keep",
                "image_codec": image_codec,
                "jpeg_quality": int(jpeg_quality),
            },
        )
        h5_out.attrs["merged"] = True
        h5_out.attrs["num_trajectories"] = merged_count
        h5_out.attrs["control_mode"] = target_control_mode or "keep"
        h5_out.attrs["image_codec"] = image_codec
        h5_out.attrs["jpeg_quality"] = int(jpeg_quality)

    print(f"[Postprocess] Merged {merged_count} trajectories -> {output_path}")


def _resolve_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path

    suffix = 1
    base = path
    while True:
        candidate = Path(f"{base}_{suffix:03d}")
        if not candidate.exists():
            print(f"[Postprocess] Output dir exists, using: {candidate}")
            return candidate
        suffix += 1


def convert_h5_to_lerobot(
    input_h5: str,
    output_dir: str,
    task_description: Optional[str] = None,
    vcodec: str = "h264",
):
    if not HAS_LEROBOT:
        raise ImportError("LeRobot is not installed; cannot convert H5 to LeRobot.")

    input_h5 = Path(input_h5)
    output_dir = _resolve_unique_dir(Path(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    task_description = task_description or output_dir.name

    with h5py.File(input_h5, "r") as h5_in:
        env_meta = _load_env_meta_from_h5(h5_in)
        features = _build_lerobot_features(env_meta)
        repo_id = output_dir.name

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=30,
            root=str(output_dir),
            features=features,
            use_videos=True,
            vcodec=vcodec,
        )

        action_keys = list(env_meta["action"].keys())
        state_keys = list(env_meta["obs"]["state"].keys())
        under_control_keys = list(env_meta["obs"].get("under_control", {}).keys())
        rgb_roles = list(env_meta["obs"].get("rgb", {}).keys())

        episode_info = []
        for episode_idx, traj_name in enumerate(_list_traj_names(h5_in)):
            traj_group = _get_traj_group(h5_in, traj_name)
            obs_group = traj_group["obs"]
            action_group = traj_group["action"]
            traj_len = _get_first_leaf_length(action_group, action_keys)

            for step_idx in range(traj_len):
                state_vec = np.concatenate(
                    [obs_group["state"][key][step_idx] for key in state_keys]
                ).astype(np.float32)
                action_vec = np.concatenate(
                    [action_group[key][step_idx] for key in action_keys]
                ).astype(np.float32)

                frame = {
                    "observation.state": torch.from_numpy(state_vec),
                    "action": torch.from_numpy(action_vec),
                    "task": task_description,
                }

                for name in under_control_keys:
                    feature_key = f"observation.under_control.{name}"
                    value = obs_group["under_control"][name][step_idx]
                    frame[feature_key] = torch.from_numpy(np.asarray(value, dtype=np.bool_))

                for role in rgb_roles:
                    frame[f"observation.images.{role}"] = torch.from_numpy(
                        read_h5_dataset(obs_group["rgb"][role], step_idx)
                    )

                dataset.add_frame(frame)

            dataset.save_episode()
            episode_info.append({
                "episode_id": episode_idx,
                "source_traj": traj_name,
                "success": bool(traj_group.attrs.get("success", True)),
                "length": traj_len,
            })

        if hasattr(dataset, "finalize"):
            dataset.finalize()

    (output_dir / "env_meta.json").write_text(
        json.dumps(env_meta, indent=2),
        encoding="utf-8",
    )
    (output_dir / "episodes.json").write_text(
        json.dumps(episode_info, indent=2),
        encoding="utf-8",
    )

    print(f"[Postprocess] Converted {input_h5} -> LeRobot dataset at {output_dir}")


def convert_lerobot_to_h5(
    input_dir: str,
    output_h5: str,
    env_meta_path: Optional[str] = None,
    image_codec: str = "jpeg",
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
):
    if not HAS_LEROBOT:
        raise ImportError("LeRobot is not installed; cannot convert LeRobot to H5.")

    input_dir = _normalize_lerobot_input_dir(input_dir)
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    if image_codec not in IMAGE_CODECS:
        raise ValueError(f"image_codec must be one of: {', '.join(IMAGE_CODECS)}")
    if not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    env_meta = _load_lerobot_env_meta(input_dir, env_meta_path)
    episodes_file = input_dir / "episodes.json"
    episode_info = []
    if episodes_file.exists():
        episode_info = json.loads(episodes_file.read_text(encoding="utf-8"))

    dataset = LeRobotDataset(repo_id=input_dir.name, root=str(input_dir))
    hf_dataset = dataset.hf_dataset

    state_keys = list(env_meta["obs"]["state"].keys())
    action_keys = list(env_meta["action"].keys())
    under_control_keys = list(env_meta["obs"].get("under_control", {}).keys())
    rgb_roles = list(env_meta["obs"].get("rgb", {}).keys())

    with h5py.File(output_h5, "w") as h5_out:
        meta_group = h5_out.create_group("meta")
        _write_json_dataset(meta_group, "env_meta", env_meta)
        h5_out.attrs["merged"] = True
        h5_out.attrs["num_trajectories"] = len(dataset.meta.episodes)
        h5_out.attrs["image_codec"] = image_codec
        h5_out.attrs["jpeg_quality"] = int(jpeg_quality)

        for episode_idx, episode in enumerate(dataset.meta.episodes):
            from_idx = episode["dataset_from_index"]
            to_idx = episode["dataset_to_index"]
            traj_len = to_idx - from_idx

            traj_group = h5_out.create_group(f"traj_{episode_idx}")
            obs_group = traj_group.create_group("obs")
            state_group = obs_group.create_group("state")
            action_group = traj_group.create_group("action")

            state_matrix = np.stack([
                _to_numpy(hf_dataset[idx]["observation.state"])
                for idx in range(from_idx, to_idx)
            ]).astype(np.float32)
            action_matrix = np.stack([
                _to_numpy(hf_dataset[idx]["action"])
                for idx in range(from_idx, to_idx)
            ]).astype(np.float32)

            cursor = 0
            for key in state_keys:
                dim = env_meta["obs"]["state"][key][0]
                create_h5_dataset(state_group, key, state_matrix[:, cursor:cursor + dim])
                cursor += dim

            cursor = 0
            for key in action_keys:
                dim = env_meta["action"][key][0]
                create_h5_dataset(action_group, key, action_matrix[:, cursor:cursor + dim])
                cursor += dim

            if under_control_keys:
                uc_group = obs_group.create_group("under_control")
                for name in under_control_keys:
                    feature_key = f"observation.under_control.{name}"
                    if feature_key in hf_dataset.features:
                        values = np.stack([
                            np.asarray(_to_numpy(hf_dataset[idx][feature_key]), dtype=np.bool_)
                            for idx in range(from_idx, to_idx)
                        ])
                    else:
                        values = np.zeros((traj_len, 1), dtype=np.bool_)
                    create_h5_dataset(uc_group, name, values)

            if rgb_roles:
                rgb_group = obs_group.create_group("rgb")
                for role in rgb_roles:
                    feature_key = f"observation.images.{role}"
                    if feature_key not in hf_dataset.features:
                        shape = env_meta["obs"]["rgb"][role]
                        values = np.zeros((traj_len,) + tuple(shape), dtype=np.uint8)
                    else:
                        try:
                            values = np.stack([
                                _to_numpy(hf_dataset[idx][feature_key]).astype(np.uint8)
                                for idx in range(from_idx, to_idx)
                            ])
                        except Exception as exc:
                            shape = env_meta["obs"]["rgb"][role]
                            values = np.zeros((traj_len,) + tuple(shape), dtype=np.uint8)
                            print(
                                f"[Postprocess] Failed to decode LeRobot video "
                                f"{feature_key}; writing zero fallback. Error: {exc}"
                            )
                    create_h5_dataset(
                        rgb_group,
                        role,
                        data=values,
                        rgb=True,
                        image_codec=image_codec,
                        jpeg_quality=jpeg_quality,
                    )

            success = True
            if episode_idx < len(episode_info):
                success = bool(episode_info[episode_idx].get("success", True))
            traj_group.attrs["success"] = success

    print(f"[Postprocess] Converted LeRobot dataset at {input_dir} -> {output_h5}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Piper data postprocess tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge_h5", help="Merge raw/merged H5 trajectories")
    merge_parser.add_argument("-i", "--input", nargs="+", required=True, help="Input H5 files or directories")
    merge_parser.add_argument("-o", "--output", required=True, help="Output merged H5 path")
    merge_parser.add_argument(
        "--control-mode",
        default=None,
        choices=["joint", "pose", "delta_pose"],
        help="Optional target control mode for merged action conversion",
    )
    merge_parser.add_argument(
        "--image-codec",
        choices=IMAGE_CODECS,
        default="jpeg",
        help="RGB H5 storage codec; jpeg saves much more space, gzip is lossless.",
    )
    merge_parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality for RGB frames (1-100, default 85).",
    )

    to_lerobot_parser = subparsers.add_parser("h5_to_lerobot", help="Convert raw/merged H5 to LeRobot")
    to_lerobot_parser.add_argument("-i", "--input", required=True, help="Input raw or merged H5 path")
    to_lerobot_parser.add_argument("-o", "--output", required=True, help="Output LeRobot root directory")
    to_lerobot_parser.add_argument(
        "--task-description",
        default=None,
        help="LeRobot per-frame task string. Default: output directory name.",
    )
    to_lerobot_parser.add_argument(
        "--vcodec",
        default="h264",
        help="LeRobot video codec. Default: h264.",
    )

    to_h5_parser = subparsers.add_parser("lerobot_to_h5", help="Convert LeRobot dataset to merged H5")
    to_h5_parser.add_argument("-i", "--input", required=True, help="Input LeRobot root directory")
    to_h5_parser.add_argument("-o", "--output", required=True, help="Output merged H5 path")
    to_h5_parser.add_argument("--env-meta", default=None, help="Optional env_meta.json path")
    to_h5_parser.add_argument(
        "--image-codec",
        choices=IMAGE_CODECS,
        default="jpeg",
        help="RGB H5 storage codec; default jpeg.",
    )
    to_h5_parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality for RGB frames (1-100, default 85).",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "merge_h5":
        merge_h5_trajectories(
            args.input,
            args.output,
            args.control_mode,
            args.image_codec,
            args.jpeg_quality,
        )
    elif args.command == "h5_to_lerobot":
        convert_h5_to_lerobot(args.input, args.output, args.task_description, args.vcodec)
    elif args.command == "lerobot_to_h5":
        convert_lerobot_to_h5(
            args.input,
            args.output,
            args.env_meta,
            args.image_codec,
            args.jpeg_quality,
        )


if __name__ == "__main__":
    main()
