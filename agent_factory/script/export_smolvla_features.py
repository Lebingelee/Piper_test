import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

if __package__ in {None, ""}:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from agent_factory.agents.impl import smolvla_vanilla as _smolvla_vanilla  # noqa: F401
from agent_factory.agents.registry import make_agent
from agent_factory.data.impl.lerobot.h5_utils import (
    _first_leaf_length,
    _read_dataset_value,
    get_trajectory_group,
    list_h5_trajectories,
    load_env_meta,
)

def _read_scalar_text(group: h5py.Group, key: str) -> str:
    if key not in group:
        raise KeyError(f"H5 group {group.name} is missing required dataset {key!r}.")
    value = _read_dataset_value(group[key])
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            value = value.reshape(-1)[0]
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def _ordered_keys(mapping: Dict[str, object]) -> List[str]:
    return list(mapping.keys())


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _copy_without_obs(src_group: h5py.Group, dst_group: h5py.Group) -> None:
    _copy_attrs(src_group.attrs, dst_group.attrs)
    for key in src_group.keys():
        if key == "obs":
            continue
        src_group.copy(key, dst_group)


def _require_env_meta_task_name(env_meta: Dict[str, object], traj_name: str) -> None:
    task_name = env_meta.get("task_name")
    if not isinstance(task_name, str) or not task_name.strip():
        raise KeyError(f"Trajectory {traj_name} requires non-empty meta/env_meta['task_name'].")


def _trajectory_length(obs_group: h5py.Group, state_keys: Sequence[str]) -> int:
    if "state" not in obs_group:
        raise KeyError(f"Observation group {obs_group.name} is missing 'state'.")
    state_node = obs_group["state"]
    if isinstance(state_node, h5py.Dataset):
        return int(state_node.shape[0])
    if isinstance(state_node, h5py.Group):
        return _first_leaf_length(state_node, list(state_keys))
    raise TypeError(f"Unsupported H5 node for {state_node.name}: {type(state_node).__name__}")


def _flatten_group_range(group: h5py.Group, keys: Sequence[str], start_idx: int, end_idx: int) -> np.ndarray:
    arrays = []
    for key in keys:
        if key not in group:
            raise KeyError(f"Missing key '{key}' under H5 group {group.name}.")
        data = np.asarray(group[key][start_idx:end_idx])
        arrays.append(data.reshape(data.shape[0], -1) if data.ndim > 2 else data)
    if not arrays:
        return np.zeros((int(end_idx) - int(start_idx), 0), dtype=np.float32)
    return np.concatenate(arrays, axis=-1).astype(np.float32, copy=False)


def _read_state_range(state_node: h5py.Dataset | h5py.Group, keys: Sequence[str], start_idx: int, end_idx: int) -> np.ndarray:
    if isinstance(state_node, h5py.Dataset):
        data = np.asarray(state_node[start_idx:end_idx])
        return data.reshape(data.shape[0], -1).astype(np.float32, copy=False)
    if isinstance(state_node, h5py.Group):
        return _flatten_group_range(state_node, keys, start_idx, end_idx)
    raise TypeError(f"Unsupported H5 node for {state_node.name}: {type(state_node).__name__}")


def _read_rgb_range(dataset: h5py.Dataset, start_idx: int, end_idx: int) -> np.ndarray:
    images = np.asarray(dataset[start_idx:end_idx])
    if images.ndim != 4:
        raise ValueError(f"RGB dataset {dataset.name} must be rank-4 over time, got shape {images.shape}.")
    if images.shape[1] == 3:
        return images
    if images.shape[-1] == 3:
        return np.transpose(images, (0, 3, 1, 2))
    raise ValueError(
        f"RGB dataset {dataset.name} must be TCHW or THWC with 3 channels, got shape {images.shape}."
    )


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32, copy=False)


def _read_packed_rgb_range(
    dataset: h5py.Dataset,
    camera_keys: Sequence[str],
    start_idx: int,
    end_idx: int,
) -> Dict[str, np.ndarray]:
    images = np.asarray(dataset[start_idx:end_idx])
    if images.ndim != 4:
        raise ValueError(
            f"Packed RGB dataset {dataset.name} must be rank-4 [T,C,H,W] or [T,H,W,C], "
            f"got shape {images.shape}."
        )

    expected_channels = 3 * len(camera_keys)
    if images.shape[1] == expected_channels:
        packed = images
    elif images.shape[-1] == expected_channels:
        packed = np.transpose(images, (0, 3, 1, 2))
    else:
        raise ValueError(
            f"Packed RGB dataset {dataset.name} has shape {images.shape}, but {len(camera_keys)} cameras "
            f"require {expected_channels} channels."
        )

    return {
        camera_name: _normalize_rgb(packed[:, 3 * index: 3 * (index + 1)])
        for index, camera_name in enumerate(camera_keys)
    }


def _read_frame_batch(
    traj_group: h5py.Group,
    *,
    state_keys: Sequence[str],
    camera_keys: Sequence[str],
    start_idx: int,
    end_idx: int,
) -> Dict[str, object]:
    obs_group = traj_group["obs"]
    state = _read_state_range(obs_group["state"], state_keys, start_idx, end_idx)
    images: Dict[str, np.ndarray] = {}
    rgb_node = obs_group["rgb"]
    if isinstance(rgb_node, h5py.Dataset):
        images = _read_packed_rgb_range(rgb_node, camera_keys, start_idx, end_idx)
    elif isinstance(rgb_node, h5py.Group):
        for camera_name in camera_keys:
            if camera_name not in rgb_node:
                raise KeyError(f"RGB group {rgb_node.name} is missing camera '{camera_name}'.")
            images[camera_name] = _normalize_rgb(
                _read_rgb_range(rgb_node[camera_name], start_idx, end_idx)
            )
    else:
        raise TypeError(f"Unsupported H5 node for {rgb_node.name}: {type(rgb_node).__name__}")
    return {
        "state": state,
        "rgb": images,
    }


def _to_numpy(tensor: torch.Tensor, dtype: Optional[np.dtype] = None) -> np.ndarray:
    value = tensor.detach()
    if value.dtype == torch.bfloat16:
        value = value.to(dtype=torch.float32)
    array = value.cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _feature_dtype(name: str) -> np.dtype:
    normalized = str(name).strip().lower()
    if normalized in {"float32", "fp32"}:
        return np.float32
    if normalized in {"float16", "fp16"}:
        return np.float16
    raise ValueError(f"Unsupported feature dtype {name!r}; expected float32 or float16.")


def _write_feature_arrays(
    feature_group: h5py.Group,
    *,
    prefix_tokens: Optional[np.ndarray],
    prefix_valid_mask: np.ndarray,
    state_token: np.ndarray,
    state_token_index: np.ndarray,
    compression: Optional[str],
) -> None:
    kwargs = {"compression": compression} if compression else {}
    if prefix_tokens is not None:
        feature_group.create_dataset("prefix_tokens", data=prefix_tokens, **kwargs)
    feature_group.create_dataset("prefix_valid_mask", data=prefix_valid_mask.astype(np.bool_, copy=False), **kwargs)
    feature_group.create_dataset("state_token", data=state_token, **kwargs)
    feature_group.create_dataset("state_token_index", data=state_token_index.astype(np.int64, copy=False), **kwargs)


def _load_agent(config_path: str, ckpt_path: Optional[str], device: Optional[str]):
    cfg = OmegaConf.load(config_path)
    if device:
        cfg.train.device = device
        cfg.device = device
    elif "device" not in cfg or not str(cfg.device or "").strip():
        cfg.device = cfg.train.device
    if ckpt_path:
        cfg.train.ckpt_path = ckpt_path
    if not bool(OmegaConf.select(cfg, "resolved", default=False)):
        raise ValueError("Feature export requires a resolved SmolVLA config.")
    agent = make_agent(str(cfg.agent_type), cfg)
    load_path = str(ckpt_path or getattr(cfg.train, "ckpt_path", "") or "").strip()
    if not load_path:
        raise ValueError("Feature export requires --ckpt-path or cfg.train.ckpt_path.")
    agent.load(load_path)
    agent.eval()
    return agent, cfg, load_path


def _export_trajectory(
    agent,
    src_h5: h5py.File,
    dst_h5: h5py.File,
    traj_key: Optional[str],
    *,
    feature_batch_size: int,
    feature_dtype: np.dtype,
    compression: Optional[str],
    save_prefix_tokens: bool,
    progress_batches: bool,
) -> None:
    src_traj = get_trajectory_group(src_h5, traj_key)
    prompt = _read_scalar_text(src_traj, "prompt")
    env_meta = load_env_meta(src_h5, src_traj)
    _require_env_meta_task_name(env_meta, src_traj.name)
    state_keys = _ordered_keys(env_meta["obs"].get("state", {}))
    camera_keys = list(getattr(agent.actor, "image_keys", []) or _ordered_keys(env_meta["obs"].get("rgb", {})))
    if not camera_keys:
        raise ValueError(f"Trajectory {traj_key} has no RGB cameras for SmolVLA feature export.")
    traj_len = _trajectory_length(src_traj["obs"], state_keys)

    if traj_key is None:
        dst_traj = dst_h5
        _copy_without_obs(src_traj, dst_traj)
    else:
        dst_traj = dst_h5.create_group(str(traj_key))
        _copy_without_obs(src_traj, dst_traj)

    feature_group = dst_traj.create_group("obs").create_group("feature")
    prefix_chunks = [] if save_prefix_tokens else None
    valid_mask_chunks = []
    state_token_chunks = []
    state_index_chunks = []

    batch_starts = range(0, traj_len, int(feature_batch_size))
    if progress_batches:
        batch_starts = tqdm(
            batch_starts,
            desc=f"frames {src_traj.name}",
            leave=False,
            unit="batch",
        )
    for start_idx in batch_starts:
        end_idx = min(start_idx + int(feature_batch_size), traj_len)
        raw_obs = _read_frame_batch(
            src_traj,
            state_keys=state_keys,
            camera_keys=camera_keys,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        prompts = [prompt for _ in range(end_idx - start_idx)]
        norm_obs = agent._normalize_smolvla_obs(raw_obs)
        features = agent.actor.extract_vlm_prefix_features(norm_obs, prompt=prompts)
        if prefix_chunks is not None:
            prefix_chunks.append(_to_numpy(features["prefix_tokens"], feature_dtype))
        valid_mask_chunks.append(_to_numpy(features["prefix_valid_mask"]))
        state_token_chunks.append(_to_numpy(features["state_token"], feature_dtype))
        state_index_chunks.append(_to_numpy(features["state_token_index"]))

    _write_feature_arrays(
        feature_group,
        prefix_tokens=np.concatenate(prefix_chunks, axis=0) if prefix_chunks is not None else None,
        prefix_valid_mask=np.concatenate(valid_mask_chunks, axis=0),
        state_token=np.concatenate(state_token_chunks, axis=0),
        state_token_index=np.concatenate(state_index_chunks, axis=0),
        compression=compression,
    )


def export_h5_file(
    agent,
    *,
    input_h5: Path,
    output_h5: Path,
    feature_batch_size: int,
    feature_dtype: np.dtype,
    compression: Optional[str],
    save_prefix_tokens: bool,
    progress_batches: bool,
    max_traj: Optional[int],
    overwrite: bool,
) -> int:
    if output_h5.exists():
        if not overwrite:
            raise FileExistsError(f"Output H5 already exists: {output_h5}")
        output_h5.unlink()
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        _copy_attrs(src.attrs, dst.attrs)
        traj_keys = list_h5_trajectories(src)
        if max_traj is not None:
            traj_keys = traj_keys[: int(max_traj)]
        print(
            f"[export_smolvla_features] exporting {input_h5} -> {output_h5} "
            f"({len(traj_keys)} trajectories)",
            flush=True,
        )
        for traj_key in tqdm(traj_keys, desc=f"trajectories {input_h5.name}", unit="traj"):
            _export_trajectory(
                agent,
                src,
                dst,
                None if traj_key is None else str(traj_key),
                feature_batch_size=feature_batch_size,
                feature_dtype=feature_dtype,
                compression=compression,
                save_prefix_tokens=save_prefix_tokens,
                progress_batches=progress_batches,
            )
    return len(traj_keys)


def _iter_input_outputs(input_path: Path, output_path: Path) -> Iterable[Tuple[Path, Path]]:
    if input_path.is_file():
        if output_path.suffix.lower() in {".h5", ".hdf5"}:
            yield input_path, output_path
        else:
            yield input_path, output_path / input_path.name
        return
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    for h5_path in sorted(input_path.rglob("*.h5")):
        rel = h5_path.relative_to(input_path)
        yield h5_path, output_path / rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic SmolVLA final VLM prefix features to H5.")
    parser.add_argument("--config", required=True, help="Resolved SmolVLA YAML config.")
    parser.add_argument("--ckpt-path", default=None, help="SmolVLA checkpoint. Defaults to cfg.train.ckpt_path.")
    parser.add_argument("--input", required=True, help="Input H5 file or folder containing H5 files.")
    parser.add_argument("--output", required=True, help="Output H5 file or output folder.")
    parser.add_argument("--device", default=None, help="Override cfg device, e.g. cuda:1.")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--feature-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    parser.add_argument(
        "--save-prefix-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save full prefix_tokens in addition to state_token, prefix_valid_mask, and state_token_index.",
    )
    parser.add_argument(
        "--progress-batches",
        action="store_true",
        help="Show nested per-frame-batch progress inside each trajectory.",
    )
    parser.add_argument("--max-traj", type=int, default=None, help="Optional smoke-test trajectory limit per input file.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    agent, _cfg, _ = _load_agent(args.config, args.ckpt_path, args.device)
    total = 0
    for input_h5, output_h5 in _iter_input_outputs(Path(args.input), Path(args.output)):
        count = export_h5_file(
            agent,
            input_h5=input_h5,
            output_h5=output_h5,
            feature_batch_size=args.feature_batch_size,
            feature_dtype=_feature_dtype(args.feature_dtype),
            compression=compression,
            save_prefix_tokens=bool(args.save_prefix_tokens),
            progress_batches=bool(args.progress_batches),
            max_traj=args.max_traj,
            overwrite=args.overwrite,
        )
        total += count
        print(f"[export_smolvla_features] {input_h5} -> {output_h5} ({count} trajectories)")
    print(f"[export_smolvla_features] done, exported {total} trajectories")


if __name__ == "__main__":
    main()
