"""Export LeRobot π₀.₅ VLM-prefix features while preserving H5 trajectories.

The destination copies every source node except ``obs`` and replaces it with
``obs/feature``.  This deliberately mirrors the established SmolVLA feature
format so existing VLA-feature datasets can consume the result.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

if __package__ in {None, ""}:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

# Import registers the pi05 implementation before make_agent is called.
from agent_factory.agents.impl import pi05_vanilla as _pi05_vanilla  # noqa: F401
from agent_factory.agents.registry import make_agent
from agent_factory.config.resolution import general_resolve
from agent_factory.script.export_smolvla_features import (
    _copy_attrs,
    _copy_without_obs,
    _feature_dtype,
    _iter_input_outputs,
    _ordered_keys,
    _read_frame_batch,
    _read_scalar_text,
    _to_numpy,
    _trajectory_length,
    _write_feature_arrays,
)
from agent_factory.data.impl.lerobot.h5_utils import (
    get_trajectory_group,
    list_h5_trajectories,
    load_env_meta,
)


FEATURE_SOURCE = "pi05_vlm_prefix_last_language_token"


def _load_pi05_agent(config_path: str, pretrained_path: Optional[str], device: Optional[str]):
    cfg = OmegaConf.load(config_path)
    if str(OmegaConf.select(cfg, "agent_type", default="")) != "pi05":
        raise ValueError("π₀.₅ feature export requires config agent_type: pi05.")
    if device:
        cfg.device = device
        cfg.train.device = device
    elif not str(OmegaConf.select(cfg, "device", default="") or "").strip():
        cfg.device = cfg.train.device
    if pretrained_path:
        cfg.actor.pretrained_path = pretrained_path
    if not bool(OmegaConf.select(cfg, "resolved", default=False)):
        cfg, _ = general_resolve(file_config=OmegaConf.to_container(cfg, resolve=False))
    if bool(OmegaConf.select(cfg, "actor.mock_mode", default=True)):
        raise ValueError(
            "Refusing to export placeholder features: set actor.mock_mode=false and provide a π₀.₅ weight directory."
        )
    if not list(OmegaConf.select(cfg, "actor.image_keys", default=[])):
        raise ValueError("actor.image_keys must explicitly declare the multi-task camera schema.")
    agent = make_agent("pi05", cfg)
    agent.eval()
    return agent, cfg


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
    state_keys = _ordered_keys(env_meta.get("obs", {}).get("state", {}))
    camera_keys = list(agent.actor.image_keys)
    traj_len = _trajectory_length(src_traj["obs"], state_keys)

    if traj_key is None:
        dst_traj = dst_h5
        _copy_without_obs(src_traj, dst_traj)
    else:
        dst_traj = dst_h5.create_group(str(traj_key))
        _copy_without_obs(src_traj, dst_traj)
    feature_group = dst_traj.create_group("obs").create_group("feature")
    feature_group.attrs["feature_source"] = FEATURE_SOURCE
    feature_group.attrs["feature_pooling"] = str(agent.actor.feature_pooling)
    feature_group.attrs["state_dim"] = int(agent.actor.state_dim)
    feature_group.attrs["camera_keys"] = np.asarray(camera_keys, dtype=h5py.string_dtype())

    prefix_chunks = [] if save_prefix_tokens else None
    valid_mask_chunks, state_token_chunks, state_index_chunks = [], [], []
    batch_starts = range(0, traj_len, int(feature_batch_size))
    if progress_batches:
        batch_starts = tqdm(batch_starts, desc=f"frames {src_traj.name}", leave=False, unit="batch")
    for start_idx in batch_starts:
        end_idx = min(start_idx + int(feature_batch_size), traj_len)
        raw_obs = _read_frame_batch(
            src_traj,
            state_keys=state_keys,
            camera_keys=camera_keys,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        features = agent.actor.extract_vlm_prefix_features(
            raw_obs,
            prompt=[prompt] * (end_idx - start_idx),
        )
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
        for traj_key in tqdm(traj_keys, desc=f"trajectories {input_h5.name}", unit="traj"):
            _export_trajectory(
                agent, src, dst, None if traj_key is None else str(traj_key),
                feature_batch_size=feature_batch_size, feature_dtype=feature_dtype,
                compression=compression, save_prefix_tokens=save_prefix_tokens,
                progress_batches=progress_batches,
            )
    return len(traj_keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic π₀.₅ VLM-prefix features to H5.")
    parser.add_argument("--config", required=True, help="Resolved π₀.₅ YAML config.")
    parser.add_argument("--pretrained-path", default=None, help="Local LeRobot π₀.₅ directory; overrides actor.pretrained_path.")
    parser.add_argument("--input", required=True, help="Input H5 file or folder containing H5 files.")
    parser.add_argument("--output", required=True, help="Output H5 file or output folder.")
    parser.add_argument("--device", default=None, help="Override cfg device, e.g. cuda:0.")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--feature-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    parser.add_argument("--save-prefix-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-batches", action="store_true")
    parser.add_argument("--max-traj", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    agent, _cfg = _load_pi05_agent(args.config, args.pretrained_path, args.device)
    compression = None if args.compression == "none" else args.compression
    total = 0
    for input_h5, output_h5 in _iter_input_outputs(Path(args.input), Path(args.output)):
        total += export_h5_file(
            agent, input_h5=input_h5, output_h5=output_h5,
            feature_batch_size=args.feature_batch_size, feature_dtype=_feature_dtype(args.feature_dtype),
            compression=compression, save_prefix_tokens=bool(args.save_prefix_tokens),
            progress_batches=bool(args.progress_batches), max_traj=args.max_traj, overwrite=args.overwrite,
        )
    print(f"[export_pi05_features] done, exported {total} trajectories")


if __name__ == "__main__":
    main()
