#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

os.environ.setdefault("IMAGEIO_USERDIR", "/tmp/imageio")

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    import imageio.v2 as imageio

    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio

        HAS_IMAGEIO = True
    except ImportError:
        imageio = None
        HAS_IMAGEIO = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample RGB keyframes from an H5 trajectory and save them as a video."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input H5 file path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output video path. Defaults to <input_stem>_rgb_strideN.mp4 next to the H5 file.",
    )
    parser.add_argument(
        "--traj-key",
        type=str,
        default=None,
        help="Trajectory group name such as traj_0000. If omitted, the first traj_* group is used.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Extract one frame every N frames. 1 means keep every frame, 10 means keep one every 10 frames.",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=10.0,
        help="Estimated original trajectory FPS, used when --fps is not provided.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video FPS. Defaults to source_fps / sample_every to preserve approximate playback speed.",
    )
    parser.add_argument(
        "--save-frames-dir",
        type=Path,
        default=None,
        help="Optional directory for saving sampled keyframes as PNG images.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable camera name labels on the output frames.",
    )
    return parser.parse_args()


def _decode_json_dataset(dataset: h5py.Dataset) -> Optional[Dict]:
    raw = dataset[()]
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _select_traj_root(h5_file: h5py.File, traj_key: Optional[str]) -> Tuple[h5py.Group, str]:
    if "obs" in h5_file and "rgb" in h5_file["obs"]:
        return h5_file, "<root>"

    traj_keys = sorted(
        key
        for key in h5_file.keys()
        if key.startswith("traj_") and isinstance(h5_file[key], h5py.Group)
    )
    if not traj_keys:
        raise KeyError("No root obs/rgb dataset or traj_* group found in the H5 file.")

    if traj_key is None:
        traj_key = traj_keys[0]
    elif traj_key not in h5_file:
        raise KeyError(f"Trajectory group not found: {traj_key}")

    return h5_file[traj_key], traj_key


def _read_camera_names(h5_file: h5py.File, traj_root: h5py.Group, num_cameras: int) -> List[str]:
    meta = None
    if "meta" in h5_file and "env_meta" in h5_file["meta"]:
        meta = _decode_json_dataset(h5_file["meta"]["env_meta"])
    elif "meta" in traj_root and "env_meta" in traj_root["meta"]:
        meta = _decode_json_dataset(traj_root["meta"]["env_meta"])

    if isinstance(meta, dict):
        rgb_meta = meta.get("obs", {}).get("rgb", {})
        if isinstance(rgb_meta, dict) and rgb_meta:
            names = list(rgb_meta.keys())
            if len(names) == num_cameras:
                return names

    return [f"camera_{idx:02d}" for idx in range(num_cameras)]


def _to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if array.dtype == np.uint8:
        return array

    array = array.astype(np.float32)
    if array.max() <= 1.0 + 1e-6:
        array *= 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _read_rgb_frame(
    rgb_node,
    frame_idx: int,
    camera_names: Sequence[str],
) -> List[np.ndarray]:
    if isinstance(rgb_node, h5py.Dataset):
        frame = rgb_node[frame_idx]
        if frame.ndim != 3 or frame.shape[0] % 3 != 0:
            raise ValueError(
                f"Expected obs/rgb dataset shape [T, 3*num_cameras, H, W], got frame shape {frame.shape}."
            )
        num_cameras = frame.shape[0] // 3
        return [_to_uint8_hwc(frame[cam_idx * 3:(cam_idx + 1) * 3]) for cam_idx in range(num_cameras)]

    frames = []
    for name in camera_names:
        if name not in rgb_node:
            raise KeyError(f"RGB camera key not found in group data: {name}")
        frames.append(_to_uint8_hwc(rgb_node[name][frame_idx]))
    return frames


def _compose_frame(camera_frames: Sequence[np.ndarray], camera_names: Sequence[str], add_labels: bool) -> np.ndarray:
    if not camera_frames:
        raise ValueError("No camera frames to compose.")

    target_h, target_w = camera_frames[0].shape[:2]
    panels = []
    for idx, frame in enumerate(camera_frames):
        panel = frame
        if panel.shape[:2] != (target_h, target_w):
            if not HAS_CV2:
                raise ValueError("Camera frame sizes differ and opencv-python is unavailable for resizing.")
            panel = cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_AREA)

        panel = panel.copy()
        if add_labels and HAS_CV2:
            cv2.putText(
                panel,
                camera_names[idx],
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                camera_names[idx],
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def _open_video_writer(output_path: Path, fps: float, frame_size: Tuple[int, int]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = frame_size

    if HAS_CV2 and output_path.suffix.lower() != ".gif":
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_path}")
        return writer, "cv2"

    if not HAS_IMAGEIO:
        raise ImportError("opencv-python or imageio is required to save the video.")

    imageio_root = Path(os.environ["IMAGEIO_USERDIR"]) / ".imageio"
    imageio_root.mkdir(parents=True, exist_ok=True)
    try:
        return imageio.get_writer(str(output_path), fps=fps), "imageio"
    except Exception as exc:
        raise RuntimeError(
            "Failed to open the video writer. For mp4 output, install opencv-python or ffmpeg support for imageio. "
            "You can also export to .gif as a fallback."
        ) from exc


def _write_video_frame(writer, backend: str, frame_rgb: np.ndarray) -> None:
    if backend == "cv2":
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        return
    writer.append_data(frame_rgb)


def _close_video_writer(writer, backend: str) -> None:
    if backend == "cv2":
        writer.release()
        return
    writer.close()


def _save_png(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_CV2:
        cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        return
    if HAS_IMAGEIO:
        imageio.imwrite(str(path), frame_rgb)
        return
    raise ImportError("opencv-python or imageio is required to save PNG keyframes.")


def _infer_num_frames(rgb_node, camera_names: Sequence[str]) -> int:
    if isinstance(rgb_node, h5py.Dataset):
        return int(rgb_node.shape[0])
    return min(int(rgb_node[name].shape[0]) for name in camera_names)


def export_h5_rgb_video(
    input_path: Path,
    output_path: Path,
    traj_key: Optional[str],
    sample_every: int,
    video_fps: float,
    save_frames_dir: Optional[Path],
    add_labels: bool,
) -> None:
    if sample_every <= 0:
        raise ValueError("--sample-every must be a positive integer.")
    if video_fps <= 0:
        raise ValueError("Video FPS must be positive.")

    with h5py.File(input_path, "r") as h5_file:
        traj_root, selected_traj_key = _select_traj_root(h5_file, traj_key)
        if "obs" not in traj_root or "rgb" not in traj_root["obs"]:
            raise KeyError(f"Trajectory {selected_traj_key} does not contain obs/rgb.")
        rgb_node = traj_root["obs"]["rgb"]

        if isinstance(rgb_node, h5py.Dataset):
            if rgb_node.ndim != 4 or rgb_node.shape[1] % 3 != 0:
                raise ValueError(
                    f"Unsupported obs/rgb dataset shape {rgb_node.shape}; expected [T, 3*num_cameras, H, W]."
                )
            num_cameras = int(rgb_node.shape[1] // 3)
        else:
            num_cameras = len(list(rgb_node.keys()))
        camera_names = _read_camera_names(h5_file, traj_root, num_cameras)
        if not isinstance(rgb_node, h5py.Dataset):
            camera_names = [name for name in camera_names if name in rgb_node]
            if not camera_names:
                camera_names = sorted(rgb_node.keys())

        num_frames = _infer_num_frames(rgb_node, camera_names)
        sampled_indices = list(range(0, num_frames, sample_every))
        if not sampled_indices:
            raise ValueError("No frames selected after sampling.")

        print(f"[Export] Input H5: {input_path}")
        print(f"[Export] Selected trajectory: {selected_traj_key}")
        print(f"[Export] RGB cameras: {camera_names}")
        print(f"[Export] Total RGB frames: {num_frames}")
        print(f"[Export] Sample every: {sample_every}")
        print(f"[Export] Sampled frames: {len(sampled_indices)}")
        print(f"[Export] Output FPS: {video_fps:.3f}")
        print(f"[Export] Output video: {output_path}")
        if save_frames_dir is not None:
            print(f"[Export] Save sampled PNGs to: {save_frames_dir}")

        writer = None
        backend = None
        try:
            for sample_id, frame_idx in enumerate(sampled_indices):
                camera_frames = _read_rgb_frame(rgb_node, frame_idx, camera_names)
                frame_rgb = _compose_frame(camera_frames, camera_names, add_labels)

                if save_frames_dir is not None:
                    frame_name = f"frame_{frame_idx:06d}_sample_{sample_id:06d}.png"
                    _save_png(save_frames_dir / frame_name, frame_rgb)

                if writer is None:
                    height, width = frame_rgb.shape[:2]
                    writer, backend = _open_video_writer(output_path, video_fps, (width, height))

                _write_video_frame(writer, backend, frame_rgb)
        finally:
            if writer is not None and backend is not None:
                _close_video_writer(writer, backend)

    print("[Export] Finished.")


def main() -> None:
    args = _parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input H5 file not found: {args.input}")

    output_path = args.output
    if output_path is None:
        output_path = args.input.with_name(
            f"{args.input.stem}_rgb_stride{args.sample_every}.mp4"
        )

    video_fps = args.fps
    if video_fps is None:
        video_fps = max(1.0, args.source_fps / args.sample_every)

    export_h5_rgb_video(
        input_path=args.input,
        output_path=output_path,
        traj_key=args.traj_key,
        sample_every=args.sample_every,
        video_fps=video_fps,
        save_frames_dir=args.save_frames_dir,
        add_labels=not args.no_labels,
    )


if __name__ == "__main__":
    main()
