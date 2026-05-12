import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("IMAGEIO_USERDIR", "/tmp/imageio")

import h5py
import matplotlib

if "--play" not in sys.argv or not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    imageio = None
    HAS_IMAGEIO = False

try:
    from PIL import Image, ImageDraw

    HAS_PIL = True
except ImportError:
    Image = None
    ImageDraw = None
    HAS_PIL = False


DEFAULT_H5_ROOT = (
    Path(__file__).resolve().parent
    / "data"
    / "piper_towel_h5_joint_task"
    / "h5_raw"
)


def _natural_traj_key(path: Path) -> Tuple[int, str]:
    stem_parts = path.stem.split("_")
    if len(stem_parts) >= 2 and stem_parts[0] == "traj":
        try:
            return int(stem_parts[1]), path.name
        except ValueError:
            pass
    return math.inf, path.name


def _collect_h5_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix != ".h5":
            raise ValueError(f"Input file is not a .h5 file: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(input_path.glob("*.h5"), key=_natural_traj_key)


def _list_traj_groups(h5_file: h5py.File) -> List[str]:
    return sorted(
        [
            key
            for key in h5_file.keys()
            if key.startswith("traj_") and isinstance(h5_file[key], h5py.Group)
        ],
        key=lambda name: int(name.split("_", 1)[1]) if name.split("_", 1)[1].isdigit() else math.inf,
    )


def _select_trajectory(input_path: Path, k: int) -> Tuple[Path, Optional[str]]:
    h5_files = _collect_h5_files(input_path)
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under: {input_path}")

    if len(h5_files) == 1:
        with h5py.File(h5_files[0], "r") as h5_file:
            traj_groups = _list_traj_groups(h5_file)
        if traj_groups:
            if k < 0 or k >= len(traj_groups):
                raise IndexError(f"k={k} is out of range for {len(traj_groups)} trajectories.")
            return h5_files[0], traj_groups[k]

    if k < 0 or k >= len(h5_files):
        raise IndexError(f"k={k} is out of range for {len(h5_files)} .h5 files.")
    return h5_files[k], None


def _read_leaf_group(group: h5py.Group) -> Dict[str, np.ndarray]:
    return {key: group[key][()] for key in group.keys() if isinstance(group[key], h5py.Dataset)}


def _get_traj_root(h5_file: h5py.File, traj_name: Optional[str]):
    return h5_file if traj_name is None else h5_file[traj_name]


def _to_hwc_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return image


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 4:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    depth = depth.astype(np.float32)
    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((depth - lo) / (hi - lo), 0.0, 1.0)


def _resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if HAS_CV2:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    if HAS_PIL:
        pil_image = Image.fromarray(image)
        return np.asarray(pil_image.resize(size, Image.BILINEAR))
    raise ImportError("Pillow or opencv-python is required to resize video frames.")


def _draw_title(image: np.ndarray, title: str) -> np.ndarray:
    if HAS_CV2:
        cv2.putText(
            image,
            title,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            title,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        return image
    if HAS_PIL:
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        draw.text((8, 8), title, fill=(255, 255, 255))
        return np.asarray(pil_image)
    return image


def _make_video_panel(title: str, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    image = _to_hwc_rgb(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    image = _resize_image(image, size)
    return _draw_title(image, title)


def _make_depth_panel(title: str, depth: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    depth = (_normalize_depth(depth) * 255).astype(np.uint8)
    if HAS_CV2:
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
    else:
        depth = np.asarray(plt.get_cmap("viridis")(depth / 255.0)[..., :3] * 255, dtype=np.uint8)
    return _make_video_panel(title, depth, size)


def _write_video(
    output_path: Path,
    rgb: Dict[str, np.ndarray],
    depth: Dict[str, np.ndarray],
    fps: float,
    step: int,
    image_size: Tuple[int, int],
    include_depth: bool,
):
    if not HAS_CV2 and not HAS_IMAGEIO:
        raise ImportError("opencv-python or imageio is required for --save-video.")
    if not rgb and not depth:
        print("[Visualize] No rgb/depth frames found to save as video.")
        return

    panel_sources = [("rgb", name, frames) for name, frames in rgb.items()]
    if include_depth:
        panel_sources.extend(("depth", name, frames) for name, frames in depth.items())
    if not panel_sources:
        print("[Visualize] No selected video panels found.")
        return

    num_frames = min(frames.shape[0] for _, _, frames in panel_sources)
    cols = min(3, len(panel_sources))
    rows = int(math.ceil(len(panel_sources) / cols))
    panel_w, panel_h = image_size
    canvas_size = (cols * panel_w, rows * panel_h)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_gif = output_path.suffix.lower() == ".gif"
    if HAS_CV2 and not use_gif:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            canvas_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_path}")
    else:
        try:
            writer = imageio.get_writer(str(output_path), fps=fps)
        except Exception as exc:
            raise RuntimeError(
                "Failed to open video writer. Install opencv-python for mp4 output, "
                "or save as .gif with --save-video output.gif."
            ) from exc

    for frame_idx in range(0, num_frames, max(1, step)):
        canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        for panel_idx, (kind, name, frames) in enumerate(panel_sources):
            row = panel_idx // cols
            col = panel_idx % cols
            title = f"{kind}/{name}"
            if kind == "depth":
                panel = _make_depth_panel(title, frames[frame_idx], image_size)
            else:
                panel = _make_video_panel(title, frames[frame_idx], image_size)
            y0 = row * panel_h
            x0 = col * panel_w
            canvas[y0:y0 + panel_h, x0:x0 + panel_w] = panel
        if HAS_CV2 and not use_gif:
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        else:
            writer.append_data(canvas)

    if HAS_CV2 and not use_gif:
        writer.release()
    else:
        writer.close()
    print(f"[Visualize] Saved video: {output_path}")


def _plot_time_series(
    ax,
    values: np.ndarray,
    title: str,
    ylabel: str,
    max_dims: int = 6,
):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    dims = min(values.shape[1], max_dims)
    for dim in range(dims):
        ax.plot(values[:, dim], label=f"d{dim}", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if dims > 1:
        ax.legend(loc="upper right", fontsize=8, ncol=2)


def _save_curves(
    output_path: Path,
    state: Dict[str, np.ndarray],
    action: Dict[str, np.ndarray],
    under_control: Dict[str, np.ndarray],
):
    plot_items = []
    for key in ["left_joint_pos", "right_joint_pos", "left_gripper_pos", "right_gripper_pos"]:
        if key in state:
            plot_items.append(("state", key, state[key]))
    for key in ["left_arm", "right_arm", "left_gripper", "right_gripper"]:
        if key in action:
            plot_items.append(("action", key, action[key]))
    for key, value in under_control.items():
        plot_items.append(("under_control", key, value.astype(np.float32)))

    if not plot_items:
        print("[Visualize] No state/action curves found to plot.")
        return

    fig, axes = plt.subplots(len(plot_items), 1, figsize=(12, max(3, 2.2 * len(plot_items))), squeeze=False)
    for ax, (group_name, key, values) in zip(axes[:, 0], plot_items):
        _plot_time_series(ax, values, f"{group_name}/{key}", key)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[Visualize] Saved curves: {output_path}")


def _save_frame_grid(
    output_path: Path,
    rgb: Dict[str, np.ndarray],
    depth: Dict[str, np.ndarray],
    frame_idx: int,
):
    panels = []
    for name, frames in rgb.items():
        panels.append((f"rgb/{name}", _to_hwc_rgb(frames[frame_idx]), "rgb"))
    for name, frames in depth.items():
        panels.append((f"depth/{name}", _normalize_depth(frames[frame_idx]), "depth"))

    if not panels:
        print("[Visualize] No rgb/depth frames found to save.")
        return

    cols = min(3, len(panels))
    rows = int(math.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, image, mode) in zip(axes.ravel(), panels):
        if mode == "depth":
            ax.imshow(image, cmap="viridis")
        else:
            ax.imshow(image)
        ax.set_title(title)
    fig.suptitle(f"frame {frame_idx}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[Visualize] Saved frame grid: {output_path}")


def _play_rgb(rgb: Dict[str, np.ndarray], step: int, fps: float):
    if not rgb:
        print("[Visualize] No RGB data found; skip playback.")
        return

    camera_names = list(rgb.keys())
    num_frames = min(frames.shape[0] for frames in rgb.values())
    interval = 1000.0 / max(fps, 1e-6)

    fig, axes = plt.subplots(1, len(camera_names), figsize=(4 * len(camera_names), 4), squeeze=False)
    image_artists = []
    for ax, name in zip(axes[0], camera_names):
        ax.axis("off")
        ax.set_title(name)
        artist = ax.imshow(_to_hwc_rgb(rgb[name][0]))
        image_artists.append((artist, name))
    fig.tight_layout()

    for frame_idx in range(0, num_frames, step):
        for artist, name in image_artists:
            artist.set_data(_to_hwc_rgb(rgb[name][frame_idx]))
        fig.suptitle(f"frame {frame_idx}/{num_frames - 1}")
        plt.pause(interval / 1000.0)
        if not plt.fignum_exists(fig.number):
            break
    plt.show(block=False)


def visualize_h5(
    input_path: Path,
    k: int,
    frame_idx: Optional[int],
    save_dir: Optional[Path],
    save_video: Optional[Path],
    play: bool,
    step: int,
    fps: float,
    video_size: Tuple[int, int],
    include_depth_video: bool,
):
    h5_path, traj_name = _select_trajectory(input_path, k)
    with h5py.File(h5_path, "r") as h5_file:
        traj = _get_traj_root(h5_file, traj_name)
        state = _read_leaf_group(traj["obs"]["state"]) if "obs" in traj and "state" in traj["obs"] else {}
        action = _read_leaf_group(traj["action"]) if "action" in traj else {}
        rgb = _read_leaf_group(traj["obs"]["rgb"]) if "obs" in traj and "rgb" in traj["obs"] else {}
        depth = _read_leaf_group(traj["obs"]["depth"]) if "obs" in traj and "depth" in traj["obs"] else {}
        under_control = (
            _read_leaf_group(traj["obs"]["under_control"])
            if "obs" in traj and "under_control" in traj["obs"]
            else {}
        )
        num_frames = min(
            [value.shape[0] for value in [*state.values(), *action.values(), *rgb.values(), *depth.values()]]
            or [0]
        )

    label = f"{h5_path.name}" if traj_name is None else f"{h5_path.name}:{traj_name}"
    print(f"[Visualize] Selected trajectory k={k}: {label}")
    print(f"[Visualize] Frames: {num_frames}")
    if state:
        print(f"[Visualize] State keys: {list(state.keys())}")
    if action:
        print(f"[Visualize] Action keys: {list(action.keys())}")
    if rgb:
        print(f"[Visualize] RGB cameras: {list(rgb.keys())}")
    if depth:
        print(f"[Visualize] Depth cameras: {list(depth.keys())}")

    if save_dir is not None:
        save_prefix = save_dir / f"k{k}_{h5_path.stem}"
        _save_curves(save_prefix.with_name(save_prefix.name + "_curves.png"), state, action, under_control)
        if num_frames > 0:
            selected_frame = frame_idx if frame_idx is not None else num_frames // 2
            selected_frame = max(0, min(selected_frame, num_frames - 1))
            _save_frame_grid(save_prefix.with_name(save_prefix.name + f"_frame_{selected_frame}.png"), rgb, depth, selected_frame)

    if save_video is not None:
        video_path = save_video
        if save_video.is_dir() or save_video.suffix == "":
            video_path = save_video / f"k{k}_{h5_path.stem}.mp4"
        _write_video(video_path, rgb, depth, fps, step, video_size, include_depth_video)

    if play:
        _play_rgb(rgb, max(1, step), fps)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the k-th Piper H5 trajectory with RGB frames and state/action curves."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_H5_ROOT,
        help="H5 raw directory or a single .h5 file. Defaults to piper_towel_h5_joint_task/h5_raw.",
    )
    parser.add_argument("-k", type=int, default=0, help="Trajectory index, zero-based.")
    parser.add_argument("--frame", type=int, default=None, help="Frame index to save as an image grid.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "visualizations",
        help="Directory for saved PNG visualizations. Use --no-save to disable.",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save PNG visualizations.")
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Save RGB camera playback as .mp4. Pass a file path or output directory.",
    )
    parser.add_argument(
        "--video-size",
        type=int,
        nargs=2,
        default=(224, 224),
        metavar=("WIDTH", "HEIGHT"),
        help="Per-camera panel size in the saved video.",
    )
    parser.add_argument("--video-depth", action="store_true", help="Include depth cameras in the saved video grid.")
    parser.add_argument("--play", action="store_true", help="Play RGB cameras interactively with matplotlib.")
    parser.add_argument("--step", type=int, default=1, help="Playback stride.")
    parser.add_argument("--fps", type=float, default=20.0, help="Playback FPS.")
    args = parser.parse_args()

    visualize_h5(
        input_path=args.input,
        k=args.k,
        frame_idx=args.frame,
        save_dir=None if args.no_save else args.save_dir,
        save_video=args.save_video,
        play=args.play,
        step=args.step,
        fps=args.fps,
        video_size=tuple(args.video_size),
        include_depth_video=args.video_depth,
    )


if __name__ == "__main__":
    main()
