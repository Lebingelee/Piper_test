"""
python -m agent_factory.script.convert_h5_to_lerobot \
  --input-h5 data/merged/can_task_merged_structure.h5 \
  --output-dir /tmp/can_lerobot_h264_smoke \
  --repo-id can_lerobot_h264_smoke \
  --fps 20 \
  --max-episodes 1 \
  --max-frames-per-episode 5 \
  --video-backend pyav \
  --video-codec h264 \
  --readback-check \
  --overwrite

python -m agent_factory.script.convert_h5_to_lerobot\
    --input-h5 data/merged/can_task_merged_structure.h5\
    --output-dir data/lerobot/can \
    --repo-id can_100 \
    --fps 20\
    --video-backend pyav   \
    --video-codec h264  \
    --readback-check  \
    --overwrite
"""

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Optional

from agent_factory.data.impl.lerobot.h5_utils import convert_h5_to_lerobot_dataset


def _ensure_local_hf_cache(output_dir: str):
    if "HF_HOME" not in os.environ:
        cache_dir = Path(output_dir) / ".hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
    if "HF_DATASETS_CACHE" not in os.environ:
        datasets_cache = Path(os.environ["HF_HOME"]) / "datasets"
        datasets_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)


def _flat_feature_names(prefix: str, dim: int) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(int(dim))]


def _image_axis_names(shape) -> list[str]:
    if len(shape) != 3:
        return [f"dim_{idx}" for idx in range(len(shape))]
    if shape[0] in (1, 3, 4):
        return ["channels", "height", "width"]
    if shape[-1] in (1, 3, 4):
        return ["height", "width", "channels"]
    return ["dim_0", "dim_1", "dim_2"]


def _ensure_lerobot_feature_names(output_dir: str) -> bool:
    info_path = Path(output_dir) / "meta" / "info.json"
    if not info_path.exists():
        return False

    info = json.loads(info_path.read_text(encoding="utf-8"))
    changed = False
    for key, feature in info.get("features", {}).items():
        if key == "observation.state" and not feature.get("names"):
            feature["names"] = _flat_feature_names("state", feature["shape"][0])
            changed = True
        elif key == "action" and not feature.get("names"):
            feature["names"] = _flat_feature_names("action", feature["shape"][0])
            changed = True
        elif feature.get("dtype") in {"image", "video"} and not feature.get("names"):
            feature["names"] = _image_axis_names(feature["shape"])
            changed = True

    if changed:
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert agent_factory structured H5 trajectories to a LeRobotDataset."
    )
    parser.add_argument("--input-h5", required=True, help="Input raw/merged structured H5 file.")
    parser.add_argument("--output-dir", required=True, help="Output LeRobotDataset root directory.")
    parser.add_argument("--repo-id", default=None, help="LeRobot repo_id. Defaults to output directory name.")
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS metadata.")
    parser.add_argument(
        "--default-prompt",
        default=None,
        help="Fallback prompt when a trajectory has no prompt dataset. Prefer running merge_h5_converter first.",
    )
    parser.add_argument("--no-videos", action="store_true", help="Store image frames instead of encoding videos.")
    parser.add_argument("--video-backend", default=None, help="Optional LeRobot video backend override.")
    parser.add_argument(
        "--video-codec",
        default=None,
        help="Optional RGB video codec for current LeRobot RGBEncoderConfig, e.g. h264, libsvtav1, auto.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit episodes for smoke conversion.")
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="Limit frames per episode for smoke conversion.",
    )
    parser.add_argument("--readback-check", action="store_true", help="Open the converted LeRobotDataset and print a small sanity check.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output directory.")
    return parser


def main(argv: Optional[list[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.readback_check:
        _ensure_local_hf_cache(args.output_dir)

    manifest = convert_h5_to_lerobot_dataset(
        input_h5=args.input_h5,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        default_prompt=args.default_prompt,
        use_videos=not args.no_videos,
        video_backend=args.video_backend,
        video_codec=args.video_codec,
        max_episodes=args.max_episodes,
        max_frames_per_episode=args.max_frames_per_episode,
        overwrite=args.overwrite,
    )
    patched_feature_names = _ensure_lerobot_feature_names(manifest["output_dir"])
    print(
        "[convert_h5_to_lerobot] converted "
        f"{len(manifest['episodes'])} episodes -> {manifest['output_dir']}"
    )
    if patched_feature_names:
        print("[convert_h5_to_lerobot] patched missing LeRobot feature names in meta/info.json")
    if args.readback_check:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset_kwargs = {
            "repo_id": manifest["repo_id"],
            "root": manifest["output_dir"],
            "video_backend": args.video_backend,
        }
        if "return_uint8" in inspect.signature(LeRobotDataset.__init__).parameters:
            dataset_kwargs["return_uint8"] = True
        dataset = LeRobotDataset(**dataset_kwargs)
        sample = dataset[0]
        sample_keys = sorted(str(key) for key in sample.keys())
        print(
            "[convert_h5_to_lerobot] readback "
            f"frames={dataset.num_frames} episodes={dataset.num_episodes} keys={sample_keys}"
        )


if __name__ == "__main__":
    main()
