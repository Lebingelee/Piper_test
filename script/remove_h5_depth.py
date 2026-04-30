import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode_json_dataset(dataset: h5py.Dataset) -> dict:
    value: Any = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return json.loads(value)


def _print_meta_keys(title: str, meta: dict) -> None:
    print(f"\n========== {title} ==========")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    obs_keys = sorted(meta.get("obs", {}).keys())
    action_keys = sorted(meta.get("action", {}).keys())
    print(f"[MetaSummary] obs keys: {obs_keys}")
    print(f"[MetaSummary] action keys: {action_keys}")


def _remove_depth_from_meta(meta: dict) -> dict:
    new_meta = json.loads(json.dumps(meta))
    new_meta.get("obs", {}).pop("depth", None)
    return new_meta


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _copy_dataset(src: h5py.Dataset, dst_group: h5py.Group, name: str) -> None:
    src.parent.copy(src, dst_group, name=name)


def _copy_group_without_depth(src_group: h5py.Group, dst_group: h5py.Group, new_meta: dict) -> int:
    removed = 0
    _copy_attrs(src_group.attrs, dst_group.attrs)

    for key in src_group.keys():
        src_item = src_group[key]

        if key == "depth" and src_group.name.endswith("/obs"):
            print(f"[RemoveDepth] Skip group: {src_item.name}")
            removed += 1
            continue

        if src_item.name == "/meta/env_meta":
            dst = dst_group.create_dataset(key, data=json.dumps(new_meta))
            _copy_attrs(src_item.attrs, dst.attrs)
            continue

        if isinstance(src_item, h5py.Group):
            child = dst_group.create_group(key)
            removed += _copy_group_without_depth(src_item, child, new_meta)
        else:
            _copy_dataset(src_item, dst_group, key)

    return removed


def strip_depth(input_path: Path, output_path: Path, overwrite: bool = False) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input H5 not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    input_size = input_path.stat().st_size
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(input_path, "r") as src:
        if "meta/env_meta" not in src:
            raise KeyError("Input H5 does not contain meta/env_meta.")

        old_meta = _decode_json_dataset(src["meta/env_meta"])
        new_meta = _remove_depth_from_meta(old_meta)
        _print_meta_keys("Original meta/env_meta", old_meta)

        with h5py.File(tmp_path, "w") as dst:
            _copy_attrs(src.attrs, dst.attrs)
            removed_groups = _copy_group_without_depth(src, dst, new_meta)

    if output_path.exists():
        output_path.unlink()
    tmp_path.rename(output_path)

    output_size = output_path.stat().st_size
    _print_meta_keys("New meta/env_meta", new_meta)
    print("\n========== Size Check ==========")
    print(f"Input : {input_path} ({input_size / (1024 ** 3):.3f} GiB)")
    print(f"Output: {output_path} ({output_size / (1024 ** 3):.3f} GiB)")
    print(f"Removed obs/depth groups: {removed_groups}")

    if output_size >= input_size:
        raise RuntimeError(
            "Output file is not smaller than input. "
            "Please inspect the H5 structure before replacing the original file."
        )

    print("[OK] New H5 file is smaller and no traj_x/obs/depth group was copied.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact H5 copy without traj_x/obs/depth.")
    parser.add_argument(
        "-i",
        "--input",
        default="data/merged/dual_merged copy.h5",
        help="Input H5 path. Defaults to the backup file with a space in its name.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/merged/dual_merged_no_depth.h5",
        help="Output H5 path.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    strip_depth(Path(args.input), Path(args.output), overwrite=args.overwrite)
