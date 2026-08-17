import argparse
import os
from pathlib import Path
from typing import Optional

import h5py


UTF8_DTYPE = h5py.string_dtype(encoding="utf-8")


def _traj_sort_key(name: str):
    parts = name.split("_")
    for token in parts[1:]:
        try:
            return int(token)
        except ValueError:
            continue
    return name


def _trajectory_keys(h5_file: h5py.File):
    keys = [
        key
        for key in h5_file.keys()
        if key.startswith("traj_") and isinstance(h5_file[key], h5py.Group)
    ]
    return sorted(keys, key=_traj_sort_key)


def _copy_h5_file(input_h5: str, output_h5: str):
    with h5py.File(input_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for key in src.keys():
            src.copy(key, dst, name=key)


def _delete_if_exists(group: h5py.Group, key: str):
    if key in group:
        del group[key]


def _write_scalar(group: h5py.Group, key: str, value):
    _delete_if_exists(group, key)
    if isinstance(value, str):
        group.create_dataset(key, data=value, dtype=UTF8_DTYPE)
    else:
        group.create_dataset(key, data=value)


def _add_prompt(group: h5py.Group, prompt: str):
    _write_scalar(group, "prompt", prompt)


def add_prompt_to_h5(
    input_h5: str,
    output_h5: str,
    prompt: str,
    overwrite: bool = False,
) -> int:
    input_path = Path(input_h5)
    output_path = Path(output_h5)
    if not input_path.exists():
        raise FileNotFoundError(f"Input H5 does not exist: {input_h5}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output H5 paths must be different.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output H5 already exists: {output_h5}")

    os.makedirs(output_path.parent or ".", exist_ok=True)
    _copy_h5_file(str(input_path), str(output_path))

    with h5py.File(output_path, "a") as h5_file:
        traj_keys = _trajectory_keys(h5_file)
        groups = [h5_file[key] for key in traj_keys] if traj_keys else [h5_file]
        for group in groups:
            _add_prompt(group, prompt=prompt)
        h5_file.attrs["has_prompt"] = True
        h5_file.attrs["prompt_mode"] = "scalar"
        h5_file.attrs["prompted_trajectories"] = len(groups)

    return len(groups)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy an H5 dataset and add a scalar prompt to every trajectory."
    )
    parser.add_argument("--input-h5", required=True, help="Input merged or single-trajectory H5 file.")
    parser.add_argument("--output-h5", required=True, help="Output H5 file with prompt metadata.")
    parser.add_argument("--prompt", required=True, help="Scalar language prompt for every trajectory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output H5 file.")
    return parser


def main(argv: Optional[list[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    count = add_prompt_to_h5(
        input_h5=args.input_h5,
        output_h5=args.output_h5,
        prompt=args.prompt,
        overwrite=args.overwrite,
    )
    print(f"[merge_h5_converter] wrote prompt metadata for {count} trajectories -> {args.output_h5}")


if __name__ == "__main__":
    main()
