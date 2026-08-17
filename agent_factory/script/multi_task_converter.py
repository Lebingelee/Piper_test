import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


UTF8_DTYPE = h5py.string_dtype(encoding="utf-8")


def _traj_sort_key(name: str):
    parts = name.split("_")
    for token in parts[1:]:
        try:
            return int(token)
        except ValueError:
            continue
    return name


def _trajectory_keys(h5_file: h5py.File) -> List[Optional[str]]:
    keys = [
        key
        for key in h5_file.keys()
        if key.startswith("traj_") and isinstance(h5_file[key], h5py.Group)
    ]
    return sorted(keys, key=_traj_sort_key) or [None]


def _trajectory_group(h5_file: h5py.File, traj_key: Optional[str]) -> h5py.Group:
    return h5_file if traj_key is None else h5_file[traj_key]


def _read_dataset_value(group: h5py.Group, key: str):
    value = group[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.generic):
            return value.item()
    return value


def _read_required_scalar(group: h5py.Group, key: str):
    if key not in group or not isinstance(group[key], h5py.Dataset):
        raise KeyError(f"Trajectory {group.name} is missing required dataset: {key}")
    return _read_dataset_value(group, key)


def _read_json_dataset(dataset: h5py.Dataset) -> Any:
    value = _read_dataset_value(dataset.parent, dataset.name.rsplit("/", 1)[-1])
    return json.loads(value)


def _load_env_meta(h5_file: h5py.File, traj_group: h5py.Group) -> Dict[str, Any]:
    if "meta" in traj_group and isinstance(traj_group["meta"], h5py.Group) and "env_meta" in traj_group["meta"]:
        return _read_json_dataset(traj_group["meta"]["env_meta"])
    if "meta" in h5_file and isinstance(h5_file["meta"], h5py.Group) and "env_meta" in h5_file["meta"]:
        return _read_json_dataset(h5_file["meta"]["env_meta"])
    raise KeyError(f"Trajectory {traj_group.name} does not contain meta/env_meta.")


def _json_signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _schema_from_meta(env_meta: Dict[str, Any]) -> Dict[str, Any]:
    obs_meta = env_meta.get("obs") or {}
    return {
        "state": obs_meta.get("state") or {},
        "rgb": obs_meta.get("rgb") or {},
        "action": env_meta.get("action") or {},
    }


def _schema_from_trajectory(traj_group: h5py.Group) -> Dict[str, Any]:
    def group_shapes(group: h5py.Group) -> Dict[str, List[int]]:
        shapes: Dict[str, List[int]] = {}
        for key in sorted(group.keys()):
            item = group[key]
            if isinstance(item, h5py.Dataset):
                shapes[key] = list(item.shape[1:])
        return shapes

    schema = {"state": {}, "rgb": {}, "action": {}}
    if "obs" in traj_group and isinstance(traj_group["obs"], h5py.Group):
        obs_group = traj_group["obs"]
        if "state" in obs_group and isinstance(obs_group["state"], h5py.Group):
            schema["state"] = group_shapes(obs_group["state"])
        if "rgb" in obs_group and isinstance(obs_group["rgb"], h5py.Group):
            schema["rgb"] = group_shapes(obs_group["rgb"])
    if "action" in traj_group and isinstance(traj_group["action"], h5py.Group):
        schema["action"] = group_shapes(traj_group["action"])
    return schema


def _schema_for_trajectory(h5_file: h5py.File, traj_group: h5py.Group) -> Dict[str, Any]:
    try:
        return _schema_from_meta(_load_env_meta(h5_file, traj_group))
    except KeyError:
        return _schema_from_trajectory(traj_group)


def _write_json_dataset(group: h5py.Group, key: str, value: Any):
    if key in group:
        del group[key]
    group.create_dataset(key, data=json.dumps(_to_jsonable(value), ensure_ascii=False), dtype=UTF8_DTYPE)


def _to_jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _delete_if_exists(group: h5py.Group, key: str):
    if key in group:
        del group[key]


def _write_scalar(group: h5py.Group, key: str, value):
    _delete_if_exists(group, key)
    if isinstance(value, str):
        group.create_dataset(key, data=value, dtype=UTF8_DTYPE)
    else:
        group.create_dataset(key, data=value)


def _trajectory_success(traj_group: h5py.Group) -> bool:
    if "success" in traj_group and isinstance(traj_group["success"], h5py.Dataset):
        raw = np.asarray(traj_group["success"][()], dtype=np.bool_)
        if raw.shape == ():
            return bool(raw.item())
        return bool(raw.reshape(-1).any())
    if "success" in traj_group.attrs:
        return bool(traj_group.attrs["success"])
    return False


def _parse_input_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        path = spec
        label = Path(path).stem
        return label, path
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --input-h5 spec: {spec}")
    return label, path


def _first_trajectory_info(label: str, path: str) -> Dict[str, Any]:
    with h5py.File(path, "r") as h5_file:
        traj_key = _trajectory_keys(h5_file)[0]
        traj_group = _trajectory_group(h5_file, traj_key)
        return {
            "input_label": label,
            "input_h5": path,
            "schema": _schema_for_trajectory(h5_file, traj_group),
            "env_meta": _load_env_meta(h5_file, traj_group),
        }


def _validate_schema_compatibility(task_infos: List[Dict[str, Any]]):
    if not task_infos:
        return
    reference = task_infos[0]["schema"]
    mismatches = []
    for info in task_infos[1:]:
        if info["schema"] != reference:
            mismatches.append(
                {
                    "input_label": info["input_label"],
                    "input_h5": info["input_h5"],
                    "schema": info["schema"],
                }
            )
    if mismatches:
        report = {
            "reference": {
                "input_label": task_infos[0]["input_label"],
                "input_h5": task_infos[0]["input_h5"],
                "schema": reference,
            },
            "mismatches": mismatches,
        }
        raise ValueError(
            "Input H5 files have incompatible action/state/camera schema: "
            + json.dumps(report, ensure_ascii=False, sort_keys=True)
        )


def merge_prompted_h5_files(input_specs: List[str], output_h5: str, overwrite: bool = False) -> int:
    parsed_inputs = [_parse_input_spec(spec) for spec in input_specs]
    if not parsed_inputs:
        raise ValueError("At least one --input-h5 is required.")
    for _, path in parsed_inputs:
        if not Path(path).exists():
            raise FileNotFoundError(f"Input H5 does not exist: {path}")

    output_path = Path(output_h5)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output H5 already exists: {output_h5}")
    os.makedirs(output_path.parent or ".", exist_ok=True)

    task_infos = [_first_trajectory_info(label, path) for label, path in parsed_inputs]
    _validate_schema_compatibility(task_infos)

    env_meta_signatures = {
        info["input_label"]: _json_signature(info["env_meta"])
        for info in task_infos
    }
    env_meta_identical = len(set(env_meta_signatures.values())) == 1

    merged_count = 0
    source_count = 0
    merge_items: List[Dict[str, Any]] = []
    task_manifest: List[Dict[str, Any]] = []

    with h5py.File(output_path, "w") as h5_out:
        for attr_key, attr_value in {
            "merged": True,
            "multi_task": True,
        }.items():
            h5_out.attrs[attr_key] = attr_value

        for info in task_infos:
            label = info["input_label"]
            path = info["input_h5"]
            task_start = merged_count
            task_traj_count = 0
            with h5py.File(path, "r") as h5_in:
                for traj_key in _trajectory_keys(h5_in):
                    src_group = _trajectory_group(h5_in, traj_key)
                    _read_required_scalar(src_group, "prompt")
                    dst_key = f"traj_{merged_count}"
                    h5_in.copy(src_group.name, h5_out, name=dst_key)
                    dst_group = h5_out[dst_key]
                    dst_group.attrs["input_label"] = label
                    dst_group.attrs["source_h5"] = path
                    if traj_key is not None:
                        dst_group.attrs["source_traj"] = traj_key
                    dst_group.attrs["success"] = _trajectory_success(dst_group)

                    merge_items.append(
                        {
                            "source_file": path,
                            "source_traj": traj_key,
                            "input_label": label,
                            "merged_traj": dst_key,
                            "prompt": _read_required_scalar(dst_group, "prompt"),
                            "success": bool(dst_group.attrs["success"]),
                        }
                    )
                    merged_count += 1
                    source_count += 1
                    task_traj_count += 1

            task_manifest.append(
                {
                    "input_label": label,
                    "input_h5": path,
                    "trajectory_count": task_traj_count,
                    "output_traj_start": task_start,
                    "output_traj_end": merged_count - 1 if task_traj_count else None,
                    "schema": info["schema"],
                    "env_meta_signature": env_meta_signatures[label],
                }
            )

        meta_group = h5_out.create_group("meta")
        _write_json_dataset(meta_group, "task_manifest", task_manifest)
        _write_json_dataset(meta_group, "source_files", [{"input_label": label, "input_h5": path} for label, path in parsed_inputs])
        _write_json_dataset(
            meta_group,
            "converter_config",
            {
                "converter": "agent_factory.script.multi_task_converter",
                "input_specs": input_specs,
                "schema_compatibility": "compatible",
                "env_meta_identical": env_meta_identical,
                "env_meta_signatures": env_meta_signatures,
                "env_meta_source": "traj_xxx/meta/env_meta",
            },
        )
        _write_json_dataset(
            meta_group,
            "merge_info",
            {
                "source_trajectories": source_count,
                "merged_trajectories": merged_count,
                "items": merge_items,
            },
        )
        h5_out.attrs["num_trajectories"] = merged_count
        h5_out.attrs["source_trajectories"] = source_count
        h5_out.attrs["env_meta_identical"] = env_meta_identical

    return merged_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multiple prompted task H5 files into one globally renumbered multi-task H5."
    )
    parser.add_argument(
        "--input-h5",
        action="append",
        required=True,
        help="Input spec in the form taskLabel=/path/to/task.h5. May be repeated.",
    )
    parser.add_argument("--output-h5", required=True, help="Output multi-task H5 file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output H5 file.")
    return parser


def main(argv: Optional[list[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    count = merge_prompted_h5_files(args.input_h5, args.output_h5, overwrite=args.overwrite)
    print(f"[multi_task_converter] merged {count} trajectories -> {args.output_h5}")


if __name__ == "__main__":
    main()
