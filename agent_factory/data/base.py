import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
from torch.utils.data.dataset import Dataset


@dataclass(frozen=True)
class TrajectoryRef:
    """
    Reference to one trajectory inside an H5 file.

    `traj_key=None` means the H5 file itself is a single trajectory with root
    groups such as `obs/` and `action/`.
    """
    file_path: str
    traj_key: Optional[str] = None


def traj_sort_key(name: str):
    parts = name.split("_")
    for token in parts[1:]:
        try:
            return int(token)
        except ValueError:
            continue
    return name


class BaseTrajectoryDataset(Dataset):
    """
    Minimal trajectory dataset base.

    This class only discovers and opens H5 trajectories. It intentionally does
    not define a training schema; subclasses should implement slicing,
    flattening, reward construction, and batch fields.
    """

    def __init__(
        self,
        h5_path: Optional[str] = None,
        folder_path: Optional[str] = None,
        num_traj: Optional[int] = None,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.folder_path = folder_path
        self.num_traj = num_traj
        self.trajectory_refs: List[TrajectoryRef] = self._discover_trajectories()
        self.handles: Dict[str, h5py.File] = {}

    def _collect_h5_files(self) -> List[str]:
        paths: List[str] = []
        if self.h5_path:
            path = Path(self.h5_path)
            if not path.exists():
                raise FileNotFoundError(f"H5 dataset not found: {self.h5_path}")
            if path.is_dir():
                paths.extend(str(p) for p in sorted(path.glob("*.h5")))
            else:
                paths.append(str(path))

        if self.folder_path:
            folder = Path(self.folder_path)
            if not folder.exists():
                raise FileNotFoundError(f"H5 folder not found: {self.folder_path}")
            paths.extend(str(p) for p in sorted(folder.glob("*.h5")))

        return paths

    def _discover_trajectories(self) -> List[TrajectoryRef]:
        refs: List[TrajectoryRef] = []
        for file_path in self._collect_h5_files():
            with h5py.File(file_path, "r") as h5_file:
                traj_keys = sorted(
                    [
                        key for key in h5_file.keys()
                        if key.startswith("traj_") and isinstance(h5_file[key], h5py.Group)
                    ],
                    key=traj_sort_key,
                )
                if traj_keys:
                    refs.extend(TrajectoryRef(file_path, key) for key in traj_keys)
                else:
                    refs.append(TrajectoryRef(file_path, None))

        if self.num_traj is not None:
            refs = refs[: self.num_traj]
        return refs

    def _get_handle(self, file_path: str) -> h5py.File:
        if file_path not in self.handles:
            self.handles[file_path] = h5py.File(file_path, "r", swmr=True)
        return self.handles[file_path]

    def get_trajectory_group(self, ref: TrajectoryRef):
        h5_file = self._get_handle(ref.file_path)
        return h5_file if ref.traj_key is None else h5_file[ref.traj_key]

    def __len__(self):
        return len(self.trajectory_refs)

    def __getitem__(self, index):
        return {}

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
