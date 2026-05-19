import os
import h5py

from agent_factory.data.impl.cpiql.expert_dataset import CPIQLExpertDataset
from agent_factory.data.impl.cpiql.replaybuffer import CPIQLFileReplayBuffer
from agent_factory.data.registry import register_dataset_type


def _build_cpiql_expert_dataset(cfg, required_keys=None, expert_path=None):
    return CPIQLExpertDataset(
        cfg=cfg,
        device="cpu",
        required_keys=required_keys,
        h5_path=expert_path,
    )


def _build_cpiql_replaybuffer(cfg, required_keys=None, replaybuffer_path=None):
    def _h5_has_intervention(path: str) -> bool:
        with h5py.File(path, "r") as h5_file:
            traj_keys = [key for key in h5_file.keys() if key.startswith("traj_") and isinstance(h5_file[key], h5py.Group)]
            groups = [h5_file[key] for key in traj_keys] if traj_keys else [h5_file]
            for group in groups:
                if "intervention" in group:
                    return True
        return False

    if not replaybuffer_path:
        raise ValueError("CPIQL replaybuffer requires a replaybuffer_path.")
    if os.path.isdir(replaybuffer_path):
        return CPIQLFileReplayBuffer(
            cfg=cfg,
            folder_path=replaybuffer_path,
            required_keys=required_keys,
        )
    if os.path.isfile(replaybuffer_path):
        if _h5_has_intervention(replaybuffer_path):
            return CPIQLFileReplayBuffer(
                cfg=cfg,
                h5_path=replaybuffer_path,
                required_keys=required_keys,
            )
        return CPIQLExpertDataset(
            cfg=cfg,
            device="cpu",
            required_keys=required_keys,
            h5_path=replaybuffer_path,
        )
    raise FileNotFoundError(f"replaybuffer_path not found: {replaybuffer_path}")


register_dataset_type("cpiql", _build_cpiql_expert_dataset, _build_cpiql_replaybuffer)

__all__ = [
    "CPIQLExpertDataset",
    "CPIQLFileReplayBuffer",
]
