import os

from .dataset import DiffusionITQCDataset, ExpertDataset
from .replaybuffer import ClassicReplayBuffer, DiffusionITQCReplayBuffer, FileReplayBuffer
from agent_factory.data.registry import register_dataset_type


def _build_diffusion_itqc_expert_dataset(cfg, required_keys=None, expert_path=None):
    if expert_path:
        cfg.dataset.expert.demo_path = expert_path
    return DiffusionITQCDataset(
        cfg=cfg,
        device="cpu",
        required_keys=required_keys,
    )


def _build_diffusion_itqc_replaybuffer(cfg, required_keys=None, replaybuffer_path=None):
    if not replaybuffer_path:
        raise ValueError("Diffusion_ITQC replaybuffer requires a replaybuffer_path.")
    if os.path.isdir(replaybuffer_path):
        return DiffusionITQCReplayBuffer(
            cfg=cfg,
            folder_path=replaybuffer_path,
            required_keys=required_keys,
        )
    if os.path.isfile(replaybuffer_path):
        folder_path = os.path.dirname(replaybuffer_path)
        return DiffusionITQCReplayBuffer(
            cfg=cfg,
            folder_path=folder_path,
            required_keys=required_keys,
        )
    raise FileNotFoundError(f"replaybuffer_path not found: {replaybuffer_path}")


register_dataset_type("diffusion_itqc", _build_diffusion_itqc_expert_dataset, _build_diffusion_itqc_replaybuffer)

__all__ = [
    "DiffusionITQCDataset",
    "ExpertDataset",
    "DiffusionITQCReplayBuffer",
    "FileReplayBuffer",
    "ClassicReplayBuffer",
]
