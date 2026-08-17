import os

from agent_factory.data.impl.tdqc.feature_dataset import TDQCFeatureDataset, tdqc_collate_fn
from agent_factory.data.registry import register_dataset_type


def _build_tdqc_expert_dataset(cfg, required_keys=None, expert_path=None):
    return TDQCFeatureDataset(
        cfg=cfg,
        h5_path=expert_path or cfg.dataset.expert.demo_path,
        required_keys=required_keys,
        num_traj=getattr(cfg.dataset.expert, "num_traj", None),
    )


def _build_tdqc_replaybuffer(cfg, required_keys=None, replaybuffer_path=None):
    if not replaybuffer_path:
        raise ValueError("TDQC feature replaybuffer requires a replaybuffer_path.")
    if os.path.isdir(replaybuffer_path):
        return TDQCFeatureDataset(
            cfg=cfg,
            folder_path=replaybuffer_path,
            required_keys=required_keys,
            num_traj=getattr(cfg.dataset.replaybuffer, "max_traj_num", None),
        )
    if os.path.isfile(replaybuffer_path):
        return TDQCFeatureDataset(
            cfg=cfg,
            h5_path=replaybuffer_path,
            required_keys=required_keys,
            num_traj=getattr(cfg.dataset.replaybuffer, "max_traj_num", None),
        )
    raise FileNotFoundError(f"TDQC replaybuffer_path not found: {replaybuffer_path}")


register_dataset_type("tdqc", _build_tdqc_expert_dataset, _build_tdqc_replaybuffer)
register_dataset_type("tdqc_feature", _build_tdqc_expert_dataset, _build_tdqc_replaybuffer)

__all__ = ["TDQCFeatureDataset", "tdqc_collate_fn"]
