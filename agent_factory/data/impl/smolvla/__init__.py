from agent_factory.data.impl.smolvla.h5_dataset import SmolVLAMultitaskH5Dataset
from agent_factory.data.registry import register_dataset_type


def _build_smolvla_expert_dataset(cfg, required_keys=None, expert_path=None):
    return SmolVLAMultitaskH5Dataset(
        cfg=cfg,
        h5_path=expert_path or cfg.dataset.expert.demo_path,
        required_keys=required_keys,
    )


register_dataset_type("smolvla_h5", _build_smolvla_expert_dataset)

__all__ = ["SmolVLAMultitaskH5Dataset"]
