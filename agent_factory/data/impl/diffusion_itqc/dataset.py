from agent_factory.data.impl.expert_dataset import ExpertDataset as _BaseExpertDataset


class DiffusionITQCDataset(_BaseExpertDataset):
    """
    Dataset implementation for `agent_factory.agents.impl.diffusion_itqc`.

    It currently reuses the shared expert dataset behavior and guarantees the
    fields required by Diffusion_ITQC:
    observations, action, next_observations, reward, terminated, value, cond.
    """


ExpertDataset = DiffusionITQCDataset

__all__ = [
    "DiffusionITQCDataset",
    "ExpertDataset",
]
