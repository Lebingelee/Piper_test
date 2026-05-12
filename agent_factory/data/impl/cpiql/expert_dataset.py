from typing import Optional, Sequence

from agent_factory.data.impl.cpiql.common import (
    CPIQLTrajectoryDataset,
    compute_n_step_progress_signals,
    compute_progress_gammas,
    compute_progress_returns,
)


class CPIQLExpertDataset(CPIQLTrajectoryDataset):
    """
    CPIQL expert/offline dataset for flattened demonstration H5 files.

    Expected trajectory keys are strictly:
        obs, action, success, terminated, truncated

    `obs` and `action` may be datasets or groups of flattened leaves. Shared
    ordering metadata is read from `meta/env_meta` when present.
    """

    def __init__(
        self,
        cfg,
        obs_space=None,
        device: str = "cpu",
        required_keys: Optional[Sequence[str]] = None,
        dataset_format: Optional[str] = None,
        h5_path: Optional[str] = None,
    ):
        del obs_space, device, dataset_format
        demo_path = h5_path or cfg.dataset.expert.demo_path
        super().__init__(
            cfg=cfg,
            h5_path=demo_path,
            num_traj=cfg.dataset.expert.num_traj,
            required_keys=required_keys,
            require_intervention=False,
        )


__all__ = [
    "CPIQLExpertDataset",
    "compute_progress_gammas",
    "compute_progress_returns",
    "compute_n_step_progress_signals",
]
