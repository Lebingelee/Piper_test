from typing import Optional, Sequence

from agent_factory.data.impl.cpiql.common import CPIQLTrajectoryDataset, cfg_get


class CPIQLFileReplayBuffer(CPIQLTrajectoryDataset):
    """
    File-backed CPIQL replay buffer.

    The folder should contain `traj_*.h5` files. Each file stores one flattened
    trajectory with the expert dataset keys plus an additional `intervention`
    bool array. `intervention[t] == 1` marks samples executed during expert
    takeover, from intervention start through expert release.

    The dataset view splits each source trajectory into contiguous execution
    segments with a single source label:
        autonomous, intervention, autonomous, intervention, autonomous, ...

    A normal trajectory with m intervention ranges becomes 2*m+1 non-empty
    child trajectories. Autonomous segments that end immediately before expert
    takeover receive a configurable pseudo terminal reward. Call
    `set_intervention_terminal_rewards()` to refresh those boundary rewards
    from a critic value estimate and recompute `reward`, `discount`, and
    `progress_return`.
    """

    def __init__(
        self,
        cfg,
        folder_path: Optional[str] = None,
        required_keys: Optional[Sequence[str]] = None,
    ):
        replay_path = (
            folder_path
            or cfg_get(cfg, "dataset.replaybuffer.folder_path", None)
            or cfg_get(cfg, "dataset.replaybuffer.replaybuffer_path", None)
            or cfg_get(cfg, "dataset.replay.folder_path", None)
            or cfg_get(cfg, "dataset.replay.replaybuffer_path", None)
            or cfg_get(cfg, "runner.save_dir", None)
        )
        if replay_path is None:
            raise ValueError(
                "CPIQLFileReplayBuffer requires folder_path or cfg.dataset.replaybuffer.folder_path."
            )
        super().__init__(
            cfg=cfg,
            folder_path=replay_path,
            num_traj=cfg_get(cfg, "dataset.replaybuffer.num_traj", cfg_get(cfg, "dataset.replay.num_traj", None)),
            required_keys=required_keys,
            require_intervention=True,
        )


__all__ = ["CPIQLFileReplayBuffer"]
