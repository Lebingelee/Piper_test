"""LeRobot-format dataset conversion and structured-H5 helpers."""

from .h5_utils import (
    build_lerobot_features,
    convert_h5_to_lerobot_dataset,
    get_trajectory_group,
    infer_lerobot_repo_id,
    list_h5_trajectories,
    load_env_meta,
    load_lerobot_policy_dataset,
    read_prompt,
)

__all__ = [
    "build_lerobot_features",
    "convert_h5_to_lerobot_dataset",
    "get_trajectory_group",
    "infer_lerobot_repo_id",
    "list_h5_trajectories",
    "load_env_meta",
    "load_lerobot_policy_dataset",
    "read_prompt",
]
