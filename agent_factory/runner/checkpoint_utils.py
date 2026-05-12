import copy
from typing import Any

import h5py
import numpy as np
import torch

from agent_factory.data.registry import build_training_bundle


def _action_normalizer_is_valid(agent: Any) -> bool:
    normalizer = getattr(agent, "action_normalizer", None)
    if normalizer is None:
        return True

    initialized = getattr(normalizer, "initialized", None)
    if initialized is not None and not bool(torch.as_tensor(initialized).item()):
        return False

    for _, value in normalizer.state_dict().items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            return False
    return True


def _flatten_action_node(action_node: Any) -> np.ndarray:
    if isinstance(action_node, h5py.Dataset):
        action = action_node[()]
        return np.asarray(action, dtype=np.float32).reshape(action.shape[0], -1)

    arrays = []
    for key in sorted(action_node.keys()):
        data = action_node[key][()]
        arrays.append(np.asarray(data, dtype=np.float32).reshape(data.shape[0], -1))
    if not arrays:
        raise KeyError(f"No action leaves found under {action_node.name}")
    return np.concatenate(arrays, axis=-1)


def _load_all_actions_from_h5(path: str) -> torch.Tensor:
    action_chunks = []
    with h5py.File(path, "r") as h5_file:
        groups = []
        if "action" in h5_file:
            groups.append(h5_file)
        groups.extend(
            h5_file[key]
            for key in sorted(h5_file.keys())
            if isinstance(h5_file[key], h5py.Group) and "action" in h5_file[key]
        )

        for group in groups:
            action = _flatten_action_node(group["action"])
            if action.size > 0:
                action_chunks.append(action)

    if not action_chunks:
        raise KeyError(f"No action datasets found in {path}")

    actions = np.concatenate(action_chunks, axis=0)
    if not np.isfinite(actions).all():
        raise ValueError(f"Non-finite action values found in {path}")
    return torch.from_numpy(actions).float()


def _fit_action_normalizer_from_h5(agent: Any, path: str) -> None:
    actions = _load_all_actions_from_h5(path)
    agent.fit_action_normalizer(actions)
    print(f"[RunnerCheckpoint] Action normalizer fitted with {actions.shape[0]} H5 actions.")


def ensure_action_normalizer_ready(agent: Any, cfg: Any) -> None:
    """
    Some stage-wise checkpoints are saved before the actor normalizer is fitted.
    Re-fit it from the configured expert dataset when the loaded buffers are
    still at their initial inf/-inf values.
    """
    if _action_normalizer_is_valid(agent):
        return

    demo_path = str(getattr(getattr(cfg.dataset, "expert", None), "demo_path", ""))
    print(
        "[RunnerCheckpoint] Loaded action normalizer is not initialized; "
        f"fitting from dataset: {demo_path}"
    )

    try:
        _fit_action_normalizer_from_h5(agent, demo_path)
    except Exception as exc:
        print(
            "[RunnerCheckpoint] Direct H5 action fit failed; falling back to dataset builder. "
            f"Reason: {exc}"
        )
        fit_cfg = copy.deepcopy(cfg)
        fit_cfg.dataset.include_rgb = False
        fit_cfg.dataset.include_depth = False
        if hasattr(fit_cfg, "train"):
            fit_cfg.train.num_workers = 0

        dataset_bundle = build_training_bundle(
            fit_cfg,
            required_keys=["action"],
        )
        agent._fit_action_normalizer_from_dataset(dataset_bundle)

    if not _action_normalizer_is_valid(agent):
        raise RuntimeError(
            "Action normalizer is still invalid after fitting from the configured dataset."
        )
