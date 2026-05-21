import argparse
import os
import re
import sys
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from agent_factory.agents.registry import make_agent
from agent_factory.data.eval.window_builder import (
    ReplayEvalWindowDataset,
    load_replay_eval_trajectory,
)
from agent_factory.script.train_universal import load_training_config


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_checkpoint_path(cfg: Any, cli_path: str) -> str:
    ckpt_path = str(cli_path or getattr(cfg.train, "ckpt_path", "") or "").strip()
    if not ckpt_path:
        raise ValueError("Checkpoint path is required. Pass --ckpt_path or set train.ckpt_path in config.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    return ckpt_path


def _resolve_replay_h5_path(cfg: Any, cli_path: str) -> str:
    replay_h5_path = str(cli_path or getattr(cfg.dataset.replaybuffer, "replaybuffer_path", "") or "").strip()
    if not replay_h5_path:
        raise ValueError(
            "Replay evaluation requires a single replay .h5 path. "
            "Pass --replay_h5_path or set dataset.replaybuffer.replaybuffer_path in config."
        )
    if os.path.isdir(replay_h5_path):
        raise ValueError(
            f"Replay evaluation first version only supports a single .h5 file, got directory: {replay_h5_path}"
        )
    if not os.path.exists(replay_h5_path):
        raise FileNotFoundError(f"Replay .h5 file not found: {replay_h5_path}")
    return replay_h5_path


def _to_numpy_1d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    return value.reshape(-1)


def _merge_eval_outputs(chunks: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    merged: Dict[str, List[np.ndarray]] = defaultdict(list)
    for chunk in chunks:
        for key, value in chunk.items():
            merged[key].append(_to_numpy_1d(value))
    return {key: np.concatenate(parts, axis=0) for key, parts in merged.items()}


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return sanitized or "figure"


def _plot_eval_results(
    aggregated: Dict[str, np.ndarray],
    save_dir: str,
    title_prefix: str = "",
    y_lim: Optional[List[float]] = None,
) -> List[str]:
    frame = aggregated.get("frame")
    if frame is None:
        example_key = next((key for key in aggregated.keys() if key.startswith("figure:")), None)
        if example_key is None:
            return []
        frame = np.arange(len(aggregated[example_key]), dtype=np.int64)

    grouped: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for key, value in aggregated.items():
        if not key.startswith("figure:"):
            continue
        payload = key[len("figure:") :]
        if "/" not in payload:
            raise ValueError(f"Figure metric key must look like 'figure:name/label', got '{key}'.")
        figure_name, label = payload.split("/", 1)
        grouped[figure_name][label] = value

    saved_paths: List[str] = []
    for figure_name, series_map in grouped.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, values in series_map.items():
            if len(values) != len(frame):
                raise ValueError(
                    f"Metric '{figure_name}/{label}' has length {len(values)} but frame has length {len(frame)}."
                )
            ax.plot(frame, values, label=label, linewidth=1.8)
        ax.set_xlabel("Action Timestep")
        ax.set_ylabel("Critic Output")
        title = f"{title_prefix} {figure_name}".strip()
        ax.set_title(title)
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()

        filename = f"{_sanitize_filename(figure_name)}.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
        saved_paths.append(save_path)
    return saved_paths


def _save_npz(aggregated: Dict[str, np.ndarray], save_dir: str) -> str:
    payload = {}
    for key, value in aggregated.items():
        safe_key = key.replace("figure:", "figure__").replace("/", "__")
        payload[safe_key] = value
    save_path = os.path.join(save_dir, "eval_metrics.npz")
    np.savez_compressed(save_path, **payload)
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description="Universal critic evaluation on one replay trajectory.")
    parser.add_argument("--config", required=True, help="Path to model_config.yaml or a compatible config file.")
    parser.add_argument("--ckpt-path", "--ckpt_path", default="", help="Checkpoint path. Falls back to train.ckpt_path in config.")
    parser.add_argument("--replay-h5-path", "--replay_h5_path", default="", help="Single replay .h5 file to evaluate. Falls back to dataset.replaybuffer.replaybuffer_path.")
    parser.add_argument("--save-dir", "--save_dir", required=True, help="Directory for saved figures.")
    parser.add_argument("--traj-idx", "--traj_idx", type=int, default=0, help="Trajectory index inside the replay .h5 file.")
    parser.add_argument("--device", default="", help="Evaluation device, e.g. cpu or cuda:0.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=64, help="Window batch size for critic evaluation.")
    parser.add_argument(
        "--y-lim",
        "--y_lim",
        nargs=2,
        type=float,
        default=[-0.3, 1.0],
        metavar=("YMIN", "YMAX"),
        help="Y-axis limits for saved plots.",
    )
    parser.add_argument("--only-obs", action="store_true", help="Only evaluate observation-based value metrics.")
    parser.add_argument("--save-npz", action="store_true", help="Also save aggregated metric arrays to eval_metrics.npz.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = str(args.device or _default_device())
    cfg, _ = load_training_config(
        config_path=args.config,
        cli_overrides={"device": device},
        finetune=False,
    )
    cfg.device = device
    cfg.train.device = device

    ckpt_path = _resolve_checkpoint_path(cfg, args.ckpt_path)
    replay_h5_path = _resolve_replay_h5_path(cfg, args.replay_h5_path)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[Eval] Loading agent {cfg.agent_type} on {device}")
    agent = make_agent(cfg.agent_type, cfg)
    agent.load(ckpt_path)
    agent.to(device)
    agent.eval()

    if not hasattr(agent, "eval_batch"):
        raise NotImplementedError(
            f"Agent '{cfg.agent_type}' does not expose critic eval_batch(). "
            "Implement it in the corresponding critic mixin first."
        )

    print(f"[Eval] Loading replay trajectory from {replay_h5_path} (traj_idx={args.traj_idx})")
    trajectory = load_replay_eval_trajectory(
        h5_path=replay_h5_path,
        include_rgb=bool(cfg.dataset.include_rgb),
        traj_idx=int(args.traj_idx),
    )
    window_dataset = ReplayEvalWindowDataset(
        trajectory=trajectory,
        obs_horizon=int(cfg.env.obs_horizon),
        pred_horizon=int(cfg.env.pred_horizon),
    )
    loader = DataLoader(
        window_dataset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    outputs: List[Dict[str, Any]] = []
    print(f"[Eval] Evaluating {len(window_dataset)} action-aligned windows")
    with torch.no_grad():
        for batch in loader:
            result = agent.eval_batch(batch, only_obs=bool(args.only_obs))
            if "frame" not in result:
                result["frame"] = batch["frame"]
            outputs.append(result)

    aggregated = _merge_eval_outputs(outputs)
    title_prefix = f"{cfg.agent_type} traj={args.traj_idx}"
    if args.y_lim[0] >= args.y_lim[1]:
        raise ValueError(f"Expected y_lim as [ymin, ymax] with ymin < ymax, got {args.y_lim}.")
    saved_figures = _plot_eval_results(
        aggregated,
        save_dir=args.save_dir,
        title_prefix=title_prefix,
        y_lim=args.y_lim,
    )

    print("[Eval] Saved figures:")
    for path in saved_figures:
        print(f"  - {path}")

    if args.save_npz:
        npz_path = _save_npz(aggregated, save_dir=args.save_dir)
        print(f"[Eval] Saved metric arrays to {npz_path}")


if __name__ == "__main__":
    main()
