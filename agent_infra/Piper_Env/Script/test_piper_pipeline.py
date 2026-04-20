#!/usr/bin/env python3
"""Hardware smoke tests for Piper teleop, data collection, and replay.

This script intentionally keeps each probe short and explicit. It is meant for
bench bring-up, not for unattended long-running dataset collection.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_infra.Piper_Env.Env.dual_piper_env import DualPiperEnv
from agent_infra.Piper_Env.Env.single_piper_env import SinglePiperEnv
from agent_infra.Piper_Env.Env.utils.piper_base_env import PiperEnv
from agent_infra.Piper_Env.Record.recorder import (
    H5TrajectoryRecorder,
    LeRobotDatasetRecorder,
)
from agent_infra.Piper_Env.Record.replay import replay_h5, replay_lerobot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test Piper single/dual arm pipelines with or without cameras. "
            "The script can probe teleop, short data collection, and replay."
        )
    )
    parser.add_argument(
        "--arm-mode",
        choices=["single", "dual", "both"],
        default="single",
        help="Which arm topology to test.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=["without", "with", "both"],
        default="without",
        help="Whether to wrap the env with configured cameras.",
    )
    parser.add_argument(
        "--function",
        choices=["teleop", "collect", "replay", "all"],
        default="teleop",
        help="Pipeline function to test.",
    )
    parser.add_argument(
        "--single-config",
        default="agent_infra/Piper_Env/Config/piper_config.yaml",
        help="Config used when --arm-mode includes single.",
    )
    parser.add_argument(
        "--dual-config",
        default="agent_infra/Piper_Env/Config/dual_piper_config.yaml",
        help="Config used when --arm-mode includes dual.",
    )
    parser.add_argument(
        "--control",
        choices=["joint", "pose", "delta_pose", "relative_pose_chunk"],
        default=None,
        help="Override config control mode. Omit to use config default.",
    )
    parser.add_argument("--hz", type=int, default=None, help="Override env loop rate.")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of step-loop frames for teleop/collection probes.",
    )
    parser.add_argument(
        "--record-format",
        choices=["h5", "lerobot"],
        default="h5",
        help="Short collection output format.",
    )
    parser.add_argument(
        "--output-root",
        default="agent_infra/Piper_Env/Record/data/pipeline_tests",
        help="Root directory for short collection probes.",
    )
    parser.add_argument(
        "--replay-input",
        default=None,
        help="H5 file or LeRobot dataset directory. If omitted after collection, the newly collected file is replayed.",
    )
    parser.add_argument(
        "--replay-format",
        choices=["h5", "lerobot"],
        default=None,
        help="Replay input format. Defaults to --record-format when omitted.",
    )
    parser.add_argument("--episode", type=int, default=0, help="LeRobot episode index.")
    parser.add_argument(
        "--reset-before-test",
        action="store_true",
        help="Move robot to config init pose before teleop/collection probes.",
    )
    parser.add_argument(
        "--start-cameras",
        action="store_true",
        help="Start camera threads when --camera-mode includes with.",
    )
    return parser.parse_args()


def _selected_modes(value: str, first: str, second: str) -> list[str]:
    if value == "both":
        return [first, second]
    return [value]


def _make_env(
    arm_mode: str,
    with_camera: bool,
    args: argparse.Namespace,
):
    config = args.single_config if arm_mode == "single" else args.dual_config
    expected_arms = 1 if arm_mode == "single" else 2

    if with_camera:
        cls = SinglePiperEnv if arm_mode == "single" else DualPiperEnv
        env = cls(config_path=config, control_mode=args.control, hz=args.hz)
        if args.start_cameras:
            env.start_cameras()
    else:
        env = PiperEnv(config_path=config, control_mode=args.control, hz=args.hz)
        if len(env.arm_names) != expected_arms:
            env.close()
            raise ValueError(
                f"{arm_mode} mode expected {expected_arms} arm(s), "
                f"but config created {len(env.arm_names)}: {env.arm_names}"
            )
    return env


def _print_meta_summary(env, arm_mode: str, camera_mode: str):
    meta = env.unwrapped.meta_keys
    obs_sections = ", ".join(meta["obs"].keys())
    state_keys = ", ".join(meta["obs"].get("state", {}).keys())
    action_keys = ", ".join(meta["action"].keys())
    rgb_keys = ", ".join(meta["obs"].get("rgb", {}).keys()) or "none"
    print(f"\n[Test] Env ready: arm={arm_mode}, camera={camera_mode}")
    print(f"[Test] obs sections: {obs_sections}")
    print(f"[Test] state keys: {state_keys}")
    print(f"[Test] action keys: {action_keys}")
    print(f"[Test] rgb roles: {rgb_keys}")


def _safe_step_loop(env, steps: int, reset_before_test: bool, record_callback=None):
    if reset_before_test:
        env.reset()

    last_obs = None
    last_info = {}
    for idx in range(max(steps, 1)):
        action = env.unwrapped.get_safe_action()
        obs, _reward, _terminated, _truncated, info = env.step(action)
        if record_callback is not None:
            record_callback(obs, action, info)
        if idx == 0:
            print(f"[Test] first obs keys: {list(obs.keys())}")
            print(f"[Test] first info keys: {list(info.keys())}")
        last_obs = obs
        last_info = info
    return last_obs, last_info


def _run_teleop_probe(env, args: argparse.Namespace):
    print("[Teleop] Press T in the terminal to toggle expert intervention during this probe.")
    print("[Teleop] The script sends safe hold-actions; PiperEnv overrides them when tele_enabled and leader is active.")
    obs, info = _safe_step_loop(env, args.steps, args.reset_before_test)
    under_control = obs.get("under_control", {}) if obs else {}
    print(f"[Teleop] final under_control: {under_control}")
    print(f"[Teleop] final intervened: {info.get('intervened')}")


def _latest_h5_file(root: Path, task_name: str) -> Optional[Path]:
    h5_dir = root / task_name / "h5_raw"
    files = sorted(h5_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _run_collection_probe(
    env,
    args: argparse.Namespace,
    arm_mode: str,
    camera_mode: str,
) -> Optional[Path]:
    output_root = Path(args.output_root)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    control_mode = getattr(env.unwrapped, "control_mode", args.control or "joint")
    task_name = f"pipeline_{arm_mode}_{camera_mode}_{args.record_format}_{control_mode}_{timestamp}"
    meta = env.unwrapped.meta_keys

    if args.record_format == "h5":
        recorder = H5TrajectoryRecorder(str(output_root), task_name, meta)
    else:
        recorder = LeRobotDatasetRecorder(str(output_root), task_name, meta, fps=env.hz)

    print(f"[Collect] Start short {args.record_format} probe: task={task_name}")
    recorder.start_episode(0)

    def _record(obs, action, info):
        recorder.add_frame(obs, action, info)

    _safe_step_loop(env, max(args.steps, 6), args.reset_before_test, _record)
    recorder.end_episode(success=True)
    recorder.finalize()

    if args.record_format == "h5":
        saved_file = _latest_h5_file(output_root, task_name)
        print(f"[Collect] saved h5: {saved_file}")
        return saved_file

    dataset_path = output_root / task_name / "lerobot"
    print(f"[Collect] saved lerobot dataset: {dataset_path}")
    return dataset_path


def _run_replay_probe(
    env,
    args: argparse.Namespace,
    collected_path: Optional[Path],
):
    replay_input = Path(args.replay_input) if args.replay_input else collected_path
    replay_format = args.replay_format or args.record_format

    if replay_input is None:
        print("[Replay] skipped: provide --replay-input, or run --function all/collect first.")
        return

    print(f"[Replay] Start replay probe: format={replay_format}, input={replay_input}")
    if replay_format == "h5":
        replay_h5(env, str(replay_input))
    else:
        replay_lerobot(env, str(replay_input), args.episode)


def _functions_to_run(name: str) -> Iterable[str]:
    if name == "all":
        return ["teleop", "collect", "replay"]
    return [name]


def main():
    args = _parse_args()
    arm_modes = _selected_modes(args.arm_mode, "single", "dual")
    camera_modes = _selected_modes(args.camera_mode, "without", "with")
    functions = list(_functions_to_run(args.function))

    print("[Test] This is a real-hardware script. Keep an emergency stop within reach.")
    print("[Test] Default behavior does not reset the robot unless --reset-before-test is set.")

    for arm_mode in arm_modes:
        for camera_mode in camera_modes:
            with_camera = camera_mode == "with"
            env = _make_env(arm_mode, with_camera, args)
            collected_path = None
            try:
                _print_meta_summary(env, arm_mode, camera_mode)
                if "teleop" in functions:
                    _run_teleop_probe(env, args)
                if "collect" in functions:
                    collected_path = _run_collection_probe(env, args, arm_mode, camera_mode)
                if "replay" in functions:
                    _run_replay_probe(env, args, collected_path)
            finally:
                env.close()
                # Give SocketCAN/SDK threads a moment to release before the next combo.
                time.sleep(0.5)


if __name__ == "__main__":
    main()
