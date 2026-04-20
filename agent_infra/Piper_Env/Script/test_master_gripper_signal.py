#!/usr/bin/env python3
"""Inspect master gripper raw and filtered signals during teleop debugging."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_infra.Piper_Env.Env.utils.piper_base_env import PiperEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print Piper leader gripper raw SDK values and env filtered cache."
    )
    parser.add_argument(
        "-cfg",
        "--config",
        default="agent_infra/Piper_Env/Config/dual_piper_config.yaml",
        help="Piper config path.",
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--hz", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    env = PiperEnv(config_path=args.config, control_mode="joint")
    interval = 1.0 / max(args.hz, 1e-6)
    deadline = time.time() + max(args.seconds, 0.1)

    print(
        "[GripperProbe] Columns: arm raw has_raw filtered valid "
        "candidate_count follower_state last_cmd"
    )
    try:
        while time.time() < deadline:
            for name, arm in env.arms.items():
                master_state = arm.get_master_state()
                raw = float(master_state["gripper_pos"][0])
                has_raw = bool(master_state.get("has_gripper", False))
                with env._lock:
                    cache = env._master_cache[name]
                    filtered = float(cache["joint"][6])
                    valid = bool(cache.get("gripper_valid", False))
                    candidate_count = int(cache.get("gripper_candidate_count", 0))
                follower_state = arm.get_state()
                follower_gripper = float(follower_state["gripper_pos"][0])
                last_cmd = float(arm.last_follower_gripper_cmd)
                print(
                    f"[GripperProbe] {name:>6s} "
                    f"raw={raw:.5f} has_raw={has_raw} "
                    f"filtered={filtered:.5f} valid={valid} "
                    f"candidate_count={candidate_count} "
                    f"follower_state={follower_gripper:.5f} "
                    f"last_cmd={last_cmd:.5f}"
                )
            time.sleep(interval)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
