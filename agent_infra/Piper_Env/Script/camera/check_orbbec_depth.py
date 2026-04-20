#!/usr/bin/env python3
"""Probe Orbbec color/depth frames without starting the robot env."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_infra.Piper_Env.Camera.orbbec_camera import (
    OrbbecCameraGroup,
    get_connected_orbbec_serials,
    get_orbbec_video_devices,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Orbbec RGB/depth availability through v4l2/sdk/auto backend."
    )
    parser.add_argument(
        "--serial",
        action="append",
        default=None,
        help="Camera serial number. Can be repeated. Default: use V4L2-discovered Orbbec serials.",
    )
    parser.add_argument(
        "--backend",
        choices=["v4l2", "sdk", "auto"],
        default="v4l2",
        help="Orbbec backend to probe. Use sdk when checking depth.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=3.0)
    return parser.parse_args()


def _discover_serials() -> list[str]:
    serials = []
    for info in get_orbbec_video_devices():
        serial = info.get("serial")
        if serial and serial not in serials:
            serials.append(serial)

    if serials:
        return serials

    return get_connected_orbbec_serials()


def _stats(name: str, frame) -> str:
    if frame is None:
        return f"{name}=None"

    arr = np.asarray(frame)
    return (
        f"{name}.shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min()}, max={arr.max()}, mean={float(arr.mean()):.2f}"
    )


def main() -> int:
    args = _parse_args()
    serials = args.serial or _discover_serials()

    print("[DepthProbe] Orbbec V4L2 devices:")
    for item in get_orbbec_video_devices():
        print(f"  - {item}")

    if not serials:
        print("[DepthProbe] No Orbbec serials found.")
        return 1

    print(f"[DepthProbe] Starting cameras: serials={serials}, backend={args.backend}")
    group = OrbbecCameraGroup(
        serials,
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=args.backend,
    )

    try:
        deadline = time.time() + max(args.seconds, 0.5)
        latest = {}
        while time.time() < deadline:
            latest = group.get_all_frames()
            time.sleep(0.1)

        for serial in serials:
            frames = latest.get(serial, {})
            print(f"[DepthProbe] {serial}: {_stats('color', frames.get('color'))}")
            print(f"[DepthProbe] {serial}: {_stats('depth', frames.get('depth'))}")
    finally:
        group.stop_all()

    if args.backend == "v4l2":
        print("[DepthProbe] Note: V4L2 usually exposes RGB only for current DaBai DCW setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
