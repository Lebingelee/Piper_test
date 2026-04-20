#!/usr/bin/env python3
"""Capture Orbbec RGB snapshots to identify left/right camera roles."""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_infra.Piper_Env.Camera.orbbec_camera import get_orbbec_video_devices


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Capture one frame from each Orbbec /dev/video device for left/right role calibration."
    )
    parser.add_argument(
        "--output-dir",
        default="agent_infra/Piper_Env/Record/data/camera_calibration",
        help="Directory for snapshot images.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show OpenCV preview windows. Press any key to close each window.",
    )
    parser.add_argument(
        "--all-video-nodes",
        action="store_true",
        help="Capture every /dev/video node. Default captures only one node per serial number.",
    )
    return parser.parse_args()


def _open_capture(video_path: str, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(video_path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main():
    args = _parse_args()
    devices = get_orbbec_video_devices()
    if not devices:
        print("[Calib] No Orbbec video devices found.")
        return

    if not args.all_video_nodes:
        unique_devices = []
        seen_serials = set()
        for info in devices:
            key = info.get("serial") or info.get("usb") or info.get("video")
            if key in seen_serials:
                continue
            seen_serials.add(key)
            unique_devices.append(info)
        devices = unique_devices

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("[Calib] Orbbec video devices:")
    for idx, info in enumerate(devices):
        print(
            f"  [{idx}] video={info.get('video')} usb={info.get('usb', 'unknown')} "
            f"serial={info.get('serial', '')} product={info.get('product', info.get('name'))}"
        )

    print("\n[Calib] Tip: cover or point one physical camera at a known left/right marker, then inspect snapshots.")

    for idx, info in enumerate(devices):
        video_path = info["video"]
        cap = _open_capture(video_path, args.width, args.height, args.fps)
        if not cap.isOpened():
            print(f"[Calib] Failed to open {video_path}")
            continue

        frame = None
        for _ in range(10):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate

        cap.release()
        if frame is None:
            print(f"[Calib] Failed to read frame from {video_path}")
            continue

        serial = info.get("serial") or info.get("usb", f"idx{idx}")
        safe_serial = serial.replace("/", "_").replace(":", "_")
        out_path = output_dir / f"orbbec_{idx}_{safe_serial}_{timestamp}.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"[Calib] Saved {video_path} serial={serial} -> {out_path}")

        if args.preview:
            window_name = f"{video_path} serial={serial}"
            cv2.imshow(window_name, frame)
            cv2.waitKey(0)

    if args.preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
