#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Adjust these parameters before running.
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_H5_PATH="${INPUT_H5_PATH:-data/test/traj_2_0514_2348.h5}"
OUTPUT_VIDEO_PATH="${OUTPUT_VIDEO_PATH:-data/video/traj_2_0514_2348.mp4}"
TRAJ_KEY="${TRAJ_KEY:-}"
SAMPLE_EVERY="${SAMPLE_EVERY:-4}"
SOURCE_FPS="${SOURCE_FPS:-30}"
OUTPUT_FPS="${OUTPUT_FPS:-15}"
SAVE_FRAMES_DIR="${SAVE_FRAMES_DIR:-}"
NO_LABELS="${NO_LABELS:-0}"

cd "${ROOT_DIR}"

CMD=(
  "${PYTHON_BIN}"
  "script/export_h5_rgb_keyframes_video.py"
  "--input" "${INPUT_H5_PATH}"
  "--output" "${OUTPUT_VIDEO_PATH}"
  "--sample-every" "${SAMPLE_EVERY}"
  "--source-fps" "${SOURCE_FPS}"
)

if [[ -n "${TRAJ_KEY}" ]]; then
  CMD+=("--traj-key" "${TRAJ_KEY}")
fi

if [[ -n "${OUTPUT_FPS}" ]]; then
  CMD+=("--fps" "${OUTPUT_FPS}")
fi

if [[ -n "${SAVE_FRAMES_DIR}" ]]; then
  CMD+=("--save-frames-dir" "${SAVE_FRAMES_DIR}")
fi

if [[ "${NO_LABELS}" == "1" ]]; then
  CMD+=("--no-labels")
fi

echo "[ExportScript] Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
