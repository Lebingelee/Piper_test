#!/bin/bash
# Piper 双臂 H5 合并脚本

TARGET_CONTROL_MODE="joint"
#TASK_NAME="piper_dual_h5_${TARGET_CONTROL_MODE}_task"
TASK_NAME="piper_STACK_h5_joint_task"

INPUT_PATH="agent_infra/Piper_Env/Record/data/${TASK_NAME}/h5_raw"
OUTPUT_PATH="data/merged/dual_${TASK_NAME}_merged.h5"
PYTHON_BIN="${PYTHON_BIN:-python3}"
H5_IMAGE_CODEC="${H5_IMAGE_CODEC:-jpeg}"
H5_JPEG_QUALITY="${H5_JPEG_QUALITY:-90}"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[Launcher] 合并双臂 H5 轨迹..."
"$PYTHON_BIN" -m agent_infra.Piper_Env.Record.postprocess merge_h5 \
  -i "$INPUT_PATH" \
  -o "$OUTPUT_PATH" \
  --control-mode "$TARGET_CONTROL_MODE" \
  --image-codec "$H5_IMAGE_CODEC" \
  --jpeg-quality "$H5_JPEG_QUALITY" \
  "$@"
