#!/bin/bash
# Piper 双臂 LeRobot -> merged H5 转换脚本

CTRL_MODE="joint"
TASK_NAME="piper_dual_lerobot_${CTRL_MODE}_task"
INPUT_DIR="agent_infra/Piper_Env/Record/data/${TASK_NAME}/lerobot"
OUTPUT_H5="agent_infra/Piper_Env/Record/data/${TASK_NAME}/merged/dual_from_lerobot.h5"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[Launcher] 双臂 LeRobot 转 merged H5..."
"$PYTHON_BIN" -m agent_infra.Piper_Env.Record.postprocess lerobot_to_h5 \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_H5" \
  "$@"
