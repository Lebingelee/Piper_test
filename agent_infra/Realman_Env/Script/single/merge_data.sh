#!/bin/bash
# merge_data.sh: 将单臂录制的轨迹合并为 H5 + JSON

INPUT_DIR="agent_infra/Realman_Env/Record/data/Single_Arm_Task"
OUTPUT_DIR="demos_RL"
NEW_NAME="Single_Arm_Merged"
CONTROL_MODE="joint_pos"
MAX_STEPS=250
RESIZE="224x224"

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "--- Realman 单臂数据合并 ---"
python3 -m agent_infra.Realman_Env.Record.merge_data \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -n "$NEW_NAME" \
    -m "$CONTROL_MODE" \
    -s "$MAX_STEPS" \
    --resize "$RESIZE"
