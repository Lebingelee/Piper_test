#!/bin/bash
# merge_data.sh: 将双臂录制的轨迹合并。由于底层已经重构为元数据驱动，合并脚本与单臂通用。

INPUT_DIR="agent_infra/Realman_Env/Record/data/Towel_folding_skill1"
OUTPUT_DIR="demos_RL"
NEW_NAME="Towel_folding_skill1"
CONTROL_MODE="joint_pos"
MAX_STEPS=600
RESIZE="224x224"

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "--- Realman 双臂数据合并 ---"
# 注意：merge_data 会自动识别 env_meta 中的双臂结构
python3 -m agent_infra.Realman_Env.Record.merge_data \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -n "$NEW_NAME" \
    -m "$CONTROL_MODE" \
    -s "$MAX_STEPS" \
    --resize "$RESIZE"
