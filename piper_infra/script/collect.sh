#!/bin/bash

###############################################################################
# Piper 数据采集启动脚本
# 
# 变量说明:
#   CUSTOM_PATH: 自定义数据集保存路径 (留空则默认使用 data/piper_recording)
#   CTRL_MODE:   控制模式，可选 'joint' (关节) 或 'pose' (位姿)
###############################################################################
export PYTHONPATH="$(pwd):$PYTHONPATH"

# --- 默认配置 ---
CUSTOM_PATH=""
CTRL_MODE="pose"  # joint 和 pose 分别代表关节和末端位姿

# --- 自动处理 ---
ARGS=""
if [ -n "$CUSTOM_PATH" ]; then ARGS="$ARGS --custom_path $CUSTOM_PATH"; fi
if [ -n "$CTRL_MODE" ]; then ARGS="$ARGS --ctrl_mode $CTRL_MODE"; fi

echo "[Launcher] 启动采集任务..."
echo "[Launcher] 控制模式: ${CTRL_MODE}"
if [ -n "$CUSTOM_PATH" ]; then echo "[Launcher] 自定义路径: ${CUSTOM_PATH}"; fi

python3 -m piper_infra.Record.record_piper_dataset $ARGS "$@"