#!/bin/bash

###############################################################################
# Piper 轨迹重播启动脚本
# 
# 变量说明:
#   ROOT_PATH:   数据集根目录 (例如 data/piper_recording_000)
#   EPISODE_IDX: 重播的轨迹索引 (-1 表示按顺序重播所有)
#   FPS:         重播频率 (留空则使用录制时的原始频率)
#   CTRL_MODE:   控制模式 (必须与录制时的模式一致: 'joint' 或 'pose')
###############################################################################

# --- 默认配置 ---
ROOT_PATH=""
EPISODE_IDX="-1"
FPS=""
CTRL_MODE="joint"

# --- 自动处理 ---
ARGS=""
if [ -n "$ROOT_PATH" ]; then ARGS="$ARGS --path $ROOT_PATH"; fi
if [ -n "$EPISODE_IDX" ]; then ARGS="$ARGS --episode $EPISODE_IDX"; fi
if [ -n "$FPS" ]; then ARGS="$ARGS --fps $FPS"; fi
if [ -n "$CTRL_MODE" ]; then ARGS="$ARGS --ctrl_mode $CTRL_MODE"; fi

echo "[Launcher] 启动轨迹重播..."
echo "[Launcher] 控制模式: ${CTRL_MODE}"
if [ -n "$ROOT_PATH" ]; then echo "[Launcher] 数据路径: ${ROOT_PATH}"; fi

# 启动 Python 模块
python3 -m piper_infra.Record.replay $ARGS "$@"
