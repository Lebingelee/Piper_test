#!/bin/bash

###############################################################################
# Piper 轨迹重播启动脚本
# 
# 变量说明:
#   ROOT_PATH:   数据集根目录 (例如 ata/piper_pose_recording_002)
#   EPISODE_IDX: 重播的轨迹索引 (-1 表示按顺序重播所有)
#   FPS:         重播频率 (留空则使用录制时的原始频率)
#   CTRL_MODE:   控制模式 (必须与录制时的模式一致: 'joint' 或 'pose')
###############################################################################

# --- 默认配置 ---
# 如果命令行没有传参，则使用以下默认值
ROOT_PATH="data/piper_joint_recording_006"
EPISODE_IDX="-1"
FPS=""
CTRL_MODE="joint"

# --- 逻辑优化：只有当用户没提供对应参数时，才使用默认值 ---
# 使用 $@ 让 Python 的 argparse 处理所有参数
echo "[Launcher] 启动轨迹重播..."

# 检查是否提供了必要的命令行参数，如果没有，则补齐默认的
ARGS=""
if [[ ! "$*" == *"--path"* ]]; then ARGS="$ARGS --path $ROOT_PATH"; fi
if [[ ! "$*" == *"--episode"* ]]; then ARGS="$ARGS --episode $EPISODE_IDX"; fi
if [[ ! "$*" == *"--ctrl_mode"* ]]; then ARGS="$ARGS --ctrl_mode $CTRL_MODE"; fi
if [[ -n "$FPS" && ! "$*" == *"--fps"* ]]; then ARGS="$ARGS --fps $FPS"; fi

# 启动 Python 模块
python3 -m piper_infra.Record.replay $ARGS "$@"
