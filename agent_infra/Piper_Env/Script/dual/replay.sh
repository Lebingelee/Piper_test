#!/bin/bash
# Piper 双臂轨迹回放启动脚本

CONFIG_PATH="agent_infra/Piper_Env/Config/dual_piper_config.yaml"
traj_PATH="agent_infra/Piper_Env/Record/data/Hang_towel_joint/h5_raw/traj_13_201932.h5"
CTRL_MODE="joint"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[Launcher] 启动双臂轨迹回放..."
"$PYTHON_BIN" -m agent_infra.Piper_Env.Record.replay -cfg "$CONFIG_PATH" -i "$traj_PATH" -ctrl "$CTRL_MODE" "$@"
