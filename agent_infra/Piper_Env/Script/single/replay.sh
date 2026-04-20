#!/bin/bash
# Piper 单臂轨迹回放启动脚本

CONFIG_PATH="agent_infra/Piper_Env/Config/piper_config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[Launcher] 启动单臂轨迹回放..."
"$PYTHON_BIN" -m agent_infra.Piper_Env.Record.replay -cfg "$CONFIG_PATH" "$@"
