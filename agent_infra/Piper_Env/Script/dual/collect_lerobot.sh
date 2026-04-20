#!/bin/bash
# Piper 双臂 LeRobot 专家示教采集启动脚本

CTRL_MODE="joint"
TASK_NAME="piper_dual_lerobot_${CTRL_MODE}_task"
TASK_DESCRIPTION="dual piper ${CTRL_MODE} teleoperation"
CONFIG_PATH="agent_infra/Piper_Env/Config/dual_piper_config.yaml"
MASTER_CAN_LEFT="can_ml"
MASTER_CAN_RIGHT="can_mr"
SLAVE_CAN_LEFT="can_sl"
SLAVE_CAN_RIGHT="can_sr"
VCODEC="h264"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --master)
      if [[ $# -lt 3 || "$2" == --* || "$3" == --* ]]; then
        echo "[Launcher] --master 需要 2 个 CAN 名称：left right。"
        exit 2
      fi
      MASTER_CAN_LEFT="$2"
      MASTER_CAN_RIGHT="$3"
      shift 3
      ;;
    --slave)
      if [[ $# -lt 3 || "$2" == --* || "$3" == --* ]]; then
        echo "[Launcher] --slave 需要 2 个 CAN 名称：left right。"
        exit 2
      fi
      SLAVE_CAN_LEFT="$2"
      SLAVE_CAN_RIGHT="$3"
      shift 3
      ;;
    *)
      echo "[Launcher] 未知参数: $1"
      echo "用法: bash $0 [--master can_ml can_mr] [--slave can_sl can_sr]"
      exit 2
      ;;
  esac
done

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[Launcher] 启动双臂 LeRobot 录制模式..."
"$PYTHON_BIN" -m agent_infra.Piper_Env.Record.recorder \
  -m lerobot \
  -t "$TASK_NAME" \
  --task-description "$TASK_DESCRIPTION" \ti
  --vcodec "$VCODEC" \
  -ctrl "$CTRL_MODE" \
  -cfg "$CONFIG_PATH" \
  -dual \
  --master "$MASTER_CAN_LEFT" "$MASTER_CAN_RIGHT" \
  --slave "$SLAVE_CAN_LEFT" "$SLAVE_CAN_RIGHT"
