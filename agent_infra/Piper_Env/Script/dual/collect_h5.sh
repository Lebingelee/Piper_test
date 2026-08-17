#!/bin/bash
# Piper 双臂 H5 专家示教采集启动脚本

CTRL_MODE="joint"
TASK_NAME="piper_STACK_h5_${CTRL_MODE}_task"
CONFIG_PATH="agent_infra/Piper_Env/Config/dual_piper_config.yaml"
MASTER_CAN_LEFT="can_ml"
MASTER_CAN_RIGHT="can_mr"
SLAVE_CAN_LEFT="can_sl"
SLAVE_CAN_RIGHT="can_sr"
MAX_STEP="${MAX_STEP:-800}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PREVIEW="${PREVIEW:-auto}"

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
    --max-step|--max_step)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] --max-step 需要 1 个整数。"
        exit 2
      fi
      MAX_STEP="$2"
      shift 2
      ;;
    --preview)
      PREVIEW="true"
      shift
      ;;
    --no-preview|--headless)
      PREVIEW="false"
      shift
      ;;
    *)
      echo "[Launcher] 未知参数: $1"
      echo "用法: bash $0 [--master can_ml can_mr] [--slave can_sl can_sr] [--max-step N|-1] [--preview|--no-preview]"
      exit 2
      ;;
  esac
done

export PYTHONPATH="$(pwd):$PYTHONPATH"

if [[ "$PREVIEW" == "auto" ]]; then
  if [[ -n "${SSH_CONNECTION:-}${SSH_TTY:-}" && -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    PREVIEW="false"
  else
    PREVIEW="true"
  fi
fi

PREVIEW_ARGS=()
case "$PREVIEW" in
  true|1|yes|on)
    PREVIEW_ARGS=(--preview)
    ;;
  false|0|no|off)
    PREVIEW_ARGS=(--no-preview)
    ;;
  *)
    echo "[Launcher] PREVIEW 只能是 auto/true/false，当前: $PREVIEW"
    exit 2
    ;;
esac

echo "[Launcher] 启动双臂 H5 录制模式..."
echo "[Launcher] OpenCV 预览: $PREVIEW"
"$PYTHON_BIN" -m agent_infra.Piper_Env.Record.recorder \
  -m h5 \
  -t "$TASK_NAME" \
  -ctrl "$CTRL_MODE" \
  -cfg "$CONFIG_PATH" \
  -dual \
  --master "$MASTER_CAN_LEFT" "$MASTER_CAN_RIGHT" \
  --slave "$SLAVE_CAN_LEFT" "$SLAVE_CAN_RIGHT" \
  --max-step "$MAX_STEP" \
  "${PREVIEW_ARGS[@]}"
