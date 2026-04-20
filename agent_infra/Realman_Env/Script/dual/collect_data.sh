#!/bin/bash
# collect_data.sh: Realman 双臂专家轨迹录制入口

TASK_NAME="bottle" 
HZ=15
MODE="joint_pos" 
MAX_STEPS=600

export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. 构建命令 (关键：增加 -dual 标志)
CMD="python3 -m agent_infra.Realman_Env.Record.recorder -dual -hz $HZ -m $MODE -max_steps $MAX_STEPS"

if [ -n "$TASK_NAME" ]; then
    CMD="$CMD -t $TASK_NAME"
fi

echo "正在启动 [双臂] 录制任务: ${TASK_NAME}"
eval $CMD
