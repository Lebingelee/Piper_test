#!/bin/bash
# collect_data.sh: Realman 单臂专家轨迹录制入口

TASK_NAME="Single_Arm_Task" 
HZ=10
MODE="joint_pos" 
MAX_STEPS=250

# 1. 确保在项目根目录下执行
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. 构建命令 (单臂模式不传 -dual 参数)
CMD="python3 -m agent_infra.Realman_Env.Record.recorder -hz $HZ -m $MODE -max_steps $MAX_STEPS"

if [ -n "$TASK_NAME" ]; then
    CMD="$CMD -t $TASK_NAME"
fi

echo "正在启动 [单臂] 录制任务: ${TASK_NAME}"
eval $CMD
