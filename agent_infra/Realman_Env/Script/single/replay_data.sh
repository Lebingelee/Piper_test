#!/bin/bash
# replay_data.sh: 对单臂轨迹进行回放

H5_PATH="agent_infra/Realman_Env/Record/data/Single_Arm_Task/traj_0_xxxxxx.h5"
HZ=10

export PYTHONPATH=$PYTHONPATH:$(pwd)

if [ ! -f "$H5_PATH" ]; then
    echo "错误: 未找到轨迹文件 [$H5_PATH]"
    exit 1
fi

echo "正在开始回放 [单臂] 轨迹: $H5_PATH"
python3 -m agent_infra.Realman_Env.Record.replay_data -i "$H5_PATH" -hz $HZ
