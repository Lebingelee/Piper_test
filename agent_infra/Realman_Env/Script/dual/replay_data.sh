#!/bin/bash
# replay_data.sh: 对双臂轨迹进行回放。回放脚本会自动通过元数据识别双臂环境。

H5_PATH="agent_infra/Realman_Env/Record/data/Towel_folding/traj_49_164016.h5"
HZ=20

export PYTHONPATH=$PYTHONPATH:$(pwd)

if [ ! -f "$H5_PATH" ]; then
    echo "错误: 未找到轨迹文件 [$H5_PATH]"
    exit 1
fi

echo "正在开始回放 [双臂] 轨迹: $H5_PATH"
python3 -m agent_infra.Realman_Env.Record.replay_data -i "$H5_PATH" -hz $HZ
