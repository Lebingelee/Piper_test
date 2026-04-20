import os
import time
import numpy as np
import argparse
import h5py
import json
from agent_infra.Realman_Env.Env.realman_env import RealManEnv
from agent_infra.Realman_Env.Env.dual_realman_env import DualRealManEnv

def replay_trajectory(file_path: str, hz: int):
    # 1. 加载数据并解析元数据
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}"); return
    
    print(f"正在加载轨迹: {file_path}")
    with h5py.File(file_path, 'r') as f:
        # A. 提取元数据判断单双臂
        if 'meta/env_kwargs' in f:
            kwargs = json.loads(f['meta/env_kwargs'][()])
            is_dual = kwargs.get('is_dual_arm', False)
            control_mode = kwargs.get('control_mode', 'joint_pos')
            robot_ips = kwargs.get('robot_ips', [])
        else:
            # 兼容旧版本
            is_dual = False
            control_mode = 'joint_pos'
            robot_ips = ["192.168.2.18"]

        # B. 还原数据为列表字典格式
        # 针对新版扁平结构，需遍历 obs 下的所有数据集
        def get_obs_at_t(group, t):
            obs = {}
            for k in group.keys():
                if isinstance(group[k], h5py.Group):
                    obs[k] = get_obs_at_t(group[k], t)
                else:
                    obs[k] = group[k][t]
            return obs

        num_steps = f['obs/state/joint_pos'].shape[0] if not is_dual else f['obs/state/left_joint_pos'].shape[0]
        data = [get_obs_at_t(f['obs'], t) for t in range(num_steps)]
        
        # 如果文件里已经有计算好的 actions，直接取
        has_actions = 'actions' in f
        actions_in_file = f['actions'][()] if has_actions else None

    T = len(data) - 1
    if T <= 0:
        print("轨迹太短，无法回放。"); return

    # 2. 初始化环境 (不开启相机以节省性能)
    if is_dual:
        print("[Replay] 初始化双臂环境...")
        env = DualRealManEnv(hz=hz, control_mode=control_mode, camera_sns=[])
    else:
        print("[Replay] 初始化单臂环境...")
        env = RealManEnv(robot_ip=robot_ips[0], hz=hz, control_mode=control_mode, camera_sns=[])
    
    env.switch_passive("true")
    
    try:
        # 3. 复位到起始点
        print("正在复位至轨迹起始位姿...")
        # 组装初始动作 (为了 movej)
        obs_start = data[0]['state']
        # 遍历所有臂进行复位
        for name in env.arm_names:
            prefix = f"{name}_" if len(env.arm_names) > 1 else ""
            start_joints = np.rad2deg(obs_start[f"{prefix}joint_pos"]).tolist()
            start_gripper = int(obs_start[f"{prefix}gripper_pos"][0] * 999 + 1)
            env.arms[name].arm.rm_movej(start_joints, 20, 0, 0, 1)
            env.arms[name].arm.rm_set_gripper_position(start_gripper, block=True, timeout=5)
        
        time.sleep(1.0)

        print(f"开始回放 (共 {T} 步)...")
        for t in range(T):
            t_start = time.time()
            
            # 4. 获取动作
            if has_actions:
                # 直接使用扁平化动作向量，Wrapper 会处理拆分
                action = actions_in_file[t]
            else:
                # 兼容模式：手动根据状态差值拼合动作字典
                action = {}
                s_next = data[t+1]['state']
                for name in env.arm_names:
                    prefix = f"{name}_" if len(env.arm_names) > 1 else ""
                    if control_mode == "joint_pos":
                        arm_act = s_next[f"{prefix}joint_pos"]
                    else:
                        arm_act = s_next[f"{prefix}ee_pose"] - data[t]['state'][f"{prefix}ee_pose"]
                    
                    action[f"{prefix}arm"] = arm_act
                    action[f"{prefix}gripper"] = s_next[f"{prefix}gripper_pos"]

            # 5. 执行
            env.step(action)
            
            if t % 10 == 0: print(f"进度: {t}/{T}")

        print("回放完成。")
        
    except Exception as e:
        print(f"回放错误: {e}")
        import traceback; traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-hz", "--hz", type=int, default=10)
    args = parser.parse_args()
    replay_trajectory(args.input, args.hz)
