import time
import numpy as np
import os
import sys

# 将项目根目录添加到 python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from agent_infra.Realman_Env.Env.realman_env import RealManEnv
from agent_infra.Realman_Env.Camera.realsense_camera import get_connected_realsense_serials

def run_test_loop(mode):
    print(f"\n>>>>>> 开始单臂模块化测试模式: {mode} <<<<<<")
    serials = get_connected_realsense_serials()
    print(serials)
    # 配置参数
    env_config = {
        "robot_ip": "192.168.2.18",
        "camera_sns": serials,
        "control_mode": mode,
        "hz": 10,
        "crop_size": (224, 224),
    }

    try:
        env = RealManEnv(**env_config)
        
        # 1. 测试 Reset
        print("正在测试 Reset (模块化回归初始位姿)...")
        obs, info = env.reset()
        # 注意：现在 state 在字典内
        print(f"Reset 完成。当前关节角度 (rad): {obs['state']['joint_pos']}")
        
        # 2. 测试 Step 循环
        print(f"开始执行 10 步动作测试...")
        for i in range(10):
            # 组装字典动作 (Base 类要求字典格式)
            if mode == 'joint_pos':
                # 保持原位不动 + 随机小扰动
                arm_action = obs['state']['joint_pos'] + np.random.uniform(-0.01, 0.01, size=6)
                action = {"arm": arm_action, "gripper": np.array([0.5], dtype=np.float32)}
            else:
                action = {"arm": np.zeros(6, dtype=np.float32), "gripper": np.array([0.5], dtype=np.float32)}
            
            # 执行一步
            env.switch_passive("true")
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 打印关键数据以确认 (假设手眼相机名为 hand_camera)
            cam_name = list(obs['rgb'].keys())[0]
            img_shape = obs['rgb'][cam_name].shape
            joint_mean = np.mean(obs['state']['joint_pos'])
            print(f"Step {i+1:02d} | 图像视角: {cam_name} | 尺寸: {img_shape} | 关节均值: {joint_mean:.4f}")
            
        print(f"模式 {mode} 测试顺利结束。")
        env.reset()

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback; traceback.print_exc()
    
    finally:
        if 'env' in locals():
            env.close()
        print(f">>>>>> 测试完成 <<<<<<\n")

if __name__ == "__main__":
    run_test_loop("joint_pos")
