import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from pynput import keyboard
from typing import Optional
from piper_infra.Env.get_pose import get_pose

# 导入 LeRobot 标准 API
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("[Error] 未检测到 lerobot 库，请先执行: pip install lerobot")
    exit(1)

# 导入同级及上级目录的组件
from piper_infra.Env.single_piper_env import PiperTeleopEnv
from piper_infra.Record.recoder import AbstractCameraWrapper
from piper_infra.Record.utils import get_next_available_path

# --- 全局控制变量 ---
is_recording = False
exit_requested = False

def on_press(key):
    global is_recording, exit_requested
    try:
        if hasattr(key, 'char'):
            if key.char == 'r' or key.char == 'R':
                is_recording = not is_recording
                status = "● 开始录制 (RECORDING)" if is_recording else "○ 停止录制 (STOPPED)"
                print(f"\n[Recorder] {status}")
            elif key.char == 'q' or key.char == 'Q':
                exit_requested = True
                print("\n[Recorder] 退出请求中...")
    except Exception as e:
        print(f"Key error: {e}")

def record_teleop_session(custom_path: Optional[str] = None, ctrl_mode: str = "joint"):
    """
    Piper 数据录制主程序 (多 episode 增强版)。

    Args:
        custom_path: 用户指定的存储路径。
        ctrl_mode: 控制模式 ('joint' 或 'pose')。
    """
    global is_recording, exit_requested

    # --- 1. 路径自适应逻辑 ---
    target_full_path = get_next_available_path(custom_path, mode=ctrl_mode)
    REPO_ID = target_full_path.name 

    # --- 2. 环境初始化 ---
    FPS = 50  
    base_env = PiperTeleopEnv(master_can="can_master", follower_can="can_slave", ctrl_mode=ctrl_mode)

    env = AbstractCameraWrapper(base_env, camera_names=["wrist", "front"])
    env.start_cameras()

    # --- 3. LeRobot 数据集特征注册 ---
    features = {
        "observation.state": {
            "dtype": "float32", 
            "shape": (19,), 
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "v1", "v2", "v3", "v4", "v5", "v6", "x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
        "observation.images.wrist": {"dtype": "video", "shape": (3, 224, 224)},
        "observation.images.front": {"dtype": "video", "shape": (3, 224, 224)},
        "action": {"dtype": "float32", "shape": (7,)},
        "meta.intervened": {"dtype": "bool", "shape": (1,)},
    }

    # 创建数据集
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=target_full_path, # 直接将 root 设置为自适应生成的完整路径
        features=features,
        use_videos=True,
    )

    # 启动录制控制监听器
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print(f"\n[Recorder] 数据录制环境就绪")
    print(f"[Recorder] 存储路径: {target_full_path}")
    print(f"------------------------------------------------")
    print(f"操作指南:")
    print(f"  [T] - 切换机械臂遥操状态 (Env 内部控制)")
    print(f"  [R] - 开启/停止单段数据录制 (Episode)")
    print(f"  [Q] - 安全退出并保存")
    print(f"------------------------------------------------\n")

    episode_count = 0
    
    # --- 4. 录制主循环 ---
    try:
        while not exit_requested:
            obs, info = env.reset()
            print(f"[Recorder] 等待录制信号... (当前 Episode: {episode_count})")
            
            # 等待按下 R 键开始录制
            while not is_recording and not exit_requested:
                time.sleep(0.1)
            
            if exit_requested:
                break
                
            print(f"[Recorder] >>> 正在录制 Episode {episode_count}...")
            
            while is_recording and not exit_requested:
                start_time = time.perf_counter()
                
                # 维持姿态的 Dummy Action
                current_state = obs["observation.state"]
                dummy_action = np.zeros(7, dtype=np.float32)

                if ctrl_mode == "pose":
                    dummy_action[:6] = current_state[12:18]
                else:    
                    dummy_action[:6] = current_state[:6]

                dummy_action[6] = current_state[18]
                
                # 环境步进
                obs, reward, terminated, truncated, info = env.step(dummy_action)
                
                # 提取拦截后的真实动作与标签
                actual_action = info["actual_action"]
                intervened_flag = info["intervened"]

                # 组装 LeRobot 帧数据
                frame_data = {
                    "observation.state": torch.from_numpy(obs["observation.state"]),
                    "observation.images.wrist": torch.from_numpy(obs["observation.images.wrist"]),
                    "observation.images.front": torch.from_numpy(obs["observation.images.front"]),
                    "action": torch.from_numpy(actual_action.astype(np.float32)),
                    "meta.intervened": torch.tensor([intervened_flag], dtype=torch.bool),
                    "task": "Piper robot teleoperation task.",
                }
                
                # 添加到缓冲区
                dataset.add_frame(frame_data)
                
                # 频率控制
                elapsed = time.perf_counter() - start_time
                time.sleep(max(0, (1.0 / FPS) - elapsed))
            
            # 保存当前 Episode
            if len(dataset.episode_buffer["action"]) > 0:
                print(f"[Recorder] <<< Episode {episode_count} 录制结束，正在保存至磁盘...")
                dataset.save_episode()
                episode_count += 1
                print(f"[Recorder] Episode {episode_count-1} 保存成功。当前总帧数: {len(dataset)}")

    except KeyboardInterrupt:
        print("\n[Recorder] 检测到强制中断...")
    finally:
        print("[Recorder] 正在执行安全退出程序...")
        listener.stop()
        env.stop_cameras()
        env.close()
        if hasattr(dataset, 'finalize'):
            dataset.finalize()
        print(f"[Recorder] 录制任务完成。总 Episode 数: {episode_count}，总帧数: {len(dataset)},保存地址:{target_full_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Piper 数据录制工具")
    parser.add_argument("--custom_path", type=str, default=None, help="自定义保存路径")
    parser.add_argument("--ctrl_mode", type=str, default="joint", choices=["joint", "pose"], help="控制模式: joint 或 pose")
    
    args = parser.parse_args()
    record_teleop_session(custom_path=args.custom_path, ctrl_mode=args.ctrl_mode)


