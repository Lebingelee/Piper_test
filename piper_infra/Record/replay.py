import argparse
import time
import numpy as np
import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from piper_infra.Env.single_piper_env import PiperTeleopEnv
from piper_infra.Record.recoder import AbstractCameraWrapper

def replay_trajectory(dataset_path: str, episode_idx: int = -1, fps: int = None, ctrl_mode: str = "joint"):
    """
    重播指定的轨迹。
    """
    # ... 
    dataset = LeRobotDataset(repo_id=path.name, root=path)
    
    # 2. 初始化环境 (传入控制模式)
    base_env = PiperTeleopEnv(master_can="can_master", follower_can="can_slave", ctrl_mode=ctrl_mode)
    # ...
    # 我们依然加上相机包装器，以保持架构一致性 (即使重播不一定需要视觉，但在 reset/step 时结构要匹配)
    env = AbstractCameraWrapper(base_env, camera_names=["wrist", "front"])
    
    # 确定要重播的 episode 列表
    episodes_to_replay = []
    if episode_idx == -1:
        episodes_to_replay = list(range(dataset.num_episodes))
    else:
        if 0 <= episode_idx < dataset.num_episodes:
            episodes_to_replay = [episode_idx]
        else:
            print(f"[Replay] 错误: 索引 {episode_idx} 超出范围。")
            return

    try:
        for ep in episodes_to_replay:
            print(f"\n[Replay] >>> 开始重播 Episode {ep} ...")
            
            # 获取该 Episode 的第一帧，进行对齐 (Reset)
            # 使用 hf_dataset 绕过视频解码，避免环境 FFmpeg 报错
            from_idx = dataset.meta.episodes[ep]["dataset_from_index"]
            to_idx = dataset.meta.episodes[ep]["dataset_to_index"]
            
            first_frame = dataset.hf_dataset[from_idx]
            # state: [6*joints, 6*vels, 6*pose, 1*gripper]
            initial_state = first_frame["observation.state"]
            initial_joints = initial_state[:6].numpy()
            initial_gripper = initial_state[18].item()
            
            # 使用 .unwrapped 访问底层环境的自定义方法 reset_to_state
            env.unwrapped.reset_to_state(initial_joints, initial_gripper, wait_time=3.0)
            
            # 开始帧步进重播
            for idx in range(from_idx, to_idx):
                start_time = time.perf_counter()
                
                # 同样通过 hf_dataset 读取，只取数值，不解码视频
                frame = dataset.hf_dataset[idx]
                target_action = frame["action"].numpy()
                
                # 执行 step。注意：重播时 Env 内部的 tele_enabled 默认为 False，
                # 所以 step 不会拦截 action，而是直接下发给从臂。
                # 由于 action 序列通常是平滑的，step 内部的 diff 逻辑会自动切换 move_js 或 move_j。
                env.step(target_action)
                
                # 严格控制频率
                elapsed = time.perf_counter() - start_time
                time.sleep(max(0, (1.0 / replay_fps) - elapsed))
            
            print(f"[Replay] <<< Episode {ep} 重播结束。")
            time.sleep(1.0) # Episode 间歇

    except KeyboardInterrupt:
        print("\n[Replay] 用户中断重播。")
    finally:
        env.close()
        print("[Replay] 环境已安全退出。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piper 轨迹重播工具")
    parser.add_argument("--path", type=str, default=None,  help="数据集根目录路径")
    parser.add_argument("--episode", type=int, default=-1, help="重播的轨迹索引")
    parser.add_argument("--fps", type=int, default=None, help="重播频率")
    parser.add_argument("--ctrl_mode", type=str, default="joint", choices=["joint", "pose"], help="控制模式: joint 或 pose")

    args = parser.parse_args()
    if args.path is None:
        args.path = "data/piper_recording_000"
    replay_trajectory(args.path, args.episode, args.fps, ctrl_mode=args.ctrl_mode)

