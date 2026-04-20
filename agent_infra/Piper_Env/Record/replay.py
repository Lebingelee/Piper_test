import time
import h5py
import argparse

from agent_infra.Piper_Env.Env.utils.piper_base_env import PiperEnv


def _extract_state_from_h5_group(state_group):
    return {key: state_group[key][0] for key in state_group.keys()}


def replay_h5(env, file_path: str):
    """回放 H5 格式的单条轨迹"""
    print(f"[Replay] 正在从 H5 加载轨迹: {file_path}")
    with h5py.File(file_path, 'r') as f:
        action_group = f['action']
        action_keys = list(action_group.keys())
        action_seq = {
            key: action_group[key][()]
            for key in action_keys
        }
        initial_state = _extract_state_from_h5_group(f["obs/state"])

    print("[Replay] 正在复位至轨迹起始点...")
    env.reset(options={"target_state": initial_state, "wait_time": 1.0})

    num_steps = len(next(iter(action_seq.values())))
    print(f"[Replay] 开始回放 ({num_steps} 步)...")
    for t in range(num_steps):
        start_t = time.perf_counter()
        action = {key: value[t] for key, value in action_seq.items()}
        env.step(action)
        elapsed = time.perf_counter() - start_t
        time.sleep(max(0, (1.0/env.hz) - elapsed))
    print("[Replay] H5 轨迹回放完成。")

def replay_lerobot(env, dataset_path: str, episode_idx: int):
    """回放 LeRobot 格式的轨迹"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print(f"[Replay] 正在从 LeRobot 加载 Episode {episode_idx}: {dataset_path}")
    
    dataset = LeRobotDataset(repo_id="replay", root=dataset_path)
    from_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    to_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]

    first_frame = dataset.hf_dataset[from_idx]
    state_vec = first_frame["observation.state"].numpy()

    initial_state = {}
    cursor = 0
    for key, shape in env.unwrapped.meta_keys["obs"]["state"].items():
        dim = shape[0]
        initial_state[key] = state_vec[cursor:cursor + dim]
        cursor += dim

    print("[Replay] 正在复位至轨迹起始点...")
    env.reset(options={"target_state": initial_state, "wait_time": 2.0})

    for idx in range(from_idx, to_idx):
        start_t = time.perf_counter()
        frame = dataset.hf_dataset[idx]
        action_vec = frame["action"].numpy()

        action = {}
        cursor = 0
        for key, shape in env.unwrapped.meta_keys["action"].items():
            dim = shape[0]
            action[key] = action_vec[cursor:cursor + dim]
            cursor += dim
        env.step(action)

        elapsed = time.perf_counter() - start_t
        time.sleep(max(0, (1.0/env.hz) - elapsed))
    print("[Replay] LeRobot 轨迹回放完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="H5文件路径或LeRobot目录")
    parser.add_argument("-f", "--format", type=str, default="h5", choices=["h5", "lerobot"])
    parser.add_argument("-ep", "--episode", type=int, default=0)
    parser.add_argument("-ctrl", "--control", type=str, default="joint")
    parser.add_argument("-cfg", "--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    # 初始化环境 (回放通常不开启相机包装器以节省资源)
    env = PiperEnv(config_path=args.config, control_mode=args.control)
    
    try:
        if args.format == "h5":
            replay_h5(env, args.input)
        else:
            replay_lerobot(env, args.input, args.episode)
    finally:
        env.close()
