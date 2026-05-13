import os
import sys
import time
import argparse
import torch

# 将根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import make_agent
from agent_factory.config.manager import ConfigManager
from agent_factory.env.env_factories import create_env
from agent_factory.runner import BaseRunner
from agent_factory.runner.checkpoint_utils import ensure_action_normalizer_ready



DEFAULT_CONFIG_PATH = "run_results/piper_dual_ITQC_plain/model_config.yaml"
DEFAULT_CHECKPOINT_PATH = (
    "run_results/piper_dual_ITQC_plain/piper_dual_ITQC_pretrain_final.pth"
)
DEFAULT_SAVE_DIR = "data/piper_dual_merged_cpiql_dac_base_runner"


def _start_cameras_if_available(env, warmup: float):
    start_cameras = None
    try:
        if hasattr(env, "get_wrapper_attr"):
            start_cameras = env.get_wrapper_attr("start_cameras")
    except AttributeError:
        start_cameras = None

    if start_cameras is None:
        start_cameras = getattr(env, "start_cameras", None)

    if start_cameras is None:
        print("[Test-Base] Camera startup hook not found; visual obs may be dummy frames.")
        return

    print("[Test-Base] Ensuring camera threads are running...")
    start_cameras()
    if warmup > 0:
        print(f"[Test-Base] Camera warmup: {warmup:.2f}s")
        time.sleep(warmup)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BaseRunner with a configured agent on Piper env."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--sleep-between", type=float, default=2.0)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--control-hz", type=int, default=0)
    parser.add_argument(
        "--no-safe-action-gap",
        action="store_true",
        help="Wait for the next policy chunk without env.step() or safe_action during planning gaps.",
    )
    parser.add_argument(
        "--planning-wait-sleep",
        type=float,
        default=0.002,
        help="Sleep interval while waiting for a policy chunk in --no-safe-action-gap mode.",
    )
    parser.add_argument(
        "--camera-warmup",
        type=float,
        default=1.0,
        help="Seconds to wait after starting camera threads before rollout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载配置
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"[Error] Config file not found at {config_path}")
        return

    print(f"[Test] Loading config from {config_path}...")
    cfg = ConfigManager.load_config(config_path)
    
    # 确保配置正确 (使用较短的 act_horizon 和 max_steps 用于测试)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device
    cfg.train.device = device
    cfg.env.server_mode = False
    cfg.runner.hitl_enabled = False
    if args.max_steps > 0:
        cfg.env.max_episode_steps = args.max_steps
    if args.control_hz > 0:
        cfg.runner.control_hz = args.control_hz
    if args.save_dir:
        cfg.runner.save_dir = args.save_dir
    os.makedirs(cfg.runner.save_dir, exist_ok=True)

    if not args.skip_load:
        if not args.checkpoint:
            print("[Error] --checkpoint is empty. Use --skip-load to run without weights.")
            return
        if not os.path.exists(args.checkpoint):
            print(f"[Error] Checkpoint file not found at {args.checkpoint}")
            return

    # 2. 初始化环境
    print("[Test-Base] Initializing Piper Environment...")
    try:
        env = create_env(cfg.env)
    except Exception as e:
        print(f"[Error] Failed to create environment: {e}")
        return
    _start_cameras_if_available(env, args.camera_warmup)

    # 3. 初始化 Agent
    print(f"[Test-Base] Initializing Agent: {cfg.agent_type} ...")
    agent = make_agent(cfg.agent_type, cfg)
    if not args.skip_load:
        print(f"[Test-Base] Loading checkpoint from {args.checkpoint} ...")
        agent.load(args.checkpoint)
        ensure_action_normalizer_ready(agent, cfg)
    agent.to(device)
    agent.eval()

    # Runner-only diagnostic knobs must be attached after make_agent().
    # make_agent() merges cfg into structured agent defaults, whose RunnerConfig
    # intentionally does not know about this temporary hardware test flag.
    cfg.runner.no_safe_action_gap = bool(args.no_safe_action_gap)
    cfg.runner.planning_wait_sleep = float(args.planning_wait_sleep)

    # 4. 初始化 BaseRunner
    print("[Test-Base] Initializing BaseRunner ...")
    if hasattr(env.unwrapped, "switch_passive"):
        env.unwrapped.switch_passive("true")
    runner = BaseRunner(cfg=cfg, agent=agent, env=env)

    # 5. 执行一次 Rollout
    print(
        f"[Test-Base] Starting rollout with {cfg.agent_type}; "
        f"saving to {cfg.runner.save_dir}"
    )
    try:
        for episode_idx in range(args.episodes):
            print(f"[Test-Base] Episode {episode_idx + 1}/{args.episodes}")
            runner.run()
            time.sleep(args.sleep_between)
    except KeyboardInterrupt:
        print("[Test-Base] KeyboardInterrupt! Stopping worker...")
    finally:
        runner.stop_worker()
        env.close()
        print("[Test-Base] Done.")

if __name__ == "__main__":
    main()
