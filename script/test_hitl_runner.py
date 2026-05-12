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
from agent_factory.runner import HITLRunner
from agent_factory.runner.checkpoint_utils import ensure_action_normalizer_ready


DEFAULT_CONFIG_PATH = "run_results/piper_dual_merged_cpiql_dac/model_config.yaml"
DEFAULT_CHECKPOINT_PATH = (
    "run_results/piper_dual_merged_cpiql_dac/cpiql_critic_step_8000.pth"
)
DEFAULT_SAVE_DIR = "data/piper_dual_merged_cpiql_dac_hitl_runner"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HITLRunner with a configured agent on Piper env."
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
    parser.add_argument("--override-key", default="o")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载配置
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"[Error] Config file not found at {config_path}")
        return

    print(f"[Test-HITL] Loading config from {config_path}...")
    cfg = ConfigManager.load_config(config_path)

    # 2. 覆盖测试配置（HITL）
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device
    cfg.train.device = device
    cfg.env.server_mode = False
    cfg.runner.hitl_enabled = True
    if args.max_steps > 0:
        cfg.env.max_episode_steps = args.max_steps
    if args.control_hz > 0:
        cfg.runner.control_hz = args.control_hz
    if args.save_dir:
        cfg.runner.save_dir = args.save_dir
    if args.override_key:
        cfg.runner.hitl_override_key = args.override_key
    cfg.runner.hitl_source = "keyboard"
    cfg.runner.hitl_finalize_intervention = True
    os.makedirs(cfg.runner.save_dir, exist_ok=True)

    if not args.skip_load:
        if not args.checkpoint:
            print("[Error] --checkpoint is empty. Use --skip-load to run without weights.")
            return
        if not os.path.exists(args.checkpoint):
            print(f"[Error] Checkpoint file not found at {args.checkpoint}")
            return

    # 3. 初始化环境
    print("[Test-HITL] Initializing Piper Environment...")
    try:
        env = create_env(cfg.env)
    except Exception as e:
        print(f"[Error] Failed to create environment: {e}")
        return

    # 4. 初始化 Agent
    print(f"[Test-HITL] Initializing Agent: {cfg.agent_type} ...")
    agent = make_agent(cfg.agent_type, cfg)
    if not args.skip_load:
        print(f"[Test-HITL] Loading checkpoint from {args.checkpoint} ...")
        agent.load(args.checkpoint)
        ensure_action_normalizer_ready(agent, cfg)
    agent.to(device)
    agent.eval()

    # 5. 初始化 HITLRunner
    print("[Test-HITL] Initializing HITLRunner...")
    print(
        f"[Test-HITL] Press '{cfg.runner.hitl_override_key}' to toggle runner override. "
        "Piper master teleop can also intervene through env info."
    )
    if hasattr(env.unwrapped, "switch_passive"):
        env.unwrapped.switch_passive("false")
    runner = HITLRunner(cfg=cfg, agent=agent, env=env)

    # 6. 执行 Rollout
    print(
        f"[Test-HITL] Starting rollout with {cfg.agent_type}; "
        f"interventions will be saved to {cfg.runner.save_dir}"
    )
    try:
        for episode_idx in range(args.episodes):
            print(f"[Test-HITL] Episode {episode_idx + 1}/{args.episodes}")
            runner.run()
            time.sleep(args.sleep_between)
    except KeyboardInterrupt:
        print("[Test-HITL] KeyboardInterrupt! Stopping runner...")
    finally:
        runner.stop_worker()
        env.close()
        print("[Test-HITL] Done.")


if __name__ == "__main__":
    main()
