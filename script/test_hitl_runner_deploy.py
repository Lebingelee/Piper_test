import argparse
import os
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import make_agent
from agent_factory.config.manager import ConfigManager
from agent_factory.env.env_factories import create_env
from agent_factory.runner import HITLDeployRunner
from agent_factory.runner.checkpoint_utils import ensure_action_normalizer_ready


DEFAULT_CONFIG_PATH = "run_results/piper_dual_merged_cpiql_dac/model_config.yaml"
DEFAULT_CHECKPOINT_PATH = (
    "run_results/piper_dual_merged_cpiql_dac/actor_step_60000.pth"
)
DEFAULT_SAVE_DIR = "data/piper_dual_merged_cpiql_dac_hitl_deploy_runner"


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
        print("[Test-HITL-Deploy] Camera startup hook not found; visual obs may be dummy frames.")
        return

    print("[Test-HITL-Deploy] Ensuring camera threads are running...")
    start_cameras()
    if warmup > 0:
        print(f"[Test-HITL-Deploy] Camera warmup: {warmup:.2f}s")
        time.sleep(warmup)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HITLDeployRunner with a configured agent on Piper env."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--control-hz", type=int, default=0)
    parser.add_argument("--risk-check-hz", type=float, default=0.0)
    parser.add_argument(
        "--risk-no-safe-action-gap",
        action="store_true",
        help="Pause env.step() after risk trigger instead of filling with safe_action.",
    )
    parser.add_argument("--save-key", default="s")
    parser.add_argument("--continue-key", default="c")
    parser.add_argument("--camera-warmup", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"[Error] Config file not found at {args.config}")
        return

    print(f"[Test-HITL-Deploy] Loading config from {args.config}...")
    cfg = ConfigManager.load_config(args.config)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device
    cfg.train.device = device
    cfg.env.server_mode = False
    cfg.runner.hitl_enabled = True
    cfg.env.max_episode_steps = int(args.max_steps)
    if args.control_hz > 0:
        cfg.runner.control_hz = int(args.control_hz)
    if args.save_dir:
        cfg.runner.save_dir = args.save_dir
    cfg.runner.risk_check_hz = float(args.risk_check_hz)
    cfg.runner.risk_use_safe_action = not bool(args.risk_no_safe_action_gap)
    cfg.runner.deploy_save_key = str(args.save_key)
    cfg.runner.deploy_continue_key = str(args.continue_key)
    os.makedirs(cfg.runner.save_dir, exist_ok=True)

    if not args.skip_load:
        if not args.checkpoint:
            print("[Error] --checkpoint is empty. Use --skip-load to run without weights.")
            return
        if not os.path.exists(args.checkpoint):
            print(f"[Error] Checkpoint file not found at {args.checkpoint}")
            return

    print("[Test-HITL-Deploy] Initializing Piper Environment...")
    try:
        env = create_env(cfg.env)
    except Exception as exc:
        print(f"[Error] Failed to create environment: {exc}")
        return
    _start_cameras_if_available(env, args.camera_warmup)

    print(f"[Test-HITL-Deploy] Initializing Agent: {cfg.agent_type} ...")
    agent = make_agent(cfg.agent_type, cfg)
    if not args.skip_load:
        print(f"[Test-HITL-Deploy] Loading checkpoint from {args.checkpoint} ...")
        agent.load(args.checkpoint)
        ensure_action_normalizer_ready(agent, cfg)
    agent.to(device)
    agent.eval()

    print("[Test-HITL-Deploy] Initializing HITLDeployRunner...")
    if hasattr(env.unwrapped, "switch_passive"):
        env.unwrapped.switch_passive("true")
    runner = HITLDeployRunner(cfg=cfg, agent=agent, env=env)

    print(
        f"[Test-HITL-Deploy] Starting rollout with {cfg.agent_type}; "
        f"session files will be saved to {cfg.runner.save_dir}"
    )
    try:
        runner.run()
    except KeyboardInterrupt:
        print("[Test-HITL-Deploy] KeyboardInterrupt! Stopping runner...")
    finally:
        runner.stop_worker()
        env.close()
        print("[Test-HITL-Deploy] Done.")


if __name__ == "__main__":
    main()
