import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import make_agent
from agent_factory.config.resolution import general_resolve
from agent_factory.env.env_factories import create_env
from agent_factory.runner import HITLDeployRunner
from agent_factory.runner.hitl_runner_deploy import (
    DEPLOY_INIT_KEY,
    DEPLOY_QUIT_KEY,
    DEPLOY_START_KEY,
    DEPLOY_STOP_KEY,
    DEPLOY_TELEOP_KEY,
)
from agent_factory.runner.checkpoint_utils import ensure_action_normalizer_ready


DEFAULT_CONFIG_PATH = "run_results/identity_piper_runner/config.yaml"
#DEFAULT_CONFIG_PATH = "run_results/piper_dual_merged_cpiql_dac/model_config.yaml"
DEFAULT_CHECKPOINT_PATH = (
    "run_results/piper_dual_merged_cpiql_dac/actor_step_60000.pth"
)
DEFAULT_SAVE_DIR = "data/hitl_deploy_runner_test"


def _load_raw_config(path: str):
    loaded = OmegaConf.load(path)
    value = OmegaConf.to_container(loaded, resolve=False)
    return value if isinstance(value, dict) else {}


def _prepare_deploy_config(raw_cfg: dict, args, device: str) -> dict:
    cfg = dict(raw_cfg)
    env_cfg = dict(cfg.get("env") or {})
    runner_cfg = dict(cfg.get("runner") or {})
    runner_config = dict(runner_cfg.get("config") or {})

    env_cfg["server_mode"] = False
    if args.max_steps is not None:
        env_cfg["max_episode_steps"] = int(args.max_steps)

    runner_cfg["type"] = "hitl_deploy"
    if args.control_hz > 0:
        runner_cfg["control_hz"] = int(args.control_hz)
    if args.save_dir:
        runner_cfg["save_dir"] = args.save_dir

    runner_config["hitl_enabled"] = True
    runner_config["risk_check_hz"] = float(args.risk_check_hz)
    runner_config["risk_use_safe_action"] = not bool(args.risk_no_safe_action_gap)
    runner_cfg["config"] = runner_config

    cfg["env"] = env_cfg
    cfg["runner"] = runner_cfg
    cfg.setdefault("train", {})
    cfg["train"] = dict(cfg["train"] or {})
    cfg["train"]["device"] = device
    return cfg


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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override cfg.env.max_episode_steps; omit to use config.yaml.",
    )
    parser.add_argument("--control-hz", type=int, default=0)
    parser.add_argument("--risk-check-hz", type=float, default=0.0)
    parser.add_argument(
        "--risk-no-safe-action-gap",
        action="store_true",
        help="Pause env.step() after risk trigger instead of filling with safe_action.",
    )
    parser.add_argument("--episodes", type=int, default=0, help="Number of rollouts to collect; 0 means until Ctrl-C.")
    parser.add_argument("--sleep-between", type=float, default=1.0)
    parser.add_argument("--camera-warmup", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"[Error] Config file not found at {args.config}")
        return

    print(f"[Test-HITL-Deploy] Loading config from {args.config}...")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    raw_cfg = _load_raw_config(args.config)
    deploy_cfg = _prepare_deploy_config(raw_cfg, args, device)
    cfg, runtime_spec = general_resolve(file_config=deploy_cfg)
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

    print(f"[Test-HITL-Deploy] Initializing Agent: {runtime_spec['agent_type']} ...")
    agent = make_agent(runtime_spec["agent_type"], cfg)
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
        episode_idx = 0
        while args.episodes <= 0 or episode_idx < args.episodes:
            episode_idx += 1
            total_msg = "∞" if args.episodes <= 0 else str(args.episodes)
            print(
                f"[Test-HITL-Deploy] Rollout {episode_idx}/{total_msg}. "
                f"Press '{DEPLOY_INIT_KEY}' to init, "
                f"'{DEPLOY_TELEOP_KEY}' to teleop, "
                f"'{DEPLOY_START_KEY}' to start, "
                f"'{DEPLOY_STOP_KEY}' to stop, "
                f"'{DEPLOY_QUIT_KEY}' to quit."
            )
            runner.run()
            if getattr(runner, "quit_requested", False):
                print("[Test-HITL-Deploy] Quit requested. Exiting collection loop...")
                break
            if args.sleep_between > 0:
                time.sleep(args.sleep_between)
    except KeyboardInterrupt:
        print("[Test-HITL-Deploy] KeyboardInterrupt! Stopping runner...")
    finally:
        runner.stop_worker()
        env.close()
        print("[Test-HITL-Deploy] Done.")


if __name__ == "__main__":
    main()
