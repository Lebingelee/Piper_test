import os
import sys
import time
import argparse
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from agent_infra.Piper_Env.Env.utils.piper_base_env import PiperEnv


def _format_array(arr: np.ndarray) -> str:
    return np.array2string(np.asarray(arr), precision=4, suppress_small=True)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Piper teleop obs/action smoke test for single or dual arms."
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Use dual-arm config and left/right meta keys.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config path. Defaults to piper_config.yaml for single and dual_piper_config.yaml for dual.",
    )
    parser.add_argument(
        "--control",
        choices=["joint", "pose", "delta_pose", "relative_pose_chunk"],
        default="joint",
        help="Control mode used by env.step().",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4000,
        help="Number of safe-action steps to run.",
    )
    parser.add_argument(
        "--master",
        nargs="+",
        default=None,
        help="Override master CAN interfaces. Single: one value. Dual: left right.",
    )
    parser.add_argument(
        "--slave",
        nargs="+",
        default=None,
        help="Override slave/follower CAN interfaces. Single: one value. Dual: left right.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print obs/action summary every N steps. Use 0 to print only init/final.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not call env.reset() before stepping. Safer for quick teleop checks.",
    )
    parser.add_argument(
        "--no-master-reset",
        action="store_true",
        help="When reset is enabled, do not move master arms to the reset target.",
    )
    return parser.parse_args()


def _build_robot_overrides(is_dual: bool, masters, slaves):
    if masters is None and slaves is None:
        return None

    arm_names = ["left", "right"] if is_dual else ["single"]
    expected = 2 if is_dual else 1
    masters = masters or []
    slaves = slaves or []

    if masters and len(masters) != expected:
        raise ValueError(f"--master expects {expected} value(s), got {len(masters)}: {masters}")
    if slaves and len(slaves) != expected:
        raise ValueError(f"--slave expects {expected} value(s), got {len(slaves)}: {slaves}")

    overrides = {}
    for idx, arm_name in enumerate(arm_names):
        arm_override = {}
        if masters:
            arm_override["master_can"] = masters[idx]
        if slaves:
            arm_override["follower_can"] = slaves[idx]
        overrides[arm_name] = arm_override
    return overrides


def _default_config_path(is_dual: bool):
    if is_dual:
        return "agent_infra/Piper_Env/Config/dual_piper_config.yaml"
    return "agent_infra/Piper_Env/Config/piper_config.yaml"


def _summarize_step(step_idx, obs, actual_action, intervened):
    print(f"========== Step {step_idx} ==========")
    print(f"under_control: {obs.get('under_control')}")
    print(f"intervened: {intervened}")
    print("obs.state:")
    for key, value in obs["state"].items():
        print(f"  {key}: {_format_array(value)}")
    print("actual_action:")
    for key, value in actual_action.items():
        print(f"  {key}: {_format_array(value)}")
    print()


def test_teleop_obs_action(
    control_mode: str = "joint",
    num_steps: int = 400000,
    is_dual: bool = False,
    config_path: str = None,
    masters=None,
    slaves=None,
    print_every: int = 30,
    reset_before_loop: bool = True,
    sync_master_reset: bool = True,
):
    mode_name = "双臂" if is_dual else "单臂"
    print(f"\n>>>>>> [PiperTeleopTest] 开始{mode_name}遥操 obs/action 联调测试 <<<<<<")
    print("按 T 切换遥操接管，按 Ctrl+C 结束测试。\n")

    env = None
    try:
        config_path = config_path or _default_config_path(is_dual)
        robot_overrides = _build_robot_overrides(is_dual, masters, slaves)
        env = PiperEnv(
            config_path=config_path,
            control_mode=control_mode,
            robot_overrides=robot_overrides,
        )

        if reset_before_loop:
            obs, info = env.reset(options={"sync_master": sync_master_reset})
        else:
            obs = env._get_obs()
            info = {"status": "skip_reset"}

        print(f"[Init] obs keys: {list(obs.keys())}")
        print(f"[Init] state keys: {list(obs['state'].keys())}")
        print(f"[Init] under_control: {obs['under_control']}")
        print(f"[Init] action keys: {list(env.meta_keys['action'].keys())}\n")
        print(f"[Init] config: {config_path}")
        print(f"[Init] arms: {env.arm_names}")
        print(f"[Init] robot configs: {env.robot_configs}\n")
        print(f"[Init] reset sync master: {sync_master_reset and reset_before_loop}\n")

        for step_idx in range(num_steps):
            safe_action = env.get_safe_action()
            obs, reward, terminated, truncated, info = env.step(safe_action)

            actual_action = info.get("actual_action", {})
            intervened = info.get("intervened", False)

            if print_every > 0 and step_idx % print_every == 0:
                _summarize_step(step_idx, obs, actual_action, intervened)

    except KeyboardInterrupt:
        print("\n[Exit] 手动结束测试。")
    except Exception as exc:
        print(f"\n[Failure] 测试出错: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    args = _parse_args()
    test_teleop_obs_action(
        control_mode=args.control,
        num_steps=args.steps,
        is_dual=args.dual,
        config_path=args.config,
        masters=args.master,
        slaves=args.slave,
        print_every=args.print_every,
        reset_before_loop=not args.no_reset,
        sync_master_reset=not args.no_master_reset,
    )
