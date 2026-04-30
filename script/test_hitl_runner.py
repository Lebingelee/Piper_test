import os
import sys
import time
import torch

# 将根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import make_agent
from agent_factory.config.manager import ConfigManager
from agent_factory.env.env_factories import create_env
from agent_factory.runner import HITLRunner


def main():
    # 1. 加载配置
    config_path = "server_result/dual_towel/diffusion_vanilla/base_config.yaml"
    if not os.path.exists(config_path):
        print(f"[Error] Config file not found at {config_path}")
        return

    print(f"[Test-HITL] Loading config from {config_path}...")
    cfg = ConfigManager.load_config(config_path)

    # 2. 覆盖测试配置（HITL）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device
    cfg.env.server_mode = False
    cfg.env_kwargs.realman.hz = 20
    cfg.env.act_horizon = 32
    cfg.runner.control_hz = 20
    cfg.runner.save_dir = "data/test_hitl_runner"
    cfg.runner.hitl_enabled = True
    cfg.runner.hitl_override_key = "t"
    cfg.runner.hitl_source = "keyboard"
    cfg.runner.hitl_finalize_intervention = True
    os.makedirs(cfg.runner.save_dir, exist_ok=True)

    # 3. 初始化环境
    print("[Test-HITL] Initializing RealMan Environment...")
    try:
        env = create_env(cfg.env, cfg.env_kwargs)
    except Exception as e:
        print(f"[Error] Failed to create environment: {e}")
        return

    # 兼容配置文件中 camera_sns 可能是字符串 "None" 的情况
    camera_sns = getattr(cfg.env_kwargs.realman, "camera_sns", None)
    if camera_sns in (None, "", "None"):
        resolved_camera_sns = getattr(env.unwrapped, "camera_sns", None)
        cfg.env_kwargs.realman.camera_sns = list(resolved_camera_sns) if resolved_camera_sns is not None else []

    # 4. 初始化 Agent
    print("[Test-HITL] Initializing Agent...")
    agent = make_agent(cfg.agent_type, cfg)
    agent.load("server_result/dual_towel/diffusion_vanilla/step_18750.pth")
    agent.to(device)
    agent.eval()

    # 5. 初始化 HITLRunner
    print("[Test-HITL] Initializing HITLRunner...")
    print("[Test-HITL] Press 't' to toggle human override during rollout.")
    env.unwrapped.switch_passive("true")
    runner = HITLRunner(cfg=cfg, agent=agent, env=env)

    # 6. 执行 Rollout
    print("[Test-HITL] Starting Rollout...")
    try:
        for _ in range(2):
            runner.run()
            time.sleep(7.0)
    except KeyboardInterrupt:
        print("[Test-HITL] KeyboardInterrupt! Stopping runner...")
    finally:
        runner.stop_worker()
        env.close()
        print("[Test-HITL] Done.")


if __name__ == "__main__":
    main()
