import os
import sys
import time
import torch

# 将根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import make_agent
from agent_factory.config.manager import ConfigManager
from agent_factory.env.env_factories import create_env
from agent_factory.runner import BaseRunner, HITLRunner

def main():
    # 1. 加载配置
    #config_path = "server_result/dual_towel/itqc/config.yaml"
    config_path = "run_results/config.yaml"
    if not os.path.exists(config_path):
        print(f"[Error] Config file not found at {config_path}")
        return

    print(f"[Test] Loading config from {config_path}...")
    cfg = ConfigManager.load_config(config_path)
    
    # 确保配置正确 (使用较短的 act_horizon 和 max_steps 用于测试)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device
    cfg.env.server_mode = False
    #cfg.env_kwargs.realman.hz = 30 # 降低测试频率
    #cfg.env.max_episode_steps = 250 # 用于测试，短一点的 trajectory
    cfg.env.act_horizon = 16
    cfg.runner.control_hz = 20
    cfg.runner.save_dir = "data/test_base_runner"
    os.makedirs(cfg.runner.save_dir, exist_ok=True)

    # 2. 初始化环境
    print("[Test] Initializing RealMan Environment...")
    try:
        env = create_env(cfg.env, cfg.env_kwargs)
    except Exception as e:
        print(f"[Error] Failed to create environment: {e}")
        return

    # 服务器导出的 YAML 里可能把 camera_sns 写成字符串 "None"。
    # create_env 可通过 config_path 正常补全环境参数，但 make_agent 的结构化配置合并要求这里是 List[str]。
    camera_sns = getattr(cfg.env_kwargs.realman, "camera_sns", None)
    if camera_sns in (None, "", "None"):
        resolved_camera_sns = getattr(env.unwrapped, "camera_sns", None)
        cfg.env_kwargs.realman.camera_sns = list(resolved_camera_sns) if resolved_camera_sns is not None else []

    # 3. 初始化 Agent
    print("[Test] Initializing Agent...")
    # 注意这里因为是测试 runner 我们可以暂时不用加载完美的预训练权重，直接用初始化或者部分权重的模型
    # 或者如果你需要权重，可以在这加上 agent.load(...)
    agent = make_agent(cfg.agent_type, cfg)
    agent.load("run_results/step_80000.pth")
    #agent.load("server_result/dual_towel/itqc/step_40000.pth")
    agent.to(device)
    agent.eval()

    # 4. 初始化 Runner（按配置选择 Base/HITL）
    hitl_enabled = bool(getattr(cfg.runner, "hitl_enabled", False))
    runner_cls = HITLRunner if hitl_enabled else BaseRunner
    print(f"[Test] Initializing Runner: {runner_cls.__name__} ...")
    env.unwrapped.switch_passive("true")
    runner = runner_cls(cfg=cfg, agent=agent, env=env)

    # 5. 执行一次 Rollout
    print("[Test] Starting Rollout...")
    try:
        for _ in range(2):
            runner.run()
            time.sleep(7.0)
    except KeyboardInterrupt:
        print("[Test] KeyboardInterrupt! Stopping worker...")
    finally:
        runner.stop_worker()
        env.close()
        print("[Test] Done.")

if __name__ == "__main__":
    main()
