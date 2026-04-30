import os
import sys
import torch
import argparse
from omegaconf import OmegaConf

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import get_default_config, make_agent
from agent_factory.env.env_factories import create_env
from agent_factory.data.dataset import ExpertDataset
from agent_factory.config.manager import ConfigManager

def setup_dual_arm_defaults(cfg):
    """
    通用双臂任务默认配置 (如需覆盖，请通过 yaml 或在 main 中根据 agent_type 进一步定制)
    """
    cfg.env.library = "realman"
    cfg.env.env_id = "Dual-Realman-Towel"
    cfg.env.control_mode = "joint_pos"
    cfg.env.action_dim = 14
    cfg.env.proprio_dim = 38
    cfg.env.num_cameras = 3
    cfg.env.max_episode_steps = 500
    cfg.env.server_mode = True

    cfg.env_kwargs.realman.is_dual = True
    cfg.env_kwargs.realman.arm_names = ["left", "right"]
    cfg.env_kwargs.realman.config_path = "dual_env_config.yaml"
    cfg.env_kwargs.realman.camera_sns = ["base", "left_hand", "right_hand"]

    cfg.dataset.expert.num_traj=10
    # 动态同步维度
    if hasattr(cfg, "actor") and cfg.actor is not None:
        cfg.actor.action_dim = cfg.env.action_dim
        if hasattr(cfg.actor, "encoder"):
            cfg.actor.encoder.proprio_dim = cfg.env.proprio_dim
            cfg.actor.encoder.visual.in_channels = 3
            cfg.actor.encoder.visual.backbone_type = "resnet"
            
    if hasattr(cfg, "critic") and cfg.critic is not None:
        # DSRL Critic 可能有自己的 action_dim (noise 维度) 和 env_action_dim
        # 这里统一同步
        if hasattr(cfg.critic, "encoder"):
            cfg.critic.encoder.proprio_dim = cfg.env.proprio_dim
            cfg.critic.encoder.visual.in_channels = 3
            cfg.critic.encoder.visual.backbone_type = "resnet"

    cfg.dataset.expert.demo_path = "data/merged/dual_merged_flattened.h5"
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Universal Training Script for agent_factory.")
    parser.add_argument("--agent", type=str, default="dsrl", help="Agent type to train (e.g., dsrl, Diffusion_ITQC).")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file to load.")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name.")
    args = parser.parse_args()

    # 1. 获取默认配置
    print(f"[Init] Loading default {args.agent} configuration...")
    cfg = get_default_config(args.agent)

    # 2. 应用通用双臂默认值 (也可以选择不应用，看需求)
    cfg = setup_dual_arm_defaults(cfg)

    # 3. 合并用户提供的 YAML
    #args.config = 'server_result/dual_towel/dsrl_from_vanilla_step25000/config.yaml'
    if args.config:
        print(f"[Config] Loading user-provided config: {args.config}")
        user_cfg = ConfigManager.load_config(args.config)
        cfg = ConfigManager.merge_configs(cfg, user_cfg)

    # 4. 命令行覆盖

    if args.exp_name:
        cfg.agent_sp.exp_name = args.exp_name

    # 5. 创建虚拟环境以验证 Spaces
    print("[Init] Creating Offline Environment to fetch spaces...")
    env = create_env(cfg.env, cfg.env_kwargs)
    obs_space = env.observation_space
    env.close()

    # 6. 初始化 Agent (此时会根据 cfg 实例化所有 Mixins)
    print(f"[Init] Initializing {args.agent} Agent...")
    agent = make_agent(args.agent, cfg)

    # 7. 加载数据集 (ExpertDataset 内部会自动根据 agent.required_keys 准备数据)
    print(f"[Train] Loading expert dataset: {cfg.dataset.expert.demo_path}")
    print(f"[Train] Required keys: {agent.required_keys}")
    expertdataset = ExpertDataset(
        cfg=cfg,
        obs_space=obs_space,
        device="cpu",
        required_keys=agent.required_keys
    )
    
    # 8. 配置保存
    save_dir = os.path.join(cfg.agent_sp.save_dir, cfg.agent_sp.exp_name or args.agent)
    os.makedirs(save_dir, exist_ok=True)
    ConfigManager.save_config(cfg, save_dir=save_dir)

    # 9. 开始训练
    dataset = {'offline': expertdataset}
    print(f"[Train] Starting {args.agent} Training...")
    agent.start_train(dataset, additional_args={"phase": "offline"})

    print("[Train] Universal training script completed.")

if __name__ == "__main__":
    main()
