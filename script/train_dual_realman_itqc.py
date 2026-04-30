import os
import sys
import torch
import argparse

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import get_default_config, make_agent
from agent_factory.env.env_factories import create_env
from agent_factory.data.dataset import ExpertDataset
from agent_factory.config.manager import ConfigManager

def get_dual_arm_config(cfg):
    """
    针对双臂 Towel Folding 任务微调配置
    """
    # --- 1. 环境基础配置 ---
    cfg.env.library = "realman"
    cfg.env.env_id = "Dual-Realman-Towel"
    cfg.env.control_mode = "joint_pos"
    cfg.env.action_dim = 14  # 双臂 (7+7)
    cfg.env.proprio_dim = 38 # 双臂 (19+19)
    cfg.env.num_cameras = 3  # 侧边 + 左手 + 右手
    cfg.env.max_episode_steps = 500
    cfg.env.server_mode = True # 离线训练模式

    # --- 2. 物理环境特定配置 ---
    # 注意：在离线训练中，这些 IP 不会真实连接，但 arm_names 用于对齐 meta_keys
    cfg.env_kwargs.realman.is_dual = True
    cfg.env_kwargs.realman.arm_names = ["left", "right"]
    cfg.env_kwargs.realman.config_path = "dual_env_config.yaml"
    cfg.env_kwargs.realman.camera_sns = ["base", "left_hand", "right_hand"]

    # 模型网络维度对齐
    cfg.actor.action_dim = cfg.env.action_dim
    cfg.actor.encoder.proprio_dim = cfg.env.proprio_dim
    cfg.critic.encoder.proprio_dim = cfg.env.proprio_dim
    
    # 🟢 修正：StateEncoder 会循环处理每个相机，所以每个视觉编码器只需 3 通道
    cfg.actor.encoder.visual.in_channels = 3
    cfg.critic.encoder.visual.in_channels = 3
    
    # 视觉通道：3个相机
    cfg.env.num_cameras = 3
    
    # 使用 ResNet 作为双臂任务的视觉 Backbone
    cfg.actor.encoder.visual.backbone_type = "resnet"
    cfg.critic.encoder.visual.backbone_type = "resnet"

    # --- 4. 数据集配置 ---
    cfg.dataset.expert.demo_path = "demos_RL/Towel_folding_skill1/Towel_folding_skill1_flattened.h5"
    cfg.dataset.expert.num_traj = None # 使用全部轨迹

    # --- 5. 训练超参数 ---
    cfg.train.batch_size = 36
    cfg.train.num_workers = 6
    
    # Agent 特定参数 (ITQC)
    cfg.agent_sp.offline_iters_critic = 200
    cfg.agent_sp.offline_iters_actor = 100000
    cfg.agent_sp.save_dir = "run_results/dual_arm_towel"
    cfg.agent_sp.exp_name = "diffusion_itqc_baseline"

    return cfg

def main():
    parser = argparse.ArgumentParser(description="Train Dual-Arm Diffusion-ITQC on Towel Folding.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file to load.")
    args = parser.parse_args()

    # 1. 获取默认配置
    print("[Init] Loading default Diffusion_ITQC configuration...")
    cfg = get_default_config("Diffusion_ITQC")

    # 2. 应用双臂任务特定修改
    if args.config:
        print(f"[Config] Loading user-provided config: {args.config}")
        user_cfg = ConfigManager.load_config(args.config)
        cfg = ConfigManager.merge_configs(cfg, user_cfg)
    else:
        cfg = get_dual_arm_config(cfg)

    # 3. 创建虚拟环境以验证 Spaces 并获取 metadata
    print("[Init] Creating Offline Environment...")
    env = create_env(cfg.env, cfg.env_kwargs)
    obs_space = env.observation_space
    env.close()

    # 4. 加载数据集
    print(f"[Train] Loading expert dataset: {cfg.dataset.expert.demo_path}")
    expertdataset = ExpertDataset(
        cfg=cfg,
        obs_space=obs_space,
        device="cpu"
    )
    
    # 5. 配置一致性检查与保存
    final_save_dir = os.path.join(cfg.agent_sp.save_dir, cfg.agent_sp.exp_name)
    os.makedirs(final_save_dir, exist_ok=True)
    print(f"[Config] Saving final configuration to: {final_save_dir}")
    ConfigManager.save_config(cfg, save_dir=final_save_dir)

    # 6. 初始化 Agent 并开始训练
    print("[Train] Initializing Diffusion_ITQC Agent...")
    agent = make_agent("Diffusion_ITQC", cfg)
    
    dataset = {'offline': expertdataset}
    
    print(f"[Train] Starting Offline Training (Critic: {cfg.agent_sp.offline_iters_critic}, Actor: {cfg.agent_sp.offline_iters_actor})...")
    agent.start_train(dataset, additional_args={"phase": "offline"})

    print("[Train] Training finished.")

if __name__ == "__main__":
    main()
