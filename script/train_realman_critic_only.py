import os
import sys
import torch
import argparse

# 将根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import get_default_config, make_agent
from agent_factory.data.dataset import ExpertDataset
from agent_factory.config.manager import ConfigManager
from agent_factory.env_utils.env_factories import create_env
from agent_infra.Realman_Env.Camera.realsense_camera import get_connected_realsense_serials

def get_debug_config(cfg):
     # 2. 修改为 Realman 物理环境配置
    # --- 环境通用配置 ---
    cfg.env.library = "realman"
    cfg.env.env_id = "Realman-Stack"
    cfg.env.control_mode = "joint_pos"
    cfg.env.action_dim = 7
    cfg.env.max_episode_steps = 250
    cfg.env.gamma = 1.0
    cfg.env.num_cameras = 2
    cfg.env.penalty = -1.0*cfg.env.max_episode_steps 
    cfg.env.reward_shape = True
    cfg.env.server_mode =  True

    # --- 环境特定配置 ---
    cfg.env_kwargs.realman.robot_ip = "192.168.2.18" # 请修改为您的IP
    cfg.env_kwargs.realman.camera_sns = get_connected_realsense_serials()

    # --- 模型配置 (维度对齐) ---
    proprio_dim = 20 # RealmanAdapterWrapper 输出的维度
    cfg.env.proprio_dim = proprio_dim
    cfg.actor.encoder.proprio_dim = proprio_dim
    cfg.critic.encoder.proprio_dim = proprio_dim
    cfg.actor.action_dim = cfg.env.action_dim

    # --- 图像配置 ---
    cfg.env.num_cameras = 2 #len(cfg.env_kwargs.realman.camera_sns)
    # 假设 RealmanAdapterWrapper 输出3通道图像
    cfg.actor.encoder.visual.in_channels = 3 
    cfg.critic.encoder.visual.in_channels = 3
    cfg.actor.encoder.visual.backbone_type = "resnet"
    cfg.critic.encoder.visual.backbone_type = "resnet"
    cfg.critic.num_critics = 2

    # --- 数据集配置 ---
    cfg.dataset.include_depth = False
    # !! 请务必修改为您的 combined_demos.h5 路径 !!
    cfg.dataset.expert.demo_path = "demos_RL/Stack_g2f_f/Stack_g2f_f.h5"
    cfg.dataset.expert.num_traj = None # 使用所有轨迹

    # --- 训练配置 ---
    cfg.train.batch_size = 48
    cfg.train.num_workers = 4
    
    # --- Agent 特定配置 ---
    cfg.agent_sp.offline_iters_critic = 50 # 示例步数
    cfg.agent_sp.save_dir = "critic_only_training/debug"
    cfg.agent_sp.exp_name = "debug_ckpt"

    return cfg
def main():
    parser = argparse.ArgumentParser(description="Train Critic-Only model for Diffusion-ITQC on Realman.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file to load.")
    parser.add_argument("--server-mode", action="store_true", help="Enable server mode to bypass hardware checks.")
    args = parser.parse_args()

    # 1. 获取 Diffusion_ITQC 默认配置
    cfg = get_default_config("Diffusion_ITQC")
    #args.config = "run_results/debug/0324/config.yaml"
    # 如果提供了配置文件，则加载并合并
    if args.config:
        print(f"[Config] Loading user-provided config: {args.config}")
        user_cfg = ConfigManager.load_config(args.config)
        cfg = ConfigManager.merge_configs(cfg, user_cfg)
    else:
        cfg = get_debug_config(cfg)

    # 2. 处理服务器模式
    if args.server_mode:
        print("[Config] Server mode enabled. Bypassing hardware initialization.")
        cfg.env.server_mode = True
        # 在服务器模式下，如果 camera_sns 为空，我们需要确保它有正确的值
        # 这里假设默认使用 2 个相机 (base + hand)
        if not getattr(cfg.env_kwargs.realman, 'camera_sns', None):
            cfg.env_kwargs.realman.camera_sns = ["offline_cam_0", "offline_cam_1"]
        cfg.env.num_cameras = len(cfg.env_kwargs.realman.camera_sns)


    # 3. 创建环境以获取 obs_space (然后关闭)
    env = create_env(cfg.env, cfg.env_kwargs)
    obs_space = env.observation_space
    env.close()

    # 4. 加载专家数据集
    print(f"[Train] Loading expert dataset: {cfg.dataset.expert.demo_path}")
    expertdataset = ExpertDataset(
        cfg=cfg,
        obs_space=obs_space,
        device="cpu" # 数据集预处理在 CPU 上进行
    )
    
    # 5. 保存最终配置并进行一致性检查
    # 如果加载了外部配置，check_consistency会使用合并后的最终配置
    print("[Config] Checking configuration consistency...")
    #ConfigManager.check_consistency(cfg)
    # 保存最终使用的配置快照
    final_save_dir = os.path.join(cfg.agent_sp.save_dir, cfg.agent_sp.exp_name)
    print(f"[Config] Saving final configuration to: {final_save_dir}")
    ConfigManager.save_config(cfg, save_dir=final_save_dir)

    # 6. 初始化 Agent 并开始训练
    print("[Train] Initializing Diffusion_ITQC Agent...")
    agent = make_agent("Diffusion_ITQC", cfg)
    
    dataset = {'offline': expertdataset}
    
    print("[Train] Starting training...")
    # start_train 将会执行 critic 训练并根据我们之前的修改保存 critic_only_checkpoint.pth
    agent.start_train(dataset, additional_args={"phase": "offline"})

    print("[Train] Training finished.")

if __name__ == "__main__":
    main()
