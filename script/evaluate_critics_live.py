import os
import sys
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from collections import deque

# 将根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import get_default_config, make_agent
from agent_factory.data.dataset import ExpertDataset
from agent_factory.config.manager import ConfigManager
from agent_factory.env_utils.env_factories import create_env

def main():
    parser = argparse.ArgumentParser(description="Live visualization of Dual-Critic value outputs on dataset.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to the critic-only checkpoint.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to the combined H5 dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to the config.yaml used during training.")
    parser.add_argument("-w", "--window_size", type=int, default=150, help="Sliding window size for visualization.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps", type=int, default=20, help="Visualization speed.")
    args = parser.parse_args()

    # 1. 加载配置
    if args.config:
        print(f"[Init] Loading config from {args.config}")
        cfg = ConfigManager.load_config(args.config)
    else:
        print("[Init] No config provided, using default Diffusion_ITQC config.")
        cfg = get_default_config("Diffusion_ITQC")
    
    cfg.device = args.device
    cfg.dataset.expert.demo_path = args.dataset

    # 2. 初始化环境 (用于获取 observation_space)
    print("[Init] Initializing metadata from environment...")
    env = create_env(cfg.env, cfg.env_kwargs)
    obs_space = env.observation_space
    env.close()

    # 3. 初始化数据集
    print(f"[Init] Loading dataset: {args.dataset}")
    dataset = ExpertDataset(
        cfg_dataset=cfg.dataset,
        cfg_env=cfg.env,
        obs_space=obs_space,
        device="cpu"
    )
    # batch_size=1 以便逐帧连续观察
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 4. 初始化 Agent 并加载权重
    print(f"[Init] Building agent and loading weights from {args.checkpoint}")
    agent = make_agent("Diffusion_ITQC", cfg)
    agent.load(args.checkpoint)
    agent.to(args.device)
    agent.eval()

    # 5. 初始化可视化窗口
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    history_v_std = deque(maxlen=args.window_size)
    history_v_suc = deque(maxlen=args.window_size)
    history_v_std_9 = deque(maxlen=args.window_size)
    history_v_std_5 = deque(maxlen=args.window_size)
    history_v_std_1 = deque(maxlen=args.window_size)
    x_indices = deque(maxlen=args.window_size)

    print(f"\n[Running] Dataset evaluation started. Total samples: {len(dataset)}")
    
    win_name = "Observation Views (Press 'q' to exit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        for i, batch in enumerate(loader):
            # 5.1 数据搬运
            obs = batch['observations'] # Dictionary of Tensors
            
            # --- 实时显示图像 (修正版) ---
            # ExpertDataset 输出的 obs['rgb'] 形状为 [1, obs_horizon, 3*N, H, W]
            # 我们取 batch[0], horizon 的最后一帧 [-1]
            if 'rgb' in obs:
                # [3*N, H, W]
                last_rgb_tensor = obs['rgb'][0, -1]
                num_cams = last_rgb_tensor.shape[0] // 3
                
                # 转为 numpy 并调整维度为 [H, W, 3*N]
                img_np = last_rgb_tensor.cpu().numpy().transpose(1, 2, 0)
                
                # 拆分并水平拼接多视角
                view_list = []
                for c_idx in range(num_cams):
                    view = img_np[:, :, c_idx*3 : (c_idx+1)*3]
                    # 如果数据是 0-1 缩放过的，转回 0-255
                    if view.max() <= 1.01:
                        view = (view * 255).astype(np.uint8)
                    view_list.append(view)
                
                if view_list:
                    combined_img = np.concatenate(view_list, axis=1)
                    # RGB -> BGR for OpenCV
                    #combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
                    cv2.imshow(win_name, combined_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 将嵌套字典中的 tensor 搬运到 device (用于推理)
            proc_obs = {}
            for k in obs:
                if isinstance(obs[k], torch.Tensor):
                    proc_obs[k] = obs[k].to(args.device)
            
            # 获取当前帧的终止状态 (用于标记背景)
            is_terminated = batch['terminated'].item() > 0.5

            # 5.2 Critic 前向传播
            with torch.no_grad():
                # 必须经过 Agent 的预处理 (归一化等)
                model_ready_obs = agent._preprocess_obs(proc_obs)
                
                # ITQC V-Net 吐出 [B, M, N]
                v_std_dist = agent.v_net(model_ready_obs)
                
                # 提取特定分位数 (索引 0=0.1, 2=0.5, 4=0.9)
                v_std_quants = v_std_dist.mean(dim=1).squeeze(0) 
                v_std_1 = v_std_quants[0].item()
                v_std_5 = v_std_quants[2].item()
                v_std_9 = v_std_quants[4].item()
                v_std_mean = v_std_quants.mean().item()

                v_suc_dist = agent.suc_v_net(model_ready_obs)
                v_suc_mean = v_suc_dist.mean().item()

            # 5.3 更新缓存
            history_v_std.append(v_std_mean)
            history_v_suc.append(v_suc_mean)
            history_v_std_9.append(v_std_9)
            history_v_std_5.append(v_std_5)
            history_v_std_1.append(v_std_1)
            x_indices.append(i)

            # 5.4 实时绘图
            ax1.clear()
            ax2.clear()

            # 子图 1: V_standard vs V_success
            ax1.plot(list(x_indices), list(history_v_std), label='V_standard', color='#1f77b4', linewidth=1.5)
            ax1.plot(list(x_indices), list(history_v_suc), label='V_success', color='#ff7f0e', linewidth=1.5)
            ax1.set_ylabel("Critic Mean Value")
            ax1.set_title(f"Dual-Critic Comparison (Index: {i})")
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # 子图 2: Standard Critic 分位数分布
            ax2.plot(list(x_indices), list(history_v_std_9), label='V_std_0.9', color="#52ff52", linewidth=1.2, alpha=0.8)
            ax2.plot(list(x_indices), list(history_v_std_5), label='V_std_0.5', color='#2ca02c', linewidth=2.0)
            ax2.plot(list(x_indices), list(history_v_std_1), label='V_std_0.1', color="#103610", linewidth=1.2, alpha=0.8)
            ax2.set_ylabel("Std Critic Quantiles")
            ax2.set_xlabel("Dataset Sample Index")
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)

            if is_terminated:
                fig.patch.set_facecolor('#e6fffa')
            else:
                fig.patch.set_facecolor('#ffffff')

            plt.tight_layout()
            plt.pause(1.0 / args.fps)

            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        print("\n[Stop] Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
