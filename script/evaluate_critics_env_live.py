import os
import sys
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from collections import deque
import time

# 将根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import get_default_config, make_agent
from agent_factory.config.manager import ConfigManager
from agent_factory.env_utils.env_factories import create_env
from agent_infra.Realman_Env.Camera.realsense_camera import get_connected_realsense_serials

def main():
    parser = argparse.ArgumentParser(description="Real-time Environment Visualization of Dual-Critic outputs.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to the critic-only checkpoint.")
    parser.add_argument("--config", type=str, default=None, help="Path to the config.yaml used during training.")
    parser.add_argument("-w", "--window_size", type=int, default=150, help="Sliding window size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. 加载配置
    if args.config:
        print(f"[Init] Loading config from {args.config}")
        cfg = ConfigManager.load_config(args.config)
    else:
        print("[Init] No config provided, using default Diffusion_ITQC config.")
        cfg = get_default_config("Diffusion_ITQC")
    
    cfg.device = args.device
    
    # 2. 初始化真实 Realman 环境
    print("[Init] Initializing REAL Realman environment...")
    cfg.env.library = "realman"
    if not cfg.env_kwargs.realman.camera_sns:
        cfg.env_kwargs.realman.camera_sns = get_connected_realsense_serials()
    
    env = create_env(cfg.env, cfg.env_kwargs)
    
    # 3. 初始化 Agent 并加载权重
    print(f"[Init] Building agent and loading weights from {args.checkpoint}")
    agent = make_agent("Diffusion_ITQC", cfg)
    agent.load(args.checkpoint)
    agent.to(args.device)

    agent.eval()

    # 4. 初始化可视化窗口
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    history_v_std = deque(maxlen=args.window_size)
    history_v_suc = deque(maxlen=args.window_size)
    history_u_s = deque(maxlen=args.window_size)
    x_indices = deque(maxlen=args.window_size)
    
    current_status_color = '#ffffff' 

    def on_key(event):
        nonlocal current_status_color
        if event.key == 's':
            current_status_color = '#e6fffa' # 成功状态背景
            print(">>> Marked as SUCCESS")
        elif event.key == 'f':
            current_status_color = '#fff5f5' # 失败状态背景
            print(">>> Marked as FAILURE")
        elif event.key == 'r':
            current_status_color = '#ffffff' # 重置
            print(">>> Reset Mark")

    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\n[Running] Environment loop started.")
    print("Controls in Plot Window:")
    print(" [S] : Mark current state as SUCCESS (Green BG)")
    print(" [F] : Mark current state as FAILURE (Red BG)")
    print(" [R] : Reset Mark (White BG)")
    print(" [Q] : Quit")

    win_name = "Real-time Multi-View (Press 'q' to stop)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        obs, _ = env.reset()
        step_idx = 0

        while True:
            # 无论环境模式如何，下发 0 动作或当前位置保持静止
            action = np.zeros(cfg.env.action_dim)
            obs, reward, terminated, truncated, info = env.step(action)

            # --- 实时显示图像 ---
            # obs['rgb'] shape: [obs_horizon, N*3, H, W]
            if 'rgb' in obs:
                # 获取最后一帧图像 (N*3, H, W)
                last_rgb = obs['rgb'][-1]
                num_cams = last_rgb.shape[0] // 3
                
                # [N*3, H, W] -> [H, W, N*3]
                img_np = last_rgb.cpu().numpy().transpose(1, 2, 0)
                
                view_list = []
                for c_idx in range(num_cams):
                    view = img_np[:, :, c_idx*3 : (c_idx+1)*3]
                    # 假设是 [0, 1] 范围
                    if view.max() <= 1.01:
                        view = (view * 255).astype(np.uint8)
                    view_list.append(view)
                
                if view_list:
                    combined_img = np.concatenate(view_list, axis=1)
                    cv2.imshow(win_name, combined_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 处理推理观测: [obs_horizon, ...] -> [1, obs_horizon, ...]
            proc_obs = {}
            for k, v in obs.items():
                if isinstance(v, torch.Tensor):
                    proc_obs[k] = v.unsqueeze(0).to(args.device)
                else:
                    proc_obs[k] = torch.from_numpy(v).unsqueeze(0).to(args.device)

            with torch.no_grad():
                model_ready_obs = proc_obs
                v_std_mean = agent.v_net(model_ready_obs).mean().item()
                v_suc_mean = agent.suc_v_net(model_ready_obs).mean().item()
                u_s = v_suc_mean - v_std_mean

            history_v_std.append(v_std_mean)
            history_v_suc.append(v_suc_mean)
            history_u_s.append(u_s)
            x_indices.append(step_idx)
            step_idx += 1

            ax1.clear()
            ax2.clear()

            ax1.plot(list(x_indices), list(history_v_std), label='V_standard', color='blue')
            ax1.plot(list(x_indices), list(history_v_suc), label='V_success', color='orange')
            ax1.set_ylabel("Value")
            ax1.set_title(f"Live Critics Output (Step: {step_idx})")
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            ax2.plot(list(x_indices), list(history_u_s), label='U(s) = V_suc - V_std', color='green', linewidth=2)
            ax2.axhline(y=0, color='black', alpha=0.5)
            ax2.set_ylabel("Difference")
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)

            fig.patch.set_facecolor(current_status_color)
            plt.pause(0.001)

            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        print("\n[Stop] Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("[Done] Environment closed.")

if __name__ == "__main__":
    main()
