import os
import sys
import torch
import time
import numpy as np
import imageio
import cv2
from tqdm import tqdm

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import make_agent
from agent_factory.config.manager import ConfigManager
from agent_factory.env.env_factories import create_env

def eval_rollout(agent, env, cfg, video_path=None, max_steps=500):
    """
    单次 Rollout 评估函数 (支持双臂多视角)
    """
    print(f"[Eval] Starting Rollout (max_steps={max_steps})...")
    obs, info = env.reset()
    
    # 如果是真机环境，切换到 passive 模式以允许动作执行
    if hasattr(env.unwrapped, "switch_passive"):
        env.unwrapped.switch_passive("true")

    frames = []
    done = False
    step_count = 0
    success = False
    
    # 获取 Horizon 参数
    act_horizon = cfg.env.act_horizon
    
    pbar = tqdm(total=max_steps, desc="Evaluation Steps")
    
    while not done and step_count < max_steps:
        # 1. 准备观测输入
        obs_tensor = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            # 添加 Batch 维度: (1, T, ...)
            obs_tensor[k] = v.unsqueeze(0).to(agent.device)
            
        # 2. 推理动作序列
        with torch.no_grad():
            action_seq = agent.sample_action(obs_tensor)
        
        # 3. 执行动作序列
        exec_len = min(act_horizon, action_seq.shape[1])
        
        for i in range(exec_len):
            action = action_seq[0, i].cpu().numpy()
            
            # Step 环境
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 4. 视频录制逻辑 (处理多相机)
            if video_path:
                # 假设 obs['rgb'] shape: (T, C_total, H, W)
                # C_total = 3 * num_cameras
                img_tensor = obs['rgb'][-1] # 取时间序列最后帧
                num_cams = img_tensor.shape[0] // 3
                
                # 拼合所有相机的画面进行显示/录制
                cam_views = []
                for c_idx in range(num_cams):
                    view = img_tensor[c_idx*3 : (c_idx+1)*3].permute(1, 2, 0).cpu().numpy()
                    if view.max() <= 1.01:
                        view = (view * 255).astype(np.uint8)
                    cam_views.append(view)
                
                # 水平拼合图像
                combined_frame = np.concatenate(cam_views, axis=1)
                frames.append(combined_frame)
                
                # 实时预览
                cv2.imshow("Dual-Arm Evaluation Preview", cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    done = True
                    break

            step_count += 1
            pbar.update(1)
            
            if terminated or truncated:
                done = True
                success = terminated
                break
        
        if done: break

    pbar.close()
    cv2.destroyAllWindows()
    print(f"[Eval] Finished. Steps: {step_count}, Success: {success}")
    
    # 保存视频
    if video_path and len(frames) > 0:
        imageio.mimsave(video_path, frames, fps=10)
        print(f"[Eval] Video saved to {video_path}")
        
    return success

def main():
    # 1. 指向训练时保存的 config
    # 注意：请修改为您实际训练产出的 config 路径
    config_path = "run_results/dual_arm_towel/diffusion_itqc_baseline/config.yaml"
    ckpt_path = "run_results/dual_arm_towel/diffusion_itqc_baseline/diffusion_itqc_baseline_pretrain_final.pth"

    if not os.path.exists(config_path):
        print(f"[Error] Config file not found at {config_path}. Please train first.")
        return

    print(f"[Eval] Loading config from {config_path}...")
    cfg = ConfigManager.load_config(config_path)
    
    # 评估设置
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.env.server_mode = False # 评估时使用真机或带硬件的测试模式

    # 2. 初始化环境 (模块化版)
    print("[Eval] Initializing Dual-Arm Environment...")
    try:
        env = create_env(cfg.env, cfg.env_kwargs)
    except Exception as e:
        print(f"[Error] Failed to create environment: {e}")
        return

    # 3. 初始化 Agent
    print("[Eval] Building Agent and loading weights...")
    agent = make_agent("Diffusion_ITQC", cfg)
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)
        agent.eval()
    else:
        print(f"[Warning] Checkpoint not found at {ckpt_path}. Running with random weights.")

    # 4. 开始评估
    os.makedirs("eval_videos", exist_ok=True)
    video_filename = f"eval_videos/dual_arm_towel_folding.mp4"
    
    eval_rollout(agent, env, cfg, video_path=video_filename)

    env.close()

if __name__ == "__main__":
    main()
