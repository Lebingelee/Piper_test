import os
import sys
import torch
import time
import numpy as np
import imageio
from tqdm import tqdm

# 将根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import make_agent
from agent_factory.config.manager import ConfigManager
from agent_factory.env_utils.env_factories import create_env

def eval_rollout(agent, env, cfg, video_path=None, max_steps=250):
    """
    单次 Rollout 评估函数
    """
    print(f"[Eval] Starting Rollout (max_steps={max_steps})...")
    obs, info = env.reset()
    env.unwrapped.switch_passive("true")

    frames = []
    done = False
    step_count = 0
    success = False
    
    # 获取 Horizon 参数
    act_horizon = 40
    obs_horizon = cfg.env.obs_horizon
    
    pbar = tqdm(total=max_steps, desc="Evaluation Steps")
    
    while not done and step_count < max_steps:
        # 1. 准备观测输入 [B, T, C, H, W]
        # UnifiedFrameStackWrapper 已经提供了 (T, C, H, W)
        obs_tensor = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            # 添加 Batch 维度: (1, T, ...)
            obs_tensor[k] = v.unsqueeze(0).to(agent.device)
            
        # 2. 推理动作序列 [1, pred_horizon, action_dim]
        with torch.no_grad():
            t_start = time.time()
            action_seq = agent.sample_action(obs_tensor)
            t_end = time.time()
            print(f"[BaseRunner] Inference time: {(t_end - t_start)*1000:.2f}ms")
        
        # 3. 执行动作序列中的 act_horizon 步
        # 注意：如果 pred_horizon < act_horizon，则取较小值
        exec_len = min(act_horizon, action_seq.shape[1])
        
        for i in range(exec_len):
            action = action_seq[0, i].cpu().numpy()
            action_copy = action[0:6]
            #print(f"max_joint {np.max(action_copy)}")
            #print(f"min_joint {np.min(action_copy)}")
            # Step 环境
            obs, reward, terminated, truncated, info = env.step(action)
            
            #print(action[0:3])
            # 记录视频帧 (假设 RealmanAdapterWrapper 返回的 rgb 是 (C, H, W))
            # 注意：如果是多相机，rgb 可能是 (N*3, H, W)，这里只取前 3 通道显示
            if video_path:
                # 可视化/录制视频 (取时间序列中的最后一帧，并取前3通道即主相机)
                # obs['rgb'] shape: (T, C, H, W) -> 取最后一次采样 [-1]
                img_tensor = obs['rgb'][-1] if obs['rgb'].dim() == 4 else obs['rgb']
                img = img_tensor[3:6].permute(1, 2, 0).cpu().numpy() # (H, W, 3)
                img = (img * 255).astype(np.uint8)
                frames.append(img)
            step_count += 1
            pbar.update(1)
            
            if terminated or truncated:
                done = True
                success = terminated
                #break
        
        if done:
            #break
            pass

    pbar.close()
    print(f"[Eval] Finished. Steps: {step_count}, Success: {success}")
    
    # 保存视频
    if video_path and len(frames) > 0:
        imageio.mimsave(video_path, frames, fps=10)
        print(f"[Eval] Video saved to {video_path}")
        
    return success

def main():
    # 1. 加载训练时的配置
    config_path = "run_results/0325/config.yaml"
    if not os.path.exists(config_path):
        print(f"[Error] Config file not found at {config_path}")
        return

    print(f"[Eval] Loading config from {config_path}...")
    cfg = ConfigManager.load_config(config_path)
    
    # 强制切换到 CPU 或指定的 GPU (评估通常 GPU 更好)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device
    cfg.env_kwargs.realman.hz = 20
    cfg.env.server_mode = False

    # 2. 初始化物理环境 (Adapter模式)
    print("[Eval] 初始化物理环境 (Adapter模式)...")
    # 如果是离线测试，可以考虑 Mock 环境，但这里我们直接使用 create_env
    # 注意：在真实物理环境下，请确保机械臂已连接且安全。
    try:
        env = create_env(cfg.env, cfg.env_kwargs)
    except Exception as e:
        print(f"[Error] Failed to create environment: {e}")
        print("Tip: If running on a machine without the arm, you might need a Mock environment.")
        return

    # 3. 初始化 Agent
    print("[Eval] 初始化 Diffusion Agent...")
    agent = make_agent("Diffusion_ITQC", cfg)
    
    # 4. 加载权重
    ckpt_path = "run_results/0325/Diffusion_ITQC/Diffusion_ITQC_pretrain_final.pth"
    if os.path.exists(ckpt_path):
        print(f"[Eval] Loading checkpoint from {ckpt_path}...")
        agent.load(ckpt_path)
    else:
        print(f"[Error] Checkpoint not found at {ckpt_path}")
        return
    
    agent.eval()

    # 5. 开始评估
    os.makedirs("eval_videos", exist_ok=True)
    video_filename = f"eval_videos/realman_diffusion_vanilla_{cfg.env.env_id}.mp4"
    
    eval_rollout(agent, env, cfg, video_path=video_filename)

    env.close()

if __name__ == "__main__":
    main()
