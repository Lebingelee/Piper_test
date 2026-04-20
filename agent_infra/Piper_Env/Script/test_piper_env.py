import time
import numpy as np
import os
import sys

# 将项目根目录添加到 python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from agent_infra.Piper_Env.Env.single_piper_env import SinglePiperEnv

def test_single_arm_teleop():
    print("\n>>>>>> [PiperTest] 开始单臂环境验证测试 <<<<<<")
    
    # 1. 初始化核心环境
    try:
        env = SinglePiperEnv(control_mode="joint")
        print("[Success] SinglePiperEnv 启动成功。")
        
        # 2. 验证 MetaKeys
        meta = env.unwrapped.meta_keys
        print(f"\n[MetaCheck] 状态 Keys: {list(meta['obs']['state'].keys())}")
        print(f"[MetaCheck] 视觉 Roles: {list(meta['obs'].get('rgb', {}).keys())}")
        print(f"[MetaCheck] 动作 Key: {list(meta['action'].keys())}")
        
        # 3. 测试 Reset
        print("\n[ActionCheck] 正在测试 Reset...")
        obs, info = env.reset()
        
        # 4. 测试 10 步动作循环
        print("[ActionCheck] 执行 10 步安全动作...")
        for i in range(10):
            # 获取安全动作 (保持当前位姿)
            action = env.unwrapped.get_safe_action()
            
            # 开启相机线程 (如果有的话)
            if i == 0: env.start_cameras()
            
            # 步进
            obs, reward, term, trunc, info = env.step(action)
            
            # 打印关键遥操状态
            itv = info.get("intervened", False)
            st = obs["state"]["joint_pos"]
            cam_info = list(obs.get("rgb", {}).keys())
            
            print(f"Step {i} | 接管: {itv} | 关节均值: {np.mean(st):.4f} | 视觉路数: {len(cam_info)}")
            time.sleep(0.1)
            
        print("\n>>>>>> [PiperTest] 测试通过！ <<<<<<\n")

    except Exception as e:
        print(f"\n[Failure] 测试出错: {e}")
        import traceback; traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    test_single_arm_teleop()
