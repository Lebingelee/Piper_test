import os
import time
import datetime
import numpy as np
import cv2
import h5py
import json
from pynput import keyboard
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

# 导入基类
from agent_infra.base_robot_env import BaseRobotEnv

@dataclass
class RealmanEnvMetadata:
    """本地元数据对象，用于解耦 agent_factory"""
    robot_ips: List[str]
    camera_sns: List[str]
    hz: int
    is_dual_arm: bool = False
    use_depth: bool = False
    control_mode: str = "joint_pos"

class DataRecorder:
    """
    专家示教轨迹录制器 - 模块化适配版
    职责：
    1. 录制原始嵌套字典观测 (支持单/双臂)。
    2. 自动保存环境的 meta_keys。
    3. 实时预览多视角画面。
    """
    def __init__(self, 
                 env: BaseRobotEnv, 
                 task_name: Optional[str] = None,
                 root_dir: str = "agent_infra/Realman_Env/Record/data",
                 sample_hz: int = 10,
                 max_episode_length: int = 200):
        self.env = env
        
        if task_name is None:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            self.task_name = f"realman-{now}"
        else:
            self.task_name = task_name

        self.hz = sample_hz
        self.dt = 1.0 / sample_hz
        self.max_episode_length = max_episode_length
        
        self.temp_dir = os.path.join(root_dir, self.task_name)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.current_trajectory: List[Dict[str, Any]] = []
        self.is_recording = False
        self.is_finished = False
        self.traj_counter = 0
        self.success = True
        
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        
        self._print_welcome()

    def _print_welcome(self):
        print(f"\n--- Realman 录制器 [{self.task_name}] ---")
        env_meta = self.env.get_env_metadata() if hasattr(self.env, "get_env_metadata") else self.env.meta_keys
        print(f" 环境元数据已就绪，Key 结构: {list(env_meta['obs'].keys())}")
        print(f" 数据将暂存在: {self.temp_dir}")
        print(" [I] : 机械臂复位 (Move to Init)")
        print(" [S] : 开始录制 (Start)")
        print(" [E] : 结束并存为成功轨迹 (End)")
        print(" [F] : 结束并存为失败轨迹 (Fail)")
        print(" [D] : 丢弃当前录制 (Discard)")
        print(" [Q] : 退出程序 (Quit)")
        print("----------------------------\n")

    def _on_press(self, key):
        try:
            char = key.char.lower()
            if char == 'i':
                print("[I] 执行复位动作中...")
                self.env.reset()
            elif char == 's':
                if not self.is_recording:
                    self.current_trajectory = []
                    self.is_recording = True
                    self.success = True
                    print(f"[S] 开始录制 Traj {self.traj_counter}...")
            elif char == 'e':
                if self.is_recording:
                    self.is_recording = False
                    self._save_temp_trajectory()
                    self.traj_counter += 1
            elif char == 'f':
                if self.is_recording:
                    self.is_recording = False
                    self.success = False
                    print("[F] 轨迹标记为失败。")
                    self._save_temp_trajectory()
                    self.traj_counter += 1
            elif char == 'd':
                if self.is_recording:
                    self.is_recording = False
                    self.current_trajectory = []
                    print("[D] 轨迹已丢弃。")
            elif char == 'q':
                print("[Q] 退出录制程序。")
                self.is_recording = False
                self.is_finished = True
                return False 
        except AttributeError:
            pass

    def _visualize(self, obs: Dict[str, Any]):
        """预览：处理多相机拼合预览"""
        if "rgb" not in obs: return
        
        # 按照字典序拼接相机画面
        sorted_roles = sorted(obs["rgb"].keys())
        imgs = [obs["rgb"][role] for role in sorted_roles]
        
        # 环境返回的是 CHW，转回 HWC 用于显示
        hwc_imgs = [img.transpose(1, 2, 0) for img in imgs]
        combined_img = np.hstack(hwc_imgs)
        
        # UI 叠加
        h, w = combined_img.shape[:2]
        status_color = (0, 0, 255) if self.is_recording else (0, 255, 0)
        status_text = f"REC - Step: {len(self.current_trajectory)}" if self.is_recording else "IDLE - Ready"
        cv2.circle(combined_img, (20, 25), 8, status_color, -1)
        cv2.putText(combined_img, status_text, (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.imshow("Realman Recorder Preview", combined_img)
        cv2.waitKey(1)

    def run(self):
        while not self.is_finished:
            t_start = time.time()
            obs = self.env._get_obs()
            if self.is_recording:
                self.current_trajectory.append(obs)
                if len(self.current_trajectory) >= self.max_episode_length:
                    self.is_recording = False
                    print(f"[Auto-Stop] 已达到最大步数 {self.max_episode_length}。")
                    self.success = False
                    self._save_temp_trajectory()
                    self.traj_counter += 1
            
            self._visualize(obs)
            elapsed = time.time() - t_start
            time.sleep(max(0, self.dt - elapsed))
        cv2.destroyAllWindows()

    def _save_temp_trajectory(self):
        if len(self.current_trajectory) < 5:
            print("[W] 轨迹太短，忽略。")
            return
        ts = datetime.datetime.now().strftime("%H%M%S")
        temp_path = os.path.join(self.temp_dir, f"traj_{self.traj_counter}_{ts}.h5")
        
        with h5py.File(temp_path, 'w') as f:
            # 1. 递归保存嵌套观测
            obs_group = f.create_group("obs")
            
            def save_dict_to_h5(group, d):
                keys = d[0].keys()
                for k in keys:
                    if isinstance(d[0][k], dict):
                        sub_group = group.create_group(k)
                        sub_list = [step[k] for step in d]
                        save_dict_to_h5(sub_group, sub_list)
                    else:
                        data = np.stack([step[k] for step in d])
                        if k in ["rgb", "image", "depth"]:
                            group.create_dataset(k, data=data, compression="gzip", compression_opts=4)
                        else:
                            group.create_dataset(k, data=data)

            save_dict_to_h5(obs_group, self.current_trajectory)

            # 2. 保存元数据
            meta_group = f.create_group("meta")
            env_meta = self.env.get_env_metadata() if hasattr(self.env, "get_env_metadata") else self.env.meta_keys
            meta_group.create_dataset("env_meta", data=json.dumps(env_meta))
            
            # 保存本地定义的 EnvMetadata (支持多臂)
            robot_ips = getattr(self.env, 'robot_ips', [getattr(self.env, 'robot_ip', "unknown")])
            is_dual = len(getattr(self.env, 'arm_names', ["arm"])) > 1
            
            meta_obj = RealmanEnvMetadata(
                robot_ips=robot_ips,
                camera_sns=self.env.camera_sns,
                hz=self.env.hz,
                is_dual_arm=is_dual,
                use_depth=self.env.use_depth,
                control_mode=self.env.control_mode
            )
            meta_group.create_dataset("env_kwargs", data=json.dumps(asdict(meta_obj)))
            
            f.attrs['success'] = self.success
            
        print(f"[E] 轨迹已保存: {temp_path}")

if __name__ == "__main__":
    import argparse
    from agent_infra.Realman_Env.Env.realman_env import RealManEnv
    from agent_infra.Realman_Env.Env.dual_realman_env import DualRealManEnv
    from agent_infra.Realman_Env.Camera.realsense_camera import get_connected_realsense_serials
    
    parser = argparse.ArgumentParser(description="Realman 专家轨迹录制工具")
    parser.add_argument("-t", "--task_name", type=str, help="任务名称")
    parser.add_argument("-dual", "--dual_arm", action="store_true", help="是否开启双臂模式")
    parser.add_argument("-hz", "--sample_hz", type=int, default=10, help="采样频率")
    parser.add_argument("-m", "--mode", type=str, default="joint_pos", choices=["joint_pos", "delta_ee_pose"])
    parser.add_argument("-max_steps", "--max_episode_length", type=int, default=200, help="最大轨迹长度")

    args = parser.parse_args()

    if args.dual_arm:
        print("[Init] 启动双臂录制模式...")
        env = DualRealManEnv(hz=args.sample_hz, control_mode=args.mode)
    else:
        print("[Init] 启动单臂录制模式...")
        serials = get_connected_realsense_serials()
        env = RealManEnv(camera_sns=serials, control_mode=args.mode, hz=args.sample_hz)
    
    recorder = DataRecorder(
        env, 
        task_name=args.task_name, 
        sample_hz=args.sample_hz, 
        max_episode_length=args.max_episode_length
    )
    
    try:
        recorder.run()
    finally:
        env.close()
