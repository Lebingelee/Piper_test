import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
import os
import yaml
from typing import List, Optional, Literal, Dict, Any

from agent_infra.Realman_Env.Env.utils.realman_base_env import RealmanBaseEnv
from agent_infra.Realman_Env.Camera.realsense_camera import RealSenseCameraGroup

class RealManEnv(RealmanBaseEnv):
    """
    睿尔曼机械臂 + 多目 RealSense 相机 + 夹爪集成 Gym 环境。
    功能聚合层：负责相机集成、配置加载及观测拼合。
    """
    def __init__(self, 
                 robot_ip: Optional[str] = None, 
                 robot_port: int = 8080,
                 camera_sns: Optional[List[str]] = None,
                 camera_res: Optional[tuple] = None,
                 crop_size: Optional[tuple] = None,
                 control_mode: Optional[Literal["joint_pos", "delta_ee_pose"]] = None,
                 hz: Optional[int] = None,
                 **kwargs):
        
        # --- 1. 加载并对齐配置 ---
        self.config = self._load_default_config()
        
        # 优先级：参数 > YAML > 默认值
        self.robot_ip = robot_ip or self.config['robot'].get('ip', "192.168.1.18")
        self.robot_port = robot_port
        self.hz = hz or self.config['robot'].get('default_hz', 10)
        self.control_mode = control_mode or self.config['robot'].get('default_control_mode', "joint_pos")
        
        # 初始化 Base 类 (目前设为单臂模式以兼容单臂 Pipeline)
        super().__init__(
            robot_ips=[self.robot_ip],
            arm_names=["arm"],
            control_mode=self.control_mode,
            hz=self.hz,
            **kwargs
        )
        
        # 相机与分辨率配置
        cam_cfg = self.config.get('cameras', {})
        self.camera_res = camera_res or tuple(cam_cfg.get('source_resolution', (1280, 720)))
        self.crop_size = crop_size or tuple(cam_cfg.get('crop_resolution', (640, 480)))
        
        if camera_sns is not None:
            self.camera_sns = camera_sns
        else:
            nodes = cam_cfg.get('nodes', [])
            sorted_nodes = sorted(nodes, key=lambda x: 0 if x['role'] == "base_camera" else 1)
            self.camera_sns = [node['serial_number'] for node in sorted_nodes]

        self.num_cameras = len(self.camera_sns)
        self.use_depth = kwargs.get('use_depth', False)
        self.exposure = kwargs.get('exposure', [100, 80])
        self.passive = False # 控制动作下发开关
        self.last_obs_time = time.time()

        # --- 2. 补全元数据 (RGB/Depth) ---
        self._setup_camera_meta()
        
        # --- 3. 定义 Gym 空间 (对齐原版) ---
        self._setup_spaces()
        
        # 🟢 修正：立即初始化硬件，以支持 recorder.py 的即时预览
        self._setup_hardware()
        self.is_setup = True
        
        print(f"RealManEnv (Modular) 初始化完成。IP: {self.robot_ip}, HZ: {self.hz}, 模式: {self.control_mode}")

    def _load_default_config(self) -> Dict:
        """从同级目录加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), "env_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {"robot": {}, "cameras": {}}

    def _setup_camera_meta(self):
        """填充 meta_keys 中的相机角色"""
        cam_cfg = self.config.get('cameras', {})
        nodes = cam_cfg.get('nodes', [])
        sn_to_role = {node['serial_number']: node['role'] for node in nodes}
        
        if "rgb" not in self.meta_keys["obs"]:
            self.meta_keys["obs"]["rgb"] = {}
            
        for sn in self.camera_sns:
            role = sn_to_role.get(sn, f"camera_{sn}")
            self.meta_keys["obs"]["rgb"][role] = (3, self.crop_size[1], self.crop_size[0])
            if self.use_depth:
                if "depth" not in self.meta_keys["obs"]:
                    self.meta_keys["obs"]["depth"] = {}
                self.meta_keys["obs"]["depth"][role] = (1, self.crop_size[1], self.crop_size[0])

    def _setup_spaces(self):
        """定义动作与观测空间，完全对齐原版实现"""
        # 动作空间 (字典)
        action_spaces = {}
        for k, v in self.meta_keys["action"].items():
            if "gripper" in k:
                action_spaces[k] = spaces.Box(low=0.0, high=1.0, shape=v, dtype=np.float32)
            else:
                low = -np.pi if self.control_mode == "joint_pos" else -0.05
                high = np.pi if self.control_mode == "joint_pos" else 0.05
                action_spaces[k] = spaces.Box(low=low, high=high, shape=v, dtype=np.float32)
        self.action_space = spaces.Dict(action_spaces)
            
        # 观测空间
        obs_spaces = {
            "rgb": spaces.Dict({
                role: spaces.Box(low=0, high=255, shape=v, dtype=np.uint8)
                for role, v in self.meta_keys["obs"]["rgb"].items()
            }),
            "state": spaces.Dict({
                k: spaces.Box(low=-np.inf, high=np.inf, shape=v, dtype=np.float32)
                for k, v in self.meta_keys["obs"]["state"].items()
            })
        }
        if self.use_depth:
            obs_spaces["depth"] = spaces.Dict({
                role: spaces.Box(low=0, high=65535, shape=v, dtype=np.uint16)
                for role, v in self.meta_keys["obs"]["depth"].items()
            })
        self.observation_space = spaces.Dict(obs_spaces)

    def _setup_hardware(self):
        """扩展硬件初始化，包含相机组"""
        # A. 初始化机械臂 (父类实现)
        super()._setup_hardware()
        
        # B. 初始化相机组
        if self.num_cameras > 0:
            print(f"正在启动相机组: {self.camera_sns}...")
            self.camera_group = RealSenseCameraGroup(
                self.camera_sns, 
                self.camera_res[0], 
                self.camera_res[1], 
                exposure=self.exposure
            )
        else:
            print("[Warning] No camera serials provided, camera integration disabled.")

    def _get_obs(self) -> Dict[str, Any]:
        """聚合机械臂状态与相机图像"""
        # A. 获取基础状态 (父类实现)
        res = super()._get_obs()
        
        # B. 获取相机数据并处理
        obs_rgb = {}
        obs_depth = {}
        
        if self.num_cameras > 0:
            all_frames = self.camera_group.get_all_frames()
            
            cam_cfg = self.config.get('cameras', {})
            nodes = cam_cfg.get('nodes', [])
            sn_to_role = {node['serial_number']: node['role'] for node in nodes}

            for sn in self.camera_sns:
                role = sn_to_role.get(sn, f"camera_{sn}")
                frame_data = all_frames.get(sn, {"color": None, "depth": None})
                color_img, depth_img = frame_data["color"], frame_data["depth"]
                
                if color_img is not None:
                    tw, th = self.crop_size
                    color_resized = cv2.resize(color_img, (tw, th), interpolation=cv2.INTER_AREA)
                    # HWC -> CHW
                    obs_rgb[role] = color_resized.transpose(2, 0, 1)
                    
                    if self.use_depth and depth_img is not None:
                        depth_resized = cv2.resize(depth_img, (tw, th), interpolation=cv2.INTER_NEAREST)
                        obs_depth[role] = depth_resized[None, ...]
                else:
                    obs_rgb[role] = np.zeros(self.meta_keys["obs"]["rgb"][role], dtype=np.uint8)
                    if self.use_depth:
                        obs_depth[role] = np.zeros(self.meta_keys["obs"]["depth"][role], dtype=np.uint16)

        res["rgb"] = obs_rgb
        if self.use_depth:
            res["depth"] = obs_depth
            
        return res

    def _apply_action(self, action: Dict[str, np.ndarray]):
        """重写动作下发，增加被动模式开关"""
        if self.passive:
            super()._apply_action(action)

    def switch_passive(self, mode: str):
        """控制是否允许动作下发"""
        self.passive = (mode.lower() == 'true')

    def close(self):
        """释放相机和机械臂资源"""
        if hasattr(self, 'camera_group'):
            self.camera_group.stop_all()
        super().close()
