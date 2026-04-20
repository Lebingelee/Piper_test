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

class DualRealManEnv(RealmanBaseEnv):
    """
    双臂睿尔曼机械臂环境。
    支持两个机械臂协同控制及多相机集成。
    """
    def __init__(self, 
                 config_path: str = "dual_env_config.yaml",
                 camera_sns: Optional[List[str]] = None,
                 hz: Optional[int] = None,
                 control_mode: Optional[Literal["joint_pos", "delta_ee_pose"]] = None,
                 **kwargs):
        
        # 1. 加载双臂配置
        self.full_config = self._load_config(config_path)
        
        # 提取机械臂信息
        robot_cfg = self.full_config.get('robots', {})
        arm_names = sorted(robot_cfg.keys()) # ["left", "right"]
        robot_ips = [robot_cfg[name]['ip'] for name in arm_names]
        
        # 🟢 关键：提取并映射各臂独立配置 (init_joint_pos, init_gripper_pos)
        arm_configs = {}
        for name in arm_names:
            arm_configs[name] = {
                'init_joint_pos': robot_cfg[name].get('init_joint_pos'),
                'init_gripper_pos': robot_cfg[name].get('init_gripper_pos')
            }
        
        # 提取公共配置
        common_cfg = self.full_config.get('common', {})
        self.hz = hz or common_cfg.get('default_hz', 10)
        self.control_mode = control_mode or common_cfg.get('default_control_mode', "joint_pos")
        
        # 初始化 Base 类
        super().__init__(
            robot_ips=robot_ips,
            arm_names=arm_names,
            control_mode=self.control_mode,
            hz=self.hz,
            arm_configs=arm_configs, # 🟢 传递独立配置
            **kwargs
        )
        
        # 2. 相机配置
        cam_cfg = self.full_config.get('cameras', {})
        self.camera_res = tuple(cam_cfg.get('source_resolution', (640, 480)))
        self.crop_size = tuple(cam_cfg.get('crop_resolution', (224, 224)))
        self.exposure = kwargs.get('exposure', [100, 80, 80]) # 假设3个相机
        
        if camera_sns is not None:
            self.camera_sns = camera_sns
        else:
            # 自动提取并按角色排序
            nodes = cam_cfg.get('nodes', [])
            self.camera_sns = [node['serial_number'] for node in nodes]

        self.num_cameras = len(self.camera_sns)
        self.use_depth = kwargs.get('use_depth', False)
        self.passive = False
        self.last_obs_time = time.time()

        # --- 3. 补全元数据与空间定义 ---
        self._setup_camera_meta()
        self._setup_spaces()
        
        # 🟢 立即初始化硬件连接
        self._setup_hardware()
        self.is_setup = True
        
        print(f"DualRealManEnv 初始化完成。Arms: {arm_names}, HZ: {self.hz}, 模式: {self.control_mode}")

    def _load_config(self, config_path: str) -> Dict:
        """从指定路径加载 YAML"""
        # 如果是相对路径，尝试在当前目录下查找
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), config_path)
            
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        raise FileNotFoundError(f"Config file not found: {config_path}")

    def _setup_camera_meta(self):
        """定义多相机元数据"""
        cam_cfg = self.full_config.get('cameras', {})
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
        """定义 Gym 空间 (基于 meta_keys 自动生成)"""
        # 动作空间
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
        """初始化双臂和相机组"""
        # A. 初始化所有机械臂 (继承自 RealmanBaseEnv)
        # 注意：Base 类会根据 self.robot_ips 和 self.arm_names 自动循环初始化
        super()._setup_hardware()
        
        # B. 初始化相机组
        if self.num_cameras > 0:
            print(f"正在启动双臂环境相机组: {self.camera_sns}...")
            self.camera_group = RealSenseCameraGroup(
                self.camera_sns, 
                self.camera_res[0], 
                self.camera_res[1], 
                exposure=self.exposure
            )

    def _get_obs(self) -> Dict[str, Any]:
        """聚合双臂状态与多路相机图像"""
        # A. 获取双臂状态 (已带 left_/right_ 前缀)
        res = super()._get_obs()
        
        # B. 获取相机图像
        obs_rgb = {}
        obs_depth = {}
        if self.num_cameras > 0:
            all_frames = self.camera_group.get_all_frames()
            nodes = self.full_config.get('cameras', {}).get('nodes', [])
            sn_to_role = {node['serial_number']: node['role'] for node in nodes}

            for sn in self.camera_sns:
                role = sn_to_role.get(sn, f"camera_{sn}")
                frame_data = all_frames.get(sn, {"color": None, "depth": None})
                color_img, depth_img = frame_data["color"], frame_data["depth"]
                
                if color_img is not None:
                    color_resized = cv2.resize(color_img, self.crop_size, interpolation=cv2.INTER_AREA)
                    obs_rgb[role] = color_resized.transpose(2, 0, 1)
                    if self.use_depth and depth_img is not None:
                        depth_resized = cv2.resize(depth_img, self.crop_size, interpolation=cv2.INTER_NEAREST)
                        obs_depth[role] = depth_resized[None, ...]
                else:
                    obs_rgb[role] = np.zeros(self.meta_keys["obs"]["rgb"][role], dtype=np.uint8)

        res["rgb"] = obs_rgb
        if self.use_depth:
            res["depth"] = obs_depth
        return res

    def _apply_action(self, action: Dict[str, np.ndarray]):
        if self.passive:
            super()._apply_action(action)

    def switch_passive(self, mode: str):
        self.passive = (mode.lower() == 'true')

    def close(self):
        if hasattr(self, 'camera_group'):
            self.camera_group.stop_all()
        super().close()
