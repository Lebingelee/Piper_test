import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import yaml
from typing import List, Optional, Literal, Dict, Any

from agent_infra.base_robot_env import BaseRobotEnv

class OfflineRealManEnv(BaseRobotEnv):
    """
    离线版本的 RealManEnv。
    不初始化任何硬件逻辑，仅通过配置生成一致的 meta_keys 和 Gym Spaces。
    用于服务器训练和 Pipeline 验证。
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
        
        # --- 1. 加载配置 ---
        self.config = self._load_default_config()
        
        self.hz = hz or self.config['robot'].get('default_hz', 10)
        self.control_mode = control_mode or self.config['robot'].get('default_control_mode', "joint_pos")
        self.rotation_type = kwargs.get('rotation_type', "euler")
        
        super().__init__(hz=self.hz)

        # 机械臂名称定义 (单臂默认为 "arm"，保持与 RealManEnv 对齐)
        self.arm_names = kwargs.get('arm_names', ["arm"])
        
        # 相机参数对齐
        cam_cfg = self.config.get('cameras', {})
        self.camera_res = camera_res or tuple(cam_cfg.get('source_resolution', (1280, 720)))
        self.crop_size = crop_size or tuple(cam_cfg.get('crop_resolution', (640, 480)))
        
        if camera_sns is not None:
            self.camera_sns = camera_sns
        else:
            nodes = cam_cfg.get('nodes', [])
            sorted_nodes = sorted(nodes, key=lambda x: 0 if x['role'] == "base_camera" else 1)
            self.camera_sns = [node['serial_number'] for node in sorted_nodes]

        self.use_depth = kwargs.get('use_depth', False)

        # --- 2. 构建元数据驱动结构 ---
        self._setup_meta_keys_offline()
        self._setup_spaces()
        
        print(f"[Offline] OfflineRealManEnv 初始化完成。模式: {self.control_mode}, 频率: {self.hz}Hz")

    def _load_default_config(self) -> Dict:
        """从同级目录加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), "env_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {"robot": {}, "cameras": {}}

    def _setup_meta_keys_offline(self):
        """完全模拟 RealmanBaseEnv 和 RealManEnv 的元数据构建逻辑"""
        # A. 状态元数据 (State)
        self.meta_keys["obs"]["state"] = {}
        for name in self.arm_names:
            prefix = f"{name}_" if len(self.arm_names) > 1 else ""
            self.meta_keys["obs"]["state"].update({
                f"{prefix}joint_pos": (6,),
                f"{prefix}joint_vel": (6,),
                f"{prefix}ee_pose": (6,),
                f"{prefix}gripper_pos": (1,)
            })

        # B. 视觉元数据 (RGB/Depth)
        cam_cfg = self.config.get('cameras', {})
        nodes = cam_cfg.get('nodes', [])
        sn_to_role = {node['serial_number']: node['role'] for node in nodes}
        
        self.meta_keys["obs"]["rgb"] = {}
        for sn in self.camera_sns:
            role = sn_to_role.get(sn, f"camera_{sn}")
            self.meta_keys["obs"]["rgb"][role] = (3, self.crop_size[1], self.crop_size[0])
            if self.use_depth:
                if "depth" not in self.meta_keys["obs"]:
                    self.meta_keys["obs"]["depth"] = {}
                self.meta_keys["obs"]["depth"][role] = (1, self.crop_size[1], self.crop_size[0])

        # C. 动作元数据 (Action)
        self.meta_keys["action"] = {}
        arm_dim = 6 if self.control_mode == "joint_pos" else (7 if self.rotation_type == "euler" else 8)
        for name in self.arm_names:
            prefix = f"{name}_" if len(self.arm_names) > 1 else ""
            self.meta_keys["action"].update({
                f"{prefix}arm": (arm_dim,),
                f"{prefix}gripper": (1,)
            })

    def _setup_spaces(self):
        """定义 Gym 空间"""
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
        """离线模式不初始化硬件"""
        pass

    def _get_obs(self) -> Dict[str, Any]:
        """返回符合结构的 Mock 全零观测"""
        obs = {"state": {}, "rgb": {}}
        
        # 填充 State
        for k, v in self.meta_keys["obs"]["state"].items():
            obs["state"][k] = np.zeros(v, dtype=np.float32)
            
        # 填充 RGB
        for role, v in self.meta_keys["obs"]["rgb"].items():
            obs["rgb"][role] = np.zeros(v, dtype=np.uint8)
            
        # 填充 Depth
        if self.use_depth:
            obs["depth"] = {}
            for role, v in self.meta_keys["obs"]["depth"].items():
                obs["depth"][role] = np.zeros(v, dtype=np.uint16)
                
        return obs

    def _apply_action(self, action: Dict[str, np.ndarray]):
        """离线模式不执行动作"""
        pass

    def get_safe_action(self) -> Dict[str, np.ndarray]:
        """返回全零的安全动作"""
        safe_action = {}
        for k, v in self.meta_keys["action"].items():
            safe_action[k] = np.zeros(v, dtype=np.float32)
        return safe_action

    def switch_passive(self, mode: str):
        pass

    def close(self):
        pass
