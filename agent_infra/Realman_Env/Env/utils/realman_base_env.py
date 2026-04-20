import numpy as np
import time
from typing import Dict, Any, List, Optional, Literal
from agent_infra.base_robot_env import BaseRobotEnv
from agent_infra.Realman_Env.Env.utils.realman_arm import RealmanArm

class RealmanBaseEnv(BaseRobotEnv):
    """
    Realman 环境基础类。
    管理机械臂硬件组，实现动作分发与状态聚合逻辑。
    """
    def __init__(self, 
                 robot_ips: List[str], 
                 arm_names: List[str] = ["arm"],
                 control_mode: Literal["joint_pos", "delta_ee_pose"] = "joint_pos",
                 hz: int = 10,
                 **kwargs):
        super().__init__(hz=hz)
        self.robot_ips = robot_ips
        self.arm_names = arm_names
        self.control_mode = control_mode
        self.rotation_type = kwargs.get('rotation_type', "euler")
        
        # 记录默认初始化参数 (兜底用)
        self.default_init_joint_pos = kwargs.get('init_joint_pos', [0, 30, 60, 0, 90, 0])
        self.default_init_gripper_pos = kwargs.get('init_gripper_pos', 500)
        self.joint_speed_ratio = kwargs.get('joint_speed_ratio', 20)
        
        # 🟢 新增：支持各臂独立配置
        self.arm_configs = kwargs.get('arm_configs', {})
        
        # 1. 构建元数据
        self._setup_meta_keys()

        # 2. 硬件句柄占位
        self.arms: Dict[str, RealmanArm] = {}

    def _setup_meta_keys(self):
        """根据机械臂数量和名称，初始化观测与动作的维度描述"""
        self.meta_keys["obs"]["state"] = {}
        for name in self.arm_names:
            prefix = f"{name}_" if len(self.arm_names) > 1 else ""
            self.meta_keys["obs"]["state"].update({
                f"{prefix}joint_pos": (6,),
                f"{prefix}joint_vel": (6,),
                f"{prefix}ee_pose": (6,),
                f"{prefix}gripper_pos": (1,)
            })

        self.meta_keys["action"] = {}
        arm_dim = 6 if self.control_mode == "joint_pos" else (7 if self.rotation_type == "euler" else 8)
        for name in self.arm_names:
            prefix = f"{name}_" if len(self.arm_names) > 1 else ""
            self.meta_keys["action"].update({
                f"{prefix}arm": (arm_dim,),
                f"{prefix}gripper": (1,)
            })

    def _setup_hardware(self):
        """
        初始化硬件连接。
        """
        for i, ip in enumerate(self.robot_ips):
            name = self.arm_names[i]
            # 🟢 修正：为每个臂提取独立配置
            spec_cfg = self.arm_configs.get(name, {})
            init_pos = spec_cfg.get('init_joint_pos', self.default_init_joint_pos)
            init_grip = spec_cfg.get('init_gripper_pos', self.default_init_gripper_pos)
            
            print(f"[Base] 正在初始化机械臂 [{name}] IP: {ip}...")
            arm = RealmanArm(
                ip=ip, 
                name=name, 
                init_joint_pos=init_pos,
                init_gripper_pos=init_grip
            )
            self.arms[name] = arm

    def _get_obs(self) -> Dict[str, Any]:
        """聚合所有机械臂的状态"""
        combined_state = {}
        for name, arm in self.arms.items():
            state = arm.get_state()
            prefix = f"{name}_" if len(self.arm_names) > 1 else ""
            for k, v in state.items():
                combined_state[f"{prefix}{k}"] = v
        
        return {"state": combined_state}

    def _apply_action(self, action: Dict[str, np.ndarray]):
        """分发动作到对应的硬件"""
        for name, arm in self.arms.items():
            prefix = f"{name}_" if len(self.arm_names) > 1 else ""
            arm_act = action[f"{prefix}arm"]
            grip_act = action[f"{prefix}gripper"][0]
            
            arm.apply_action(arm_act, grip_act, self.control_mode, self.rotation_type)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        执行硬件初始化（仅一次）和机械臂归位动作。
        """
        if not self.is_setup:
            self._setup_hardware()
            self.is_setup = True
        
        print(f"Reset: 机械臂组正在回归初始位姿...")
        for arm in self.arms.values():
            arm.move_to_init(self.joint_speed_ratio)
            
        return self._get_obs(), {}

    def get_safe_action(self) -> Dict[str, np.ndarray]:
        obs = self._get_obs()
        safe_action = {}
        for name in self.arm_names:
            prefix = f"{name}_" if len(self.arm_names) > 1 else ""
            if self.control_mode == "joint_pos":
                safe_action[f"{prefix}arm"] = obs["state"][f"{prefix}joint_pos"]
            else:
                arm_dim = self.meta_keys["action"][f"{prefix}arm"][0]
                safe_action[f"{prefix}arm"] = np.zeros(arm_dim, dtype=np.float32)
            safe_action[f"{prefix}gripper"] = obs["state"][f"{prefix}gripper_pos"]
        return safe_action

    def close(self):
        for arm in self.arms.values():
            arm.close()
        self.arms = {}
        self.is_setup = False
