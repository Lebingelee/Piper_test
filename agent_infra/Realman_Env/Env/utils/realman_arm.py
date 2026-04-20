import numpy as np
import time
from typing import Dict, Any, List, Optional, Literal

# 导入睿尔曼 API
try:
    from Robotic_Arm.rm_robot_interface import *
except ImportError:
    print("Error: Robotic_Arm SDK not found. RealmanArm will run in dummy mode.")

class RealmanArm:
    """
    单台睿尔曼机械臂的硬件包装类。
    """
    def __init__(self, 
                 ip: str, 
                 port: int = 8080, 
                 name: str = "arm",
                 init_joint_pos: List[float] = [0, 30, 60, 0, 90, 0],
                 init_gripper_pos: int = 500):
        self.ip = ip
        self.port = port
        self.name = name
        self.init_joint_pos = init_joint_pos
        self.init_gripper_pos = init_gripper_pos
        
        # 初始化 SDK 句柄
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(self.ip, self.port)
        
        if getattr(self.handle, 'id', -1) < 0:
            raise ConnectionError(f"[{self.name}] 机械臂连接失败: {self.ip}")
        
        print(f"[{self.name}] 机械臂连接成功: {self.ip}")

    def move_to_init(self, joint_speed: int = 20):
        """
        回归初始位姿逻辑 (对应原环境 reset 中的核心动作)。
        不重新定义机械臂，仅执行动作。
        """
        # 1. 机械臂回零 (使用原版参数)
        self.arm.rm_movej(self.init_joint_pos, joint_speed, 0, 0, 1)
        # 2. 夹爪初始化
        self.arm.rm_set_gripper_position(self.init_gripper_pos, block=True, timeout=5)
        time.sleep(1.0)

    def get_state(self) -> Dict[str, np.ndarray]:
        """获取当前机械臂和夹爪的标准化状态"""
        ret, state = self.arm.rm_get_current_arm_state()
        if ret == 0:
            joint_pos = np.deg2rad(state['joint']).astype(np.float32)
            ee_pose = np.array(state['pose'], dtype=np.float32)
        else:
            joint_pos = np.zeros(6, dtype=np.float32)
            ee_pose = np.zeros(6, dtype=np.float32)

        ret_g, g_state = self.arm.rm_get_rm_plus_state_info()
        if ret_g == 0:
            raw_pos = g_state.get('pos', 500)[0]
            gripper_pos = np.array([raw_pos / 999.0], dtype=np.float32)
        else:
            gripper_pos = np.array([0.5], dtype=np.float32)

        return {
            "joint_pos": joint_pos,
            "ee_pose": ee_pose,
            "gripper_pos": gripper_pos,
            "joint_vel": np.zeros(6, dtype=np.float32)
        }

    def apply_action(self, 
                     arm_action: np.ndarray, 
                     gripper_action: float, 
                     mode: Literal["joint_pos", "delta_ee_pose"],
                     rotation_type: str = "euler"):
        """执行动作"""
        if mode == "joint_pos":
            target_joints_deg = np.rad2deg(arm_action).tolist()
            # 保持原版 canfd 参数: trajectory_mode=1, radio=10
            self.arm.rm_movej_canfd(target_joints_deg, follow=False, trajectory_mode=1, radio=10)
        elif mode == "delta_ee_pose":
            ret, state = self.arm.rm_get_current_arm_state()
            if ret == 0:
                current_pose = np.array(state['pose'])
                # 对齐原版逻辑: target = current + delta
                if rotation_type == "euler":
                    target_pose = current_pose + arm_action
                else:
                    target_pose = current_pose + arm_action[:6]
                self.arm.rm_movep_canfd(target_pose.tolist(), follow=False, trajectory_mode=1, radio=0)

        # 执行夹爪动作 (非阻塞)
        target_gripper_val = int(gripper_action * 999 + 1)
        self.arm.rm_set_gripper_position(target_gripper_val, block=False, timeout=1)

    def close(self):
        """
        显式释放 SDK 资源。
        """
        if hasattr(self, 'arm'):
            self.arm.rm_delete_robot_arm()
            print(f"[{self.name}] SDK 句柄已销毁 (rm_delete_robot_arm)。")
