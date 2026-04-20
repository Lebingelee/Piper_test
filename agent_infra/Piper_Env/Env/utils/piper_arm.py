import numpy as np
import time
from typing import Dict, Any, List, Literal
from pyAgxArm import create_agx_arm_config, AgxArmFactory
from agent_infra.Piper_Env.Env.utils.get_pose import get_pose


GRIPPER_MIN_WIDTH_M = 0.0
GRIPPER_MAX_WIDTH_M = 0.1
JOINT_LIMIT_EPS_RAD = 1e-6
JOINT5_LIMIT_RAD = float(np.deg2rad(70.0) + JOINT_LIMIT_EPS_RAD)
PIPER_SDK_JOINT_LIMIT_OVERRIDES = {
    "joint5": [-JOINT5_LIMIT_RAD, JOINT5_LIMIT_RAD],
}


class PiperArm:
    """
    Piper 机械臂硬件包装类。
    管理主臂 (Leader) 与从臂 (Follower) 的连接、状态获取及动作下发。
    """
    def __init__(self, 
                 master_can: str = "can_master", 
                 follower_can: str = "can_slave",
                 name: str = "arm",
                 init_joint_pos: List[float] = [0.0]*6,
                 init_gripper_pos: float = 0.05):
        
        self.master_can = master_can
        self.follower_can = follower_can
        self.name = name
        self.init_joint_pos = init_joint_pos
        self.init_gripper_pos = self._clip_gripper_width(init_gripper_pos)
        self.last_follower_gripper_cmd = self.init_gripper_pos
        self.last_master_gripper_cmd = self.init_gripper_pos
        self.joint_hold_deadband = 1e-4
        
        # 1. 硬件句柄占位
        self.master = None
        self.follower = None
        self.master_eff = None
        self.follower_eff = None
        self.is_setup = False

    def connect(self):
        """初始化硬件连接并进行基本配置"""
        if self.is_setup:
            return
            
        print(f"[{self.name}] 正在连接 Piper 机械臂 (Leader: {self.master_can}, Follower: {self.follower_can})...")
        
        try:
            # 主臂 (Leader)
            self.cfg_m = create_agx_arm_config(
                robot="piper",
                channel=self.master_can,
                joint_limits=PIPER_SDK_JOINT_LIMIT_OVERRIDES,
            )
            self.master = AgxArmFactory.create_arm(self.cfg_m)
            self.master.connect()
            self.master.set_leader_mode()
            self.master_eff = self.master.init_effector(self.master.OPTIONS.EFFECTOR.AGX_GRIPPER)

            # 从臂 (Follower)
            self.cfg_f = create_agx_arm_config(
                robot="piper",
                channel=self.follower_can,
                joint_limits=PIPER_SDK_JOINT_LIMIT_OVERRIDES,
            )
            self.follower = AgxArmFactory.create_arm(self.cfg_f)
            self.follower.connect()
            self.follower.set_joint_angle_vel_limits(joint_index=255, max_joint_spd=3.0)
            self.follower.set_joint_acc_limits(joint_index=255, max_joint_acc=5.0)
            
            self.follower_eff = self.follower.init_effector(self.follower.OPTIONS.EFFECTOR.AGX_GRIPPER)
            
            self.follower.set_speed_percent(100)
            self.is_setup = True
            print(f"[{self.name}] 硬件连接就绪。")
        except Exception as e:
            print(f"[{self.name}] 硬件连接失败: {e}")
            self.is_setup = False
            raise e

    def get_state(self) -> Dict[str, np.ndarray]:
        """获取从臂 (Follower) 的标准化状态，对齐 agent_infra 规范"""
        # A. 关节角度
        ja = self.follower.get_joint_angles()
        joint_pos = np.array(ja.msg if ja else [0.0]*6, dtype=np.float32)
        
        # B. 关节速度 (通过 motor_states 获取)
        joint_vel = []
        for i in range(1, 7):
            ms = self.follower.get_motor_states(i)
            joint_vel.append(ms.msg.velocity if ms else 0.0)
        joint_vel = np.array(joint_vel, dtype=np.float32)
        
        # C. 末端法兰位姿 [x, y, z, roll, pitch, yaw]
        fp = self.follower.get_flange_pose()
        ee_pose = np.array(fp.msg if fp else [0.0]*6, dtype=np.float32)
        
        # D. 夹爪宽度 (m). Prefer physical feedback over control feedback; the
        # latter may report a default 0.0 while the gripper is simply idle.
        gripper_width, _ = self._read_gripper_width(
            self.follower_eff,
            fallback=self.last_follower_gripper_cmd,
            guard_direct_zero=True,
        )
        gripper_pos = np.array([gripper_width], dtype=np.float32)

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "ee_pose": ee_pose,
            "gripper_pos": gripper_pos
        }

    def get_master_state(self) -> Dict[str, Any]:
        """获取主臂 (Leader) 原始状态，用于遥操作控制"""
        mja = self.master.get_leader_joint_angles()
        ok = self.master.is_ok()
    
        gripper_width, has_gripper = self._read_gripper_width(
            self.master_eff,
            fallback=None,
        )
        
        # 使用外部 FK 计算主臂位姿以对齐控制模式
        mep = get_pose(mja.msg) if mja else np.zeros(6, dtype=np.float32)
        
        return {
            "joint_pos": np.array(mja.msg if mja else [0.0]*6, dtype=np.float32),
            "ee_pose": mep.astype(np.float32),
            "gripper_pos": np.array([gripper_width], dtype=np.float32),
            "is_ok": ok,
            "has_leader_joint": mja is not None,
            "has_gripper": has_gripper,
        }

    def apply_action(
        self,
        arm_action: np.ndarray,
        gripper_action: float,
        mode: str = "joint",
    ):
        """
        执行动作下发。
        - joint 模式：根据差异动态切换 move_j (大差异) 和 move_js (透传)。
        - pose 模式：直接执行绝对 move_p。
        - delta_pose 模式：将 6D 末端增量加到当前末端位姿后执行 move_p。
        - relative_pose_chunk 模式：按展平的多步 6D 增量依次执行 move_p。
        - 夹爪：直接使用 SDK 物理宽度，单位 m，范围 [0.0, 0.1]。
        """
        arm_action = np.asarray(arm_action, dtype=np.float32).reshape(-1)
        if mode not in ("joint", "pose", "delta_pose", "relative_pose_chunk"):
            raise ValueError(f"[{self.name}] Unsupported control mode: {mode}")
        if mode != "relative_pose_chunk" and arm_action.shape[0] != 6:
            raise ValueError(
                f"[{self.name}] Piper {mode} action must have 6 values, "
                f"got shape {arm_action.shape}."
            )
        if mode == "relative_pose_chunk" and arm_action.shape[0] % 6 != 0:
            raise ValueError(
                f"[{self.name}] relative_pose_chunk action length must be "
                f"a multiple of 6, got shape {arm_action.shape}."
            )

        if mode == "pose":
            self.follower.move_p(arm_action.tolist())
        elif mode == "delta_pose":
            self.follower.move_p(self._pose_from_delta(arm_action).tolist())
        elif mode == "relative_pose_chunk":
            current_pose = self._get_current_ee_pose()
            for delta_pose in arm_action.reshape(-1, 6):
                if np.allclose(delta_pose, 0.0, atol=1e-7):
                    continue
                current_pose = current_pose + delta_pose.astype(np.float32)
                self.follower.move_p(current_pose.tolist())
        else:
            curr_ja = self.follower.get_joint_angles()
            if curr_ja is not None:
                diff = np.sum(np.abs(arm_action - np.asarray(curr_ja.msg, dtype=np.float32)))
                if diff < self.joint_hold_deadband:
                    self._move_gripper(
                        self.follower_eff,
                        gripper_action,
                        remember_as="follower",
                    )
                    return
                # 差异大于 0.7 弧度使用带规划的 move_j，否则使用低延迟 move_js
                if diff > 0.7:
                    self.follower.move_j(arm_action.tolist())
                else:
                    self.follower.move_js(arm_action.tolist())
        
        self._move_gripper(self.follower_eff, gripper_action, remember_as="follower")

    @staticmethod
    def _clip_gripper_width(gripper_pos: float) -> float:
        return float(np.clip(float(gripper_pos), GRIPPER_MIN_WIDTH_M, GRIPPER_MAX_WIDTH_M))

    def _get_current_ee_pose(self) -> np.ndarray:
        fp = self.follower.get_flange_pose()
        return np.array(fp.msg if fp else [0.0] * 6, dtype=np.float32)

    def _pose_from_delta(self, delta_pose: np.ndarray) -> np.ndarray:
        return self._get_current_ee_pose() + np.asarray(delta_pose, dtype=np.float32).reshape(6)

    def _read_gripper_width(self, effector, fallback=None, guard_direct_zero: bool = False):
        if effector is None:
            if fallback is None:
                return 0.0, False
            return self._clip_gripper_width(fallback), False

        for method_name in ("get_gripper_status", "get_gripper_ctrl_states"):
            if not hasattr(effector, method_name):
                continue
            try:
                msg = getattr(effector, method_name)()
            except Exception:
                msg = None
            if msg is not None and hasattr(msg, "msg") and hasattr(msg.msg, "width"):
                width = self._clip_gripper_width(msg.msg.width)
                if (
                    guard_direct_zero
                    and fallback is not None
                    and width <= 1e-3
                    and float(fallback) >= 0.03
                ):
                    return self._clip_gripper_width(fallback), False
                return width, True

        if fallback is None:
            return 0.0, False
        return self._clip_gripper_width(fallback), False

    def _move_gripper(self, effector, gripper_pos: float, remember_as: str = ""):
        if effector is not None:
            target = self._clip_gripper_width(gripper_pos)
            effector.move_gripper(target)
            if remember_as == "follower":
                self.last_follower_gripper_cmd = target
            elif remember_as == "master":
                self.last_master_gripper_cmd = target

    def _move_master_to_joint(self, joint_pos: np.ndarray, gripper_pos: float, wait_time: float = 3.0):
        if self.master is None:
            return

        target_joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
        print(f"[{self.name}] 同步主臂回归目标位姿...")
        try:
            is_home_target = bool(np.allclose(target_joint_pos, 0.0, atol=1e-4))
            if is_home_target and hasattr(self.master, "move_leader_to_home"):
                print(f"[{self.name}] 主臂执行 SDK leader home 归位。")
                self.master.move_leader_to_home()
                time.sleep(wait_time)
            elif hasattr(self.master, "set_follower_mode"):
                self.master.set_follower_mode()
                time.sleep(0.2)

                if hasattr(self.master, "move_j"):
                    print(f"[{self.name}] 主臂尝试执行 joint 目标同步。")
                    self.master.move_j(target_joint_pos.tolist())
                    time.sleep(wait_time)
                else:
                    print(f"[{self.name}] 当前 SDK 不支持主臂任意 joint 目标同步。")
            elif hasattr(self.master, "enable") and hasattr(self.master, "move_j"):
                self.master.enable()
                time.sleep(0.2)
                print(f"[{self.name}] 主臂尝试执行 joint 目标同步。")
                self.master.move_j(target_joint_pos.tolist())
                time.sleep(wait_time)
            else:
                print(f"[{self.name}] 当前 SDK 不支持主动移动主臂。")

            self._move_gripper(self.master_eff, gripper_pos, remember_as="master")
            if hasattr(self.master, "restore_leader_drag_mode"):
                self.master.restore_leader_drag_mode()
            elif hasattr(self.master, "set_leader_mode"):
                self.master.set_leader_mode()
        except Exception as exc:
            print(f"[{self.name}] 主臂同步归位失败: {exc}")

    def move_to_init(self, wait_time: float = 3.0, sync_master: bool = False):
        """回归初始位姿逻辑"""
        print(f"[{self.name}] 执行回归初始位姿 (wait: {wait_time}s)...")
        if sync_master:
            self._move_master_to_joint(
                self.init_joint_pos,
                self.init_gripper_pos,
                wait_time=wait_time,
            )

        self.follower.enable()
        time.sleep(0.5)
        self.follower.move_j(self.init_joint_pos)
        self._move_gripper(
            self.follower_eff,
            self.init_gripper_pos,
            remember_as="follower",
        )
        time.sleep(wait_time)

    def move_to_state(
        self,
        joint_pos: np.ndarray,
        gripper_pos: float,
        wait_time: float = 2.0,
        sync_master: bool = False,
    ):
        """统一的 reset-to-state 接口，回归到指定关节位姿和夹爪状态。"""
        target_joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
        target_gripper_pos = self._clip_gripper_width(
            float(np.asarray(gripper_pos, dtype=np.float32).reshape(-1)[0])
        )

        print(f"[{self.name}] 执行回归到指定状态 (wait: {wait_time}s)...")
        if sync_master:
            self._move_master_to_joint(
                target_joint_pos,
                target_gripper_pos,
                wait_time=wait_time,
            )

        self.follower.enable()
        time.sleep(0.5)
        self.follower.move_j(target_joint_pos.tolist())
        self._move_gripper(
            self.follower_eff,
            target_gripper_pos,
            remember_as="follower",
        )
        time.sleep(wait_time)

    def enable(self):
        if self.follower: self.follower.enable()
        if self.master: self.master.enable()

    def disable(self):
        #if self.follower: self.follower.disable()
        #if self.master: self.master.disable()
        pass

    def close(self):
        self.disable()
        self.is_setup = False
        try:
            if self.follower: self.follower.disconnect()
            if self.master: self.master.disconnect()
        except:
            pass
