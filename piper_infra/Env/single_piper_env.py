import gymnasium as gym
import numpy as np
import threading
import time
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
from pynput import keyboard 
from pyAgxArm import create_agx_arm_config, AgxArmFactory


class PiperTeleopEnv(gym.Env):
    """
    Piper 机械臂自适应遥操作强化学习环境。
    支持控制模式：
    1. 'joint': 6关节角度 (rad) + 1夹爪宽度 (m)
    2. 'pose': 末端 6D 位姿 [x, y, z, roll, pitch, yaw] + 1夹爪宽度
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, master_can: str = "can_master", follower_can: str = "can_slave", ctrl_mode: str = "joint"):
        super(PiperTeleopEnv, self).__init__()
        self.ctrl_mode = ctrl_mode

        # --- 1. 硬件初始化 ---
        self.cfg_m = create_agx_arm_config(robot="piper", channel=master_can)
        self.master = AgxArmFactory.create_arm(self.cfg_m)
        self.master.connect()
        self.master.set_leader_mode()
        self.master_eff = self.master.init_effector(self.master.OPTIONS.EFFECTOR.AGX_GRIPPER)

        self.cfg_f = create_agx_arm_config(robot="piper", channel=follower_can)
        self.follower = AgxArmFactory.create_arm(self.cfg_f)
        self.follower.connect()
        self.follower.set_joint_angle_vel_limits(joint_index=255, max_joint_spd=3.0)
        self.follower.set_joint_acc_limits(joint_index=255, max_joint_acc=5.0)
        self.follower_eff = self.follower.init_effector(self.follower.OPTIONS.EFFECTOR.AGX_GRIPPER)
        
        self.follower.set_speed_percent(100)

        # --- 2. 空间定义 (基于模式动态调整限值) ---
        if self.ctrl_mode == "pose":
            # 位姿空间限制 (单位: 米/弧度)
            # x, y, z 限制在 [-0.8, 0.8], [-0.8, 0.8], [-0.2, 1.0]
            # r, p, y 限制在 [-pi, pi]
            low_act = np.array([-0.8, -0.8, -0.2, -np.pi, -np.pi, -np.pi, 0.01], dtype=np.float32)
            high_act = np.array([0.8, 0.8, 1.0, np.pi, np.pi, np.pi, 0.1], dtype=np.float32)
        else:
            # 关节空间限制 (单位: 弧度)
            low_act = np.array([-np.pi] * 6 + [0.01], dtype=np.float32)
            high_act = np.array([np.pi] * 6 + [0.1], dtype=np.float32)

        self.action_space = spaces.Box(low=low_act, high=high_act, shape=(7,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32),
            "observation.images.wrist": spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8),
            "observation.images.front": spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8),
            "under_control": spaces.Discrete(2)
        })

        # --- 3. 状态与并发控制 ---
        self._lock = threading.Lock()
        self._master_running = True
        self._latest_master_joints = np.zeros(7, dtype=np.float32)
        self._latest_master_pose = np.zeros(7, dtype=np.float32)
        self._master_is_ok = False
        self.tele_enabled = False 
        self._last_ok_time = time.time()
        self._debounce_threshold = 0.5 

        # --- 4. 线程启动 ---
        self.read_thread = threading.Thread(target=self._master_read_thread, daemon=True)
        self.read_thread.start()
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

        print(f"[Env] PiperTeleopEnv Ready (Mode: {self.ctrl_mode}). Press 'T' to toggle.")

    def _on_press(self, key):
        try:
            if key.char == 't' or key.char == 'T':
                self.tele_enabled = not self.tele_enabled
                print(f"\n[Env] Teleoperation Mode: {'ENABLED' if self.tele_enabled else 'DISABLED'}")
        except AttributeError:
            pass

    def _master_read_thread(self):
        while self._master_running:
            start_t = time.perf_counter()
            mja = self.master.get_leader_joint_angles()
            mep = self.master.get_flange_pose()
            ok = self.master.is_ok()
            gcs = self.master_eff.get_gripper_ctrl_states()

            with self._lock:
                self._master_is_ok = ok
                if ok: self._last_ok_time = time.time()
                if mja is not None: self._latest_master_joints[:6] = mja.msg
                if mep is not None: self._latest_master_pose[:6] = mep.msg
                if gcs is not None:
                    # 统一物理映射：主臂反馈的是 0-0.08 左右的宽度
                    w = gcs.msg.width
                    self._latest_master_joints[6] = w
                    self._latest_master_pose[6] = w

            elapsed = time.perf_counter() - start_t
            time.sleep(max(0, 0.005 - elapsed))

    def _check_under_control(self) -> bool:
        current_time = time.time()
        with self._lock:
            hardware_ok = self._master_is_ok or (current_time - self._last_ok_time < self._debounce_threshold)
            return self.tele_enabled and hardware_ok

    def _get_obs(self) -> Dict[str, Any]:
        ja = self.follower.get_joint_angles()
        follower_angles = np.array(ja.msg if ja else [0.0]*6, dtype=np.float32)
        follower_vels = []
        for i in range(1, 7):
            ms = self.follower.get_motor_states(i)
            follower_vels.append(ms.msg.velocity if ms else 0.0)
        follower_vels = np.array(follower_vels, dtype=np.float32)
        fp = self.follower.get_flange_pose()
        follower_pose = np.array(fp.msg if fp else [0.0]*6, dtype=np.float32)
        fg = self.follower_eff.get_gripper_ctrl_states()
        follower_gripper = np.array([fg.msg.width if fg else 0.0], dtype=np.float32)

        state_vector = np.concatenate([follower_angles, follower_vels, follower_pose, follower_gripper])
        under_control = 1 if self._check_under_control() else 0

        return {
            "observation.state": state_vector,
            "observation.images.wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            "observation.images.front": np.zeros((3, 224, 224), dtype=np.uint8),
            "under_control": under_control
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        is_intervened = self._check_under_control()
        executed_action = action.copy() 

        if is_intervened:
            with self._lock:
                if self.ctrl_mode == "pose":
                    expert_val = self._latest_master_pose[:6].copy()
                    expert_gripper = self._latest_master_pose[6]
                else:
                    expert_val = self._latest_master_joints[:6].copy()
                    expert_gripper = self._latest_master_joints[6]
            executed_action[:6] = expert_val
            executed_action[6] = expert_gripper

        # --- 硬件执行 ---
        if self.ctrl_mode == "pose":
            self.follower.move_p(executed_action[:6].tolist())
        else:
            curr_ja = self.follower.get_joint_angles()
            if curr_ja is not None:
                diff = np.sum(np.abs(executed_action[:6] - np.array(curr_ja.msg)))
                if diff > 0.7: self.follower.move_j(executed_action[:6].tolist())
                else: self.follower.move_js(executed_action[:6].tolist())
        
        # 夹爪控制：统一采用 0.5 映射系数 (基于 Piper 协议规范)
        # 注意：无论是 tele 还是 model，都应用相同的映射以保证数据分布一致
        target_width = np.clip(executed_action[6] * 0.5, 0.01, 0.1)
        self.follower_eff.move_gripper(target_width)

        obs = self._get_obs()
        return obs, 0.0, False, False, {"actual_action": executed_action, "intervened": is_intervened}

    def reset_to_state(self, joint_angles: np.ndarray, gripper_width: float, wait_time: float = 3.0):
        print(f"[Env] 正在平滑移动至起始位姿... (关节模式对齐)")
        self.follower.move_j(joint_angles.tolist())
        self.follower_eff.move_gripper(np.clip(gripper_width * 0.5, 0.01, 0.1))
        time.sleep(wait_time)
        print("[Env] 到位完成。")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.follower.enable()
        time.sleep(0.5)
        return self._get_obs(), {"status": "reset_done"}

    def close(self):
        print("[Env] Shutting down...")
        self._master_running = False
        self.listener.stop()
        if hasattr(self, 'read_thread'): self.read_thread.join(timeout=1.0)
        try:
            self.follower.disable()
            self.master.disable()
        except: pass
        super().close()
