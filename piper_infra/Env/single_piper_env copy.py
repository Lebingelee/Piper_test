import gymnasium as gym
import numpy as np
import threading
import time
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
from pynput import keyboard # 需要安装 pynput: pip install pynput
from pyAgxArm import create_agx_arm_config, AgxArmFactory


class PiperTeleopEnv(gym.Env):
    """
    Piper 机械臂自适应遥操作强化学习环境 (完整版)。
    
    核心特性：
    1. 键盘监听：'T' 键切换遥操激活状态。
    2. 防抖逻辑：0.5s 硬件断连防抖，确保控制平滑。
    3. 动作覆写：人类接管时自动覆盖模型 Action，并记录专家动作。
    4. 读写分离：200Hz 后台同步，50Hz 环境步进。
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, master_can: str = "can_master", follower_can: str = "can_slave"):
        super(PiperTeleopEnv, self).__init__()

        # --- 1. 硬件初始化 ---
        # 主臂 (Leader)
        self.cfg_m = create_agx_arm_config(robot="piper", channel=master_can)
        self.master = AgxArmFactory.create_arm(self.cfg_m)
        self.master.connect()
        self.master.set_leader_mode()
        self.master_eff = self.master.init_effector(self.master.OPTIONS.EFFECTOR.AGX_GRIPPER)

        # 从臂 (Follower)
        self.cfg_f = create_agx_arm_config(robot="piper", channel=follower_can)
        self.follower = AgxArmFactory.create_arm(self.cfg_f)
        self.follower.connect()
        self.follower.set_joint_angle_vel_limits(joint_index=255, max_joint_spd=3.0)
        self.follower.set_joint_acc_limits(joint_index=255, max_joint_acc=5.0)
        self.follower_eff = self.follower.init_effector(self.follower.OPTIONS.EFFECTOR.AGX_GRIPPER)

        # --- 2. 空间定义 (Spaces) ---
        self.action_space = spaces.Box(
            low=np.array([-np.pi] * 6 + [0.01], dtype=np.float32),
            high=np.array([np.pi] * 6 + [0.1], dtype=np.float32),
            shape=(7,),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32),
            "observation.images.wrist": spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8),
            "observation.images.front": spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8),
            "under_control": spaces.Discrete(2)
        })

        # --- 3. 状态与并发控制 ---
        self._lock = threading.Lock()
        self._master_running = True
        
        # 共享变量
        self._latest_master_state = np.zeros(7, dtype=np.float32)
        self._master_is_ok = False
        self.tele_enabled = False # 'T' 键切换的激活标志
        
        # 防抖机制变量
        self._last_ok_time = time.time()
        self._debounce_threshold = 0.5 # 0.5秒防抖

        # --- 4. 线程启动 ---
        # 后台硬件读取线程 (200Hz)
        self.read_thread = threading.Thread(target=self._master_read_thread, daemon=True)
        self.read_thread.start()

        # 键盘监听线程
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

        print(f"[Env] PiperTeleopEnv Ready. Press 'T' to toggle teleoperation.")

    def _on_press(self, key):
        """键盘监听回调：按下 T 键翻转遥操状态"""
        try:
            if key.char == 't' or key.char == 'T':
                self.tele_enabled = not self.tele_enabled
                status = "ENABLED" if self.tele_enabled else "DISABLED"
                print(f"\n[Env] Teleoperation Mode: {status}")
        except AttributeError:
            pass

    def _master_read_thread(self):
        """后台高频读取线程：200Hz"""
        while self._master_running:
            start_t = time.perf_counter()
            mja = self.master.get_leader_joint_angles()
            ok = self.master.is_ok()
            gcs = self.master_eff.get_gripper_ctrl_states()

            with self._lock:
                self._master_is_ok = ok
                if ok:
                    self._last_ok_time = time.time() # 更新最后一次通信正常的时间
                
                if mja is not None:
                    self._latest_master_state[:6] = mja.msg
                if gcs is not None:
                    # 夹爪控制映射逻辑：主臂宽度 -> 目标宽度
                    self._latest_master_state[6] = gcs.msg.width

            elapsed = time.perf_counter() - start_t
            time.sleep(max(0, 0.005 - elapsed))

    def _check_under_control(self) -> bool:
        """核心判定逻辑：检查当前是否处于接管状态（包含防抖）"""
        current_time = time.time()
        with self._lock:
            # 基础判定：开关开启且硬件 OK
            hardware_ok = self._master_is_ok
            # 防抖判定：如果硬件当前不 OK，但距离上次 OK 没超过 0.5s，依然视为 OK
            if not hardware_ok:
                if (current_time - self._last_ok_time) < self._debounce_threshold:
                    hardware_ok = True
            
            return self.tele_enabled and hardware_ok

    def _get_obs(self) -> Dict[str, Any]:
        """获取观测值字典"""
        # 获取从臂状态 (用于 observation.state)
        ja = self.follower.get_joint_angles()
        follower_angles = np.array(ja.msg if ja else [0.0]*6, dtype=np.float32)
        
        # 采集关节速度
        follower_vels = []
        for i in range(1, 7):
            ms = self.follower.get_motor_states(i)
            follower_vels.append(ms.msg.velocity if ms else 0.0)
        follower_vels = np.array(follower_vels, dtype=np.float32)
        
        # 采集法兰位姿
        fp = self.follower.get_flange_pose()
        follower_pose = np.array(fp.msg if fp else [0.0]*6, dtype=np.float32)
        
        # 采集夹爪宽度
        fg = self.follower_eff.get_gripper_status()
        follower_gripper = np.array([fg.msg.width if fg else 0.0], dtype=np.float32)

        # 拼接 19D 向量
        state_vector = np.concatenate([follower_angles, follower_vels, follower_pose, follower_gripper])
        
        # 判定是否受控
        under_control = 1 if self._check_under_control() else 0

        return {
            "observation.state": state_vector,
            "observation.images.wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            "observation.images.front": np.zeros((3, 224, 224), dtype=np.uint8),
            "under_control": under_control
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        执行环境步进逻辑。
        
        1. 检查接管：判断当前是人类遥操还是模型控制。
        2. 动作覆盖：若接管，提取主臂状态作为 actual_action 覆盖传入的 action。
        3. 硬件执行：下发指令至从臂。
        4. 信息反馈：在 info 中记录真实动作以供 LeRobot 保存。
        """
        # --- A. 接管判定与动作获取 ---
        is_intervened = self._check_under_control()
        executed_action = action.copy() # 默认执行传入动作

        if is_intervened:
            # 【核心逻辑】从共享锁中提取人类专家的实时动作
            with self._lock:
                expert_joints = self._latest_master_state[:6].copy()
                expert_gripper = self._latest_master_state[6]
            
            # 更新执行动作向量
            executed_action[:6] = expert_joints
            executed_action[6] = expert_gripper

        # --- B. 硬件控制下发 ---
        # 1. 关节控制 (带平滑/透传自适应切换)
        curr_ja = self.follower.get_joint_angles()
        if curr_ja is not None:
            # 计算目标与当前角度差异
            diff = np.sum(np.abs(executed_action[:6] - np.array(curr_ja.msg)))
            if diff > 0.7:
                # 差异大，使用 move_j (带梯形规划，防冲击)
                self.follower.move_j(executed_action[:6].tolist())
            else:
                # 差异小，使用 move_js (高频透传，低延迟)
                self.follower.move_js(executed_action[:6].tolist())
        
        # 2. 夹爪控制
        # 映射逻辑：夹爪反馈宽度 * 0.5 (取决于 Piper 底层协议比例), 限制在 [0.01, 0.1] 范围内
        target_width = np.clip(executed_action[6] * 0.5, 0.01, 0.1) if is_intervened else np.clip(executed_action[6],0.01,0.1) 
        self.follower_eff.move_gripper(target_width)

        # --- C. 封装返回值 ---
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        
        # info 字典极其关键，它决定了 LeRobot 最终保存到磁盘的数据集是否包含真实的专家动作
        info = {
            "actual_action": executed_action, # 这是该步真正被执行的 7D 动作
            "intervened": is_intervened      # 接管标记
        }
        
        return obs, reward, terminated, truncated, info

    def reset_to_state(self, joint_angles: np.ndarray, gripper_width: float, wait_time: float = 2.0):
        """
        平滑重置到指定姿态。
        用于轨迹重播前，将从臂从当前位置平滑移动到轨迹的第一帧。
        """
        print(f"[Env] 正在平滑移动至起始位姿... 预期耗时: {wait_time}s")
        # 强制使用 move_j 进行慢速平滑插值
        self.follower.move_j(joint_angles.tolist())
        self.follower_eff.move_gripper(gripper_width)
        # 等待机械臂到位
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
        if hasattr(self, 'read_thread'):
            self.read_thread.join(timeout=1.0)
        try:
            #self.follower.disable()
            #self.master.disable()
            pass
        except:
            pass
        super().close()


if __name__ == "__main__":
    # 模拟安全控制循环测试 (50Hz)
    # 建议使用: python piper_infra/Env/single_piper_env.py
    try:
        # 请确保 CAN 通道名称与你的硬件连接一致
        env = PiperTeleopEnv(master_can="can_master", follower_can="can_slave")
        obs, info = env.reset()
        
        print("\n" + "="*50)
        print("测试启动：安全模式 (Stay-Still Action)")
        print("1. 默认状态：从臂将维持在 reset 后的初始位置。")
        print("2. 遥操模式：按下键盘 'T' 键，主臂将接管控制。")
        print("3. 安全退出：按 Ctrl+C 终止程序。")
        print("="*50 + "\n")

        while True:
            # --- 安全动作生成 ---
            # 从当前的观测中提取从臂的真实关节角度 (index 0-5) 和 夹爪宽度 (index 18)
            # 这确保了在模型不介入时，下发给 step 的是一个“维持当前姿态”的动作，而非随机动作
            current_state = obs["observation.state"]
            safe_action = np.zeros(7, dtype=np.float32)
            safe_action[:6] = current_state[:6]   # 维持当前 6 个关节角度
            safe_action[6] = current_state[18]    # 维持当前夹爪宽度
            
            # 执行环境步进
            # 如果人类按下了 'T' 键，step 内部会自动将 safe_action 覆写为专家动作
            obs, reward, done, trunc, info = env.step(safe_action)
            
            # 状态打印
            if info['intervened']:
                print(f"\r[STATUS] 人类接管中 (HITL) | 专家关节1: {info['actual_action'][0]:.3f} rad", end="")
            else:
                print(f"\r[STATUS] 模型自主/维持模式 | 从臂信息: {current_state}")
            
            time.sleep(0.02) # 严格控制 50Hz (20ms) 循环频率

    except KeyboardInterrupt:
        print("\n\n[Test] 检测到中断，正在关闭环境...")
    except Exception as e:
        print(f"\n\n[Error] 运行出错: {e}")
    finally:
        env.close()
        print("[Test] 环境已安全退出。")
