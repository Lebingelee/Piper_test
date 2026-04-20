import numpy as np
from typing import Dict, Any, List, Optional, Literal

import os
import time
import yaml
import threading

from pynput import keyboard



from agent_infra.base_robot_env import BaseRobotEnv
from agent_infra.Piper_Env.Env.utils.piper_arm import PiperArm


PIPER_CONTROL_MODES = ("joint", "pose", "delta_pose", "relative_pose_chunk")


class PiperBaseEnv(BaseRobotEnv):
    """
    Piper 环境基础类。
    采用与 Realman 一致的 prefix 化设计，便于后续扩展到双臂。
    """

    def __init__(
        self,
        arm_names: List[str],
        robot_configs: Dict[str, Dict[str, Any]],
        hz: int = 10,
        control_mode: str = "joint",
        **kwargs,
    ):
        super().__init__(hz=hz)
        self.arm_names = arm_names
        self.robot_configs = robot_configs
        self._validate_control_mode(control_mode)
        self.control_mode = control_mode
        self.relative_pose_chunk_size = int(kwargs.get("relative_pose_chunk_size", 8))
        self.default_init_joint_pos = kwargs.get("init_joint_pos", [0.0] * 6)
        self.default_init_gripper_pos = kwargs.get("init_gripper_pos", 0.05)
        self.default_init_wait_time = kwargs.get("init_wait_time", 3.0)

        self._setup_meta_keys()

        self.arms: Dict[str, PiperArm] = {}
        self.arm: Optional[PiperArm] = None

    @staticmethod
    def _validate_control_mode(control_mode: str):
        if control_mode not in PIPER_CONTROL_MODES:
            raise ValueError(
                f"Unsupported Piper control_mode '{control_mode}'. "
                f"Expected one of: {', '.join(PIPER_CONTROL_MODES)}."
            )

    def _prefix(self, name: str) -> str:
        return f"{name}_" if len(self.arm_names) > 1 else ""

    def _setup_meta_keys(self):
        self.meta_keys["obs"]["state"] = {}
        self.meta_keys["action"] = {}

        arm_dim = 6
        if self.control_mode == "relative_pose_chunk":
            arm_dim = self.relative_pose_chunk_size * 6
        for name in self.arm_names:
            prefix = self._prefix(name)
            self.meta_keys["obs"]["state"].update(
                {
                    f"{prefix}joint_pos": (6,),
                    f"{prefix}joint_vel": (6,),
                    f"{prefix}ee_pose": (6,),
                    f"{prefix}gripper_pos": (1,),
                }
            )
            self.meta_keys["action"].update(
                {
                    f"{prefix}arm": (arm_dim,),
                    f"{prefix}gripper": (1,),
                }
            )

    def _setup_hardware(self):
        for name in self.arm_names:
            spec_cfg = self.robot_configs.get(name, {})
            master_can = spec_cfg.get("master_can", "can_master")
            follower_can = spec_cfg.get("follower_can", "can_slave")
            init_joint_pos = spec_cfg.get("init_joint_pos", self.default_init_joint_pos)
            init_gripper_pos = spec_cfg.get("init_gripper_pos", self.default_init_gripper_pos)

            print(
                f"[PiperBase] 正在初始化机械臂 [{name}] "
                f"(Master: {master_can}, Follower: {follower_can})..."
            )
            arm = PiperArm(
                master_can=master_can,
                follower_can=follower_can,
                name=name,
                init_joint_pos=init_joint_pos,
                init_gripper_pos=init_gripper_pos,
            )
            arm.connect()
            self.arms[name] = arm

        if len(self.arm_names) == 1:
            self.arm = self.arms[self.arm_names[0]]

    def _get_obs(self) -> Dict[str, Any]:
        combined_state = {}
        for name, arm in self.arms.items():
            prefix = self._prefix(name)
            state = arm.get_state()
            for key, value in state.items():
                combined_state[f"{prefix}{key}"] = value

        return {"state": combined_state}

    def _apply_action(self, action: Dict[str, np.ndarray]):
        for name, arm in self.arms.items():
            prefix = self._prefix(name)
            arm_act = action[f"{prefix}arm"]
            grip_act = action[f"{prefix}gripper"][0]
            arm.apply_action(arm_act, grip_act, mode=self.control_mode)

    def reset_to_state(
        self,
        target_state: Dict[str, np.ndarray],
        wait_time: Optional[float] = None,
        sync_master: bool = False,
    ):
        actual_wait_time = self.default_init_wait_time if wait_time is None else wait_time
        print("[PiperBase] Reset: 机械臂组正在回归指定状态...")
        for name, arm in self.arms.items():
            prefix = self._prefix(name)
            arm.move_to_state(
                target_state[f"{prefix}joint_pos"],
                target_state[f"{prefix}gripper_pos"],
                wait_time=actual_wait_time,
                sync_master=sync_master,
            )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if not self.is_setup:
            self._setup_hardware()
            self.is_setup = True

        options = options or {}
        target_state = options.get("target_state")
        wait_time = options.get("wait_time", self.default_init_wait_time)
        sync_master = bool(options.get("sync_master", False))

        if target_state is not None:
            self.reset_to_state(target_state, wait_time=wait_time, sync_master=sync_master)
        else:
            print("[PiperBase] Reset: 机械臂组正在回归初始位姿...")
            for arm in self.arms.values():
                arm.move_to_init(wait_time, sync_master=sync_master)

        return self._get_obs(), {"status": "reset_done"}

    def get_safe_action(self) -> Dict[str, np.ndarray]:
        obs = self._get_obs()
        state = obs["state"]
        safe_action = {}
        for name in self.arm_names:
            prefix = self._prefix(name)
            if self.control_mode == "joint":
                arm_action = state[f"{prefix}joint_pos"]
            elif self.control_mode == "pose":
                arm_action = state[f"{prefix}ee_pose"]
            elif self.control_mode == "delta_pose":
                arm_action = np.zeros(6, dtype=np.float32)
            else:
                arm_action = np.zeros(self.relative_pose_chunk_size * 6, dtype=np.float32)
            safe_action[f"{prefix}arm"] = arm_action.copy()
            safe_action[f"{prefix}gripper"] = state[f"{prefix}gripper_pos"].copy()
        return safe_action

    def close(self):
        for arm in self.arms.values():
            arm.close()
        self.arms = {}
        self.arm = None
        self.is_setup = False



class PiperEnv(PiperBaseEnv):
    """
    Piper 机械臂 teleop 核心环境类。
    在 PiperBaseEnv 之上增加 leader 读取、专家介入和 under_control 观测。
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        control_mode: Optional[Literal["joint", "pose"]] = None,
        hz: Optional[int] = None,
        robot_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ):
        self.config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "Config",
        )
        full_config_path = self._resolve_config_path(config_path)

        with open(full_config_path, "r", encoding="utf-8") as f:
            self.full_config = yaml.safe_load(f)

        robot_cfg = self.full_config.get("robots", {})
        if robot_overrides:
            robot_cfg = self._merge_robot_overrides(robot_cfg, robot_overrides)
            self.full_config["robots"] = robot_cfg

        common_cfg = self.full_config.get("common", {})
        arm_names = list(robot_cfg.keys()) or ["arm"]

        self.control_mode = control_mode or common_cfg.get("default_control_mode", "joint")
        self.hz = hz or common_cfg.get("default_hz", 10)

        super().__init__(
            arm_names=arm_names,
            robot_configs=robot_cfg,
            control_mode=self.control_mode,
            hz=self.hz,
            init_wait_time=common_cfg.get("init_wait_time", 3.0),
            relative_pose_chunk_size=common_cfg.get("relative_pose_chunk_size", 8),
            **kwargs,
        )
        self._setup_teleop_meta_keys()

        self.tele_enabled = False
        self._lock = threading.Lock()
        self._master_running = True
        self._debounce_threshold = common_cfg.get("teleop_debounce_sec", 0.5)
        self._master_gripper_jump_threshold = float(
            common_cfg.get("master_gripper_jump_threshold_m", 0.04)
        )
        self._master_gripper_confirm_frames = int(
            common_cfg.get("master_gripper_confirm_frames", 6)
        )
        self._master_gripper_zero_epsilon = float(
            common_cfg.get("master_gripper_zero_epsilon_m", 1e-3)
        )
        self._master_gripper_zero_guard_threshold = float(
            common_cfg.get("master_gripper_zero_guard_threshold_m", 0.03)
        )
        self._master_cache = {
            name: {
                "joint": np.zeros(7, dtype=np.float32),
                "pose": np.zeros(7, dtype=np.float32),
                "is_ok": False,
                "last_ok_time": time.time(),
                "gripper_valid": False,
                "gripper_candidate": np.nan,
                "gripper_candidate_count": 0,
            }
            for name in self.arm_names
        }

        self._setup_hardware()
        self.is_setup = True
        self._initialize_master_gripper_cache_from_followers()

        self.read_thread = threading.Thread(target=self._master_read_thread, daemon=True)
        self.read_thread.start()
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

        print(
            f"[PiperEnv] Robot Core Ready. Arms: {self.arm_names}, "
            f"Mode: {self.control_mode}, Hz: {self.hz}"
        )

    @staticmethod
    def _merge_robot_overrides(
        robot_cfg: Dict[str, Dict[str, Any]],
        robot_overrides: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        merged = {
            name: dict(cfg or {})
            for name, cfg in robot_cfg.items()
        }

        for name, override in robot_overrides.items():
            merged.setdefault(name, {})
            merged[name].update(override or {})

        return merged

    def _setup_teleop_meta_keys(self):
        self.meta_keys["obs"]["under_control"] = {
            name: (1,) for name in self.arm_names
        }

    def _initialize_master_gripper_cache_from_followers(self):
        for name, arm in self.arms.items():
            try:
                state = arm.get_state()
                gripper_width = self._clip_gripper_width(state["gripper_pos"][0])
                with self._lock:
                    self._master_cache[name]["joint"][6] = gripper_width
                    self._master_cache[name]["pose"][6] = gripper_width
            except Exception as exc:
                print(f"[PiperEnv] 初始化主臂夹爪缓存失败 [{name}]: {exc}")

    def _resolve_config_path(self, config_path: Optional[str]) -> str:
        default_path = os.path.join(self.config_dir, "piper_config.yaml")
        if config_path is None:
            return default_path

        if os.path.isabs(config_path):
            return config_path

        cwd_candidate = os.path.abspath(config_path)
        if os.path.exists(cwd_candidate):
            return cwd_candidate

        config_candidate = os.path.join(self.config_dir, config_path)
        if os.path.exists(config_candidate):
            return config_candidate

        raise FileNotFoundError(
            "Config file not found. Checked paths: "
            f"{cwd_candidate} and {config_candidate}"
        )

    def _on_press(self, key):
        try:
            if key.char in ("t", "T"):
                self.tele_enabled = not self.tele_enabled
                print(f"\n[PiperEnv] Teleoperation Toggle: {'ON' if self.tele_enabled else 'OFF'}")
        except AttributeError:
            pass

    @staticmethod
    def _clip_gripper_width(width: float) -> float:
        return float(np.clip(float(width), 0.0, 0.1))

    def _filter_master_gripper(self, cache: Dict[str, Any], raw_width: float) -> float:
        width = self._clip_gripper_width(raw_width)
        current = float(cache["joint"][6])

        # Some leader gripper reads may briefly report exact zero while the
        # physical leader gripper is open and idle. Treat a direct open->0 jump
        # as a missing/stale read; a real close should pass through intermediate
        # widths and will then be accepted normally.
        if (
            width <= self._master_gripper_zero_epsilon
            and current >= self._master_gripper_zero_guard_threshold
        ):
            cache["gripper_candidate"] = np.nan
            cache["gripper_candidate_count"] = 0
            return current

        if not cache["gripper_valid"]:
            cache["gripper_valid"] = True
            cache["gripper_candidate"] = np.nan
            cache["gripper_candidate_count"] = 0
            return width

        if abs(width - current) <= self._master_gripper_jump_threshold:
            cache["gripper_candidate"] = np.nan
            cache["gripper_candidate_count"] = 0
            return width

        candidate = cache["gripper_candidate"]
        if np.isfinite(candidate) and abs(width - float(candidate)) <= 1e-3:
            cache["gripper_candidate_count"] += 1
        else:
            cache["gripper_candidate"] = width
            cache["gripper_candidate_count"] = 1

        if cache["gripper_candidate_count"] >= self._master_gripper_confirm_frames:
            cache["gripper_candidate"] = np.nan
            cache["gripper_candidate_count"] = 0
            return width

        return current

    def _master_read_thread(self):
        while self._master_running:
            start_t = time.perf_counter()
            try:
                for name, arm in self.arms.items():
                    master_state = arm.get_master_state()
                    with self._lock:
                        cache = self._master_cache[name]
                        cache["is_ok"] =master_state["is_ok"] 
                        
                        if cache["is_ok"]:
                            cache["last_ok_time"] = time.time()

                        if master_state.get("has_leader_joint", False):
                            cache["joint"][:6] = master_state["joint_pos"]
                            cache["pose"][:6] = master_state["ee_pose"]

                        if master_state.get("has_gripper", False):
                            gripper_width = self._filter_master_gripper(
                                cache,
                                master_state["gripper_pos"][0],
                            )
                            cache["joint"][6] = gripper_width
                            cache["pose"][6] = gripper_width
            except Exception as exc:
                print(f"\n[PiperEnv] 主臂读取线程异常: {exc}\n")

            elapsed = time.perf_counter() - start_t
            time.sleep(max(0.0, 0.005 - elapsed))

    def _check_under_control(self, name: str) -> bool:
        current_time = time.time()
        with self._lock:
            cache = self._master_cache[name]
            hardware_ok = cache["is_ok"] or (
                current_time - cache["last_ok_time"] < self._debounce_threshold
            )
            #print(hardware_ok)
            return self.tele_enabled and hardware_ok

    def _get_under_control_obs(self) -> Dict[str, np.ndarray]:
        return {
            name: np.array([self._check_under_control(name)], dtype=np.bool_)
            for name in self.arm_names
        }

    def _get_obs(self) -> Dict[str, Any]:
        obs = super()._get_obs()
        obs["under_control"] = self._get_under_control_obs()
        return obs

    def step(self, action: Dict[str, np.ndarray]):
        executed_action = {k: v.copy() for k, v in action.items()}
        intervened = {}

        for name in self.arm_names:
            prefix = self._prefix(name)
            is_intervened = self._check_under_control(name)
            intervened[name] = is_intervened
            if not is_intervened:
                continue

            with self._lock:
                cache = self._master_cache[name]
                if self.control_mode == "joint":
                    source = cache["joint"]
                    expert_val = source[:6].copy()
                else:
                    source = cache["pose"]
                    expert_pose = source[:6].copy()
                    if self.control_mode == "pose":
                        expert_val = expert_pose
                    else:
                        current_pose = self.arms[name].get_state()["ee_pose"]
                        delta_pose = expert_pose - current_pose
                        if self.control_mode == "delta_pose":
                            expert_val = delta_pose.astype(np.float32)
                        else:
                            expert_val = np.zeros(
                                self.relative_pose_chunk_size * 6,
                                dtype=np.float32,
                            )
                            expert_val[:6] = delta_pose.astype(np.float32)
                if cache.get("gripper_valid", False):
                    expert_gripper = self._clip_gripper_width(source[6])
                else:
                    expert_gripper = self._clip_gripper_width(
                        self.arms[name].last_follower_gripper_cmd
                    )

            executed_action[f"{prefix}arm"] = expert_val
            executed_action[f"{prefix}gripper"] = np.array([expert_gripper], dtype=np.float32)

        obs, reward, terminated, truncated, info = super().step(executed_action)
        info["actual_action"] = executed_action
        info["intervened"] = intervened if len(self.arm_names) > 1 else intervened[self.arm_names[0]]
        return obs, reward, terminated, truncated, info

    def close(self):
        self._master_running = False
        if hasattr(self, "listener"):
            self.listener.stop()
        if hasattr(self, "read_thread"):
            self.read_thread.join(timeout=1.0)
        super().close()
