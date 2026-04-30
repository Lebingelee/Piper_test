import numpy as np
from typing import Dict, Any, List, Optional, Literal

import os
import time
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        self.default_init_wait_time = kwargs.get("init_wait_time", 0.50)
        self.parallel_reset = bool(kwargs.get("parallel_reset", False))

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

    def _begin_master_sync_reset(self):
        """Hook: 子类可在主臂同步复位前暂停后台线程。"""
        return

    def _end_master_sync_reset(self):
        """Hook: 子类可在主臂同步复位后恢复后台线程。"""
        return

    def _run_reset_jobs(self, jobs: Dict[str, Any]):
        if not jobs:
            return

        if (not self.parallel_reset) or len(jobs) <= 1:
            for name, fn in jobs.items():
                fn()
            return

        errors = []
        with ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="piper_reset") as executor:
            future_to_name = {
                executor.submit(fn): name
                for name, fn in jobs.items()
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    future.result()
                except Exception as exc:
                    errors.append(f"[{name}] {exc}")

        if errors:
            raise RuntimeError("并行复位存在失败项: " + " | ".join(errors))

    def reset_to_state(
        self,
        target_state: Dict[str, np.ndarray],
        wait_time: Optional[float] = None,
        sync_master: bool = False,
    ):
        actual_wait_time = self.default_init_wait_time if wait_time is None else wait_time
        print("[PiperBase] Reset: 机械臂组正在回归指定状态...")
        if sync_master:
            self._begin_master_sync_reset()
        try:
            jobs = {}
            for name, arm in self.arms.items():
                prefix = self._prefix(name)
                target_joint = target_state[f"{prefix}joint_pos"]
                target_gripper = target_state[f"{prefix}gripper_pos"]
                jobs[name] = (
                    lambda arm=arm, target_joint=target_joint, target_gripper=target_gripper:
                    arm.move_to_state(
                        target_joint,
                        target_gripper,
                        wait_time=actual_wait_time,
                        sync_master=sync_master,
                    )
                )
            self._run_reset_jobs(jobs)
        finally:
            if sync_master:
                self._end_master_sync_reset()

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
            if sync_master:
                self._begin_master_sync_reset()
            try:
                print("[PiperBase] Reset: 机械臂组正在回归初始位姿...")
                jobs = {
                    name: (
                        lambda arm=arm: arm.move_to_init(wait_time, sync_master=sync_master)
                    )
                    for name, arm in self.arms.items()
                }
                self._run_reset_jobs(jobs)
            finally:
                if sync_master:
                    self._end_master_sync_reset()

        return self._get_obs(), {"status": "reset_done"}

    def get_safe_action(self, state: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        if state is None:
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
            parallel_reset=common_cfg.get("parallel_reset", False),
            relative_pose_chunk_size=common_cfg.get("relative_pose_chunk_size", 8),
            **kwargs,
        )
        self._setup_teleop_meta_keys()

        self.tele_enabled = False
        self._lock = threading.Lock()
        self._master_running = True
        self._master_read_pause = False
        self._master_read_pause_depth = 0
        self._master_read_pause_lock = threading.Lock()
        self._teleop_before_reset = False
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
        self._master_read_hz = float(common_cfg.get("master_read_hz", 60.0))
        if self._master_read_hz <= 0.0:
            self._master_read_hz = 60.0
        self._master_read_dt = 1.0 / self._master_read_hz
        self.master_follow = bool(common_cfg.get("master_follow", False))
        self._sync_joint_threshold = float(common_cfg.get("sync_joint_threshold_rad", 0.12))
        self._sync_pose_threshold = float(common_cfg.get("sync_pose_threshold", 0.03))
        self._master_follow_warned = set()
        if self.master_follow:
            print(
                "[PiperEnv] Warning: master_follow 主臂主动跟随路径已临时禁用（安全回退）。"
                "当前将保持 master 分支稳定语义：仅 follower 跟随 leader。"
            )
        self._master_cache = {
            name: {
                "joint": np.zeros(7, dtype=np.float32),
                "pose": np.zeros(7, dtype=np.float32),
                "is_ok": False,
                "last_ok_time": time.time(),
                "has_leader_joint": False,
                "last_leader_joint_time": 0.0,
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
        return float(np.clip(float(width), 0.0, 0.2))

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
            if self._master_read_pause:
                time.sleep(0.005)
                continue
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
                            cache["has_leader_joint"] = True
                            cache["last_leader_joint_time"] = time.time()
                            cache["joint"][:6] = master_state["joint_pos"]
                            cache["pose"][:6] = master_state["ee_pose"]
                        else:
                            cache["has_leader_joint"] = False

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
            time.sleep(max(0.0, self._master_read_dt - elapsed))

    def _begin_master_sync_reset(self):
        with self._master_read_pause_lock:
            self._master_read_pause_depth += 1
            self._master_read_pause = True
            if self._master_read_pause_depth == 1:
                self._teleop_before_reset = bool(self.tele_enabled)
        if self.tele_enabled:
            print("[PiperEnv] Reset期间暂停Teleop输入。")
        self.tele_enabled = False
        print("[PiperEnv] 主臂读取线程已暂停，优先执行主臂复位。")
        time.sleep(0.03)

    def _end_master_sync_reset(self):
        restored_teleop = None
        with self._master_read_pause_lock:
            self._master_read_pause_depth = max(0, self._master_read_pause_depth - 1)
            if self._master_read_pause_depth == 0:
                self._master_read_pause = False
                restored_teleop = bool(self._teleop_before_reset)
                self._teleop_before_reset = False
        if restored_teleop is not None:
            self.tele_enabled = restored_teleop
            print(f"[PiperEnv] Reset后恢复Teleop状态: {'ON' if self.tele_enabled else 'OFF'}")
        print("[PiperEnv] 主臂读取线程已恢复。")

    def _check_under_control(self, name: str) -> bool:
        current_time = time.time()
        with self._lock:
            cache = self._master_cache[name]
            hardware_ok = cache["is_ok"] or (
                current_time - cache["last_ok_time"] < self._debounce_threshold
            )
            leader_joint_ready = cache.get("has_leader_joint", False) or (
                current_time - float(cache.get("last_leader_joint_time", 0.0))
                < self._debounce_threshold
            )
            return self.tele_enabled and hardware_ok and leader_joint_ready

    def _check_hardware_override(self, name: str) -> bool:
        """不依赖 tele_enabled 的硬件接管检测。"""
        current_time = time.time()
        with self._lock:
            cache = self._master_cache[name]
            hardware_ok = cache["is_ok"] or (
                current_time - cache["last_ok_time"] < self._debounce_threshold
            )
            leader_joint_ready = cache.get("has_leader_joint", False) or (
                current_time - float(cache.get("last_leader_joint_time", 0.0))
                < self._debounce_threshold
            )
            return hardware_ok and leader_joint_ready

    def _get_under_control_obs(self) -> Dict[str, np.ndarray]:
        return {
            name: np.array([self._check_under_control(name)], dtype=np.bool_)
            for name in self.arm_names
        }

    def _get_obs(self) -> Dict[str, Any]:
        obs = super()._get_obs()
        obs["under_control"] = self._get_under_control_obs()
        return obs

    def _is_follower_synced(self, name: str, expert_arm_action: np.ndarray) -> bool:
        follower_state = self.arms[name].get_state()
        if self.control_mode == "joint":
            err = np.max(np.abs(expert_arm_action - follower_state["joint_pos"]))
            return bool(err <= self._sync_joint_threshold)
        if self.control_mode == "pose":
            err = np.linalg.norm(expert_arm_action - follower_state["ee_pose"])
            return bool(err <= self._sync_pose_threshold)
        if self.control_mode == "delta_pose":
            err = np.linalg.norm(expert_arm_action)
            return bool(err <= self._sync_pose_threshold)

        if expert_arm_action.shape[0] >= 6:
            err = np.linalg.norm(expert_arm_action[:6])
            return bool(err <= self._sync_pose_threshold)
        return True

    def _mirror_follower_to_master(self, obs: Dict[str, Any], intervened: Dict[str, bool]):
        # 安全回退：暂时禁用主臂主动跟随，避免主臂锁住/异常位姿。
        for arm in self.arms.values():
            arm.end_master_follow()

    def step(self, action: Dict[str, np.ndarray]):
        policy_action = {k: v.copy() for k, v in action.items()}
        executed_action = {k: v.copy() for k, v in action.items()}
        intervened = {}
        action_source = {}
        syncing = {}

        for name in self.arm_names:
            prefix = self._prefix(name)
            is_intervened = self._check_under_control(name)
            intervened[name] = is_intervened
            if not is_intervened:
                action_source[name] = "policy"
                syncing[name] = False
                continue

            # 一旦检测到人工拖动，立即解除主臂跟随从臂关系，避免信息错位。
            self.arms[name].end_master_follow()
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

            is_syncing = not self._is_follower_synced(name, expert_val)
            syncing[name] = is_syncing
            action_source[name] = "sync" if is_syncing else "expert"
            executed_action[f"{prefix}arm"] = expert_val
            executed_action[f"{prefix}gripper"] = np.array([expert_gripper], dtype=np.float32)

        obs, reward, terminated, truncated, info = super().step(executed_action)
        self._mirror_follower_to_master(obs, intervened)
        info["actual_action"] = executed_action
        info["policy_action"] = policy_action
        info["intervened"] = intervened if len(self.arm_names) > 1 else intervened[self.arm_names[0]]
        info["intervened_map"] = intervened
        info["action_source"] = action_source if len(self.arm_names) > 1 else action_source[self.arm_names[0]]
        info["action_source_map"] = action_source
        info["syncing"] = syncing if len(self.arm_names) > 1 else syncing[self.arm_names[0]]
        info["syncing_map"] = syncing
        info["master_follow_enabled"] = self.master_follow
        action_type_map = {
            name: (0 if action_source[name] == "policy" else 1 if action_source[name] == "expert" else 2)
            for name in self.arm_names
        }
        info["action_type"] = action_type_map if len(self.arm_names) > 1 else action_type_map[self.arm_names[0]]
        info["action_type_map"] = action_type_map
        return obs, reward, terminated, truncated, info

    def close(self):
        self._master_running = False
        if hasattr(self, "listener"):
            self.listener.stop()
        if hasattr(self, "read_thread"):
            self.read_thread.join(timeout=1.0)
        for arm in self.arms.values():
            arm.end_master_follow()
        super().close()
