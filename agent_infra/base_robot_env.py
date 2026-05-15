from abc import ABC, abstractmethod
import time
import numpy as np
from typing import Dict, Any, Optional, List

import gymnasium as gym


ENV_STATE_ALIASES = {
    "teleop": "teleop",
    "tele_enabled": "teleop",
    "teleop_enabled": "teleop",
    "passive": "passive",
    "action_dispatch": "passive",
    "master_follow": "master_follow",
    "master_follow_enabled": "master_follow",
}

class BaseRobotEnv(gym.Env, ABC):
    """
    机器人环境抽象基类 (BaseRobotEnv)。
    为所有真实/仿真机器人环境提供统一接口，确保与 MetadataAdapterWrapper 兼容。
    """
    def __init__(self, hz: int = 10, **kwargs):
        super().__init__()
        self.hz = hz
        self.dt = 1.0 / hz
        self.last_step_time = time.time()
        
        # 核心：元数据字典，用于描述 obs 和 action 的结构
        # 必须由子类在初始化完成前填充
        self.meta_keys = {
            "obs": {},
            "action": {}
        }
        
        # 状态标志
        self.is_setup = False

    @staticmethod
    def parse_bool_mode(value: Any) -> bool:
        """Parse a user-facing boolean mode used by switch_* interfaces."""
        if isinstance(value, bool):
            return value
        if value is None:
            raise ValueError("Boolean mode cannot be None.")
        text = str(value).strip().lower()
        if text in ("true", "1", "on", "yes", "y", "enable", "enabled"):
            return True
        if text in ("false", "0", "off", "no", "n", "disable", "disabled"):
            return False
        raise ValueError(
            f"Unsupported boolean mode '{value}'. "
            "Expected true/false, 1/0, on/off, yes/no, or enable/disable."
        )

    @classmethod
    def resolve_switch_mode(cls, mode: Any, current: bool) -> bool:
        """Resolve switch mode. No argument, None, or 'toggle' means invert."""
        if mode is None:
            return not bool(current)
        text = str(mode).strip().lower()
        if text in ("", "toggle"):
            return not bool(current)
        return cls.parse_bool_mode(mode)

    def get_env_state(self, key: Optional[str] = None) -> Any:
        """
        Read environment-owned state through one public interface.

        Canonical keys:
        - "teleop": whether human teleoperation is active.
        - "passive": whether hardware action dispatch is enabled.
        - "master_follow": whether master arms should follow policy-controlled followers.
        """
        state = {}
        if hasattr(self, "tele_enabled"):
            state["teleop"] = bool(getattr(self, "tele_enabled"))
        if hasattr(self, "passive"):
            state["passive"] = bool(getattr(self, "passive"))
        if hasattr(self, "master_follow"):
            state["master_follow"] = bool(getattr(self, "master_follow"))

        if key is None:
            return state

        canonical_key = ENV_STATE_ALIASES.get(str(key), str(key))
        if canonical_key not in state:
            raise KeyError(f"Environment state '{key}' is not available.")
        return state[canonical_key]

    def switch_passive(self, mode: Any = "toggle") -> bool:
        """Switch hardware action dispatch state when the environment supports it."""
        if not hasattr(self, "passive"):
            raise NotImplementedError(f"{type(self).__name__} does not expose passive state.")
        self.passive = self.resolve_switch_mode(mode, bool(getattr(self, "passive")))
        return bool(self.passive)

    def switch_tele(self, mode: Any = "toggle") -> bool:
        """Switch human teleoperation state when the environment supports it."""
        if not hasattr(self, "tele_enabled"):
            raise NotImplementedError(f"{type(self).__name__} does not expose teleop state.")
        self.tele_enabled = self.resolve_switch_mode(
            mode,
            bool(getattr(self, "tele_enabled")),
        )
        return bool(self.tele_enabled)

    def switch_master_follow(self, mode: Any = "toggle") -> bool:
        """Switch master-follow state when the environment supports it."""
        if not hasattr(self, "master_follow"):
            raise NotImplementedError(
                f"{type(self).__name__} does not expose master_follow state."
            )
        self.master_follow = self.resolve_switch_mode(
            mode,
            bool(getattr(self, "master_follow")),
        )
        return bool(self.master_follow)

    @abstractmethod
    def _setup_hardware(self):
        """初始化硬件连接（机械臂、相机、传感器等）"""
        pass

    @abstractmethod
    def _get_obs(self) -> Dict[str, Any]:
        """从硬件获取原始观测数据（通常返回嵌套字典）"""
        pass

    @abstractmethod
    def _apply_action(self, action: Dict[str, np.ndarray]):
        """将动作指令下发至硬件"""
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """标准 Gym Reset 逻辑"""
        super().reset(seed=seed)
        if not self.is_setup:
            self._setup_hardware()
            self.is_setup = True
        
        # 子类应在 reset 中实现回到初始位姿的逻辑
        # 这里返回初始观测
        return self._get_obs(), {}

    def step(self, action: Dict[str, np.ndarray]):
        """标准 Gym Step 逻辑，包含严格的频率控制"""
        # 1. 下发动作
        self._apply_action(action)
        
        # 2. 频率控制 (维持稳定的 Control Loop)
        now = time.time()
        elapsed = now - self.last_step_time
        sleep_time = self.dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_step_time = time.time()
        
        # 3. 获取观测
        obs = self._get_obs()
        
        # 在真实机器人环境中，reward/terminated 逻辑通常由 Wrapper 或外部定义
        return obs, 0.0, False, False, {}

    @abstractmethod
    def close(self):
        """释放硬件资源"""
        pass

    def get_safe_action(self) -> Dict[str, np.ndarray]:
        """
        [可选] 获取安全动作（如保持当前位姿）。
        建议子类实现，用于紧急制动或初始化。
        """
        raise NotImplementedError

    def get_env_metadata(self) -> Dict[str, Any]:
        return {
            "obs": self.meta_keys.get("obs", {}),
            "action": self.meta_keys.get("action", {}),
            "env_control_mode": getattr(self, "env_control_mode", ""),
            "controller_backend": getattr(self, "controller_backend", ""),
            "control": getattr(self, "control_meta", {}),
        }

    def get_control_state(self) -> Dict[str, Any]:
        obs = self._get_obs()
        state = obs.get("state", {}) if isinstance(obs, dict) else {}
        control_meta = dict(getattr(self, "control_meta", {}) or {})
        arm_entries = list(control_meta.get("arm_entries", []) or [])
        arm_poses = {}
        for entry in arm_entries:
            if not isinstance(entry, dict):
                continue
            action_key = entry.get("action_key")
            state_pose_key = entry.get("state_pose_key")
            if not action_key or not state_pose_key or state_pose_key not in state:
                continue
            arm_poses[str(action_key)] = np.asarray(state[state_pose_key], dtype=np.float32).reshape(-1)

        result: Dict[str, Any] = {"arm_poses": arm_poses}
        if len(arm_poses) == 1:
            result["arm_pose"] = next(iter(arm_poses.values()))
        return result
