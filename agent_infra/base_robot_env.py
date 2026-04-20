import gymnasium as gym
from abc import ABC, abstractmethod
import time
import numpy as np
from typing import Dict, Any, Optional, List

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
