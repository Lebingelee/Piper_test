import gymnasium as gym
import numpy as np
import threading
import time
from typing import Dict, Any, Tuple, Optional, List

class AbstractCameraWrapper(gym.Wrapper):
    """
    多视角异步视觉包装器 (抽象基类)。
    
    设计理念：
    1. 软同步 (Soft-sync)：相机采集频率 (30Hz) 与环境步进频率 (50Hz) 解耦。
    2. 防阻塞 (Non-blocking)：step 方法永远只拿最新的缓存帧。
    3. 线程安全：使用 threading.Lock 保护共享图像缓冲区。
    """

    def __init__(self, env: gym.Env, camera_names: List[str] = ["wrist", "front"]):
        super().__init__(env)
        self.camera_names = camera_names
        
        # --- 1. 图像状态锁与缓冲区 ---
        self._lock = threading.Lock()
        self._latest_frames: Dict[str, np.ndarray] = {}
        self._running = False
        self._threads: List[threading.Thread] = []

        # 初始化 Dummy 数据 (对齐 LeRobot 标准: C, H, W)
        for name in self.camera_names:
            # 默认填充 (3, 224, 224) 全零 uint8 
            self._latest_frames[name] = np.zeros((3, 224, 224), dtype=np.uint8)

    def start_cameras(self):
        """由子类实现或启动所有相机的后台读取线程"""
        if self._running:
            return
        self._running = True
        # 子类需具体实现 _camera_thread_loop

    def stop_cameras(self):
        """安全停止相机线程"""
        self._running = False
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []

    def _apply_vision_to_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """将最新的视觉数据注入到 obs/rgb 字典中"""
        # 确保 obs 结构存在
        if "rgb" not in obs:
            obs["rgb"] = {}
            
        with self._lock:
            for name in self.camera_names:
                obs["rgb"][name] = self._latest_frames[name].copy()
        
        return obs

    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """覆写 reset：在环境重置后注入视觉数据"""
        obs, info = self.env.reset(**kwargs)
        obs = self._apply_vision_to_obs(obs)
        return obs, info

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """覆写 step：在环境步进后注入视觉数据"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._apply_vision_to_obs(obs)
        return obs, reward, terminated, truncated, info

    def close(self):
        """确保相机线程随环境一起关闭"""
        self.stop_cameras()
        super().close()
