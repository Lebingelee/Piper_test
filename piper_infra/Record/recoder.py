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
    2. 防阻塞 (Non-blocking)：step 方法永远只拿最新的缓存帧，绝不等待相机 I/O。
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
        self._hz = 20

        # 初始化 Dummy 数据 (对齐 LeRobot 标准: C, H, W)
        # RGB: (3, 224, 224) uint8
        for name in self.camera_names:
            self._latest_frames[f"observation.images.{name}"] = np.zeros((3, 224, 224), dtype=np.uint8)

    def start_cameras(self):
        """启动所有相机的后台读取线程"""
        if self._running:
            return
        
        self._running = True
        for name in self.camera_names:
            t = threading.Thread(
                target=self._camera_thread_loop, 
                args=(name,), 
                name=f"CameraThread_{name}",
                daemon=True
            )
            t.start()
            self._threads.append(t)
        print(f"[CameraWrapper] Started {len(self.camera_names)} camera threads.")

    def stop_cameras(self):
        """安全停止相机线程"""
        self._running = False
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []
        print("[CameraWrapper] Camera threads stopped.")

    def _camera_thread_loop(self, camera_name: str):
        """
        后台相机采集主循环。
        当前为抽象占位实现，使用 time.sleep 模拟 30Hz 采样耗时。
        """
        key = f"observation.images.{camera_name}"
        while self._running:
            start_time = time.perf_counter()
            
            # --- 模拟采集逻辑 (实际硬件时在此调用 SDK) ---
            # 模拟 30Hz 采样耗时 (约 33ms)
            time.sleep(1/self._hz) 
            
            # 生成 Dummy 数据 (全零矩阵)
            # 在实际子类中，这里将替换为真实的 cv2.imread 或 RealSense 读取
            dummy_frame = np.zeros((3, 224, 224), dtype=np.uint8)
            
            # --- 原子更新缓冲区 ---
            with self._lock:
                self._latest_frames[key] = dummy_frame
            
            # 控制频率
            elapsed = time.perf_counter() - start_time
            # 维持采集循环频率

    def _apply_vision_to_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """将最新的视觉数据注入到 observation 字典中"""
        with self._lock:
            # 深拷贝最新的帧，防止外部修改影响缓冲区
            vision_update = {k: v.copy() for k, v in self._latest_frames.items()}
        
        obs.update(vision_update)
        return obs

    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """覆写 reset：在环境重置后注入视觉数据"""
        obs, info = self.env.reset(**kwargs)
        obs = self._apply_vision_to_obs(obs)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """覆写 step：在环境步进后注入视觉数据"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._apply_vision_to_obs(obs)
        return obs, reward, terminated, truncated, info

    def close(self):
        """确保相机线程随环境一起关闭"""
        self.stop_cameras()
        super().close()
