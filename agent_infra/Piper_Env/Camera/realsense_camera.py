import numpy as np
import cv2
import threading
import time
from typing import List, Dict, Tuple, Optional

try:
    import pyrealsense2 as rs
    HAS_PYREALSENSE2 = True
except ImportError:
    rs = None
    HAS_PYREALSENSE2 = False


def get_connected_realsense_serials() -> List[str]:
    """查询并返回所有已连接的 RealSense 相机序列号"""
    if not HAS_PYREALSENSE2:
        print("[RealSense] pyrealsense2 未安装，无法枚举设备。")
        return []

    context = rs.context()
    devices = context.query_devices()
    serials = []
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"检测到设备: {name}, 序列号: {serial}")
        serials.append(serial)
    return serials

class RealSenseCamera:
    """
    单台 RealSense 相机封装 - 异步线程版
    """
    def __init__(self, serial_number: str = None, width=640, height=480, fps=30, exposure: Optional[int] = None):
        self.serial_number = str(serial_number) if serial_number else None
        self.width = width
        self.height = height
        self.fps = fps
        self.exposure = exposure

        self.pipeline = None
        self.config = None
        self.profile = None
        self.align = None
        self.is_running = False
        self.thread = None
        self.frame_lock = threading.Lock()
        self.latest_color = None
        self.latest_depth = None

        if not HAS_PYREALSENSE2:
            print(
                f"[RealSense] pyrealsense2 未安装，设备 [{self.serial_number or 'Default'}] 将进入 fallback 模式。"
            )
            return
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        if self.serial_number:
            self.config.enable_device(self.serial_number)
        
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        try:
            self.profile = self.pipeline.start(self.config)
            
            # ======== 新增：曝光控制 ========
            if self.exposure is not None:
                color_sensor = self.profile.get_device().first_color_sensor()
                # 关闭自动曝光
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                # 设置固定曝光值 (越小越防抖，但画面越暗)
                color_sensor.set_option(rs.option.exposure, self.exposure)
                print(f"相机 [{self.serial_number or 'Default'}] 已应用手动曝光: {self.exposure}")
            # ===============================

            self.align = rs.align(rs.stream.color)
            self.is_running = True
            # 启动采集线程
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            print(f"相机 [{self.serial_number or 'Default'}] 异步采集已启动.")
        except Exception as e:
            print(f"相机 [{self.serial_number}] 启动失败: {e}")
            self.is_running = False

    def _update(self):
        """后台线程：不断抓取最新帧"""
        while self.is_running:
            try:
                # 这里的 timeout 设短一点，避免线程卡死
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # RealSense stream is configured as bgr8; expose RGB to wrappers/recorders.
                curr_color = cv2.cvtColor(
                    np.asanyarray(color_frame.get_data()),
                    cv2.COLOR_BGR2RGB,
                )
                curr_depth = np.asanyarray(depth_frame.get_data())

                # 更新缓冲区
                with self.frame_lock:
                    self.latest_color = curr_color.copy()
                    self.latest_depth = curr_depth.copy()
            
            except Exception as e:
                # 降低报错频率，避免刷屏
                # print(f"相机 [{self.serial_number}] 采集异常: {e}")
                time.sleep(0.01)

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """主线程调用：瞬间返回缓冲区内的最新帧"""
        with self.frame_lock:
            if self.latest_color is None:
                return None, None
            return self.latest_color, self.latest_depth

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except:
            pass

class RealSenseCameraGroup:
    """多相机集成管理类 - 异步模式"""
    def __init__(self, serial_numbers: List[str], width=640, height=480, fps=30, exposure: Optional[list] = None):
        self.serial_numbers = serial_numbers
        self.cameras: Dict[str, RealSenseCamera] = {}
        for idx, sn in enumerate(self.serial_numbers):
            camera_exposure = exposure[idx] if exposure and idx < len(exposure) else None
            self.cameras[sn] = RealSenseCamera(sn, width, height, fps, camera_exposure)
        
        # 等待相机稳定并获取首帧
        print("正在等待相机组预热 (2s)...")
        time.sleep(2.0)

    def get_all_frames(self) -> Dict[str, Dict[str, np.ndarray]]:
        """瞬间获取所有相机的最新帧"""
        results = {}
        for sn, cam in self.cameras.items():
            color, depth = cam.get_frames()
            results[sn] = {"color": color, "depth": depth}
        return results

    def stop_all(self):
        for cam in self.cameras.values():
            cam.stop()
        print("相机组已关闭。")
