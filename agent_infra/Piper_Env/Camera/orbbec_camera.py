import threading
import time
import subprocess
import contextlib
import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Config = None
Context = None
OBFormat = None
OBSensorType = None
Pipeline = None
HAS_PYORBBECSDK = None


def _load_pyorbbecsdk() -> bool:
    """Lazy-load pyorbbecsdk so importing this module never starts SDK probing."""
    global Config, Context, OBFormat, OBSensorType, Pipeline, HAS_PYORBBECSDK

    if HAS_PYORBBECSDK is not None:
        return HAS_PYORBBECSDK

    try:
        from pyorbbecsdk import (
            Config as _Config,
            Context as _Context,
            OBFormat as _OBFormat,
            OBSensorType as _OBSensorType,
            Pipeline as _Pipeline,
        )
    except ImportError:
        HAS_PYORBBECSDK = False
        return False

    Config = _Config
    Context = _Context
    OBFormat = _OBFormat
    OBSensorType = _OBSensorType
    Pipeline = _Pipeline
    HAS_PYORBBECSDK = True
    return True


def _device_list_count(device_list) -> int:
    if hasattr(device_list, "get_count"):
        return int(device_list.get_count())
    return len(device_list)


@contextlib.contextmanager
def _maybe_suppress_native_output(enabled: bool):
    if not enabled:
        yield
        return

    stdout_fd = 1
    stderr_fd = 2
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


def get_connected_orbbec_serials(suppress_native_logs: bool = True) -> List[str]:
    """查询并返回所有已连接的 Orbbec 相机序列号。

    枚举过程默认放在子进程中执行，避免 pyorbbecsdk/libusb 的 native
    错误直接终止主进程。
    """
    marker = "ORBBEC_SERIALS_JSON="
    child_code = r"""
import json
import os

marker = "ORBBEC_SERIALS_JSON="
try:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)

    from pyorbbecsdk import Context

    ctx = Context()
    device_list = ctx.query_devices()
    count = device_list.get_count() if hasattr(device_list, "get_count") else len(device_list)
    serials = []
    for idx in range(int(count)):
        device = device_list.get_device_by_index(idx)
        info = device.get_device_info()
        serials.append(
            {
                "name": info.get_name(),
                "serial": info.get_serial_number(),
            }
        )

    os.dup2(saved_stdout, 1)
    os.dup2(saved_stderr, 2)
    os.close(devnull_fd)
    os.close(saved_stdout)
    os.close(saved_stderr)
    print(marker + json.dumps(serials))
except BaseException as exc:
    try:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
    except Exception:
        pass
    print(marker + json.dumps({"error": repr(exc)}))
    raise
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", child_code],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except FileNotFoundError:
        print("[Orbbec] 无法找到当前 Python 解释器，无法枚举设备。")
        _print_orbbec_usb_fallback()
        return []
    except subprocess.TimeoutExpired:
        print("[Orbbec] SDK 枚举超时。")
        _print_orbbec_usb_fallback()
        return []

    if not suppress_native_logs:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")

    serials: List[str] = []
    payload = None
    for line in result.stdout.splitlines():
        if line.startswith(marker):
            try:
                payload = json.loads(line[len(marker):])
            except json.JSONDecodeError:
                payload = None

    if isinstance(payload, list):
        for item in payload:
            serial = item.get("serial")
            name = item.get("name")
            print(f"检测到 Orbbec 设备: {name}, 序列号: {serial}")
            serials.append(serial)
    elif isinstance(payload, dict) and "error" in payload:
        print(f"[Orbbec] SDK 枚举失败: {payload['error']}")
    elif result.returncode != 0:
        print(f"[Orbbec] SDK 枚举子进程退出码: {result.returncode}")

    if not serials:
        print("[Orbbec] SDK 未返回可用序列号。下面继续检查系统 USB 层是否能看到 Orbbec 设备。")
        _print_orbbec_usb_fallback()

    return serials


def get_orbbec_usb_devices() -> List[str]:
    """粗略枚举 Orbbec USB 设备，用于 SDK 枚举失败时诊断。"""
    sysfs_devices = _get_usb_devices_from_sysfs(include_ignored=False)
    if sysfs_devices:
        return sysfs_devices

    try:
        result = subprocess.run(
            ["lsusb"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    lines = []
    for line in result.stdout.splitlines():
        lower = line.lower()
        is_orbbec_named = "orbbec" in lower
        is_orbbec_vendor_candidate = "2bc5:" in lower
        is_known_can_adapter = "candlelight" in lower or "can adapter" in lower
        if is_orbbec_named or (is_orbbec_vendor_candidate and not is_known_can_adapter):
            lines.append(line.strip())
    return lines


def get_orbbec_video_devices() -> List[Dict[str, str]]:
    """Return Orbbec RGB /dev/video mappings discovered through sysfs."""
    video_root = Path("/sys/class/video4linux")
    if not video_root.exists():
        return []

    usb_devices = _get_usb_device_records(include_ignored=False)
    by_usb_path = {record["usb_path"]: record for record in usb_devices}
    results: List[Dict[str, str]] = []

    for video_dir in sorted(video_root.glob("video*"), key=lambda p: p.name):
        name = _read_sysfs_text(video_dir / "name")
        device_path = Path(os.path.realpath(video_dir / "device"))
        lower_name = name.lower()
        if "orbbec" not in lower_name:
            continue

        matched = None
        for usb_path, record in by_usb_path.items():
            try:
                device_path.relative_to(usb_path)
                matched = record
                break
            except ValueError:
                continue

        info = {
            "video": f"/dev/{video_dir.name}",
            "name": name,
        }
        if matched:
            info.update(
                {
                    "usb": matched["usb"],
                    "id": matched["id"],
                    "product": matched["product"],
                    "manufacturer": matched["manufacturer"],
                    "serial": matched.get("serial", ""),
                }
            )
        results.append(info)

    return results


def get_ignored_orbbec_usb_candidates() -> List[str]:
    """列出被 Orbbec USB fallback 忽略的已知非相机 2bc5 设备。"""
    sysfs_devices = _get_usb_devices_from_sysfs(include_ignored=True, ignored_only=True)
    if sysfs_devices:
        return sysfs_devices

    try:
        result = subprocess.run(
            ["lsusb"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    ignored = []
    for line in result.stdout.splitlines():
        lower = line.lower()
        if "2bc5:" in lower and ("candlelight" in lower or "can adapter" in lower):
            ignored.append(line.strip())
    return ignored


def _read_sysfs_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def _get_usb_devices_from_sysfs(
    include_ignored: bool = False,
    ignored_only: bool = False,
) -> List[str]:
    return [
        _format_usb_record(record)
        for record in _get_usb_device_records(
            include_ignored=include_ignored,
            ignored_only=ignored_only,
        )
    ]


def _get_usb_device_records(
    include_ignored: bool = False,
    ignored_only: bool = False,
) -> List[Dict[str, str]]:
    usb_root = Path("/sys/bus/usb/devices")
    if not usb_root.exists():
        return []

    devices = []
    for device_dir in sorted(usb_root.iterdir(), key=lambda p: p.name):
        vendor = _read_sysfs_text(device_dir / "idVendor").lower()
        product_id = _read_sysfs_text(device_dir / "idProduct").lower()
        product = _read_sysfs_text(device_dir / "product")
        manufacturer = _read_sysfs_text(device_dir / "manufacturer")
        serial = _read_sysfs_text(device_dir / "serial")

        lower_desc = f"{manufacturer} {product}".lower()
        is_orbbec_named = "orbbec" in lower_desc
        is_orbbec_vendor_candidate = vendor == "2bc5"
        is_known_can_adapter = "candlelight" in lower_desc or "can adapter" in lower_desc

        if ignored_only:
            if not (is_orbbec_vendor_candidate and is_known_can_adapter):
                continue
        else:
            if not (is_orbbec_named or (is_orbbec_vendor_candidate and not is_known_can_adapter)):
                continue
            if is_known_can_adapter and not include_ignored:
                continue

        record = {
            "usb": device_dir.name,
            "id": f"{vendor}:{product_id}",
            "product": product or "unknown",
            "manufacturer": manufacturer or "unknown",
            "serial": serial,
            "usb_path": str(device_dir.resolve()),
        }
        devices.append(record)

    return devices


def _format_usb_record(record: Dict[str, str]) -> str:
    parts = [
        f"usb={record['usb']}",
        f"id={record['id']}",
        f"product={record['product']}",
        f"manufacturer={record['manufacturer']}",
    ]
    if record.get("serial"):
        parts.append(f"serial={record['serial']}")
    return ", ".join(parts)


def _print_orbbec_usb_fallback():
    usb_devices = get_orbbec_usb_devices()
    ignored_devices = get_ignored_orbbec_usb_candidates()
    if usb_devices:
        print("[Orbbec] lsusb 可以看到疑似 Orbbec 设备，但 pyorbbecsdk 没有拿到可用序列号：")
        for item in usb_devices:
            print(f"  - {item}")
        print("[Orbbec] 建议检查 udev rules、USB3 线/口、设备供电，并重新插拔后重试。")
    else:
        print("[Orbbec] lsusb 没有看到明确的 Orbbec 相机设备。请先检查 USB 连接、供电和线材。")

    if ignored_devices:
        print("[Orbbec] 已忽略以下 2bc5 非相机设备，避免误判为 Orbbec：")
        for item in ignored_devices:
            print(f"  - {item}")


class OrbbecCamera:
    """
    单台 Orbbec 相机封装 - 异步线程版。

    说明：
    1. 优先使用 `pyorbbecsdk` 官方 Python SDK。
    2. SDK 不可用或枚举失败时，自动按 serial_number 走 V4L2 RGB fallback。
    3. 当前优先面向 RGB 采集，depth 先按 SDK 原始 uint16 缓冲读取。
    """

    def __init__(
        self,
        serial_number: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        exposure: Optional[int] = None,
        backend: str = "v4l2",
        fourcc: str = "MJPG",
        auto_exposure: bool = True,
        brightness: Optional[int] = None,
        gain: Optional[int] = None,
        enable_depth: bool = True,
    ):
        self.serial_number = str(serial_number) if serial_number else None
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.exposure = exposure
        self.preferred_backend = str(backend or "v4l2").lower()
        self.fourcc = str(fourcc or "MJPG").upper()
        self.auto_exposure = bool(auto_exposure)
        self.brightness = brightness
        self.gain = gain
        self.enable_depth = bool(enable_depth)

        self.pipeline = None
        self.config = None
        self.device = None
        self.capture = None
        self.backend = "none"
        self.is_running = False
        self.thread: Optional[threading.Thread] = None

        self.frame_lock = threading.Lock()
        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None

        if self.preferred_backend == "sdk":
            if self._try_start_sdk():
                return
            self._try_start_v4l2()
        elif self.preferred_backend == "auto":
            if self.enable_depth and self._try_start_sdk():
                return
            if self._try_start_v4l2():
                return
            self._try_start_sdk()
        elif self.preferred_backend == "v4l2":
            if self._try_start_v4l2():
                return
            self._try_start_sdk()
        else:
            raise ValueError(
                f"Unsupported Orbbec backend '{backend}'. "
                "Expected one of: v4l2, sdk, auto."
            )

    def _try_start_sdk(self) -> bool:
        if not _load_pyorbbecsdk():
            print(f"[Orbbec] pyorbbecsdk 未安装，尝试 V4L2 fallback: {self.serial_number or 'default'}")
            return False

        try:
            self.pipeline = self._create_pipeline()
            self.config = Config()

            color_profile = self._select_color_profile()
            depth_profile = self._select_depth_profile() if self.enable_depth else None

            if color_profile is not None:
                self.config.enable_stream(color_profile)
            if depth_profile is not None:
                self.config.enable_stream(depth_profile)

            self.pipeline.start(self.config)
            self.is_running = True
            self.backend = "sdk"
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            print(f"[Orbbec] 相机 [{self.serial_number or 'default'}] 异步采集已启动。")
            return True
        except Exception as exc:
            print(f"[Orbbec] SDK 启动失败，尝试 V4L2 fallback [{self.serial_number or 'default'}]: {exc}")
            self.is_running = False
            self.pipeline = None
            self.config = None
            self.device = None
            return False

    def _find_v4l2_device(self) -> Optional[str]:
        devices = get_orbbec_video_devices()
        if self.serial_number:
            for info in devices:
                if info.get("serial") == self.serial_number:
                    return info["video"]

        if devices:
            return devices[0]["video"]
        return None

    def _try_start_v4l2(self) -> bool:
        video_path = self._find_v4l2_device()
        if not video_path:
            print(f"[Orbbec] 未找到可用 V4L2 RGB 设备: {self.serial_number or 'default'}")
            return False

        capture = cv2.VideoCapture(video_path, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)

        if self.exposure is not None:
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            capture.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))
        elif self.auto_exposure:
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        if self.brightness is not None:
            capture.set(cv2.CAP_PROP_BRIGHTNESS, float(self.brightness))
        if self.gain is not None:
            capture.set(cv2.CAP_PROP_GAIN, float(self.gain))

        if not capture.isOpened():
            print(f"[Orbbec] V4L2 打开失败: {video_path}")
            capture.release()
            return False

        self.capture = capture
        self.backend = "v4l2"
        self.is_running = True
        self.thread = threading.Thread(target=self._update_v4l2, daemon=True)
        self.thread.start()
        print(
            f"[Orbbec] V4L2 相机 [{self.serial_number or 'default'}] "
            f"已启动: {video_path} "
            f"(format={self.fourcc}, auto_exposure={self.auto_exposure}, "
            f"exposure={self.exposure}, brightness={self.brightness}, gain={self.gain})"
        )
        return True

    def _create_pipeline(self):
        if not self.serial_number:
            return Pipeline()

        ctx = Context()
        device_list = ctx.query_devices()
        device = device_list.get_device_by_serial_number(self.serial_number)
        self.device = device
        return Pipeline(device)

    def _select_video_profile(self, sensor_type, fmt):
        profile_list = self.pipeline.get_stream_profile_list(sensor_type)

        try:
            return profile_list.get_video_stream_profile(
                self.width,
                self.height,
                fmt,
                self.fps,
            )
        except Exception:
            try:
                return profile_list.get_video_stream_profile(
                    0,
                    0,
                    fmt,
                    0,
                )
            except Exception:
                return profile_list.get_default_video_stream_profile()

    def _select_color_profile(self):
        return self._select_video_profile(OBSensorType.COLOR_SENSOR, OBFormat.RGB)

    def _select_depth_profile(self):
        return self._select_video_profile(OBSensorType.DEPTH_SENSOR, OBFormat.Y16)

    def _decode_color_frame(self, color_frame) -> Optional[np.ndarray]:
        if color_frame is None:
            return None

        width = color_frame.get_width()
        height = color_frame.get_height()
        frame_format = color_frame.get_format()
        frame_data = color_frame.get_data()
        format_name = getattr(frame_format, "name", str(frame_format)).upper()

        if "MJPG" in format_name or "JPEG" in format_name:
            encoded = np.frombuffer(frame_data, dtype=np.uint8)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                return None
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        color = np.frombuffer(frame_data, dtype=np.uint8)
        if color.size < width * height * 3:
            return None

        color = color[: width * height * 3].reshape((height, width, 3))

        if "BGR" in format_name:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        return color

    def _decode_depth_frame(self, depth_frame) -> Optional[np.ndarray]:
        if depth_frame is None:
            return None

        width = depth_frame.get_width()
        height = depth_frame.get_height()
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        if depth.size < width * height:
            return None
        return depth[: width * height].reshape((height, width))

    def _update(self):
        while self.is_running and self.pipeline is not None:
            try:
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                curr_color = self._decode_color_frame(color_frame)
                curr_depth = self._decode_depth_frame(depth_frame)

                if curr_color is None and curr_depth is None:
                    continue

                with self.frame_lock:
                    if curr_color is not None:
                        self.latest_color = curr_color.copy()
                    if curr_depth is not None:
                        self.latest_depth = curr_depth.copy()
            except Exception:
                time.sleep(0.01)

    def _update_v4l2(self):
        while self.is_running and self.capture is not None:
            ok, color = self.capture.read()
            if ok and color is not None:
                # OpenCV/V4L2 returns BGR; expose RGB to wrappers/recorders.
                color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                with self.frame_lock:
                    self.latest_color = color.copy()
            else:
                time.sleep(0.01)

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self.frame_lock:
            color = None if self.latest_color is None else self.latest_color.copy()
            depth = None if self.latest_depth is None else self.latest_depth.copy()
            return color, depth

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        try:
            if self.capture is not None:
                self.capture.release()
        except Exception:
            pass


class OrbbecCameraGroup:
    """多台 Orbbec 相机集成管理类。"""

    def __init__(
        self,
        serial_numbers: List[str],
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        exposure: Optional[List[int]] = None,
        backend: str = "v4l2",
        fourcc: str = "MJPG",
        auto_exposure: Optional[List[bool]] = None,
        brightness: Optional[List[int]] = None,
        gain: Optional[List[int]] = None,
        enable_depth: Optional[List[bool]] = None,
    ):
        self.serial_numbers = serial_numbers
        self.cameras: Dict[str, OrbbecCamera] = {}
        exposure = exposure or [None] * len(serial_numbers)
        auto_exposure = auto_exposure or [True] * len(serial_numbers)
        brightness = brightness or [None] * len(serial_numbers)
        gain = gain or [None] * len(serial_numbers)
        enable_depth = enable_depth or [True] * len(serial_numbers)
        self.backend = str(backend or "v4l2").lower()
        self.fourcc = str(fourcc or "MJPG").upper()

        for idx, sn in enumerate(self.serial_numbers):
            self.cameras[sn] = OrbbecCamera(
                serial_number=sn,
                width=width,
                height=height,
                fps=fps,
                exposure=exposure[idx] if idx < len(exposure) else None,
                backend=self.backend,
                fourcc=self.fourcc,
                auto_exposure=auto_exposure[idx] if idx < len(auto_exposure) else True,
                brightness=brightness[idx] if idx < len(brightness) else None,
                gain=gain[idx] if idx < len(gain) else None,
                enable_depth=enable_depth[idx] if idx < len(enable_depth) else True,
            )

        if self.cameras:
            print("正在等待 Orbbec 相机组预热 (2s)...")
            time.sleep(2.0)

    def get_all_frames(self) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        results: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        for sn, cam in self.cameras.items():
            color, depth = cam.get_frames()
            results[sn] = {"color": color, "depth": depth}
        return results

    def stop_all(self):
        for cam in self.cameras.values():
            cam.stop()
        print("Orbbec 相机组已关闭。")
