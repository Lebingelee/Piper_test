import numpy as np
import cv2
import time
import threading
from typing import List, Optional, Dict, Any

from agent_infra.Piper_Env.Env.utils.camera_wrapper import AbstractCameraWrapper
from agent_infra.Piper_Env.Camera.orbbec_camera import OrbbecCameraGroup
from agent_infra.Piper_Env.Camera.realsense_camera import RealSenseCameraGroup

class PiperCameraWrapper(AbstractCameraWrapper):
    """
    具体的 Piper 视觉包装器。
    1. 负责实例化本地的 RealSense / Orbbec 相机组。
    2. 后台线程采集 RGB / 可选 depth 图像并处理 (Resize, HWC->CHW)。
    3. 动态更新底层环境的 meta_keys，确保数据对齐。
    """
    def __init__(self, 
                 env, 
                 camera_sns: Optional[List[str]] = None,
                 crop_size: Optional[tuple] = None,
                 exposure: Optional[List[int]] = None,
                 source_resolution: Optional[tuple] = None):
        
        camera_cfg = getattr(env, "full_config", {}).get("cameras", {})
        nodes = camera_cfg.get("nodes", [])
        default_enable_depth = bool(camera_cfg.get("enable_depth", True))
        self._depth_missing_warned = set()
        self.camera_specs = []
        for node in nodes:
            role = node.get("role")
            if not role:
                continue

            fallback_color = str(node.get("fallback_color", "black")).lower()
            fallback_value = 255 if fallback_color == "white" else 0
            self.camera_specs.append({
                "role": role,
                "serial_number": node.get("serial_number"),
                "camera_type": str(node.get("camera_type", "realsense")).lower(),
                "camera_backend": str(node.get("camera_backend", "v4l2")).lower(),
                "camera_format": str(node.get("camera_format", "MJPG")).upper(),
                "exposure": node.get("exposure"),
                "auto_exposure": node.get("auto_exposure", True),
                "brightness": node.get("brightness"),
                "gain": node.get("gain"),
                "enable_depth": bool(node.get("enable_depth", default_enable_depth)),
                "depth_required": bool(node.get("depth_required", camera_cfg.get("depth_required", False))),
                "fallback_value": fallback_value,
            })

        if camera_sns is not None:
            sns_filter = set(camera_sns)
            self.camera_specs = [
                spec for spec in self.camera_specs
                if spec["serial_number"] in sns_filter
            ]
            if not self.camera_specs:
                self.camera_specs = [
                    {
                        "role": f"camera_{sn}",
                        "serial_number": sn,
                        "camera_type": "realsense",
                        "camera_backend": "sdk",
                        "camera_format": "MJPG",
                        "exposure": None,
                        "auto_exposure": True,
                        "brightness": None,
                        "gain": None,
                        "enable_depth": default_enable_depth,
                        "depth_required": False,
                        "fallback_value": 0,
                    }
                    for sn in camera_sns
                ]

        if crop_size is None:
            crop_size = tuple(camera_cfg.get("crop_resolution", (224, 224)))

        if source_resolution is None:
            source_resolution = tuple(camera_cfg.get("source_resolution", (640, 480)))

        self.camera_roles = [spec["role"] for spec in self.camera_specs]
        self.role_to_spec = {spec["role"]: spec for spec in self.camera_specs}
        self.active_camera_specs = [
            spec for spec in self.camera_specs if spec["serial_number"]
        ]
        self.depth_roles = [
            spec["role"] for spec in self.camera_specs if spec.get("enable_depth", False)
        ]
        
        super().__init__(env, camera_names=self.camera_roles)
        
        self.camera_sns = [spec["serial_number"] for spec in self.active_camera_specs]
        self.crop_size = crop_size
        self.source_resolution = source_resolution
        self.exposure = exposure or [
            spec.get("exposure") for spec in self.active_camera_specs
        ]
        self.camera_groups: Dict[str, Any] = {}
        self._latest_depth_frames: Dict[str, np.ndarray] = {}

        for role in self.camera_roles:
            fallback_frame = self._get_fallback_frame(role)
            self._latest_frames[role] = np.zeros(
                fallback_frame.shape,
                dtype=np.uint8,
            )
            self._latest_frames[role][:] = fallback_frame
            if role in self.depth_roles:
                self._latest_depth_frames[role] = self._get_fallback_depth_frame()

        # --- 动态更新底层 MetaKeys ---
        self._update_unwrapped_meta()
        if hasattr(self.env.unwrapped, "_setup_spaces"):
            self.env.unwrapped._setup_spaces()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def _update_unwrapped_meta(self):
        """将视觉维度注入到底层环境的 meta_keys 中"""
        if "rgb" not in self.env.unwrapped.meta_keys["obs"]:
            self.env.unwrapped.meta_keys["obs"]["rgb"] = {}
        
        for role in self.camera_roles:
            self.env.unwrapped.meta_keys["obs"]["rgb"][role] = (3, self.crop_size[1], self.crop_size[0])
        if self.depth_roles:
            if "depth" not in self.env.unwrapped.meta_keys["obs"]:
                self.env.unwrapped.meta_keys["obs"]["depth"] = {}
            for role in self.depth_roles:
                self.env.unwrapped.meta_keys["obs"]["depth"][role] = (1, self.crop_size[1], self.crop_size[0])

    def _get_fallback_frame(self, role: str) -> np.ndarray:
        spec = self.role_to_spec.get(role, {})
        fallback_value = spec.get("fallback_value", 0)
        return np.full(
            (3, self.crop_size[1], self.crop_size[0]),
            fallback_value,
            dtype=np.uint8,
        )

    def _get_fallback_depth_frame(self) -> np.ndarray:
        return np.zeros((1, self.crop_size[1], self.crop_size[0]), dtype=np.uint16)

    def _center_crop_and_resize(self, image: np.ndarray) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        target_w, target_h = self.crop_size

        if src_h <= 0 or src_w <= 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        src_ratio = src_w / src_h
        target_ratio = target_w / target_h

        if abs(src_ratio - target_ratio) > 1e-6:
            if src_ratio > target_ratio:
                cropped_w = max(1, int(round(src_h * target_ratio)))
                x0 = max(0, (src_w - cropped_w) // 2)
                image = image[:, x0:x0 + cropped_w]
            else:
                cropped_h = max(1, int(round(src_w / target_ratio)))
                y0 = max(0, (src_h - cropped_h) // 2)
                image = image[y0:y0 + cropped_h, :]

        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _center_crop_and_resize_depth(self, depth: np.ndarray) -> np.ndarray:
        src_h, src_w = depth.shape[:2]
        target_w, target_h = self.crop_size

        if src_h <= 0 or src_w <= 0:
            return np.zeros((target_h, target_w), dtype=np.uint16)

        src_ratio = src_w / src_h
        target_ratio = target_w / target_h

        if abs(src_ratio - target_ratio) > 1e-6:
            if src_ratio > target_ratio:
                cropped_w = max(1, int(round(src_h * target_ratio)))
                x0 = max(0, (src_w - cropped_w) // 2)
                depth = depth[:, x0:x0 + cropped_w]
            else:
                cropped_h = max(1, int(round(src_w / target_ratio)))
                y0 = max(0, (src_h - cropped_h) // 2)
                depth = depth[y0:y0 + cropped_h, :]

        return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    def _apply_vision_to_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if "rgb" not in obs:
            obs["rgb"] = {}
        if self.depth_roles and "depth" not in obs:
            obs["depth"] = {}

        with self._lock:
            for name in self.camera_names:
                obs["rgb"][name] = self._latest_frames[name].copy()
            for name in self.depth_roles:
                obs["depth"][name] = self._latest_depth_frames[name].copy()

        return obs

    def start_cameras(self):
        """启动硬件并开启后台线程"""
        if self._running:
            return
        
        try:
            self._start_camera_groups()
        except Exception as e:
            print(f"[PiperCamera] 相机硬件启动失败 (进入 Dummy 模式): {e}")
            self.camera_groups = {}

        self._running = True
        for spec in self.camera_specs:
            sn = spec.get("serial_number")
            role = spec["role"]
            camera_type = spec["camera_type"]
            t = threading.Thread(
                target=self._camera_thread_loop, 
                args=(sn, role, camera_type), 
                daemon=True
            )
            t.start()
            self._threads.append(t)

    def _split_specs_by_type(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped_specs: Dict[str, List[Dict[str, Any]]] = {}
        for spec in self.active_camera_specs:
            grouped_specs.setdefault(spec["camera_type"], []).append(spec)
        return grouped_specs

    def _exposure_map(self) -> Dict[str, Optional[int]]:
        exposure_map: Dict[str, Optional[int]] = {}
        for idx, spec in enumerate(self.active_camera_specs):
            exposure_map[spec["serial_number"]] = (
                self.exposure[idx] if idx < len(self.exposure) else None
            )
        return exposure_map

    def _camera_control_map(self, key: str) -> Dict[str, Any]:
        return {
            spec["serial_number"]: spec.get(key)
            for spec in self.active_camera_specs
        }

    def _start_camera_groups(self):
        grouped_specs = self._split_specs_by_type()
        exposure_map = self._exposure_map()

        if not grouped_specs:
            print("[PiperCamera] 未检测到可用相机配置，全部 role 进入 Dummy 模式。")
            self.camera_groups = {}
            return

        self.camera_groups = {}

        realsense_specs = grouped_specs.get("realsense", [])
        if realsense_specs:
            serials = [spec["serial_number"] for spec in realsense_specs]
            exposures = [exposure_map.get(sn) for sn in serials]
            print(f"[PiperCamera] 正在启动 RealSense 相机: {serials}")
            self.camera_groups["realsense"] = RealSenseCameraGroup(
                serials,
                width=self.source_resolution[0],
                height=self.source_resolution[1],
                exposure=exposures,
            )

        orbbec_specs = grouped_specs.get("orbbec", [])
        if orbbec_specs:
            serials = [spec["serial_number"] for spec in orbbec_specs]
            exposures = [exposure_map.get(sn) for sn in serials]
            auto_exposures = [self._camera_control_map("auto_exposure").get(sn) for sn in serials]
            brightness = [self._camera_control_map("brightness").get(sn) for sn in serials]
            gains = [self._camera_control_map("gain").get(sn) for sn in serials]
            enable_depth = [self._camera_control_map("enable_depth").get(sn) for sn in serials]
            backends = {spec.get("camera_backend", "v4l2") for spec in orbbec_specs}
            backend = next(iter(backends)) if len(backends) == 1 else "v4l2"
            formats = {spec.get("camera_format", "MJPG") for spec in orbbec_specs}
            camera_format = next(iter(formats)) if len(formats) == 1 else "MJPG"
            if len(backends) > 1:
                print(f"[PiperCamera] Orbbec backend 配置不一致 {backends}，默认使用 v4l2。")
            if len(formats) > 1:
                print(f"[PiperCamera] Orbbec format 配置不一致 {formats}，默认使用 MJPG。")
            print(
                f"[PiperCamera] 正在启动 Orbbec 相机: {serials} "
                f"(backend={backend}, format={camera_format})"
            )
            self.camera_groups["orbbec"] = OrbbecCameraGroup(
                serials,
                width=self.source_resolution[0],
                height=self.source_resolution[1],
                exposure=exposures,
                backend=backend,
                fourcc=camera_format,
                auto_exposure=auto_exposures,
                brightness=brightness,
                gain=gains,
                enable_depth=enable_depth,
            )

    def _get_camera_frames(
        self,
        camera_type: str,
        serial_number: Optional[str],
    ) -> Dict[str, Optional[np.ndarray]]:
        if not serial_number:
            return {"color": None, "depth": None}

        camera_group = self.camera_groups.get(camera_type)
        if camera_group is None:
            return {"color": None, "depth": None}

        all_frames = camera_group.get_all_frames()
        return all_frames.get(serial_number, {"color": None, "depth": None})

    def _camera_thread_loop(self, sn: str, role: str, camera_type: str):
        """采集循环：处理图像并存入缓存"""
        while self._running:
            start_t = time.perf_counter()
            
            frame_data = self._get_camera_frames(camera_type, sn)
            color_img = frame_data.get("color")
            depth_img = frame_data.get("depth")

            if color_img is not None:
                processed = self._center_crop_and_resize(color_img).transpose(2, 0, 1)
            else:
                processed = self._get_fallback_frame(role)

            if role in self.depth_roles:
                if depth_img is not None:
                    processed_depth = self._center_crop_and_resize_depth(depth_img)[None, ...]
                else:
                    processed_depth = self._get_fallback_depth_frame()
                    spec = self.role_to_spec.get(role, {})
                    if role not in self._depth_missing_warned:
                        level = "ERROR" if spec.get("depth_required", False) else "Warning"
                        print(
                            f"[PiperCamera] {level}: depth enabled for role [{role}], "
                            "but current camera backend returned no depth frame. "
                            "Using zero fallback; set enable_depth: false to disable this role."
                        )
                        self._depth_missing_warned.add(role)
            
            with self._lock:
                self._latest_frames[role] = processed
                if role in self.depth_roles:
                    self._latest_depth_frames[role] = processed_depth

            elapsed = time.perf_counter() - start_t
            time.sleep(max(0, 0.033 - elapsed)) # 维持约 30Hz

    def stop_cameras(self):
        super().stop_cameras()
        for camera_group in self.camera_groups.values():
            camera_group.stop_all()
        self.camera_groups = {}
