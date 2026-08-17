import time
import datetime
import os
import numpy as np
import cv2
import h5py
import json
import torch
import threading
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Literal

# 尝试导入 LeRobot
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False

# --- 1. 录制器抽象基类 ---

class BaseRecorder(ABC):
    @abstractmethod
    def start_episode(self, episode_id: int):
        pass

    @abstractmethod
    def add_frame(self, obs: Dict[str, Any], action: Dict[str, Any], info: Dict[str, Any]):
        pass

    @abstractmethod
    def end_episode(self, success: bool):
        pass

    def discard_episode(self):
        if hasattr(self, "current_episode_data"):
            self.current_episode_data = []

        dataset = getattr(self, "dataset", None)
        if dataset is None:
            return

        clear_buffer = getattr(dataset, "clear_episode_buffer", None)
        if callable(clear_buffer):
            clear_buffer()
            return

        episode_buffer = getattr(dataset, "episode_buffer", None)
        if isinstance(episode_buffer, dict):
            for value in episode_buffer.values():
                if hasattr(value, "clear"):
                    value.clear()
        elif hasattr(episode_buffer, "clear"):
            episode_buffer.clear()

    @abstractmethod
    def finalize(self):
        pass

# --- 2. H5 录制实现 (每条轨迹独立文件) ---

class H5TrajectoryRecorder(BaseRecorder):
    def __init__(self, root_dir: str, task_name: str, env_meta: Dict):
        self.output_dir = os.path.join(root_dir, task_name, "h5_raw")
        os.makedirs(self.output_dir, exist_ok=True)
        self.env_meta = env_meta
        self.current_episode_data = []
        self.current_episode_id = 0

    def start_episode(self, episode_id: int):
        self.current_episode_data = []
        self.current_episode_id = episode_id

    def add_frame(self, obs: Dict[str, Any], action: Dict[str, Any], info: Dict[str, Any]):
        # H5 录制通常保存完整的 obs 和执行的 actual_action
        frame = {
            "obs": obs,
            "action": info.get("actual_action", action)
        }
        self.current_episode_data.append(frame)

    def _save_dict_to_h5(self, group, d_list):
        """递归保存嵌套字典"""
        first_step = d_list[0]
        for k in first_step.keys():
            if isinstance(first_step[k], dict):
                sub_group = group.create_group(k)
                sub_list = [step[k] for step in d_list]
                self._save_dict_to_h5(sub_group, sub_list)
            else:
                data = np.stack([step[k] for step in d_list])
                # 对图像进行压缩
                if k in ["rgb", "image", "depth"] or any(x in k for x in ["camera", "wrist", "front"]):
                    group.create_dataset(k, data=data, compression="gzip", compression_opts=4)
                else:
                    group.create_dataset(k, data=data)

    def end_episode(self, success: bool):
        if len(self.current_episode_data) < 5: return
        
        ts = datetime.datetime.now().strftime("%H%M%S")
        file_path = os.path.join(self.output_dir, f"traj_{self.current_episode_id}_{ts}.h5")
        
        with h5py.File(file_path, 'w') as f:
            # 1. 保存观测与动作
            self._save_dict_to_h5(f.create_group("obs"), [step["obs"] for step in self.current_episode_data])
            self._save_dict_to_h5(f.create_group("action"), [step["action"] for step in self.current_episode_data])
            
            # 2. 保存元数据
            meta_g = f.create_group("meta")
            meta_g.create_dataset("env_meta", data=json.dumps(self.env_meta))
            f.attrs['success'] = success
            
        print(f"[H5Recorder] 轨迹已保存: {file_path}")

    def finalize(self):
        print("[H5Recorder] 所有 H5 轨迹录制完成。")

# --- 3. LeRobot 录制实现 (聚合数据集) ---

class LeRobotDatasetRecorder(BaseRecorder):
    def __init__(
        self,
        root_dir: str,
        task_name: str,
        env_meta: Dict,
        fps: int,
        task_description: Optional[str] = None,
        vcodec: str = "h264",
    ):
        if not HAS_LEROBOT:
            raise ImportError("LeRobot 库未安装，无法使用 LeRobotRecorder")
        
        self.repo_id = task_name
        self.root = os.path.join(root_dir, task_name, "lerobot")
        self.fps = fps
        self.env_meta = env_meta
        self.task_description = task_description or task_name
        self.vcodec = vcodec
        self._dark_image_warned = set()
        if "depth" in env_meta.get("obs", {}):
            print(
                "[LeRobotRecorder] 当前 LeRobot 写入路径暂不保存 obs/depth；"
                "如需 depth，请先使用 h5 格式采集。"
            )
        
        # 构建 LeRobot 特征注册
        features = self._build_features(env_meta)
        
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=fps,
            root=self.root,
            features=features,
            use_videos=True,
            vcodec=self.vcodec,
        )
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, "env_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.env_meta, f, indent=2)

    def _build_features(self, env_meta: Dict) -> Dict:
        state_dim = sum(shape[0] for shape in env_meta["obs"]["state"].values())
        action_dim = sum(shape[0] for shape in env_meta["action"].values())
        features = {
            "observation.state": {"dtype": "float32", "shape": (state_dim,)},
            "action": {"dtype": "float32", "shape": (action_dim,)},
        }
        if "under_control" in env_meta["obs"]:
            for name in env_meta["obs"]["under_control"].keys():
                features[f"observation.under_control.{name}"] = {"dtype": "bool", "shape": (1,)}
        # 自动注入相机
        if "rgb" in env_meta["obs"]:
            for role, shape in env_meta["obs"]["rgb"].items():
                features[f"observation.images.{role}"] = {"dtype": "video", "shape": shape}
        return features

    def start_episode(self, episode_id: int):
        # LeRobot 内部管理 buffer，外部不需要特殊操作
        pass

    def add_frame(self, obs: Dict[str, Any], action: Dict[str, Any], info: Dict[str, Any]):
        act_dict = info.get("actual_action", action)
        state_vec = np.concatenate([obs["state"][key] for key in self.env_meta["obs"]["state"].keys()])
        act_vec = np.concatenate([act_dict[key] for key in self.env_meta["action"].keys()])

        frame = {
            "observation.state": torch.from_numpy(state_vec),
            "action": torch.from_numpy(act_vec.astype(np.float32)),
            "task": self.task_description,
        }

        if "under_control" in obs:
            for name, value in obs["under_control"].items():
                frame[f"observation.under_control.{name}"] = torch.from_numpy(
                    np.asarray(value, dtype=np.bool_)
                )

        for role, img in obs.get("rgb", {}).items():
            if role not in self._dark_image_warned and float(np.asarray(img).mean()) < 5.0:
                print(
                    f"[LeRobotRecorder] Warning: obs/rgb/{role} mean is very low "
                    f"({float(np.asarray(img).mean()):.2f}); video may look black."
                )
                self._dark_image_warned.add(role)
            frame[f"observation.images.{role}"] = torch.from_numpy(img)

        self.dataset.add_frame(frame)

    def end_episode(self, success: bool):
        # LeRobot 仅保存成功的轨迹？或者全部保存并在 finalize 时处理
        # 按照 Piper 习惯，我们保存当前 Episode
        if len(self.dataset.episode_buffer["action"]) > 5:
            self.dataset.save_episode()
            print(f"[LeRobotRecorder] Episode 保存成功。总帧数: {len(self.dataset)}")

    def finalize(self):
        if hasattr(self.dataset, 'finalize'):
            self.dataset.finalize()
        print(f"[LeRobotRecorder] 数据集已 Finalize: {self.root}")

# --- 4. 统一执行层 (Manager) ---

def build_robot_overrides(
    is_dual: bool,
    masters: Optional[List[str]],
    slaves: Optional[List[str]],
) -> Optional[Dict[str, Dict[str, str]]]:
    """Build robot CAN overrides from CLI arguments."""
    if masters is None and slaves is None:
        return None

    arm_names = ["left", "right"] if is_dual else ["single"]
    expected = len(arm_names)

    if masters is not None and len(masters) != expected:
        raise ValueError(
            f"--master expects {expected} value(s), got {len(masters)}: {masters}"
        )
    if slaves is not None and len(slaves) != expected:
        raise ValueError(
            f"--slave expects {expected} value(s), got {len(slaves)}: {slaves}"
        )

    overrides = {}
    for idx, arm_name in enumerate(arm_names):
        arm_override = {}
        if masters is not None:
            arm_override["master_can"] = masters[idx]
        if slaves is not None:
            arm_override["follower_can"] = slaves[idx]
        if arm_override:
            overrides[arm_name] = arm_override

    return overrides or None

class DataCollectionManager:
    def __init__(self, 
                 env, 
                 mode: Literal["h5", "lerobot"] = "h5",
                 task_name: Optional[str] = None,
                 root_dir: str = "agent_infra/Piper_Env/Record/data",
                 preview: bool = False,
                 teleop_on_start: bool = False,
                 task_description: Optional[str] = None,
                 vcodec: str = "h264",
                 max_step: int = -1):
        
        self.env = env
        self.mode = mode
        self.preview = preview
        if self.preview and self._is_headless_display():
            print("[Recorder] 未检测到 DISPLAY/WAYLAND_DISPLAY，自动关闭 OpenCV 预览。")
            self.preview = False
        self.hz = self.env.unwrapped.hz
        self.teleop_on_start = teleop_on_start
        self.task_description = task_description
        self.vcodec = vcodec
        if max_step == -1:
            self.max_step = 9999
            self._auto_delete_on_max_step = True
        elif max_step > 0:
            self.max_step = max_step
            self._auto_delete_on_max_step = False
        else:
            raise ValueError("--max-step must be a positive integer or -1")
        
        # 基础信息
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.task_name = task_name or f"piper-{mode}-{now}"
        self.task_name = self._ensure_control_mode_in_task_name(self.task_name)
        if self.mode == "lerobot":
            self.task_name = self._resolve_unique_lerobot_task_name(root_dir, self.task_name)
        
        # 获取底层 Meta
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "get_env_metadata"):
            self.env_meta = unwrapped.get_env_metadata()
        else:
            self.env_meta = unwrapped.meta_keys
        if hasattr(unwrapped, "switch_master_follow"):
            unwrapped.switch_master_follow("false")
        
        # 初始化录制器
        if mode == "h5":
            self.recorder = H5TrajectoryRecorder(root_dir, self.task_name, self.env_meta)
        else:
            self.recorder = LeRobotDatasetRecorder(
                root_dir,
                self.task_name,
                self.env_meta,
                fps=self.hz,
                task_description=self.task_description or self.task_name,
                vcodec=self.vcodec,
            )

        # 控制变量
        self.is_recording = False
        self.is_finished = False
        self.is_resetting = False
        self.traj_counter = 0
        self.success = True
        self._reset_lock = threading.Lock()
        self._record_frame_idx = 0
        self._is_prompting_max_step = False
        self._safe_action_state_cache: Optional[Dict[str, np.ndarray]] = None
        
        # 监听与显示
        self.listener = self._start_keyboard_listener()
        if self.teleop_on_start:
            self._switch_teleop("true")

    @staticmethod
    def _is_headless_display() -> bool:
        if os.name == "nt":
            return False
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    def _ensure_control_mode_in_task_name(self, task_name: str) -> str:
        control_mode = str(getattr(self.env.unwrapped, "control_mode", "") or "").lower()
        if control_mode not in {"joint", "pose", "delta_pose", "relative_pose_chunk"}:
            return task_name

        parts = task_name.split("_")
        if (
            control_mode in parts
            or "joint" in parts
            or "pose" in parts
            or "delta" in parts
            or "relative" in parts
        ):
            return task_name

        if task_name.endswith("_task"):
            normalized = f"{task_name[:-5]}_{control_mode}_task"
        else:
            normalized = f"{task_name}_{control_mode}"

        print(f"[Recorder] 任务名自动补充控制模式: {task_name} -> {normalized}")
        return normalized

    @staticmethod
    def _resolve_unique_lerobot_task_name(root_dir: str, task_name: str) -> str:
        base_task_name = task_name
        candidate = base_task_name
        suffix = 1

        while os.path.exists(os.path.join(root_dir, candidate, "lerobot")):
            candidate = f"{base_task_name}_{suffix:03d}"
            suffix += 1

        if candidate != base_task_name:
            print(
                f"[Recorder] LeRobot 数据集目录已存在，自动切换任务名: "
                f"{base_task_name} -> {candidate}"
            )

        return candidate

    def _start_keyboard_listener(self):
        if self._is_headless_display():
            print("[Recorder] 未检测到图形显示，使用终端输入快捷键后端。")
            return self._start_terminal_keyboard_listener()

        try:
            from pynput import keyboard

            listener = keyboard.Listener(on_press=self._on_press)
            listener.start()
            print("[Recorder] 键盘监听后端: pynput")
            return listener
        except Exception as exc:
            print(f"[Recorder] pynput 键盘监听不可用，切换到终端输入后端: {exc}")

        return self._start_terminal_keyboard_listener()

    def _start_terminal_keyboard_listener(self):
        if not sys.stdin.isatty():
            print("[Recorder] stdin 不是 TTY，无法启用终端快捷键；请用 Ctrl-C 结束进程。")
            return None

        listener = threading.Thread(target=self._terminal_key_loop, daemon=True)
        listener.start()
        print("[Recorder] 键盘监听后端: terminal（输入快捷键后回车）")
        return listener

    def _terminal_key_loop(self):
        try:
            import select
        except Exception as exc:
            print(f"[Recorder] 终端快捷键后端不可用: {exc}")
            return

        while not self.is_finished:
            if self._is_prompting_max_step:
                time.sleep(0.1)
                continue

            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue

            line = sys.stdin.readline()
            if not line:
                time.sleep(0.1)
                continue
            text = line.strip().lower()
            if not text:
                continue
            if self._handle_key_char(text[0]) is False:
                return

    def _handle_key_char(self, char: str):
        if self._is_prompting_max_step:
            return None
        if char == 't':
            self._toggle_teleop()
        elif char == 'i':
            if not self._reset_lock.acquire(blocking=False):
                print("[I] 复位正在执行，忽略重复按键。")
                return None
            self.is_resetting = True
            try:
                print("[I] 执行复位...")
                self.env.reset(options={"sync_master": True})
                self._safe_action_state_cache = None
            finally:
                self.is_resetting = False
                self._reset_lock.release()
        elif char == 's':
            if not self.is_recording:
                self.recorder.start_episode(self.traj_counter)
                self._record_frame_idx = 0
                self.is_recording = True
                print(f"[S] 开始录制 Traj {self.traj_counter} ({self.mode})...")
        elif char == 'e':
            if self.is_recording:
                self.is_recording = False
                self.recorder.end_episode(success=True)
                self._record_frame_idx = 0
                self.traj_counter += 1
        elif char == 'f':
            if self.is_recording:
                self.is_recording = False
                self.recorder.end_episode(success=False)
                self._record_frame_idx = 0
                self.traj_counter += 1
        elif char == 'd':
            if self.is_recording:
                self.is_recording = False
                self.recorder.discard_episode()
                self._record_frame_idx = 0
                print("[D] 丢弃当前轨迹。")
        elif char == 'q':
            print("[Q] 退出。")
            self.is_finished = True
            return False
        return None

    def _on_press(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            return None
        return self._handle_key_char(char)

    def _switch_teleop(self, mode: str):
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "switch_tele"):
            unwrapped.switch_tele(mode)
            return
        print("[T] 当前环境不支持 switch_tele。")

    def _toggle_teleop(self):
        self._switch_teleop("toggle")

    def _handle_max_step_reached(self):
        frame_count = self._record_frame_idx
        self.is_recording = False

        if self._auto_delete_on_max_step:
            self.recorder.discard_episode()
            self._record_frame_idx = 0
            print(f"[MaxStep] Traj {self.traj_counter} 已达到 9999 帧，自动删除该轨迹。")
            return

        self._is_prompting_max_step = True
        try:
            print(f"\n[MaxStep] Traj {self.traj_counter} 已达到最大帧数 {frame_count}。")
            while True:
                choice = input("[MaxStep] 保存该轨迹还是删除？输入 s 保存 / d 删除: ").strip().lower()
                if choice in {"s", "save", "y", "yes"}:
                    self.recorder.end_episode(success=True)
                    self.traj_counter += 1
                    print("[MaxStep] 轨迹已保存。")
                    break
                if choice in {"d", "delete", "discard", "n", "no"}:
                    self.recorder.discard_episode()
                    print("[MaxStep] 轨迹已删除。")
                    break
                print("[MaxStep] 无效输入，请输入 s 或 d。")
        except (EOFError, KeyboardInterrupt):
            self.recorder.discard_episode()
            print("\n[MaxStep] 未确认保存，轨迹已删除。")
        finally:
            self._record_frame_idx = 0
            self._is_prompting_max_step = False

    def run(self):
        print(f"\n--- Piper 数据采集 [{self.mode}] ---")
        print(f" 任务: {self.task_name}")
        print(f" 最大轨迹长度: {self.max_step} 帧" + (" (到达后自动删除)" if self._auto_delete_on_max_step else ""))
        print(" [T]遥操开关 [I]复位 [S]开始 [E]成功结束 [F]失败结束 [D]丢弃 [Q]退出")
        
        # 如果是包装器环境，启动相机
        if hasattr(self.env, 'start_cameras'):
            self.env.start_cameras()

        while not self.is_finished:
            t_start = time.time()

            if self.is_resetting:
                time.sleep(0.05)
                continue
            
            # 1. 获取安全动作（维持位姿）
            # 注意：如果 PiperEnv 处于接管模式，step 内部会自动覆写此动作
            if self._safe_action_state_cache is not None:
                try:
                    safe_action = self.env.unwrapped.get_safe_action(
                        state=self._safe_action_state_cache
                    )
                except TypeError:
                    safe_action = self.env.unwrapped.get_safe_action()
            else:
                safe_action = self.env.unwrapped.get_safe_action()
            
            # 2. 步进
            obs, reward, terminated, truncated, info = self.env.step(safe_action)
            state = obs.get("state") if isinstance(obs, dict) else None
            if isinstance(state, dict):
                self._safe_action_state_cache = {
                    key: np.asarray(value).copy()
                    for key, value in state.items()
                }
            else:
                self._safe_action_state_cache = None
            
            # 3. 录制
            if self.is_recording:
                self.recorder.add_frame(obs, safe_action, info)
                self._record_frame_idx += 1
                if self._record_frame_idx >= self.max_step:
                    self._handle_max_step_reached()
            
            # 4. 预览
            if self.preview:
                try:
                    self._visualize(obs, info)
                except cv2.error as exc:
                    print(f"[Recorder] OpenCV 预览失败，已自动关闭预览: {exc}")
                    self.preview = False
                    try:
                        cv2.destroyAllWindows()
                    except cv2.error:
                        pass
            
            # 频率维持
            elapsed = time.time() - t_start
            #print(elapsed)
            time.sleep(max(0, (1.0/self.hz) - elapsed))
            
        self.recorder.finalize()
        if self.preview:
            cv2.destroyAllWindows()

    def _visualize(self, obs: Dict[str, Any], info: Optional[Dict[str, Any]] = None):
        if "rgb" not in obs or not obs["rgb"]:
            return

        rgb_obs = obs["rgb"]
        roles = sorted(rgb_obs.keys())
        if not roles:
            return

        def _to_hwc_rgb(img: np.ndarray) -> np.ndarray:
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr

        tiles: List[np.ndarray] = []
        for role in roles:
            rgb = _to_hwc_rgb(rgb_obs[role])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # 每路相机单独标注：角色 + 分辨率（直接附着在图像上）
            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (8, 8), (min(w - 8, 280), 34), (30, 30, 30), thickness=-1)
            cv2.putText(
                bgr,
                f"{role}",
                (14, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(bgr)

        tile_h, tile_w = tiles[0].shape[:2]
        cols = 1 if len(tiles) == 1 else 2
        rows = int(np.ceil(len(tiles) / cols))
        grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
        for idx, tile in enumerate(tiles):
            r = idx // cols
            c = idx % cols
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            grid[y0:y1, x0:x1] = tile

        info = info or {}
        status_text = "RECORDING" if self.is_recording else "IDLE"
        status_color = (0, 0, 255) if self.is_recording else (0, 200, 0)
        try:
            teleop_enabled = bool(self.env.unwrapped.get_env_state("teleop"))
        except Exception:
            teleop_enabled = False
        intervened = info.get("intervened", False)
        under_control = obs.get("under_control", {})
        uc_parts = []
        if isinstance(under_control, dict):
            for name in sorted(under_control.keys()):
                flag = bool(np.asarray(under_control[name]).reshape(-1)[0])
                uc_parts.append(f"{name}:{'Y' if flag else 'N'}")
        uc_text = ",".join(uc_parts) if uc_parts else "-"

        # 统一窗口宽度到 640，保持纵横比，避免低分辨率时预览窗口过小。
        target_w = 640
        grid_h, grid_w = grid.shape[:2]
        if grid_w > 0 and grid_w != target_w:
            scale = target_w / float(grid_w)
            resized_h = max(1, int(round(grid_h * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            grid = cv2.resize(grid, (target_w, resized_h), interpolation=interp)
        else:
            target_w = grid_w

        # 将全局状态文本放在图像外的下方区域，提升可读性。
        footer_h = 72
        canvas = np.zeros((grid.shape[0] + footer_h, target_w, 3), dtype=np.uint8)
        canvas[:grid.shape[0], :target_w] = grid
        canvas[grid.shape[0]:, :] = (20, 20, 20)

        dot_color = status_color
        cv2.circle(canvas, (20, grid.shape[0] + 24), 8, dot_color, -1)
        cv2.putText(
            canvas,
            f"{status_text} | Traj:{self.traj_counter} | Frame:{self._record_frame_idx}",
            (38, grid.shape[0] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            status_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Teleop:{'ON' if teleop_enabled else 'OFF'} | Intervened:{intervened}",
            (10, grid.shape[0] + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Piper Record Preview", canvas)
        cv2.waitKey(1)

if __name__ == "__main__":
    import argparse
    from agent_infra.Piper_Env.Env.single_piper_env import SinglePiperEnv
    from agent_infra.Piper_Env.Env.dual_piper_env import DualPiperEnv
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="h5", choices=["h5", "lerobot"])
    parser.add_argument("-t", "--task", type=str, default=None)
    parser.add_argument(
        "-ctrl",
        "--control",
        type=str,
        default="joint",
        choices=["joint", "pose", "delta_pose", "relative_pose_chunk"],
    )
    parser.add_argument("-cfg", "--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("-dual", "--dual_arm", action="store_true", help="是否使用双臂环境")
    parser.add_argument(
        "--master",
        nargs="+",
        default=None,
        help="覆盖主臂 CAN 接口。单臂传 1 个值；双臂按 left right 传 2 个值。",
    )
    parser.add_argument(
        "--slave",
        nargs="+",
        default=None,
        help="覆盖从臂/follower CAN 接口。单臂传 1 个值；双臂按 left right 传 2 个值。",
    )
    parser.add_argument(
        "--preview",
        dest="preview",
        action="store_true",
        help="显示 OpenCV 录制预览窗口（默认开启）",
    )
    parser.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="关闭 OpenCV 录制预览窗口",
    )
    parser.set_defaults(preview=True)
    parser.add_argument(
        "--teleop-on-start",
        action="store_true",
        help="启动采集后立即开启主臂遥操接管",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default=None,
        help="写入 LeRobot 每帧 task 字段的自然语言任务描述；默认使用任务名",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        help="LeRobot 视频编码，默认 h264，兼容性优于默认 AV1/libsvtav1",
    )
    parser.add_argument(
        "--max-step",
        "--max_step",
        type=int,
        default=-1,
        help="单条轨迹最大帧数；-1 表示 9999 帧且到达后自动删除",
    )
    args = parser.parse_args()

    robot_overrides = build_robot_overrides(args.dual_arm, args.master, args.slave)
    if robot_overrides:
        print(f"[Recorder] CAN override: {robot_overrides}")

    if args.dual_arm:
        env = DualPiperEnv(
            config_path=args.config,
            control_mode=args.control,
            robot_overrides=robot_overrides,
        )
    else:
        env = SinglePiperEnv(
            config_path=args.config,
            control_mode=args.control,
            robot_overrides=robot_overrides,
        )
    
    manager = DataCollectionManager(
        env,
        mode=args.mode,
        task_name=args.task,
        preview=args.preview,
        teleop_on_start=args.teleop_on_start,
        task_description=args.task_description,
        vcodec=args.vcodec,
        max_step=args.max_step,
    )
    try:
        manager.run()
    finally:
        env.close()
