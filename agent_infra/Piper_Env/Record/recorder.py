import time
import datetime
import os
import numpy as np
import cv2
import h5py
import json
import torch
import threading
from abc import ABC, abstractmethod
from pynput import keyboard
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
                 vcodec: str = "h264"):
        
        self.env = env
        self.mode = mode
        self.preview = preview
        self.hz = self.env.unwrapped.hz
        self.teleop_on_start = teleop_on_start
        self.task_description = task_description
        self.vcodec = vcodec
        
        # 基础信息
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.task_name = task_name or f"piper-{mode}-{now}"
        self.task_name = self._ensure_control_mode_in_task_name(self.task_name)
        if self.mode == "lerobot":
            self.task_name = self._resolve_unique_lerobot_task_name(root_dir, self.task_name)
        
        # 获取底层 Meta
        unwrapped = self.env.unwrapped
        self.env_meta = unwrapped.meta_keys
        
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
        
        # 监听与显示
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        if self.teleop_on_start:
            self._set_teleop(True)

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

    def _on_press(self, key):
        try:
            char = key.char.lower()
            if char == 'i':
                if not self._reset_lock.acquire(blocking=False):
                    print("[I] 复位正在执行，忽略重复按键。")
                    return
                self.is_resetting = True
                try:
                    print("[I] 执行复位...")
                    self.env.reset(options={"sync_master": True})
                finally:
                    self.is_resetting = False
                    self._reset_lock.release()
            elif char == 's':
                if not self.is_recording:
                    self.recorder.start_episode(self.traj_counter)
                    self.is_recording = True
                    print(f"[S] 开始录制 Traj {self.traj_counter} ({self.mode})...")
            elif char == 'e':
                if self.is_recording:
                    self.is_recording = False
                    self.recorder.end_episode(success=True)
                    self.traj_counter += 1
            elif char == 'f':
                if self.is_recording:
                    self.is_recording = False
                    self.recorder.end_episode(success=False)
                    self.traj_counter += 1
            elif char == 'd':
                if self.is_recording:
                    self.is_recording = False
                    print("[D] 丢弃当前轨迹。")
            elif char == 'q':
                print("[Q] 退出。")
                self.is_finished = True
                return False 
        except AttributeError:
            pass

    def _set_teleop(self, enabled: bool):
        unwrapped = self.env.unwrapped
        if not hasattr(unwrapped, "tele_enabled"):
            print("[T] 当前环境不支持 teleop。")
            return

        unwrapped.tele_enabled = bool(enabled)
        print(f"[T] Teleoperation: {'ON' if unwrapped.tele_enabled else 'OFF'}")

    def run(self):
        print(f"\n--- Piper 数据采集 [{self.mode}] ---")
        print(f" 任务: {self.task_name}")
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
            safe_action = self.env.unwrapped.get_safe_action()
            
            # 2. 步进
            obs, reward, terminated, truncated, info = self.env.step(safe_action)
            
            # 3. 录制
            if self.is_recording:
                self.recorder.add_frame(obs, safe_action, info)
            
            # 4. 预览
            if self.preview:
                self._visualize(obs)
            
            # 频率维持
            elapsed = time.time() - t_start
            time.sleep(max(0, (1.0/self.hz) - elapsed))
            
        self.recorder.finalize()
        if self.preview:
            cv2.destroyAllWindows()

    def _visualize(self, obs: Dict[str, Any]):
        if "rgb" not in obs or not obs["rgb"]: return
        
        # 拼接预览
        roles = sorted(obs["rgb"].keys())
        imgs = []
        for r in roles:
            # CHW -> HWC
            imgs.append(obs["rgb"][r].transpose(1, 2, 0))
        
        combined = np.hstack(imgs)
        # BGR 转换（如果相机是 RGB）
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        
        status_color = (0, 0, 255) if self.is_recording else (0, 255, 0)
        text = "RECORDING" if self.is_recording else "IDLE"
        cv2.putText(combined, f"{text} | Traj: {self.traj_counter}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow("Piper Record Preview", combined)
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
    parser.add_argument("--preview", action="store_true", help="显示 OpenCV 录制预览窗口")
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
    )
    try:
        manager.run()
    finally:
        env.close()
