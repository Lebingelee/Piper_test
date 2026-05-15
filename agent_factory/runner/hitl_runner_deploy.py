import copy
from dataclasses import asdict, is_dataclass
import json
import math
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from agent_factory.runner.hitl_runner import HITLRunner


DEPLOY_START_KEY = "s"
DEPLOY_STOP_KEY = "e"
DEPLOY_CONTINUE_KEY = "c"
DEPLOY_INIT_KEY = "i"
DEPLOY_QUIT_KEY = "q"
DEPLOY_TELEOP_KEY = "t"


class HITLDeployRunner(HITLRunner):
    """
    Deployment-oriented HITL runner.

    Compared with the generic HITL runner, this class adds:
    - asynchronous risk checks via ``agent.get_risk(obs, action)``
    - session-level recording with multiple ``traj_xxxx`` groups per H5
    - periodic max-step prompts without forcing immediate truncation online
    - manual save / continue controls for deployment operators
    """

    def __init__(self, cfg: DictConfig, agent: Any, env: Any):
        self.start_key: str = str(DEPLOY_START_KEY).lower()
        self.stop_key: str = str(DEPLOY_STOP_KEY).lower()
        self.continue_key: str = str(DEPLOY_CONTINUE_KEY).lower()
        self.init_key: str = str(DEPLOY_INIT_KEY).lower()
        self.quit_key: str = str(DEPLOY_QUIT_KEY).lower()
        self.teleop_key: str = str(DEPLOY_TELEOP_KEY).lower()
        self.risk_check_hz: float = float(getattr(cfg.runner, "risk_check_hz", 0.0))
        self.risk_use_safe_action: bool = bool(
            getattr(cfg.runner, "risk_use_safe_action", True)
        )
        self._key_lock = threading.Lock()
        self._init_requested = False
        self._start_requested = False
        self._stop_requested = False
        self._continue_requested = False
        self._quit_requested = False
        self._teleop_toggle_requested = False
        self.quit_requested = False
        self._prompt_active = False

        self._risk_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        self._risk_result_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        self._risk_stop = False
        self._risk_submit_token = 0
        self._risk_last_submit_time = 0.0
        self._risk_last_handled_token = -1
        self.risk_thread: Optional[threading.Thread] = None

        super().__init__(cfg=cfg, agent=agent, env=env)

        self.risk_thread = threading.Thread(target=self._risk_worker, daemon=True)
        self.risk_thread.start()

    def _start_override_listener(self) -> None:
        try:
            from pynput import keyboard
        except Exception:
            print("[HITLDeployRunner] pynput not available; save/continue hotkeys disabled.")
            return

        def _on_press(key):
            try:
                if self._prompt_active:
                    return
                char = key.char.lower()
            except Exception:
                return

            with self._key_lock:
                if char == self.init_key:
                    self._init_requested = True
                elif char == self.start_key:
                    self._start_requested = True
                elif char == self.stop_key:
                    self._stop_requested = True
                elif char == self.continue_key:
                    self._continue_requested = True
                elif char == self.quit_key:
                    self._quit_requested = True
                    self.quit_requested = True
                elif char == self.teleop_key:
                    self._teleop_toggle_requested = True

        self._listener = keyboard.Listener(on_press=_on_press)
        self._listener.start()

    def _read_override_flag(self) -> bool:
        # Deploy runner relies on env teleop state instead of a second runner-side
        # toggle to avoid conflicting override sources.
        return False

    def _consume_init_requested(self) -> bool:
        with self._key_lock:
            value = self._init_requested
            self._init_requested = False
            return value

    def _consume_start_requested(self) -> bool:
        with self._key_lock:
            value = self._start_requested
            self._start_requested = False
            return value

    def _consume_stop_requested(self) -> bool:
        with self._key_lock:
            value = self._stop_requested
            self._stop_requested = False
            return value

    def _consume_continue_requested(self) -> bool:
        with self._key_lock:
            value = self._continue_requested
            self._continue_requested = False
            return value

    def _consume_quit_requested(self) -> bool:
        with self._key_lock:
            value = self._quit_requested
            self._quit_requested = False
            if value:
                self.quit_requested = True
            return value

    def _consume_teleop_toggle_requested(self) -> bool:
        with self._key_lock:
            value = self._teleop_toggle_requested
            self._teleop_toggle_requested = False
            return value

    def _toggle_env_teleop(self) -> bool:
        base_env = self._unwrapped_env()
        if hasattr(base_env, "switch_tele"):
            try:
                base_env.switch_tele("toggle")
                return True
            except Exception as exc:
                print(f"[HITLDeployRunner] env teleop toggle failed: {exc}")
                return False
        return False

    def _risk_worker(self) -> None:
        print("[HITLDeployRunner] 风险线程已启动。")
        while not self._risk_stop:
            try:
                payload = self._risk_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                obs = payload["obs"]
                action = np.asarray(payload["action"], dtype=np.float32)
                token = int(payload["token"])

                obs_batch: Dict[str, Any] = {}
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        obs_batch[key] = torch.from_numpy(value).unsqueeze(0).to(self.device)
                    elif isinstance(value, torch.Tensor):
                        obs_batch[key] = value.unsqueeze(0).to(self.device)
                    else:
                        obs_batch[key] = value

                action_batch = torch.from_numpy(action).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    risk_value = bool(self.agent.get_risk(obs_batch, action_batch))

                result = {"token": token, "risk": risk_value}
                try:
                    if self._risk_result_queue.full():
                        try:
                            self._risk_result_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._risk_result_queue.put_nowait(result)
                except queue.Full:
                    pass
            except Exception as exc:
                print(f"[HITLDeployRunner] 风险评估失败: {exc}")

    def _submit_risk_check(self, obs: Dict[str, Any], policy_action: np.ndarray) -> None:
        if self.risk_check_hz <= 0.0:
            return

        now = time.monotonic()
        period = 1.0 / max(self.risk_check_hz, 1e-6)
        if (now - self._risk_last_submit_time) < period:
            return
        self._risk_last_submit_time = now

        self._risk_submit_token += 1
        payload = {
            "token": self._risk_submit_token,
            "obs": copy.deepcopy(obs),
            "action": np.asarray(policy_action, dtype=np.float32).copy(),
        }

        try:
            if self._risk_queue.full():
                try:
                    self._risk_queue.get_nowait()
                except queue.Empty:
                    pass
            self._risk_queue.put_nowait(payload)
        except queue.Full:
            pass

    def _poll_latest_risk_result(self) -> Optional[Dict[str, Any]]:
        latest = None
        while not self._risk_result_queue.empty():
            try:
                latest = self._risk_result_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def _disable_teleop(self) -> None:
        base_env = self._unwrapped_env()
        if not hasattr(base_env, "switch_tele"):
            return
        try:
            teleop_enabled = bool(base_env.get_env_state("teleop"))
        except Exception:
            teleop_enabled = False
        if teleop_enabled:
            base_env.switch_tele("false")
            print("[HITLDeployRunner] 已强制退出遥操状态。")

    def _refresh_obs_with_safe_step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        action = self._get_safe_action(obs)
        next_obs, _, _, _, _ = self.env.step(action)
        return next_obs

    def _reset_for_deploy_init(self) -> Dict[str, Any]:
        print("[HITLDeployRunner] 执行初始化复位并同步主从臂...")
        obs, _ = self.env.reset(options={"sync_master": True})
        self._env_override_active = False
        self._clear_action_queue()
        while not self.obs_queue.empty():
            try:
                self.obs_queue.get_nowait()
            except queue.Empty:
                break
        while not self._risk_queue.empty():
            try:
                self._risk_queue.get_nowait()
            except queue.Empty:
                break
        while not self._risk_result_queue.empty():
            try:
                self._risk_result_queue.get_nowait()
            except queue.Empty:
                break
        return obs

    def _reset_session_buffers(self) -> None:
        self.current_traj = {
            "obs": [],
            "action": [],
            "policy_action": [],
            "action_type": [],
            "runner_action_type": [],
            "env_action_type": [],
            "rewards": [],
            "success": [],
            "intervention": [],
            "terminated": [],
            "truncated": [],
            "risk": [],
            "boundary_reason": [],
        }
        self._last_closed_action_idx = -1

    def _append_record(
        self,
        obs: Dict[str, Any],
        executed_action: np.ndarray,
        policy_action: np.ndarray,
        runner_action_type: int,
        env_action_type: int,
        reward: float,
        success: bool,
        intervention: bool,
        terminated: bool,
        truncated: bool,
        risk_active: bool,
    ) -> None:
        self.current_traj["obs"].append(copy.deepcopy(obs))
        self.current_traj["action"].append(np.asarray(executed_action, dtype=np.float32))
        self.current_traj["policy_action"].append(np.asarray(policy_action, dtype=np.float32))
        self.current_traj["action_type"].append(np.int32(env_action_type))
        self.current_traj["runner_action_type"].append(np.int32(runner_action_type))
        self.current_traj["env_action_type"].append(np.int32(env_action_type))
        self.current_traj["rewards"].append(float(reward))
        self.current_traj["success"].append(bool(success))
        self.current_traj["intervention"].append(bool(intervention))
        self.current_traj["terminated"].append(bool(terminated))
        self.current_traj["truncated"].append(bool(truncated))
        self.current_traj["risk"].append(bool(risk_active))
        self.current_traj["boundary_reason"].append("")

    def _mark_last_boundary(
        self,
        reason: str,
        *,
        terminated: bool = False,
        truncated: bool = False,
        success: Optional[bool] = None,
        force: bool = False,
    ) -> bool:
        if not self.current_traj["action"]:
            return False
        idx = len(self.current_traj["action"]) - 1
        if idx <= self._last_closed_action_idx and not force:
            return False
        if terminated:
            self.current_traj["terminated"][idx] = True
        if truncated:
            self.current_traj["truncated"][idx] = True
        if success is not None:
            self.current_traj["success"][idx] = bool(success)
        self.current_traj["boundary_reason"][idx] = str(reason)
        self._last_closed_action_idx = idx
        return True

    def _prompt_save_outcome(self) -> str:
        self._prompt_active = True
        try:
            while True:
                try:
                    choice = input(
                        "[Save] 请选择本次 rollout 处理方式：1 成功 / 2 失败 / d 删除: "
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n[Save] 未确认处理方式，默认删除本次 rollout。")
                    return "delete"
                if choice in {"1", "s", "success", "y", "yes"}:
                    return "success"
                if choice in {"2", "f", "fail", "failure", "n", "no"}:
                    return "failure"
                if choice in {"d", "delete", "discard"}:
                    return "delete"
                print("[Save] 无效输入，请输入 1 / 2 / d。")
        finally:
            self._prompt_active = False

    def _prompt_binary_success(self) -> Optional[bool]:
        outcome = self._prompt_save_outcome()
        if outcome == "delete":
            return None
        return outcome == "success"

    def _prompt_max_step_choice(self) -> str:
        self._prompt_active = True
        try:
            while True:
                try:
                    choice = input(
                        "[MaxStep] 已达到最大迭代步数。输入 s 保存并退出 / d 删除并退出 / c 继续: "
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n[MaxStep] 未确认，默认继续录制。")
                    return "continue"
                if choice in {"s", "save"}:
                    return "save"
                if choice in {"d", "delete", "discard"}:
                    return "delete"
                if choice in {"c", "continue"}:
                    return "continue"
                print("[MaxStep] 无效输入，请输入 s / d / c。")
        finally:
            self._prompt_active = False

    def _ensure_final_obs(self, obs: Dict[str, Any]) -> None:
        target_len = len(self.current_traj["action"]) + 1
        if len(self.current_traj["obs"]) < target_len:
            self.current_traj["obs"].append(copy.deepcopy(obs))

    def _prepare_traj_for_save(self) -> None:
        super()._prepare_traj_for_save()
        num_actions = len(self.current_traj["action"])
        if num_actions <= 0:
            return

        if len(self.current_traj["risk"]) < num_actions:
            self.current_traj["risk"] = self.current_traj["risk"] + [False] * (
                num_actions - len(self.current_traj["risk"])
            )
        else:
            self.current_traj["risk"] = self.current_traj["risk"][:num_actions]

        boundary_reason = [str(v) for v in self.current_traj["boundary_reason"][:num_actions]]
        if len(boundary_reason) < num_actions:
            boundary_reason = boundary_reason + [""] * (num_actions - len(boundary_reason))
        self.current_traj["boundary_reason"] = boundary_reason

    def _apply_tail_aligned_max_boundaries(self) -> None:
        num_actions = len(self.current_traj["action"])
        if num_actions <= 0 or self.max_steps <= 0:
            return

        num_segments = int(math.ceil(num_actions / float(self.max_steps)))
        first_len = num_actions - self.max_steps * (num_segments - 1)
        boundary_idx = first_len - 1
        while boundary_idx < num_actions:
            self.current_traj["truncated"][boundary_idx] = True
            if not self.current_traj["boundary_reason"][boundary_idx]:
                self.current_traj["boundary_reason"][boundary_idx] = "max_step_tail"
            boundary_idx += self.max_steps

    @staticmethod
    def _latest_frame(value: Any, obs_horizon: int) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value)
        if arr.ndim >= 2 and arr.shape[0] == int(obs_horizon):
            arr = arr[-1]
        return arr

    def _write_obs_group(self, obs_group: h5py.Group, raw_obs_list: List[Dict[str, Any]]) -> None:
        if not raw_obs_list:
            return

        obs_horizon = int(getattr(self.cfg.env, "obs_horizon", 1))
        obs_keys = list(raw_obs_list[0].keys())
        for key in obs_keys:
            values = []
            for obs in raw_obs_list:
                if key not in obs:
                    continue
                arr = self._latest_frame(obs[key], obs_horizon)
                if key == "rgb" and arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
                elif key != "rgb" and arr.dtype.kind == "f":
                    arr = arr.astype(np.float32)
                values.append(arr)
            if not values:
                continue
            kwargs = {"compression": "gzip", "compression_opts": 4} if key == "rgb" else {}
            obs_group.create_dataset(key, data=np.stack(values), **kwargs)

    def _save_session_trajectory(self) -> None:
        self._prepare_traj_for_save()
        self._apply_tail_aligned_max_boundaries()

        num_actions = len(self.current_traj["action"])
        if num_actions < 1:
            print("[HITLDeployRunner] [W] 当前 session 没有可保存的动作，忽略保存。")
            return

        boundary_mask = (
            np.asarray(self.current_traj["success"], dtype=bool)
            | np.asarray(self.current_traj["terminated"], dtype=bool)
            | np.asarray(self.current_traj["truncated"], dtype=bool)
        )
        boundary_indices = np.flatnonzero(boundary_mask)
        if len(boundary_indices) == 0 or int(boundary_indices[-1]) != num_actions - 1:
            self.current_traj["terminated"][-1] = True
            boundary_indices = np.append(boundary_indices, num_actions - 1)

        import datetime

        ts = datetime.datetime.now().strftime("%m%d_%H%M")
        file_name = f"traj_{self.traj_counter}_{ts}.h5"
        file_path = os.path.join(self.save_dir, file_name)

        env_meta = self._get_env_metadata()
        root_success = bool(np.any(np.asarray(self.current_traj["success"], dtype=bool)))

        with h5py.File(file_path, "w") as h5_file:
            meta_group = h5_file.create_group("meta")
            meta_group.create_dataset("env_cfg", data=self._serialize_env_cfg())
            meta_group.create_dataset("env_meta", data=json.dumps(env_meta))

            start_idx = 0
            for segment_idx, end_idx in enumerate(boundary_indices.tolist()):
                if end_idx < start_idx:
                    continue
                group = h5_file.create_group(f"traj_{segment_idx:04d}")
                length = end_idx - start_idx + 1

                group.create_dataset(
                    "action",
                    data=np.stack(self.current_traj["action"][start_idx : end_idx + 1]).astype(np.float32),
                )
                group.create_dataset(
                    "policy_action",
                    data=np.stack(
                        self.current_traj["policy_action"][start_idx : end_idx + 1]
                    ).astype(np.float32),
                )
                group.create_dataset(
                    "action_type",
                    data=np.asarray(
                        self.current_traj["action_type"][start_idx : end_idx + 1],
                        dtype=np.int32,
                    ),
                )
                group.create_dataset(
                    "runner_action_type",
                    data=np.asarray(
                        self.current_traj["runner_action_type"][start_idx : end_idx + 1],
                        dtype=np.int32,
                    ),
                )
                group.create_dataset(
                    "env_action_type",
                    data=np.asarray(
                        self.current_traj["env_action_type"][start_idx : end_idx + 1],
                        dtype=np.int32,
                    ),
                )

                obs_group = group.create_group("obs")
                self._write_obs_group(
                    obs_group,
                    self.current_traj["obs"][start_idx : end_idx + 2],
                )

                group.create_dataset(
                    "rewards",
                    data=np.asarray(
                        self.current_traj["rewards"][start_idx : end_idx + 1],
                        dtype=np.float32,
                    ),
                )
                group.create_dataset(
                    "success",
                    data=np.asarray(
                        self.current_traj["success"][start_idx : end_idx + 1], dtype=bool
                    ),
                )
                group.create_dataset(
                    "terminated",
                    data=np.asarray(
                        self.current_traj["terminated"][start_idx : end_idx + 1], dtype=bool
                    ),
                )
                group.create_dataset(
                    "truncated",
                    data=np.asarray(
                        self.current_traj["truncated"][start_idx : end_idx + 1], dtype=bool
                    ),
                )
                group.create_dataset(
                    "intervention",
                    data=np.asarray(
                        self.current_traj["intervention"][start_idx : end_idx + 1],
                        dtype=bool,
                    ),
                )
                group.create_dataset(
                    "risk",
                    data=np.asarray(
                        self.current_traj["risk"][start_idx : end_idx + 1], dtype=bool
                    ),
                )

                group.attrs["success"] = bool(
                    np.any(self.current_traj["success"][start_idx : end_idx + 1])
                )
                group.attrs["length"] = int(length)
                group.attrs["boundary_reason"] = str(
                    self.current_traj["boundary_reason"][end_idx]
                )
                start_idx = end_idx + 1

            h5_file.attrs["success"] = root_success
            h5_file.attrs["length"] = int(num_actions)
            h5_file.attrs["num_traj"] = len(boundary_indices)

        print(f"[HITLDeployRunner] 多子轨迹 session 已保存至: {file_path}")

        self.traj_counter += 1
        self.saved_files_fifo.append(file_path)

        max_files = self.buffer_capacity + self.redundancy_margin
        while len(self.saved_files_fifo) > max_files:
            file_to_delete = self.saved_files_fifo.pop(0)
            if os.path.exists(file_to_delete):
                try:
                    os.remove(file_to_delete)
                    print(f"[HITLDeployRunner] [FIFO] 已删除旧 session: {file_to_delete}")
                except Exception as exc:
                    print(f"[HITLDeployRunner] [FIFO] 删除失败 {file_to_delete}: {exc}")

    def run(self):
        print(
            f"[HITLDeployRunner] 开始部署 HITL 控制循环 "
            f"(Hz: {self.control_hz}, RiskHz: {self.risk_check_hz}, "
            f"RiskMode: {'safe_hold' if self.risk_use_safe_action else 'pause_no_step'})..."
        )
        self._env_override_active = False
        self._risk_last_submit_time = 0.0
        self._risk_last_handled_token = -1
        self.quit_requested = False
        self._reset_session_buffers()

        step_count = 0
        next_prompt_step = self.max_steps if self.max_steps > 0 else 0
        current_chunk: List[np.ndarray] = []
        current_policy_chunk: List[np.ndarray] = []
        chunk_pointer = 0
        inference_requested = False
        self.episode_done = False
        risk_triggered = False
        is_human_prev = False

        while not self.obs_queue.empty():
            try:
                self.obs_queue.get_nowait()
            except queue.Empty:
                break
        self._clear_action_queue()
        while not self._risk_queue.empty():
            try:
                self._risk_queue.get_nowait()
            except queue.Empty:
                break
        while not self._risk_result_queue.empty():
            try:
                self._risk_result_queue.get_nowait()
            except queue.Empty:
                break

        with self._key_lock:
            self._init_requested = False
            self._start_requested = False
            self._stop_requested = False
            self._quit_requested = False
            self._teleop_toggle_requested = False

        print(
            f"[HITLDeployRunner] 空闲待命：按 {self.init_key} 初始化，"
            f"按 {self.teleop_key} 切换遥操，"
            f"按 {self.start_key} 从当前状态开始录制，"
            f"录制中按 {self.stop_key} 结束，"
            f"按 {self.quit_key} 退出。"
        )
        obs: Dict[str, Any] = self._refresh_obs_with_safe_step({})
        while not self.episode_done:
            if self._consume_quit_requested():
                self._disable_teleop()
                print("[HITLDeployRunner] 收到退出指令，结束部署循环。")
                self.episode_done = True
                return
            if self._consume_teleop_toggle_requested():
                self._toggle_env_teleop()
                continue
            if self._consume_init_requested():
                obs = self._reset_for_deploy_init()
                continue
            if self._consume_start_requested():
                print("[HITLDeployRunner] 开始录制当前 rollout。")
                break
            obs = self._refresh_obs_with_safe_step(obs)

        self._queue_latest_obs_for_inference(obs)
        inference_requested = True

        while not self.episode_done:
            if self._consume_quit_requested():
                self._disable_teleop()
                print("[HITLDeployRunner] 收到退出指令，当前 rollout 不保存。")
                self.episode_done = True
                break
            if self._consume_teleop_toggle_requested():
                self._toggle_env_teleop()

            stop_requested = self._consume_stop_requested()
            continue_requested = self._consume_continue_requested()

            env_override_requested = self._read_env_override_requested()
            is_human_override = bool(env_override_requested)

            if is_human_override and not is_human_prev:
                self._mark_last_boundary("teleop_start", terminated=True)
                self._clear_action_queue()
                current_chunk = []
                current_policy_chunk = []
                chunk_pointer = 0
                inference_requested = False

            if is_human_prev and not is_human_override:
                self._mark_last_boundary("teleop_end", terminated=True)
                self._clear_action_queue()
                current_chunk = []
                current_policy_chunk = []
                chunk_pointer = 0
                inference_requested = False
                obs = self._refresh_obs_with_safe_step(obs)
                self._queue_latest_obs_for_inference(obs)
                inference_requested = True
                risk_triggered = False
                is_human_prev = False
                continue

            risk_result = self._poll_latest_risk_result()
            if (
                risk_result is not None
                and int(risk_result["token"]) > self._risk_last_handled_token
            ):
                self._risk_last_handled_token = int(risk_result["token"])
                if bool(risk_result["risk"]) and (not is_human_override) and (not risk_triggered):
                    self._mark_last_boundary("risk_trigger", terminated=True)
                    self._clear_action_queue()
                    current_chunk = []
                    current_policy_chunk = []
                    chunk_pointer = 0
                    inference_requested = False
                    risk_triggered = True
                    print("[HITLDeployRunner] 风险指标触发，等待专家接管或继续指令。")

            if stop_requested:
                self._disable_teleop()
                final_success = self._prompt_binary_success()
                if final_success is None:
                    print("[HITLDeployRunner] 已删除当前 rollout，不保存。")
                    self.episode_done = True
                    break
                self._mark_last_boundary(
                    "manual_save",
                    terminated=True,
                    success=final_success,
                    force=True,
                )
                self._ensure_final_obs(obs)
                self._save_session_trajectory()
                self.episode_done = True
                break

            if risk_triggered and (not is_human_override):
                if continue_requested:
                    self._mark_last_boundary("risk_continue", terminated=True)
                    risk_triggered = False
                    self._clear_action_queue()
                    current_chunk = []
                    current_policy_chunk = []
                    chunk_pointer = 0
                    inference_requested = False
                    obs = self._refresh_obs_with_safe_step(obs)
                    self._queue_latest_obs_for_inference(obs)
                    inference_requested = True
                    continue
                if not self.risk_use_safe_action:
                    time.sleep(max(0.0, self.planning_wait_sleep))
                    is_human_prev = False
                    continue

            if is_human_override:
                policy_action = self._get_safe_action(obs)
                logged_policy_action = np.asarray(policy_action, dtype=np.float32)
                fallback_action_type = 2
            elif risk_triggered and self.risk_use_safe_action:
                policy_action = self._get_safe_action(obs)
                logged_policy_action = np.asarray(policy_action, dtype=np.float32)
                fallback_action_type = 1
            else:
                need_chunk = chunk_pointer >= len(current_chunk)

                # 和 BaseRunner 对齐：chunk 耗尽后只请求一次新推理。
                # 等待期间反复投递 obs 会让 Identity replay cursor 被提前推进；
                # 当前 chunk 未耗尽时接收新 chunk 会造成 replay 跳帧。
                if need_chunk and not inference_requested:
                    self._queue_latest_obs_for_inference(obs)
                    inference_requested = True

                if need_chunk:
                    try:
                        new_chunk = self.action_queue.get_nowait()
                        current_policy_chunk = np.asarray(new_chunk, dtype=np.float32)
                        current_chunk = self._transform_policy_chunk_to_env_chunk(
                            current_policy_chunk
                        )
                        exec_horizon = min(
                            int(self.act_horizon),
                            len(current_policy_chunk),
                            len(current_chunk),
                        )
                        current_policy_chunk = current_policy_chunk[:exec_horizon]
                        current_chunk = current_chunk[:exec_horizon]
                        chunk_pointer = 0
                        inference_requested = False
                    except queue.Empty:
                        pass

                if chunk_pointer < len(current_chunk):
                    policy_action = np.asarray(current_chunk[chunk_pointer], dtype=np.float32)
                    logged_policy_action = np.asarray(
                        current_policy_chunk[chunk_pointer], dtype=np.float32
                    )
                    chunk_pointer += 1
                    fallback_action_type = 0
                    self._submit_risk_check(obs, logged_policy_action)
                else:
                    policy_action = self._get_safe_action(obs)
                    logged_policy_action = np.asarray(policy_action, dtype=np.float32)
                    fallback_action_type = 1

            next_obs, reward, terminated, truncated, info = self.env.step(policy_action)
            executed_action = self._flatten_env_action(
                info.get("actual_action"),
                policy_action,
            )
            env_action_type = self._extract_env_action_type(
                info, fallback_type=fallback_action_type
            )
            env_intervened = self._extract_env_intervened(info)

            self._append_record(
                obs=obs,
                executed_action=executed_action,
                policy_action=logged_policy_action,
                runner_action_type=fallback_action_type,
                env_action_type=env_action_type,
                reward=float(reward),
                success=self._extract_success(info, terminated),
                intervention=bool(env_intervened or is_human_override),
                terminated=bool(terminated),
                truncated=bool(truncated),
                risk_active=bool(risk_triggered),
            )

            step_count += 1
            obs = next_obs
            is_human_prev = is_human_override

            if next_prompt_step > 0 and step_count >= next_prompt_step:
                if is_human_override:
                    print(
                        f"[HITLDeployRunner] 遥操状态下达到 {step_count} 步，默认继续。"
                    )
                    next_prompt_step += self.max_steps
                else:
                    choice = self._prompt_max_step_choice()
                    if choice == "save":
                        final_success = self._prompt_binary_success()
                        if final_success is None:
                            print("[HITLDeployRunner] 已删除当前 rollout，不保存。")
                            self.episode_done = True
                            break
                        self._mark_last_boundary(
                            "max_step_save",
                            terminated=True,
                            success=final_success,
                            force=True,
                        )
                        self._ensure_final_obs(obs)
                        self._save_session_trajectory()
                        self.episode_done = True
                        break
                    if choice == "delete":
                        print("[HITLDeployRunner] 已丢弃当前 rollout session。")
                        self.episode_done = True
                        break
                    next_prompt_step += self.max_steps

            if terminated or truncated:
                self._disable_teleop()
                self._mark_last_boundary(
                    "episode_end",
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    force=True,
                )
                self._ensure_final_obs(obs)
                self._save_session_trajectory()
                self.episode_done = True
                print(
                    f"[HITLDeployRunner] Episode finished. Steps: {step_count} "
                    f"(Terminated: {terminated}, Truncated: {truncated})"
                )
                break

    def stop_worker(self):
        self._risk_stop = True
        super().stop_worker()
        if self.risk_thread is not None:
            self.risk_thread.join(timeout=1.0)

    def _serialize_env_cfg(self) -> str:
        try:
            return OmegaConf.to_yaml(self.cfg.env)
        except Exception:
            env_cfg = self.cfg.env
            if is_dataclass(env_cfg):
                env_cfg = asdict(env_cfg)
            elif hasattr(env_cfg, "__dict__"):
                env_cfg = {
                    key: value
                    for key, value in vars(env_cfg).items()
                    if not key.startswith("_")
                }
            try:
                return json.dumps(env_cfg, ensure_ascii=True, default=str, indent=2)
            except Exception:
                return repr(env_cfg)
