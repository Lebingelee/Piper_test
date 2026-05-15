import threading
import queue
import copy
from enum import Enum
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig

from agent_factory.runner.base_runner import BaseRunner


class RunnerState(str, Enum):
    POLICY = "policy"
    HUMAN_OVERRIDE = "human_override"
    REPLAN = "replan"


class HITLRunner(BaseRunner):
    """
    通用 HITL Runner（环境无关）。

    设计原则：
    - 继承 BaseRunner，复用推理线程与轨迹存储机制。
    - 通过状态机扩展人类接管流程，尽量不侵入 BaseRunner 现有结构。
    - 使用键盘或 env.info 的干预信号维护接管状态。
    - 不依赖 env 私有原始观测接口，不重建机器人专有动作语义。
    """

    def __init__(self, cfg: DictConfig, agent: Any, env: Any):
        super().__init__(cfg=cfg, agent=agent, env=env)

        self.state: RunnerState = RunnerState.POLICY
        self.prev_state: RunnerState = RunnerState.POLICY

        self.override_key: str = str(getattr(cfg.runner, "hitl_override_key", "t")).lower()
        self.teleop_key: str = str(getattr(cfg.runner, "hitl_teleop_key", "t")).lower()
        self.override_active: bool = False
        self._override_lock = threading.Lock()

        self._env_override_active: bool = False

        self._listener = None
        self._start_override_listener()

    def _start_override_listener(self) -> None:
        """
        启动键盘监听线程（占位实现）。
        """
        try:
            from pynput import keyboard
        except Exception:
            print("[HITLRunner] pynput not available; keyboard override disabled.")
            return

        def _on_press(key):
            try:
                if not key.char:
                    return
                char = key.char.lower()
                if char == self.teleop_key:
                    self._toggle_env_teleop()
                elif char == self.override_key:
                    with self._override_lock:
                        self.override_active = not self.override_active
                    print(f"[HITLRunner] override toggled -> {self.override_active}")
            except Exception:
                pass

        self._listener = keyboard.Listener(on_press=_on_press)
        self._listener.start()

    def _toggle_env_teleop(self) -> bool:
        base_env = self._unwrapped_env()
        if hasattr(base_env, "switch_tele"):
            try:
                base_env.switch_tele("toggle")
                return True
            except Exception as exc:
                print(f"[HITLRunner] env teleop toggle failed: {exc}")
                return False
        return False

    def _read_override_flag(self) -> bool:
        with self._override_lock:
            return self.override_active

    def _read_env_override_requested(self) -> bool:
        base_env = self._unwrapped_env()
        if hasattr(base_env, "get_env_state"):
            try:
                return bool(base_env.get_env_state("teleop"))
            except (KeyError, AttributeError, TypeError, ValueError):
                pass
        return False

    def _align_action_dim(self, action: np.ndarray) -> np.ndarray:
        """
        将动作向量裁剪/补零到环境动作维度，避免维度漂移导致 step 失败。
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        try:
            target_dim = int(self.env.action_space.shape[0])
        except Exception:
            return action
        if action.shape[0] == target_dim:
            return action
        if action.shape[0] < target_dim:
            pad = np.zeros((target_dim - action.shape[0],), dtype=np.float32)
            return np.concatenate([action, pad], axis=0)
        return action[:target_dim]

    def _clear_action_queue(self) -> None:
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except queue.Empty:
                break

    def _queue_latest_obs_for_inference(self, obs: Dict[str, Any]) -> None:
        while not self.obs_queue.empty():
            try:
                self.obs_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.obs_queue.put_nowait(obs)
        except queue.Full:
            pass

    @staticmethod
    def _any_true(value: Any) -> bool:
        if isinstance(value, dict):
            return any(HITLRunner._any_true(v) for v in value.values())
        arr = np.asarray(value).reshape(-1)
        return bool(arr.size > 0 and np.any(arr))

    def _extract_env_intervened(self, info: Dict[str, Any]) -> bool:
        if "intervened_map" in info:
            return self._any_true(info["intervened_map"])
        if "intervened" in info:
            return self._any_true(info["intervened"])
        return False

    def _extract_env_action_type(self, info: Dict[str, Any], fallback_type: int) -> int:
        if "action_type_map" in info and isinstance(info["action_type_map"], dict):
            vals = [int(v) for v in info["action_type_map"].values()]
            if vals:
                return int(max(vals))
        if "action_type" in info:
            value = info["action_type"]
            if isinstance(value, dict):
                vals = [int(v) for v in value.values()]
                if vals:
                    return int(max(vals))
            try:
                return int(value)
            except Exception:
                pass
        return int(fallback_type)

    def _flatten_env_action(self, action_from_info: Any, fallback_action: np.ndarray) -> np.ndarray:
        if action_from_info is None:
            return self._align_action_dim(fallback_action)

        try:
            if isinstance(action_from_info, dict):
                if (
                    hasattr(self.env, "has_wrapper_attr")
                    and self.env.has_wrapper_attr("flatten_action")
                ):
                    flatten_fn = self.env.get_wrapper_attr("flatten_action")
                    return self._align_action_dim(
                        np.asarray(flatten_fn(action_from_info), dtype=np.float32)
                    )
                return self._align_action_dim(fallback_action)
            return self._align_action_dim(np.asarray(action_from_info, dtype=np.float32))
        except Exception:
            return self._align_action_dim(fallback_action)

    def _switch_state(self, new_state: RunnerState) -> None:
        self.prev_state = self.state
        self.state = new_state

    def _prepare_traj_for_save(self) -> None:
        """
        保存前统一整理轨迹：
        - 规范 action/action_type 的长度和维度
        - 对 action_type 做 0/1/2/3 约束
        - 保证 obs 长度为 T+1
        """
        num_actions = len(self.current_traj["action"])
        if num_actions == 0:
            return

        # 1) 动作维度统一
        for i in range(num_actions):
            self.current_traj["action"][i] = self._align_action_dim(
                self.current_traj["action"][i]
            )
            if "policy_action" in self.current_traj and i < len(self.current_traj["policy_action"]):
                self.current_traj["policy_action"][i] = self._align_action_dim(
                    self.current_traj["policy_action"][i]
                )

        # 2) action_type 归一化到 {0,1,2,3}
        norm_types = []
        for t in self.current_traj["action_type"][:num_actions]:
            t_int = int(t)
            if t_int not in (0, 1, 2, 3):
                t_int = 1
            norm_types.append(np.int32(t_int))
        self.current_traj["action_type"] = norm_types
        for key, default_val in (("runner_action_type", 1), ("env_action_type", 1)):
            if key not in self.current_traj:
                continue
            seq = [np.int32(x) for x in self.current_traj[key][:num_actions]]
            if len(seq) < num_actions:
                seq = seq + [np.int32(default_val)] * (num_actions - len(seq))
            self.current_traj[key] = seq

        if "policy_action" in self.current_traj:
            seq = self.current_traj["policy_action"][:num_actions]
            if len(seq) < num_actions:
                if len(seq) > 0:
                    last = self._align_action_dim(seq[-1])
                else:
                    action_dim = len(self.current_traj["action"][0]) if self.current_traj["action"] else 1
                    last = self._align_action_dim(np.zeros((action_dim,), dtype=np.float32))
                seq = seq + [last.copy() for _ in range(num_actions - len(seq))]
            self.current_traj["policy_action"] = [
                self._align_action_dim(np.asarray(x, dtype=np.float32)) for x in seq
            ]

        # 3) rewards/success/intervention/terminated/truncated 与动作长度对齐
        for key, default_val in (
            ("rewards", 0.0),
            ("success", False),
            ("intervention", False),
            ("terminated", False),
            ("truncated", False),
        ):
            seq = self.current_traj[key]
            if len(seq) < num_actions:
                seq = seq + [default_val] * (num_actions - len(seq))
            elif len(seq) > num_actions:
                seq = seq[:num_actions]
            self.current_traj[key] = seq

        # 4) obs 需要满足 T+1，若不满足则兜底补齐/裁剪
        target_obs_len = num_actions + 1
        obs_seq = self.current_traj["obs"]
        if len(obs_seq) < target_obs_len and len(obs_seq) > 0:
            last_obs = copy.deepcopy(obs_seq[-1])
            obs_seq = obs_seq + [last_obs] * (target_obs_len - len(obs_seq))
        elif len(obs_seq) > target_obs_len:
            obs_seq = obs_seq[:target_obs_len]
        self.current_traj["obs"] = obs_seq

    def run(self):
        """
        HITL 主循环（A/B/C 状态机）：
        A) 人类接管：清空 policy chunk，记录干预动作（type=2）
        B) 接管释放瞬间：强制重规划，本帧下发安全动作（type=1）
        C) 正常阶段：消费最新 chunk；若无 chunk 或超时则安全动作（type=1）
        """
        print(f"[HITLRunner] 开始执行 HITL 控制循环 (Hz: {self.control_hz})...")
        obs, _ = self.env.reset()
        self._env_override_active = False

        step_count = 0
        current_chunk = []
        current_policy_chunk = []
        chunk_pointer = 0
        inference_requested = False
        self.episode_done = False
        is_human_prev = False

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
        }

        while not self.obs_queue.empty():
            try:
                self.obs_queue.get_nowait()
            except queue.Empty:
                break
        self._clear_action_queue()

        self._queue_latest_obs_for_inference(obs)
        inference_requested = True

        while not self.episode_done:
            manual_override = self._read_override_flag()
            env_override_requested = self._read_env_override_requested()
            is_human_override = bool(
                manual_override or env_override_requested or self._env_override_active
            )

            # 情况 A: 进入或处于人类接管
            if is_human_override:
                self._switch_state(RunnerState.HUMAN_OVERRIDE)
                if not is_human_prev:
                    self._clear_action_queue()
                    current_chunk = []
                    current_policy_chunk = []
                    chunk_pointer = 0
                    inference_requested = False
                else:
                    # 接管持续期间持续清理，丢弃潜在 stale chunk。
                    self._clear_action_queue()
                policy_action = self._get_safe_action(obs)
                fallback_action_type = 2

            # 情况 B: 人类刚放弃接管 (True -> False)
            elif is_human_prev and (not is_human_override):
                self._switch_state(RunnerState.REPLAN)
                self._clear_action_queue()
                current_chunk = []
                current_policy_chunk = []
                chunk_pointer = 0
                inference_requested = False

                self._queue_latest_obs_for_inference(obs)
                inference_requested = True
                policy_action = self._get_safe_action(obs)
                fallback_action_type = 1

            # 情况 C: 正常模型控制阶段
            else:
                self._switch_state(RunnerState.POLICY)

                need_chunk = chunk_pointer >= len(current_chunk)

                # chunk 耗尽时触发一次新推理；等待期间不要反复推进
                # Identity replay 的内部 cursor，也不要预取后中途替换当前 chunk。
                if need_chunk and not inference_requested:
                    self._queue_latest_obs_for_inference(obs)
                    inference_requested = True

                # 仅当当前 chunk 已耗尽时接收新 chunk。HITL 不能像
                # latest-policy 模式那样中途替换，否则 replay 会跳帧。
                if need_chunk:
                    try:
                        new_chunk = self.action_queue.get_nowait()
                        current_policy_chunk = np.asarray(new_chunk, dtype=np.float32)
                        current_chunk = self._transform_policy_chunk_to_env_chunk(current_policy_chunk)
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

                # 执行动作或降级 safe_action
                if chunk_pointer < len(current_chunk):
                    policy_action = np.asarray(current_chunk[chunk_pointer], dtype=np.float32)
                    logged_policy_action = np.asarray(current_policy_chunk[chunk_pointer], dtype=np.float32)
                    chunk_pointer += 1
                    fallback_action_type = 0
                else:
                    policy_action = self._get_safe_action(obs)
                    logged_policy_action = policy_action
                    fallback_action_type = 1
            if is_human_override or (is_human_prev and (not is_human_override)):
                logged_policy_action = policy_action

            next_obs, reward, terminated, truncated, info = self.env.step(policy_action)
            #print(policy_action)
            executed_action = self._flatten_env_action(
                info.get("actual_action"),
                policy_action,
            )
            env_action_type = self._extract_env_action_type(info, fallback_type=fallback_action_type)
            env_intervened = self._extract_env_intervened(info)
            self._env_override_active = bool(env_intervened or self._read_env_override_requested())

            self.current_traj["obs"].append(copy.deepcopy(obs))
            self.current_traj["action"].append(np.asarray(executed_action, dtype=np.float32))
            self.current_traj["policy_action"].append(np.asarray(logged_policy_action, dtype=np.float32))
            self.current_traj["action_type"].append(np.int32(env_action_type))
            self.current_traj["runner_action_type"].append(np.int32(fallback_action_type))
            self.current_traj["env_action_type"].append(np.int32(env_action_type))
            self.current_traj["rewards"].append(float(reward))
            self.current_traj["success"].append(self._extract_success(info, terminated))
            self.current_traj["intervention"].append(bool(env_intervened))
            self.current_traj["terminated"].append(bool(terminated))
            self.current_traj["truncated"].append(bool(truncated))

            step_count += 1
            if step_count >= self.max_steps:
                truncated = True
                self.current_traj["truncated"][-1] = True

            obs = next_obs
            is_human_prev = is_human_override

            if terminated or truncated:
                self.episode_done = True
                self.current_traj["obs"].append(copy.deepcopy(obs))
                self._prepare_traj_for_save()
                print(
                    f"[HITLRunner] Episode finished. Steps: {step_count} "
                    f"(Terminated: {terminated}, Truncated: {truncated})"
                )
                #self._save_trajectory()
                break

    def stop_worker(self):
        super().stop_worker()
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
