import os
import queue
import threading
import time
import torch
import numpy as np
from typing import Dict, Any, Optional

from agent_factory.agents.registry import make_agent
from agent_factory.control import build_action_transform_meta, forward_transform_action
from agent_factory.env.env_factories import create_env
from agent_factory.runner.config_utils import runner_config_get
from omegaconf import DictConfig

class BaseRunner:
    """
    BaseRunner: 底层异步基座
    负责构建非阻塞的推理流水线、维持严格的控制频率、执行基础的动作分发与数据存储。
    适用于纯模型的自主采集与验证。
    """
    def __init__(self, cfg: DictConfig, agent: Any, env: Any):
        """
        📍 Step 1: 基础设施与依赖注入
        
        Args:
            cfg (DictConfig): 全局配置
            agent (Any): 已经初始化的 Policy 模型 (处于 eval 模式)
            env (Any): 已经实例化的环境
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # 1. 注入环境与模型
        self.env = env
        self.agent = agent
        
        # 3. 读取配置文件，初始化参数
        self.control_hz = cfg.runner.control_hz
        self.act_horizon = cfg.env.act_horizon
        self.buffer_capacity = cfg.runner.buffer_capacity
        self.redundancy_margin = cfg.runner.redundancy_margin
        
        # 从环境配置中读取最大步数 (Single Source of Truth)
        self.max_steps = getattr(cfg.env, 'max_episode_steps', 250)
        self.agent_control_mode = getattr(
            cfg,
            "agent_control_mode",
            getattr(cfg.env, "env_control_mode", "delta_pose"),
        )
        self.env_control_mode = getattr(
            cfg.env,
            "env_control_mode",
            getattr(cfg.env, "control_mode", "delta_pose"),
        )
        self.no_safe_action_gap = bool(
            getattr(cfg.runner, "no_safe_action_gap", False)
        )
        self.planning_wait_sleep = float(
            getattr(cfg.runner, "planning_wait_sleep", 0.002)
        )
        
        # 4. 定义核心容器
        self.obs_queue = queue.Queue(maxsize=1)
        self.action_queue = queue.Queue(maxsize=1)
        
        # 轨迹内存字典：增加 RL 标准字段 (rewards, terminated, truncated)
        self.current_traj = {
            'obs': [], 
            'action': [],
            'action_type': [],
            'prompt': None,
            'success': [],
            'intervention': [],
            'rewards': [],
            'terminated': [],
            'truncated': []
        }
        
        # 文件存储与 FIFO 管理跨重启恢复
        self.save_dir = getattr(self.cfg.runner, 'save_dir', 'data/online_rl')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.saved_files_fifo = []
        import glob
        existing_files = glob.glob(os.path.join(self.save_dir, "*.h5"))
        # 按修改时间排序，确保最早的文件在队列前面
        existing_files.sort(key=os.path.getmtime)
        self.saved_files_fifo.extend(existing_files)
        
        # 解析最大序号或直接用现有长度
        max_id = -1
        for f in existing_files:
            try:
                # 假设格式为 traj_XXXX_YYYYMMDD_HHMMSS.h5
                basename = os.path.basename(f)
                traj_id = int(basename.split('_')[1])
                if traj_id > max_id: max_id = traj_id
            except:
                pass
        self.traj_counter = max_id + 1 if max_id >= 0 else len(self.saved_files_fifo)
        
        # 线程运行状态
        self._stop_thread = False
        
        # 📍 Step 2: 启动推理线程 (Daemon Thread)
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        
        gap_mode = "wait_no_step" if self.no_safe_action_gap else "safe_action_step"
        print(f"[BaseRunner] 基础设施初始化完成。Hz: {self.control_hz}, ActHorizon: {self.act_horizon}, GapMode: {gap_mode}, 当前存储队列长度: {len(self.saved_files_fifo)}")

    def _unwrapped_env(self):
        return getattr(self.env, "unwrapped", self.env)

    def _build_forward_transform_meta(self) -> Dict[str, Any]:
        base_env = self._unwrapped_env()
        if not hasattr(base_env, "get_control_state"):
            raise NotImplementedError(
                "Forward control transform requires env.get_control_state() when agent/env control modes differ."
            )
        control_state = base_env.get_control_state()
        env_meta = (
            base_env.get_env_metadata()
            if hasattr(base_env, "get_env_metadata")
            else {
                "obs": getattr(base_env, "meta_keys", {}).get("obs", {}),
                "action": getattr(base_env, "meta_keys", {}).get("action", {}),
            }
        )
        return build_action_transform_meta(
            current_pose=control_state.get("arm_pose"),
            current_poses=control_state.get("arm_poses"),
            env_meta=env_meta,
        )

    def _transform_policy_chunk_to_env_chunk(self, action_chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if bool(getattr(self.agent, "uses_env_safe_action", False)):
            return chunk
        if self.agent_control_mode == self.env_control_mode:
            return chunk
        meta = self._build_forward_transform_meta()
        return forward_transform_action(
            obs=None,
            agent_action=chunk,
            agent_control_mode=self.agent_control_mode,
            env_control_mode=self.env_control_mode,
            meta=meta,
        )

    def _get_env_metadata(self) -> Dict[str, Any]:
        base_env = self._unwrapped_env()
        if hasattr(base_env, "get_env_metadata"):
            return base_env.get_env_metadata()
        return {
            "obs": getattr(base_env, "meta_keys", {}).get("obs", {}),
            "action": getattr(base_env, "meta_keys", {}).get("action", {}),
            "env_control_mode": getattr(base_env, "env_control_mode", self.env_control_mode),
            "controller_backend": getattr(base_env, "controller_backend", ""),
            "control": getattr(base_env, "control_meta", {}),
        }

    def _get_env_attr(self, attr: str, default: Any = None) -> Any:
        try:
            if hasattr(self.env, "get_wrapper_attr"):
                return self.env.get_wrapper_attr(attr)
        except AttributeError:
            pass
        except Exception:
            pass
        for candidate in (self.env, self._unwrapped_env()):
            if hasattr(candidate, attr):
                return getattr(candidate, attr)
        return default

    @staticmethod
    def _prompt_to_str(prompt: Any) -> Optional[str]:
        if prompt is None:
            return None
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        if isinstance(prompt, np.ndarray):
            if prompt.size == 0:
                return None
            prompt = prompt.reshape(-1)[0]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
        if isinstance(prompt, (list, tuple)):
            if not prompt:
                return None
            return BaseRunner._prompt_to_str(prompt[0])
        text = str(prompt)
        return text if text.strip() else None

    def _resolve_prompt(self, info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if info and "prompt" in info:
            prompt = self._prompt_to_str(info.get("prompt"))
            if prompt is not None:
                return prompt

        get_prompt = self._get_env_attr("get_prompt", None)
        if callable(get_prompt):
            try:
                prompt = self._prompt_to_str(get_prompt())
                if prompt is not None:
                    return prompt
            except Exception:
                pass

        task_description = self._get_env_attr("task_description", None)
        prompt = self._prompt_to_str(task_description)
        if prompt is not None:
            return prompt

        prompt = self._prompt_to_str(getattr(self.cfg.env, "prompt", None))
        if prompt is not None:
            return prompt

        prompt = self._prompt_to_str(runner_config_get(self.cfg, "prompt", None))
        if prompt is not None:
            return prompt
        return None

    def _make_inference_request(self, obs: Dict[str, Any], prompt: Optional[str]):
        return {"obs": obs, "prompt": prompt}

    def _agent_accepts_prompt(self) -> bool:
        return bool(getattr(self.agent, "requires_prompt", False))

    def _prepare_obs_for_agent(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if bool(getattr(self.agent, "uses_raw_observations", False)):
            return obs

        obs_batch = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                obs_batch[k] = torch.from_numpy(v).unsqueeze(0).to(self.device)
            elif isinstance(v, torch.Tensor):
                obs_batch[k] = v.unsqueeze(0).to(self.device)
            else:
                obs_batch[k] = v
        return obs_batch

    def _recorded_env_action(self, action_from_info: Any, fallback_action: np.ndarray) -> Any:
        import copy
        if action_from_info is None:
            return np.asarray(fallback_action, dtype=np.float32)
        return copy.deepcopy(action_from_info)

    def _prepare_action_for_storage(self, action_value: Any) -> Any:
        if isinstance(action_value, dict):
            return {
                key: self._prepare_action_for_storage(value)
                for key, value in action_value.items()
            }
        return np.asarray(action_value, dtype=np.float32)

    @staticmethod
    def _any_true(value: Any) -> bool:
        if isinstance(value, dict):
            return any(BaseRunner._any_true(v) for v in value.values())
        arr = np.asarray(value).reshape(-1)
        return bool(arr.size > 0 and np.any(arr))

    def _extract_env_intervened(self, info: Dict[str, Any]) -> bool:
        if "intervened_map" in info:
            return self._any_true(info["intervened_map"])
        if "intervened" in info:
            return self._any_true(info["intervened"])
        if "intervention" in info:
            return self._any_true(info["intervention"])
        return False

    def _extract_success(self, info: Dict[str, Any], terminated: bool) -> bool:
        for key in ("success", "is_success"):
            if key in info:
                return self._any_true(info[key])
        return bool(terminated)

    def _normalize_step_flags(
        self,
        info: Dict[str, Any],
        terminated: bool,
        truncated: bool,
        success: bool,
    ):
        return bool(terminated), bool(truncated), bool(success)

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

    def stop_worker(self):
        """安全停止推理线程"""
        self._stop_thread = True

    def _inference_worker(self):
        """
        📍 Step 2: 多线程异步推理模块
        作为一个守护线程运行，阻塞监听 obs_queue，调用 agent 推理并放入 action_queue。
        """
        print("[BaseRunner] 推理线程已启动。")
        while not self._stop_thread:
            try:
                # 1. 阻塞监听 self.obs_queue (带有超时以便退出循环)
                # maxsize=1 确保我们总是拿到最实时的一帧
                request = self.obs_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if isinstance(request, dict) and "obs" in request and "prompt" in request:
                obs = request["obs"]
                prompt = request.get("prompt")
            else:
                obs = request
                prompt = self._resolve_prompt()

            # 2. 数据准备：注意 obs 的形状格式
            # 环境返回的通常是单帧 (H, W, C)，模型需要 (B, T, H, W, C) 或 (B, T, D)
            # 在这里，obs 已经是 Wrapper 包装过的，通常包含 obs_horizon 长度的序列
            # 我们需要为其添加 Batch 维度并移动到设备
            obs_batch = self._prepare_obs_for_agent(obs)

            # 3. 调用 agent.sample_action(obs)
            try:
                # 记录推理开始时间
                t_start = time.time()
                
                if bool(getattr(self.agent, "uses_env_safe_action", False)):
                    if self.no_safe_action_gap:
                        raise RuntimeError(
                            "runner.no_safe_action_gap=True is incompatible with "
                            "agents that require env safe actions."
                        )
                    safe_action = self._get_safe_action(obs)
                    #print(f"[BaseRunner] 使用 env-safe-action 进行填充: {safe_action}")
                    pred_horizon = int(
                        getattr(
                            getattr(self.cfg, "actor", None),
                            "pred_horizon",
                            getattr(self.cfg.env, "pred_horizon", self.act_horizon),
                        )
                    )
                    action_chunk = np.repeat(
                        np.asarray(safe_action, dtype=np.float32).reshape(1, -1),
                        pred_horizon,
                        axis=0,
                    )
                else:
                    with torch.no_grad():
                        # 推理结果 action_chunk 通常形状为 (B, pred_horizon, action_dim)
                        # sample_action 内部已完成反归一化
                        if self._agent_accepts_prompt():
                            action_chunk = self.agent.sample_action(obs_batch, prompt=prompt)
                        else:
                            action_chunk = self.agent.sample_action(obs_batch)
                        
                        # 4. 后处理：去掉 Batch 维度，并确保数据在 CPU
                        if isinstance(action_chunk, torch.Tensor):
                            action_chunk = action_chunk.detach().cpu().numpy()
                        
                        if action_chunk.ndim == 3: # (1, pred_horizon, action_dim)
                            action_chunk = action_chunk[0]
                
                t_end = time.time()
                print(f"[BaseRunner] Inference time: {(t_end - t_start)*1000:.2f}ms")

            except Exception as e:
                print(f"[BaseRunner] 推理失败: {e}")
                import traceback
                traceback.print_exc()
                continue

            # 5. 将生成的 action_chunk 放入 action_queue
            # 若队列满则替换旧任务 (maxsize=1)
            try:
                if self.action_queue.full():
                    try:
                        self.action_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.action_queue.put_nowait(action_chunk)
            except queue.Full:
                pass # 理论上不会发生，因为前面已经 get_nowait 了

    def _get_safe_action(self, obs: dict) -> np.ndarray:
        """
        优先通过 wrapper 链获取 get_safe_action，确保命中 MetadataAdapterWrapper
        并返回与 Policy 输出维度一致的扁平化向量。
        """
        # 1) 首选 wrapper 链：MetadataAdapterWrapper 明确暴露扁平 safe action。
        try:
            safe_flatten_fn = self.env.get_wrapper_attr("get_safe_flatten_action")
            return np.asarray(safe_flatten_fn(), dtype=np.float32)
        except AttributeError:
            pass

        # 2) 回退到底层环境
        if hasattr(self.env.unwrapped, "get_safe_action"):
            safe_action = self.env.unwrapped.get_safe_action()
            if isinstance(safe_action, dict):
                try:
                    flatten_fn = self.env.get_wrapper_attr("flatten_action")
                    return flatten_fn(safe_action).astype(np.float32, copy=False)
                except AttributeError:
                    pass
                action_dim = getattr(self.cfg.env, "action_dim", 7)
                return np.zeros(action_dim, dtype=np.float32)
            return np.asarray(safe_action, dtype=np.float32)

        # 3) 兼容极简环境
        action_dim = getattr(self.cfg.env, "action_dim", 7)
        return np.zeros(action_dim, dtype=np.float32)

    def run(self):
        """
        📍 Step 3: 核心控制循环
        严格遵循 `eval` 脚本的 "规划-执行" 模式。
        只有在一个 action_chunk 完全消耗完 act_horizon 步后，才获取最新的 obs 进行推理。
        在推理期间的帧用 safe_action 填充。
        """
        print(f"[BaseRunner] 开始执行核心控制循环 (Hz: {self.control_hz})...")
        #self.env.unwrapped.reset(options={"sync_master": True})
        obs, _ = self.env.reset()

        step_count = 0
        current_chunk = []
        chunk_pointer = 0  # 追踪当前 chunk 的执行位置

        # 清空当前轨迹缓存
        self.current_traj = {
            'obs': [], 'action': [], 'action_type': [],
            'prompt': self._resolve_prompt(),
            'success': [], 'intervention': [],
            'rewards': [], 'terminated': [], 'truncated': []
        }
        self.episode_done = False

        # --- 初始化状态：进入规划模式 ---
        is_planning = True
        has_started = False  # 只有在获取到第一个有效 chunk 后才开始录制
        # 清空队列防止旧数据残留 (跨 episode 污染)
        while not self.obs_queue.empty():
            try: self.obs_queue.get_nowait()
            except queue.Empty: break
        while not self.action_queue.empty():
            try: self.action_queue.get_nowait()
            except queue.Empty: break
        # 首次强制唤醒推理线程
        self.obs_queue.put_nowait(self._make_inference_request(obs, self.current_traj.get("prompt")))
        planning_wait_start = time.time()
        planning_wait_logged = False

        import copy

        while not self.episode_done:
            # 1. 检查新计划是否到达
            if is_planning:
                try:
                    # 非阻塞获取新计划
                    new_chunk = self.action_queue.get_nowait()
                    current_chunk = self._transform_policy_chunk_to_env_chunk(new_chunk)
                    chunk_pointer = 0  # 重置执行指针
                    is_planning = False # 切换到执行模式
                    has_started = True  # 首次获取到动作，标记开始录制
                    if self.no_safe_action_gap:
                        wait_ms = (time.time() - planning_wait_start) * 1000.0
                        print(
                            f"[BaseRunner] action_chunk ready after no-step wait: "
                            f"{wait_ms:.2f}ms"
                        )
                except queue.Empty:
                    # 计划还没来，继续保持 is_planning = True
                    if self.no_safe_action_gap:
                        if not planning_wait_logged:
                            print(
                                "[BaseRunner] waiting for action_chunk without "
                                "env.step() or safe_action..."
                            )
                            planning_wait_logged = True
                        time.sleep(max(0.0, self.planning_wait_sleep))
                        continue
            
            # 2. 动作分发：执行或等待
            if not is_planning:
                # --- 执行模式 ---
                action = current_chunk[chunk_pointer]
                action_type = 0
                chunk_pointer += 1
            else:
                # --- 规划模式 (等待 new_chunk) ---
                pass
                action = self._get_safe_action(obs)
                #print(action)
                action_type = 1

            # 3. 执行物理动作 (无论是否开始录制，都要 step 环境以维持频率)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            step_prompt = self._resolve_prompt(info)
            if step_prompt is not None:
                self.current_traj["prompt"] = step_prompt
            #print(action)
            executed_action = self._recorded_env_action(info.get("actual_action"), action)
            intervention = self._extract_env_intervened(info)
            env_action_type = self._extract_env_action_type(
                info,
                fallback_type=2 if intervention else action_type,
            )

            # --- 核心改进：若尚未获取第一个 chunk，则跳过记录逻辑 ---
            if not has_started:
                obs = next_obs
                continue

            # 打印调试信息（仅在正式开始后）
            if is_planning:
                print(f"safe_action at {step_count} frame (re-planning gap)")

            # 存入观测和动作
            self.current_traj['obs'].append(copy.deepcopy(obs))
            self.current_traj['action'].append(self._prepare_action_for_storage(executed_action))
            self.current_traj['action_type'].append(np.int32(env_action_type))

            step_count += 1

            # 【兜底截断】依赖 cfg.env.max_episode_steps 控制最大长度
            if step_count >= self.max_steps:
                truncated = True

            # 存入 RL 环境转移状态
            success = self._extract_success(info, terminated)
            terminated, truncated, success = self._normalize_step_flags(
                info=info,
                terminated=terminated,
                truncated=truncated,
                success=success,
            )
            self.current_traj['rewards'].append(reward)
            self.current_traj['success'].append(bool(success))
            self.current_traj['intervention'].append(bool(intervention))
            self.current_traj['terminated'].append(bool(terminated))
            self.current_traj['truncated'].append(bool(truncated))

            obs = next_obs

            # 4. 检查是否需要重新进入规划模式
            # 当 chunk 执行完毕时，切换回规划模式
            if not is_planning and chunk_pointer >= self.act_horizon:
                is_planning = True
                planning_wait_start = time.time()
                planning_wait_logged = False
                # 使用最新 obs 请求下一计划, 放入前清空以防 obs 堆积
                if self.obs_queue.empty():
                    self.obs_queue.put_nowait(self._make_inference_request(obs, self.current_traj.get("prompt"))) 

            if terminated or truncated:
                self.episode_done = True
                # 追加最终的 observation，以满足 Offline RL 数据集规范
                self.current_traj['obs'].append(copy.deepcopy(obs))
                print(f"[BaseRunner] Episode finished. Total Steps: {step_count} (Terminated: {terminated}, Truncated: {truncated})")
                self._save_trajectory()
                break

    def _save_trajectory(self):
        """
        📍 Step 4: 统一规范存储 (Unified Specification compliant)
        将 self.current_traj 序列化为标准化 .h5 文件。
        对观测数据进行降维（仅保留最新帧）并转换图像为 uint8 以节省空间。
        """
        min_save_steps = int(runner_config_get(self.cfg, "min_save_steps", 5))
        if len(self.current_traj['action']) < min_save_steps:
            print("[BaseRunner] [W] 轨迹太短，忽略保存。")
            return
            
        import h5py
        import datetime
        
        # 获取保存目录
        save_dir = getattr(self.cfg.runner, 'save_dir', 'data/online_rl')
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        ts = datetime.datetime.now().strftime("%m%d_%H%M")
        traj_name = f"traj_{self.traj_counter}_{ts}.h5"
        temp_path = os.path.join(save_dir, traj_name)
        
        with h5py.File(temp_path, 'w') as f:
            num_actions = len(self.current_traj['action'])
            env_meta = self._get_env_metadata()
            # 1. 存储动作与辅助信息
            def _to_h5_array(values):
                arr = np.stack([np.asarray(value) for value in values])
                if arr.dtype.kind == "f":
                    arr = arr.astype(np.float32)
                return arr

            def _write_sequence_node(parent_group, name, values):
                sample = values[0]
                if isinstance(sample, dict):
                    child_group = parent_group.create_group(name)
                    for child_key in sorted(sample.keys()):
                        child_values = [value[child_key] for value in values]
                        _write_sequence_node(child_group, child_key, child_values)
                    return
                parent_group.create_dataset(name, data=_to_h5_array(values))

            _write_sequence_node(f, "action", self.current_traj["action"])
            f.create_dataset("action_type", data=np.stack(self.current_traj['action_type']).astype(np.int32))
            if "runner_action_type" in self.current_traj and len(self.current_traj["runner_action_type"]) == len(self.current_traj["action"]):
                f.create_dataset(
                    "runner_action_type",
                    data=np.asarray(self.current_traj["runner_action_type"], dtype=np.int32),
                )
            if "env_action_type" in self.current_traj and len(self.current_traj["env_action_type"]) == len(self.current_traj["action"]):
                f.create_dataset(
                    "env_action_type",
                    data=np.asarray(self.current_traj["env_action_type"], dtype=np.int32),
                )
            
            # 2. 存储标准化观测 (obs/)
            obs_group = f.create_group("obs")
            
            # 提取 obs 序列 (处理 FrameStack 产生的时间轴)
            raw_obs_list = self.current_traj['obs']

            def _to_numpy(value):
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                return np.asarray(value)

            def _latest_frame(value):
                arr = _to_numpy(value)
                if arr.ndim >= 2 and arr.shape[0] == int(getattr(self.cfg.env, "obs_horizon", 1)):
                    arr = arr[-1]
                return arr

            def _get_nested(mapping, key_path):
                value = mapping
                for part in key_path:
                    if not isinstance(value, dict) or part not in value:
                        return None
                    value = value[part]
                return value

            def _leaf_to_save_array(value, key_path):
                arr = _latest_frame(value)
                top_key = key_path[0] if key_path else ""
                if top_key == "rgb" and arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
                elif arr.dtype.kind == "f":
                    arr = arr.astype(np.float32)
                return arr

            def _write_obs_node(parent_group, key_path, sample_value):
                leaf_name = key_path[-1]
                if isinstance(sample_value, dict):
                    child_group = parent_group.create_group(leaf_name)
                    for child_key in sorted(sample_value.keys()):
                        _write_obs_node(child_group, [*key_path, child_key], sample_value[child_key])
                    return

                values = []
                for obs in raw_obs_list:
                    value = _get_nested(obs, key_path)
                    if value is None:
                        continue
                    values.append(_leaf_to_save_array(value, key_path))
                if not values:
                    return
                kwargs = {}
                if key_path and key_path[0] == "rgb":
                    kwargs = {"compression": "gzip", "compression_opts": 4}
                parent_group.create_dataset(leaf_name, data=np.stack(values), **kwargs)

            if raw_obs_list:
                for key in sorted(raw_obs_list[0].keys()):
                    _write_obs_node(obs_group, [key], raw_obs_list[0][key])

            prompt = self._prompt_to_str(self.current_traj.get("prompt")) or ""
            string_dtype = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("prompt", data=prompt, dtype=string_dtype)

            # 3. 存储信号位
            f.create_dataset("rewards", data=np.array(self.current_traj['rewards'], dtype=np.float32))
            success = self.current_traj.get("success", [])
            intervention = self.current_traj.get("intervention", [])
            if len(success) < num_actions:
                success = list(success) + [False] * (num_actions - len(success))
            if len(intervention) < num_actions:
                intervention = list(intervention) + [False] * (num_actions - len(intervention))
            f.create_dataset("success", data=np.asarray(success[:num_actions], dtype=bool))
            f.create_dataset("terminated", data=np.array(self.current_traj['terminated'], dtype=bool))
            f.create_dataset("truncated", data=np.array(self.current_traj['truncated'], dtype=bool))
            f.create_dataset("intervention", data=np.asarray(intervention[:num_actions], dtype=bool))
            
            # 4. 存储元数据 (meta/) - 实现全生命周期溯源
            meta_group = f.create_group("meta")
            
            # 存储环境配置 (YAML)
            from omegaconf import OmegaConf
            env_cfg_yaml = OmegaConf.to_yaml(self.cfg.env)
            meta_group.create_dataset("env_cfg", data=env_cfg_yaml)
            
            # 存储元数据 Key 结构 (JSON/YAML)
            if env_meta:
                import json
                meta_keys_json = json.dumps(env_meta)
                meta_group.create_dataset("env_meta", data=meta_keys_json)

            # 属性
            f.attrs['success'] = bool(np.any(success[:num_actions]))
            f.attrs['length'] = num_actions
            f.attrs['prompt'] = prompt
            
        print(f"[BaseRunner] 统一格式轨迹已保存至: {temp_path} (Images compressed as uint8, with meta/)")
        
        # 4. 更新 FIFO 和 计数器
        self.traj_counter += 1
        self.saved_files_fifo.append(temp_path)
        
        # 5. 延迟删除逻辑
        max_files = self.buffer_capacity + self.redundancy_margin
        while len(self.saved_files_fifo) > max_files:
            file_to_delete = self.saved_files_fifo.pop(0)
            if os.path.exists(file_to_delete):
                try:
                    os.remove(file_to_delete)
                    print(f"[BaseRunner] [FIFO] 已删除冗余旧轨迹: {file_to_delete}")
                except Exception as e:
                    print(f"[BaseRunner] [FIFO] 删除旧轨迹失败 {file_to_delete}: {e}")
