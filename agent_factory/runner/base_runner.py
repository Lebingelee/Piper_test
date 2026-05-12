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
        
        # 4. 定义核心容器
        self.obs_queue = queue.Queue(maxsize=1)
        self.action_queue = queue.Queue(maxsize=1)
        
        # 轨迹内存字典：增加 RL 标准字段 (rewards, terminated, truncated)
        self.current_traj = {
            'obs': [], 
            'action': [],
            'action_type': [],
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
        
        print(f"[BaseRunner] 基础设施初始化完成。Hz: {self.control_hz}, ActHorizon: {self.act_horizon}, 当前存储队列长度: {len(self.saved_files_fifo)}")

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
                obs = self.obs_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # 2. 数据准备：注意 obs 的形状格式
            # 环境返回的通常是单帧 (H, W, C)，模型需要 (B, T, H, W, C) 或 (B, T, D)
            # 在这里，obs 已经是 Wrapper 包装过的，通常包含 obs_horizon 长度的序列
            # 我们需要为其添加 Batch 维度并移动到设备
            obs_batch = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    obs_batch[k] = torch.from_numpy(v).unsqueeze(0).to(self.device)
                elif isinstance(v, torch.Tensor):
                    obs_batch[k] = v.unsqueeze(0).to(self.device)
                else:
                    obs_batch[k] = v

            # 3. 调用 agent.sample_action(obs)
            try:
                # 记录推理开始时间
                t_start = time.time()
                
                with torch.no_grad():
                    # 推理结果 action_chunk 通常形状为 (B, pred_horizon, action_dim)
                    # sample_action 内部已完成反归一化
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
        # 1) 首选 wrapper 链：FrameStack -> MetadataAdapter -> Base Env
        if hasattr(self.env, "has_wrapper_attr") and self.env.has_wrapper_attr("get_safe_action"):
            safe_fn = self.env.get_wrapper_attr("get_safe_action")
            safe_action = safe_fn()
            if isinstance(safe_action, dict):
                # 防御式处理：若返回 dict，尝试通过 wrapper 链做扁平化
                if self.env.has_wrapper_attr("flatten_action"):
                    flatten_fn = self.env.get_wrapper_attr("flatten_action")
                    return flatten_fn(safe_action).astype(np.float32, copy=False)
                action_dim = getattr(self.cfg.env, "action_dim", 7)
                return np.zeros(action_dim, dtype=np.float32)
            return np.asarray(safe_action, dtype=np.float32)

        # 2) 回退到底层环境
        if hasattr(self.env.unwrapped, "get_safe_action"):
            safe_action = self.env.unwrapped.get_safe_action()
            if isinstance(safe_action, dict):
                if hasattr(self.env, "has_wrapper_attr") and self.env.has_wrapper_attr("flatten_action"):
                    flatten_fn = self.env.get_wrapper_attr("flatten_action")
                    return flatten_fn(safe_action).astype(np.float32, copy=False)
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
        obs, _ = self.env.reset()

        step_count = 0
        current_chunk = []
        chunk_pointer = 0  # 追踪当前 chunk 的执行位置

        # 清空当前轨迹缓存
        self.current_traj = {
            'obs': [], 'action': [], 'action_type': [],
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
        self.obs_queue.put_nowait(obs)

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
                except queue.Empty:
                    # 计划还没来，继续保持 is_planning = True
                    pass
            
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
                action_type = 1

            # 3. 执行物理动作 (无论是否开始录制，都要 step 环境以维持频率)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # --- 核心改进：若尚未获取第一个 chunk，则跳过记录逻辑 ---
            if not has_started:
                obs = next_obs
                continue

            # 打印调试信息（仅在正式开始后）
            if is_planning:
                print(f"safe_action at {step_count} frame (re-planning gap)")

            # 存入观测和动作
            self.current_traj['obs'].append(copy.deepcopy(obs))
            self.current_traj['action'].append(np.array(action))
            self.current_traj['action_type'].append(action_type)

            step_count += 1

            # 【兜底截断】依赖 cfg.env.max_episode_steps 控制最大长度
            if step_count >= self.max_steps:
                truncated = True

            # 存入 RL 环境转移状态
            self.current_traj['rewards'].append(reward)
            self.current_traj['terminated'].append(terminated)
            self.current_traj['truncated'].append(truncated)

            obs = next_obs

            # 4. 检查是否需要重新进入规划模式
            # 当 chunk 执行完毕时，切换回规划模式
            if not is_planning and chunk_pointer >= self.act_horizon:
                is_planning = True
                # 使用最新 obs 请求下一计划, 放入前清空以防 obs 堆积
                if self.obs_queue.empty():
                    self.obs_queue.put_nowait(obs) 

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
        if len(self.current_traj['action']) < 5:
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
            # 1. 存储动作与辅助信息
            f.create_dataset("action", data=np.stack(self.current_traj['action']).astype(np.float32))
            f.create_dataset("action_type", data=np.stack(self.current_traj['action_type']).astype(np.int32))
            if "policy_action" in self.current_traj and len(self.current_traj["policy_action"]) == len(self.current_traj["action"]):
                f.create_dataset(
                    "policy_action",
                    data=np.stack(self.current_traj["policy_action"]).astype(np.float32),
                )
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
            # 原始 obs 结构示例: {'rgb': (T_stack, C, H, W), 'state': (T_stack, D)}
            raw_obs_list = self.current_traj['obs']
            
            # 只提取每个 Step 堆叠序列中的最后一帧 (即当前真实观测)
            # 最终形状: (T_episode, C, H, W) 和 (T_episode, D)
            processed_rgb = []
            processed_state = []
            for obs in raw_obs_list:
                # RGB 处理: (T_stack, C, H, W) -> (C, H, W)
                img = obs['rgb']
                if isinstance(img, torch.Tensor): img = img.cpu().numpy()
                if img.ndim == 4: img = img[-1] # 取最后一帧
                
                # 转换 float32 (0-1) 为 uint8 (0-255)
                if img.dtype != np.uint8:
                    img = (img * 255.0).astype(np.uint8)
                processed_rgb.append(img)
                
                # State 处理: (T_stack, D) -> (D)
                st = obs['state']
                if isinstance(st, torch.Tensor): st = st.cpu().numpy()
                if st.ndim == 2: st = st[-1]
                processed_state.append(st)
                
            obs_group.create_dataset("rgb", data=np.stack(processed_rgb), compression="gzip", compression_opts=4)
            obs_group.create_dataset("state", data=np.stack(processed_state).astype(np.float32))

            # 3. 存储信号位
            f.create_dataset("rewards", data=np.array(self.current_traj['rewards'], dtype=np.float32))
            f.create_dataset("terminated", data=np.array(self.current_traj['terminated'], dtype=bool))
            f.create_dataset("truncated", data=np.array(self.current_traj['truncated'], dtype=bool))
            
            # 4. 存储元数据 (meta/) - 实现全生命周期溯源
            meta_group = f.create_group("meta")
            
            # 存储环境配置 (YAML)
            from omegaconf import OmegaConf
            env_cfg_yaml = OmegaConf.to_yaml(self.cfg.env)
            meta_group.create_dataset("env_cfg", data=env_cfg_yaml)
            
            # 存储元数据 Key 结构 (JSON/YAML)
            env_meta = self._get_env_metadata()
            if env_meta:
                import json
                meta_keys_json = json.dumps(env_meta)
                meta_group.create_dataset("env_meta", data=meta_keys_json)

            # 属性
            f.attrs['success'] = bool(np.any(self.current_traj['terminated']))
            f.attrs['length'] = len(self.current_traj['action'])
            
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
