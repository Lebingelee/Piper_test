import gymnasium as gym
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from gymnasium.spaces import Box, Dict as GymDict
from collections import deque
from agent_factory.env.model_transform import ModelTransformSpec


def _normalize_flatten_obs_obj(flatten_obs_obj: Any, legal_keys: List[str]) -> set:
    if flatten_obs_obj is None:
        values = ["all"]
    elif isinstance(flatten_obs_obj, str):
        values = [flatten_obs_obj]
    elif isinstance(flatten_obs_obj, (list, tuple)) or (
        hasattr(flatten_obs_obj, "__iter__") and not isinstance(flatten_obs_obj, (bytes, bytearray, dict))
    ):
        values = list(flatten_obs_obj)
    else:
        raise TypeError(
            "env.flatten_obs_obj must be a string or a list of strings, "
            f"got {type(flatten_obs_obj).__name__}."
        )

    if not all(isinstance(value, str) for value in values):
        raise TypeError("env.flatten_obs_obj must contain only strings.")

    requested = [value.strip() for value in values]
    legal = set(legal_keys)
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError("env.flatten_obs_obj='all' cannot be combined with other obs keys.")
        return set(legal_keys)

    invalid = sorted(value for value in requested if value not in legal)
    if invalid:
        raise ValueError(
            "env.flatten_obs_obj contains invalid obs keys "
            f"{invalid}. Legal keys are {sorted(legal)} or ['all']."
        )
    return set(requested)


def _box_from_tensor_like(value: Any):
    if isinstance(value, torch.Tensor):
        shape = tuple(value.shape)
    else:
        shape = tuple(np.asarray(value).shape)
    return Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)


def _space_from_observation_value(value: Any):
    if isinstance(value, dict):
        return GymDict({key: _space_from_observation_value(val) for key, val in value.items()})
    return _box_from_tensor_like(value)


def _ordered_keys(mapping: Dict[str, Any], explicit_order: Any = None) -> List[str]:
    """Use a declared model order, falling back to deterministic lexical order."""
    keys = list(mapping.keys())
    if explicit_order is None:
        return sorted(keys)
    order = list(explicit_order)
    unknown = [key for key in order if key not in mapping]
    missing = [key for key in keys if key not in order]
    if unknown or missing or len(set(order)) != len(order):
        raise ValueError(
            "explicit metadata order must contain every key exactly once; "
            f"unknown={unknown}, missing={missing}"
        )
    return order


class MetadataAdapterWrapper(gym.Wrapper):
    """
    [核心适配器] 元数据驱动的扁平化适配器 (Metadata-Driven Flattened Adapter)。
    职责：
    1. 自动读取 unwrapped.meta_keys。
    2. 执行局部字典序排序 (Lexicographical Alignment)。
    3. Obs: 将层级字典拼接为扁平的 'rgb', 'state', 'depth'。
    4. Action: 将扁平的向量动作 (Vector) 切割为层级字典 (Dict)。
    5. 转换所有输出为 torch.Tensor 并执行基础归一化 (RGB / 255.0)。
    """
    def __init__(self, env, device="cpu", flatten_obs_obj=None, flatten_action: bool = True):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped
        self.device = device
        self.flatten_action_output = bool(flatten_action)
        
        if not hasattr(self.unwrapped_env, "meta_keys"):
            raise AttributeError("Environment must have 'meta_keys' defined for MetadataAdapterWrapper.")

        # 1. Cache declared model order; legacy environments retain lexical order.
        self.meta_keys = self.unwrapped_env.meta_keys
        env_meta_getter = getattr(self.unwrapped_env, "get_env_metadata", None)
        self.env_meta = env_meta_getter() if callable(env_meta_getter) else {}
        if not isinstance(self.env_meta, dict):
            self.env_meta = {}
        obs_orders = self.env_meta.get("observation_order", self.env_meta.get("obs_order", {}))
        if not isinstance(obs_orders, dict):
            obs_orders = {}
        self.model_transform_spec = ModelTransformSpec.from_env_meta(self.env_meta, self.meta_keys)
        self.sorted_obs_keys = {
            modality: list(self.model_transform_spec.leaf_order(modality, self.meta_keys["obs"][modality]))
            for modality in self.meta_keys["obs"].keys()
        }
        self.flatten_obs_keys = _normalize_flatten_obs_obj(
            flatten_obs_obj if flatten_obs_obj is not None else ["all"],
            list(self.sorted_obs_keys.keys()),
        )
        self.vector_action_passthrough = bool(
            getattr(self.unwrapped_env, "action_is_vector", False)
            and isinstance(self.env.action_space, Box)
        )
        self.sorted_action_keys = (
            []
            if self.vector_action_passthrough
            else _ordered_keys(self.meta_keys["action"], self.env_meta.get("action_order"))
        )

        # 2. 缓存 Action 切片偏移量
        self.action_offsets = {}
        curr_offset = 0
        flat_low = []
        flat_high = []
        for k in self.sorted_action_keys:
            shape = self.meta_keys["action"][k]
            length = np.prod(shape)
            self.action_offsets[k] = (curr_offset, curr_offset + length, shape)
            curr_offset += length
            if isinstance(self.env.action_space, GymDict) and k in self.env.action_space.spaces:
                space = self.env.action_space.spaces[k]
                flat_low.append(np.asarray(space.low, dtype=np.float32).reshape(-1))
                flat_high.append(np.asarray(space.high, dtype=np.float32).reshape(-1))
            else:
                flat_low.append(np.full((int(length),), -1.0, dtype=np.float32))
                flat_high.append(np.full((int(length),), 1.0, dtype=np.float32))
        if self.vector_action_passthrough:
            self.action_dim = int(np.prod(self.env.action_space.shape))
            action_low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
            action_high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1)
        else:
            self.action_dim = curr_offset
            action_low = np.concatenate(flat_low, axis=0) if flat_low else np.full((self.action_dim,), -1.0, dtype=np.float32)
            action_high = np.concatenate(flat_high, axis=0) if flat_high else np.full((self.action_dim,), 1.0, dtype=np.float32)

        # 3. 重新推断并构建观测空间 (Flattened)
        # 获取一次真实观测来推断最终维度
        sample_obs = self.observation(self.env.observation_space.sample())
        
        new_obs_spaces = {}
        for k, v in sample_obs.items():
            new_obs_spaces[k] = _space_from_observation_value(v)
        self.observation_space = GymDict(new_obs_spaces)
        
        # 4. 重新构建动作空间 (Flattened)
        self.action_space = Box(
            low=action_low,
            high=action_high,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        print(
            "[MetadataAdapterWrapper] Initialized. "
            f"Action dim: {self.action_dim}; flatten_obs_obj={sorted(self.flatten_obs_keys)}; "
            f"flatten_action={self.flatten_action_output}; vector_passthrough={self.vector_action_passthrough}"
        )

    def _to_tensor(self, value: Any, *, modality: str = "", leaf: str = "", normalize_rgb: bool = False) -> torch.Tensor:
        result = self.model_transform_spec.tensor(modality, leaf, value, device=self.device)
        if normalize_rgb and (modality, leaf) not in self.model_transform_spec.rgb_leaves and result.numel() > 0 and result.max() > 1.01:
            result = result / 255.0
        return result

    def _collect_modality_dict(self, obs: dict, modality: str, *, normalize_rgb: bool = False) -> Dict[str, torch.Tensor]:
        ret = {}
        source = obs.get(modality, {})
        if not isinstance(source, dict):
            raise TypeError(
                f"obs['{modality}'] must be a dict when env.flatten_obs_obj does not include '{modality}'."
            )

        keys = self.sorted_obs_keys.get(modality, [])
        if not keys:
            keys = sorted(source.keys())
        for role in keys:
            if role not in source:
                continue
            ret[role] = self._to_tensor(
                source[role],
                normalize_rgb=normalize_rgb,
                modality=modality,
                leaf=role,
            )
        return ret

    def observation(self, obs: dict):
        """
        将原始层级字典转换为扁平 Torch Dict
        Input: { 'rgb': { 'cam1': (3,H,W) }, 'state': { 'qpos': (7,) } }
        Output: { 'rgb': (C_total, H, W), 'state': (D_total,) }
        """
        ret = {}
        
        # 1. 处理 RGB
        if "rgb" in self.sorted_obs_keys:
            if "rgb" in self.flatten_obs_keys:
                rgb_list = list(self._collect_modality_dict(obs, "rgb", normalize_rgb=True).values())

                # 回退：若 meta_keys 为空或角色不匹配，尝试直接遍历观测中的 rgb 字典
                if len(rgb_list) == 0 and "rgb" in obs and isinstance(obs["rgb"], dict):
                    for _, img in sorted(obs["rgb"].items()):
                        rgb_list.append(self._to_tensor(img, modality="rgb", leaf="", normalize_rgb=True))

                # 在 Channel 维度拼接 (C1, H, W) + (C2, H, W) -> (C1+C2, H, W)
                if len(rgb_list) > 0:
                    ret["rgb"] = torch.cat(rgb_list, dim=0).to(self.device)
            else:
                rgb_dict = self._collect_modality_dict(obs, "rgb", normalize_rgb=True)
                if rgb_dict:
                    ret["rgb"] = rgb_dict

        # 2. 处理 State
        if "state" in self.sorted_obs_keys:
            state_dict = self._collect_modality_dict(obs, "state")
            if "state" in self.flatten_obs_keys:
                state_list = [val.flatten() for val in state_dict.values()]
                if len(state_list) > 0:
                    ret["state"] = torch.cat(state_list, dim=0).to(self.device)
            elif state_dict:
                ret["state"] = state_dict

        # 3. 处理 Depth (如果存在)
        if "depth" in self.sorted_obs_keys:
            depth_dict = self._collect_modality_dict(obs, "depth")
            if "depth" in self.flatten_obs_keys:
                depth_list = list(depth_dict.values())
                if len(depth_list) > 0:
                    ret["depth"] = torch.cat(depth_list, dim=0).to(self.device)
            elif depth_dict:
                ret["depth"] = depth_dict

        return ret

    def _copy_action_dict(self, dict_action: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return {
            key: np.asarray(value, dtype=np.float32).copy()
            for key, value in dict_action.items()
        }

    def _vector_to_dict_action(self, action_vector: np.ndarray) -> Dict[str, np.ndarray]:
        action_vector = np.asarray(action_vector, dtype=np.float32).reshape(-1)
        dict_action = {}
        for k in self.sorted_action_keys:
            start, end, shape = self.action_offsets[k]
            dict_action[k] = action_vector[start:end].reshape(shape)
        return dict_action

    def _prepare_action_info(self, info: Dict[str, Any], dict_action: Dict[str, np.ndarray]):
        info = dict(info or {})
        backend_actual = info.get("actual_action", None)
        if backend_actual is not None:
            info["backend_actual_action"] = np.asarray(backend_actual, dtype=np.float32).reshape(-1).copy()
        actual_flat = info.get("desired_action_vector", self.flatten_action(dict_action))
        actual_flat = np.asarray(actual_flat, dtype=np.float32).reshape(-1)
        info["actual_action_flat"] = actual_flat.copy()
        if self.flatten_action_output:
            info["actual_action"] = actual_flat.copy()
        else:
            info["actual_action"] = self._copy_action_dict(dict_action)
        return info

    def step(self, action_vector: np.ndarray):
        """
        将扁平向量动作为字典动作分发给环境
        """
        if isinstance(action_vector, torch.Tensor):
            action_vector = action_vector.detach().cpu().numpy()

        if self.vector_action_passthrough:
            if isinstance(action_vector, dict):
                raise TypeError("this environment advertises an external vector action, not a dict action")
            outbound_action = np.asarray(action_vector, dtype=np.float32).reshape(self.env.action_space.shape)
            obs, reward, terminated, truncated, info = self.env.step(outbound_action)
            info = dict(info or {})
            info.setdefault("actual_action_flat", outbound_action.reshape(-1).copy())
            if self.flatten_action_output:
                info.setdefault("actual_action", outbound_action.reshape(-1).copy())
            return self.observation(obs), reward, terminated, truncated, info
        if isinstance(action_vector, dict):
            dict_action = self._copy_action_dict(action_vector)
        else:
            dict_action = self._vector_to_dict_action(action_vector)

        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        info = self._prepare_action_info(info, dict_action)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def get_safe_flatten_action(self):
        """
        获取扁平化的安全动作向量。
        1. 调用底层环境的字典式 safe_action。
        2. 使用 flatten_action 自动对齐并拼接。
        """
        if hasattr(self.env, "get_safe_action"):
            dict_action = self.env.get_safe_action()
            return self.flatten_action(dict_action)
        else:
            # 兜底：如果底层没实现，尝试返回全 0
            return np.zeros(self.action_dim, dtype=np.float32)

    def get_safe_action(self):
        if self.flatten_action_output:
            return self.get_safe_flatten_action()
        if hasattr(self.env, "get_safe_action"):
            return self.env.get_safe_action()
        return self._vector_to_dict_action(np.zeros(self.action_dim, dtype=np.float32))

    def flatten_action(self, dict_action: Dict[str, np.ndarray]) -> np.ndarray:
        """
        辅助方法：将字典动作根据 meta_keys 顺序扁平化
        """
        vec_list = []
        for k in self.sorted_action_keys:
            vec_list.append(dict_action[k].flatten())
        return np.concatenate(vec_list, axis=0)

    def get_prompt(self):
        get_prompt = getattr(self.env, "get_prompt", None)
        if callable(get_prompt):
            return get_prompt()
        return getattr(self.env, "task_description", None)

    @property
    def task_description(self):
        return self.get_prompt()

class ManiSkillAdapterWrapper(gym.ObservationWrapper):
    """
    [Deprecated] Legacy ManiSkill-specific adapter.

    New ManiSkill integrations must use ``agent_infra.maniskill_env`` and
    ``MetadataAdapterWrapper`` instead.  The class remains temporarily for
    external callers during migration, but env_factories no longer selects it.
    职责：
    1. 从 sensor_data 中提取 RGB 和 Depth。
    2. 将 HWC (Image) 转换为 CHW (PyTorch)。
    3. 归一化 RGB 到 [0, 1] (可选，取决于 Encoder，这里默认做 float 转换)。
    4. 统一输出 Key 为: 'rgb', 'depth', 'state'。
    """
    def __init__(self, env, rgb=True, depth=True, state=True, sep_depth=True, flatten_obs_obj=None):
        try:
            from mani_skill.utils import common as mani_skill_common
        except ImportError as exc:
            raise ImportError(
                "ManiSkillAdapterWrapper requires mani_skill. "
                "Please install mani_skill when env.library='mani_skill'."
            ) from exc
        super().__init__(env)
        self.base_env = env.unwrapped
        self._common = mani_skill_common
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.sep_depth = sep_depth
        self.observation_space = env.observation_space


        ## check if rgb/depth data exists in first camera's sensor data
        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "depth" not in first_cam:
            self.include_depth = False
        if "rgb" not in first_cam:
            self.include_rgb = False
        legal_obs_keys = []
        if self.include_rgb:
            legal_obs_keys.append("rgb")
        if self.include_depth:
            legal_obs_keys.append("depth")
        if self.include_state:
            legal_obs_keys.append("state")
        self.flatten_obs_keys = _normalize_flatten_obs_obj(
            flatten_obs_obj if flatten_obs_obj is not None else ["all"],
            legal_obs_keys,
        )
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)
        
    def observation(self, observation: dict):
        """
        Input: ManiSkill 原始 Obs (包含 sensor_data, sensor_param, extra 等)
        Output: 标准化 Dict {'rgb': (C,H,W), 'depth': (1,H,W), 'state': (D,)}
        """
        # 1. 提取 Sensor Data
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]

        rgb_images = []
        rgb_by_camera = {}
        depth_images = []
        depth_by_camera = {}
        
        # 遍历所有相机
        for cam_name, cam_data in sensor_data.items():
            if self.include_rgb and "rgb" in cam_data:
                rgb_images.append(cam_data["rgb"]) # (H, W, 3)
                rgb_by_camera[cam_name] = cam_data["rgb"]
            if self.include_depth and "depth" in cam_data:
                depth_images.append(cam_data["depth"]) # (H, W, 1)
                depth_by_camera[cam_name] = cam_data["depth"]

        ret = dict()

        # 2. 处理 RGB (Concat -> Permute -> Normalize)
        if len(rgb_images) > 0:
            if "rgb" in self.flatten_obs_keys:
                rgb_concat = torch.cat(rgb_images, dim=-1) # (H, W, C_total)
                # Permute to (C, H, W)
                ret["rgb"] = rgb_concat.permute(0, 3, 1, 2).float()
            else:
                ret["rgb"] = {
                    cam_name: img.permute(0, 3, 1, 2).float()
                    for cam_name, img in sorted(rgb_by_camera.items())
                }

        # 3. 处理 Depth
        if len(depth_images) > 0:
            if "depth" in self.flatten_obs_keys:
                depth_concat = torch.cat(depth_images, dim=-1) # (H, W, C_depth)
                ret["depth"] = depth_concat.permute(0, 3, 1, 2).float() # 通常 Depth 已经是 float，不需要除 255
            else:
                ret["depth"] = {
                    cam_name: depth.permute(0, 3, 1, 2).float()
                    for cam_name, depth in sorted(depth_by_camera.items())
                }

        # 4. 如果不分离 Depth (按照你提供的旧代码逻辑，虽然我们推荐分离)
        if (
            not self.sep_depth
            and "rgb" in ret
            and "depth" in ret
            and isinstance(ret["rgb"], torch.Tensor)
            and isinstance(ret["depth"], torch.Tensor)
        ):
            ret["rgbd"] = torch.cat([ret["rgb"], ret["depth"]], dim=0)
            del ret["rgb"]
            del ret["depth"]

        observation = self._common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
        )
        # 5. 处理 State (Agent Proprioception)
        if self.include_state:
            if "state" in self.flatten_obs_keys:
                ret["state"] = observation
            else:
                ret["state"] = {"state": observation}
        return ret


class GymnasiumAdapterWrapper(gym.ObservationWrapper):
    """
    [特定适配器] 适配标准 Gymnasium 环境 (如 CartPole, Mujoco)。
    """
    def __init__(self, env):
        super().__init__(env)
        # Standard Gym usually returns numpy, but our pipeline prefers Torch
        self.device = "cpu" 
        
    def observation(self, observation):
        ret = {}
        # 1. Visual: 如果 Render Mode 是 rgb_array，这里很难直接获取，
        # 通常需要 PixelObservationWrapper 在外部包裹。
        # 这里假设输入已经是 pixels 或者我们需要从 info/render 获取。
        # 为简化，针对 state-based gym 环境：
        
        if isinstance(observation, np.ndarray):
             ret["state"] = torch.from_numpy(observation).float()
             # 创建 Dummy RGB 用于兼容 pipeline
             ret["rgb"] = torch.zeros((3, 64, 64), dtype=torch.float32)
             
        elif isinstance(observation, dict):
            if "state" in observation:
                ret["state"] = torch.from_numpy(observation["state"]).float()
            # ... handle other keys
            
        return ret
    
class UnifiedFrameStackWrapper(gym.Wrapper):
    """
    通用 FrameStack。
    支持 Dict Observation。
    输出格式保持 PyTorch 友好的 Tensor。
    """
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        #self.observation_space = env.observation_space
        # 初始化填充
        # 我们需要在 reset 时处理，这里无法预知 shape
        pass

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def get_safe_flatten_action(self):
        return self.env.get_safe_flatten_action()

    def get_safe_action(self):
        return self.env.get_safe_action()

    def flatten_action(self, dict_action: Dict[str, np.ndarray]) -> np.ndarray:
        return self.env.flatten_action(dict_action)

    def get_prompt(self):
        get_prompt = getattr(self.env, "get_prompt", None)
        if callable(get_prompt):
            return get_prompt()
        return getattr(self.env, "task_description", None)

    @property
    def task_description(self):
        return self.get_prompt()

    def _get_ob(self):
        # 假设 obs 是 {'visual': Tensor(C,H,W), 'state': Tensor(D)}
        # 我们将其 Stack 为 {'visual': Tensor(T, C, H, W), 'state': Tensor(T, D)}
        # 如果是向量化环境 (B, C, H, W)，则 Stack 为 (B, T, C, H, W)
        
        keys = self.frames[0].keys()
        stacked_obs = {}
        
        for k in keys:
            tensors = [f[k] for f in self.frames]
            if isinstance(tensors[0], dict):
                stacked_obs[k] = {}
                for sub_key in tensors[0].keys():
                    sub_tensors = [tensor[sub_key] for tensor in tensors]
                    sample_dim = sub_tensors[0].dim()
                    if sample_dim >= 3:
                        stack_dim = 1 if sample_dim == 4 else 0
                    else:
                        stack_dim = 1 if sample_dim == 2 else 0
                    stacked_obs[k][sub_key] = torch.stack(sub_tensors, dim=stack_dim)
                continue
            # 根据输入维度判断是否是向量化环境
            # 这里的逻辑是：如果图像是 3 维 (C, H, W) 或 状态是 1 维 (D,)，说明是单环境，在 dim=0 堆叠产生时间维
            # 如果图像是 4 维 (B, C, H, W) 或 状态是 2 维 (B, D)，说明是向量化环境，在 dim=1 堆叠产生时间维
            sample_dim = tensors[0].dim()
            if sample_dim >= 3: # 图像 (C,H,W) 或 (B,C,H,W)
                stack_dim = 1 if sample_dim == 4 else 0
            else: # 状态 (D,) 或 (B,D)
                stack_dim = 1 if sample_dim == 2 else 0
                
            stacked_obs[k] = torch.stack(tensors, dim=stack_dim)
            
        return stacked_obs
