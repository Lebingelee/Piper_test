import gymnasium as gym
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from gymnasium.spaces import Box, Dict as GymDict
from collections import deque

try:
    from mani_skill.utils import common as mani_skill_common
except ImportError:
    mani_skill_common = None

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
    def __init__(self, env, device="cpu"):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped
        self.device = device
        
        if not hasattr(self.unwrapped_env, "meta_keys"):
            raise AttributeError("Environment must have 'meta_keys' defined for MetadataAdapterWrapper.")

        # 1. 缓存排序后的 Key 顺序
        self.meta_keys = self.unwrapped_env.meta_keys
        self.sorted_obs_keys = {
            modality: sorted(self.meta_keys["obs"][modality].keys())
            for modality in self.meta_keys["obs"].keys()
        }
        self.sorted_action_keys = sorted(self.meta_keys["action"].keys())

        # 2. 缓存 Action 切片偏移量
        self.action_offsets = {}
        curr_offset = 0
        for k in self.sorted_action_keys:
            shape = self.meta_keys["action"][k]
            length = np.prod(shape)
            self.action_offsets[k] = (curr_offset, curr_offset + length, shape)
            curr_offset += length
        self.action_dim = curr_offset

        # 3. 重新推断并构建观测空间 (Flattened)
        # 获取一次真实观测来推断最终维度
        sample_obs = self.observation(self.env.observation_space.sample())
        
        new_obs_spaces = {}
        for k, v in sample_obs.items():
            new_obs_spaces[k] = Box(
                low=-np.inf, high=np.inf, 
                shape=v.shape, 
                dtype=np.float32
            )
        self.observation_space = GymDict(new_obs_spaces)
        
        # 4. 重新构建动作空间 (Flattened)
        self.action_space = Box(
            low=-1.0, high=1.0, # 假设动作经过预缩放或环境内部处理
            shape=(self.action_dim,),
            dtype=np.float32
        )
        print(f"[MetadataAdapterWrapper] Initialized. Action dim: {self.action_dim}")

    def observation(self, obs: dict):
        """
        将原始层级字典转换为扁平 Torch Dict
        Input: { 'rgb': { 'cam1': (3,H,W) }, 'state': { 'qpos': (7,) } }
        Output: { 'rgb': (C_total, H, W), 'state': (D_total,) }
        """
        ret = {}
        
        # 1. 处理 RGB
        if "rgb" in self.sorted_obs_keys:
            rgb_list = []
            for role in self.sorted_obs_keys["rgb"]:
                if "rgb" not in obs or role not in obs["rgb"]:
                    continue
                img = obs["rgb"][role]
                if not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(img).float()
                # 归一化到 [0, 1]
                if img.max() > 1.01:
                    img = img / 255.0
                rgb_list.append(img)

            # 回退：若 meta_keys 为空或角色不匹配，尝试直接遍历观测中的 rgb 字典
            if len(rgb_list) == 0 and "rgb" in obs and isinstance(obs["rgb"], dict):
                for _, img in sorted(obs["rgb"].items()):
                    if not isinstance(img, torch.Tensor):
                        img = torch.from_numpy(img).float()
                    if img.max() > 1.01:
                        img = img / 255.0
                    rgb_list.append(img)

            # 在 Channel 维度拼接 (C1, H, W) + (C2, H, W) -> (C1+C2, H, W)
            if len(rgb_list) > 0:
                ret["rgb"] = torch.cat(rgb_list, dim=0).to(self.device)

        # 2. 处理 State
        if "state" in self.sorted_obs_keys:
            state_list = []
            for k in self.sorted_obs_keys["state"]:
                if "state" not in obs or k not in obs["state"]:
                    continue
                val = obs["state"][k]
                if not isinstance(val, torch.Tensor):
                    val = torch.from_numpy(np.array(val)).float()
                state_list.append(val.flatten())
            if len(state_list) > 0:
                ret["state"] = torch.cat(state_list, dim=0).to(self.device)

        # 3. 处理 Depth (如果存在)
        if "depth" in self.sorted_obs_keys:
            depth_list = []
            for role in self.sorted_obs_keys["depth"]:
                if "depth" not in obs or role not in obs["depth"]:
                    continue
                d = obs["depth"][role]
                if not isinstance(d, torch.Tensor):
                    d = torch.from_numpy(d).float()
                depth_list.append(d)
            if len(depth_list) > 0:
                ret["depth"] = torch.cat(depth_list, dim=0).to(self.device)

        return ret

    def step(self, action_vector: np.ndarray):
        """
        将扁平向量动作为字典动作分发给环境
        """
        if isinstance(action_vector, torch.Tensor):
            action_vector = action_vector.detach().cpu().numpy()
            
        dict_action = {}
        for k in self.sorted_action_keys:
            start, end, shape = self.action_offsets[k]
            dict_action[k] = action_vector[start:end].reshape(shape)
            
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def get_safe_action(self):
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

    def flatten_action(self, dict_action: Dict[str, np.ndarray]) -> np.ndarray:
        """
        辅助方法：将字典动作根据 meta_keys 顺序扁平化
        """
        vec_list = []
        for k in self.sorted_action_keys:
            vec_list.append(dict_action[k].flatten())
        return np.concatenate(vec_list, axis=0)

class ManiSkillAdapterWrapper(gym.ObservationWrapper):
    """
    [特定适配器] 专门适配 ManiSkill 环境。
    职责：
    1. 从 sensor_data 中提取 RGB 和 Depth。
    2. 将 HWC (Image) 转换为 CHW (PyTorch)。
    3. 归一化 RGB 到 [0, 1] (可选，取决于 Encoder，这里默认做 float 转换)。
    4. 统一输出 Key 为: 'rgb', 'depth', 'state'。
    """
    def __init__(self, env, rgb=True, depth=True, state=True, sep_depth=True):
        if mani_skill_common is None:
            raise ImportError(
                "ManiSkillAdapterWrapper requires mani_skill. "
                "Please install mani_skill when env.library='mani_skill'."
            )
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
        depth_images = []
        
        # 遍历所有相机
        for cam_name, cam_data in sensor_data.items():
            if self.include_rgb and "rgb" in cam_data:
                rgb_images.append(cam_data["rgb"]) # (H, W, 3)
            if self.include_depth and "depth" in cam_data:
                depth_images.append(cam_data["depth"]) # (H, W, 1)

        ret = dict()

        # 2. 处理 RGB (Concat -> Permute -> Normalize)
        if len(rgb_images) > 0:
            #
            rgb_concat = torch.cat(rgb_images, dim=-1) # (H, W, C_total)
            # Permute to (C, H, W)
            ret["rgb"] = rgb_concat.permute(0, 3, 1, 2).float()

        # 3. 处理 Depth
        if len(depth_images) > 0:
            depth_concat = torch.cat(depth_images, dim=-1) # (H, W, C_depth)
            ret["depth"] = depth_concat.permute(0, 3, 1, 2).float() # 通常 Depth 已经是 float，不需要除 255

        # 4. 如果不分离 Depth (按照你提供的旧代码逻辑，虽然我们推荐分离)
        if not self.sep_depth and "rgb" in ret and "depth" in ret:
            ret["rgbd"] = torch.cat([ret["rgb"], ret["depth"]], dim=0)
            del ret["rgb"]
            del ret["depth"]

        observation = self._common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
        )
        # 5. 处理 State (Agent Proprioception)
        if self.include_state:
            ret["state"] = observation
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

    def _get_ob(self):
        # 假设 obs 是 {'visual': Tensor(C,H,W), 'state': Tensor(D)}
        # 我们将其 Stack 为 {'visual': Tensor(T, C, H, W), 'state': Tensor(T, D)}
        # 如果是向量化环境 (B, C, H, W)，则 Stack 为 (B, T, C, H, W)
        
        keys = self.frames[0].keys()
        stacked_obs = {}
        
        for k in keys:
            tensors = [f[k] for f in self.frames]
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
