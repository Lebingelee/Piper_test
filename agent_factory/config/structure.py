from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

# ==================== 0. 基础组件配置 ====================

@dataclass
class VisualEncoderConfig:
    in_channels: int = 3
    out_dim: int = 256
    backbone_type: str = "plain"  # 'plain' or 'resnet'
    pool_feature_map: bool = True
    use_group_norm: bool = True

@dataclass
class StateEncoderConfig:
    """
    StateEncoder 现在作为 Actor/Critic 的子组件存在
    """
    visual: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    proprio_dim: int = 0
    out_dim: int = 256  # 融合后的 embedding 维度
    view_fusion: str = "concat"  # 'concat' or 'mean'

@dataclass
class UNetConfig:
    down_dims: List[int] = field(default_factory=lambda: [128, 256, 512])
    diffusion_step_embed_dim: int = 64
    n_groups: int = 8

@dataclass
class SchedulerConfig:
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"

@dataclass
class NormalizationConfig:
    """
    动作归一化配置
    """
    type: Optional[str] = 'min_max'  # 'min_max', 'mean_std', 'quantile', or None
    params: Dict[str, Any] = field(default_factory=dict)  # 存储特定算法参数，如 q_low/q_high

# ==================== 1. Actor 配置 ====================

@dataclass
class BaseActorConfig:
    type: str = "base"
    obs_horizon: int = 2
    action_dim: int = 10
    pred_horizon: int = 16
    require_env_action_dim_match: bool = True
    # 每个 Actor 都有自己的归一化配置
    norm: NormalizationConfig = field(default_factory=NormalizationConfig)
    # 每个 Actor 都有自己的 Encoder 配置

@dataclass
class DiffusionBasePolicyConfig:
    type: str = ""
    ckpt_path: str = ""
    base_config_path: str = ""


@dataclass
class DSRLActorConfig(BaseActorConfig):
    type: str = "dsrl_policy"
    action_dim: int = 7
    pred_horizon: int = 16
    require_env_action_dim_match: bool = True
    lr: float = 3e-4
    temp_lr: float = 3e-4
    init_temperature: float = 1.0
    target_entropy: Any = "auto"
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    action_limit: float = 1.0
    encoder: StateEncoderConfig = field(default_factory=StateEncoderConfig)
    
@dataclass
class DiffusionActorConfig(BaseActorConfig):
    type: str = "diffusion"  # 这里的 type 主要用于序列化标识
    use_extra_cond: bool = False
    lr: float = 1e-4
    weight_decay: float = 1e-6
    # 兼容 use_extra_cond=True 场景，避免运行时缺字段
    cond_dim: int = 1
    cond_embed_dim: int = 32
    use_cfg_loss: bool = False
    cfg_drop_rate: float = 0.1
    guidance_scale: float = 1.0
    
    # Diffusion Specific
    unet: UNetConfig = field(default_factory=UNetConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    encoder: StateEncoderConfig = field(default_factory=StateEncoderConfig)


@dataclass
class Conditional_DiffusionActorConfig(DiffusionActorConfig):
    type: str = "conditional_diffusion"  # 这里的 type 主要用于序列化标识
    # Conditional Logic
    cond_dim: int = 1
    cond_embed_dim: int = 32
    CFG_alpha: float = 0.4
    cfg_drop_rate: float = 0.1
    guidance_scale: float = 2.0

# ==================== 2. Critic 配置 ====================

@dataclass
class BaseCriticConfig:
    type: str = "base"
    encoder: StateEncoderConfig = field(default_factory=StateEncoderConfig)
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

@dataclass
class IQLCriticConfig(BaseCriticConfig):
    type: str = "iql"
    q_lr: float = 3e-4
    v_lr: float = 3e-4
    expectile: float = 0.7


@dataclass
class DSRLCriticConfig(BaseCriticConfig):
    type: str = "dsrl"
    lr: float = 3e-4
    num_qs: int = 2
    critic_reduction: str = "min"
    backup_entropy: bool = True
    noise_critic_grad_steps: int = 10
    critic_backup_combine_type: str = "min"
    action_norm: NormalizationConfig = field(default_factory=NormalizationConfig)


# ==================== 3. 全局/通用配置 ====================

@dataclass
class EnvConfig:
    env_id: str = "StackCube-v1"
    library: str = "mani_skill"
    env_config_path: str = ""
    action_dim: int = 7
    proprio_dim: int = 25
    server_mode : bool = False

    obs_horizon : int = 2
    pred_horizon: int = 16
    act_horizon: int = 8

    num_cameras: int = 2
    obs_mode: str = "rgb"
    control_mode: str = "pd_ee_delta_pose"
    reward_mode: str = "sparse"
    render_mode: str = "rgb_array"
    max_episode_steps: int = 150

    gamma: float = 0.99
    penalty: float = -150.0
    reward_scale: int = 100
    reward_shape: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_workers: int = 0
    n_epochs: int = 1000
    save_interval: int = 5000

@dataclass
class ExpertDatasetConfig:
    demo_path: str = "data/demos/expert_data.hdf5"
    num_traj: Optional[int] = None  # None 表示加载全部轨迹
    format: str = "flat"  # flat | structured | auto

@dataclass
class ReplayBufferConfig:
    max_traj_num: int = 100

@dataclass
class RunnerConfig:
    control_hz: int = 10
    buffer_capacity: int = 100
    redundancy_margin: int = 100
    save_dir: str = "data"
    hitl_enabled: bool = False
    hitl_override_key: str = "t"
    hitl_source: str = "keyboard"  # reserved: keyboard / hardware
    hitl_finalize_intervention: bool = True

@dataclass
class DatasetConfig:
    
    include_rgb: bool = True
    include_depth: bool = False
    expert: ExpertDatasetConfig = field(default_factory=ExpertDatasetConfig)
    replay: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)

# ==================== 4. 全局配置 (Top-Level) ====================

@dataclass
class GlobalConfig:
    """
    最终组装的顶层 Config。
    注意：actor 和 critic 字段默认是 None 或 Base，
    实际构建时会被 registry 替换为具体的子类 (如 DiffusionActorConfig)。
    """
    agent_type: str = "unknown"
    device: str = "cuda"
    seed: int = 42

    # Shared Hyperparams
    soft_update_tau: float = 0.005
    reward_scale: float = 100

    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)


    # Placeholders for mixin injection
    actor: Any = None 
    critic: Any = None
    agent_sp: Any = None  # Agent Special Config Slot
