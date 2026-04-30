from dataclasses import dataclass, field, asdict
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

# ==================== 4. 环境特定参数 (Environment-Specific Kwargs) ====================

@dataclass
class RealmanEnvKwargs:
    """Realman 物理环境的特定构造参数"""
    robot_ip: str = "192.168.1.19"
    camera_sns: List[str] = field(default_factory=list)
    hz: int = 10
    use_depth: bool = False
    
    # 🟢 新增：双臂与模块化适配参数
    is_dual: bool = False           # 是否开启双臂模式
    config_path: str = "env_config.yaml" # 配置文件路径
    arm_names: List[str] = field(default_factory=lambda: ["arm"]) 

@dataclass
class ManiSkillEnvKwargs:
    """ManiSkill 仿真环境的特定构造参数"""
    reward_mode: str = 'sparse'
    render_mode: str = 'rgb_array'

@dataclass
class EnvKwargsConfig:
    """
    环境特定参数的容器。
    create_env 会根据 env.library 的值选择对应的子配置。
    """
    mani_skill: ManiSkillEnvKwargs = field(default_factory=ManiSkillEnvKwargs)
    realman: RealmanEnvKwargs = field(default_factory=RealmanEnvKwargs)


def _default_env_kwargs() -> Dict[str, Any]:
    """
    默认环境参数容器（动态字典版）。
    保留现有 mani_skill / realman 默认值，同时允许未来新增任意 library 键，
    避免每次新增环境都修改 dataclass 字段。
    """
    return {
        "mani_skill": asdict(ManiSkillEnvKwargs()),
        "realman": asdict(RealmanEnvKwargs()),
    }

# ==================== 5. 全局配置 (Top-Level) ====================

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
    # NOTE:
    # 1) 继续兼容现有 YAML: env_kwargs.realman / env_kwargs.mani_skill
    # 2) 允许未来新增 env_kwargs.<new_library> 而不修改 structure.py
    env_kwargs: Dict[str, Any] = field(default_factory=_default_env_kwargs)
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)


    # Placeholders for mixin injection
    actor: Any = None 
    critic: Any = None
    agent_sp: Any = None  # Agent Special Config Slot
