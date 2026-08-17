from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

# ==================== 0. 基础组件配置 ====================

@dataclass
class VisualEncoderConfig:
    in_channels: int = 3
    out_dim: int = 256
    backbone_type: str = 'resnet'#  # 'plain' or 'resnet'
    pool_feature_map: bool = True
    use_group_norm: bool = True

@dataclass
class StateEncoderConfig:
    """
    StateEncoder 现在作为 Actor/Critic 的子组件存在
    """
    type: str = "CNN_state_encoder"
    include_rgb: bool = True
    visual: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    proprio_dim: int = 0
    out_dim: int = 256  # 融合后的 embedding 维度
    obs_key: str = "feature"
    hidden_dims: List[int] = field(default_factory=lambda: [256])
    num_cameras: int = 1
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
class FlowMatchingActorConfig(BaseActorConfig):
    type: str = "flow_matching"
    lr: float = 1e-4
    weight_decay: float = 1e-6

    # SmolVLA-style flow matching controls.
    num_inference_steps: int = 10
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0
    time_eps: float = 1e-3
    time_embed_scale: float = 100.0
    clip_sample: bool = True

    unet: UNetConfig = field(default_factory=UNetConfig)
    encoder: StateEncoderConfig = field(default_factory=StateEncoderConfig)


@dataclass
class Pi05ActorConfig(BaseActorConfig):
    """LeRobot π₀.₅-specific actor settings.

    Public environment/runner schemas stay unchanged.  The resolver derives
    ``state_dim`` and ``action_dim`` from the selected environment contract;
    the remaining fields describe the selected π₀.₅ checkpoint and prefix
    feature representation.
    """

    type: str = "pi05"
    norm: NormalizationConfig = field(default_factory=lambda: NormalizationConfig(type=None))
    pretrained_path: str = ""
    pretrained_revision: Optional[str] = None
    local_files_only: bool = True
    mock_mode: bool = True
    strict_load: bool = True
    state_dim: int = 0
    feature_dim: int = 2048
    max_state_dim: int = 32
    max_action_dim: int = 32
    chunk_size: int = 50
    n_action_steps: int = 50
    num_inference_steps: int = 10
    image_keys: List[str] = field(default_factory=list)
    prompt_key: str = "prompt"
    state_key: str = "observation.state"
    action_key: str = "action"
    feature_pooling: str = "last_language_token"


@dataclass
class SmolVLAActorConfig(BaseActorConfig):
    type: str = "smolvla"
    norm: NormalizationConfig = field(default_factory=lambda: NormalizationConfig(type="mean_std"))
    obs_norm: NormalizationConfig = field(default_factory=lambda: NormalizationConfig(type="mean_std"))
    pretrained_path: str = (
        "/home/lilinyi/.cache/huggingface/hub/models--lerobot--smolvla_base/"
        "snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
    )
    pretrained_revision: Optional[str] = None
    local_files_only: bool = True
    strict_load: bool = False
    lr: float = 1e-4
    weight_decay: float = 1e-10
    max_state_dim: int = 32
    max_action_dim: int = 32
    chunk_size: int = 50
    n_action_steps: int = 50
    num_inference_steps: int = 10
    image_keys: List[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])
    image_shape: List[int] = field(default_factory=lambda: [3, 256, 256])
    prompt_key: str = "prompt"
    state_key: str = "observation.state"
    action_key: str = "action"
    tokenizer_max_length: int = 48
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True
    load_vlm_weights: bool = True
    normalization_source: str = "agent_factory"  # agent_factory | none


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
    return_mse_weight: float = 1.0
    gamma_default_mode: str = "one_minus_margin_over_max_episode_steps"
    gamma_default_margin: float = 2.0
    gamma_use_act_horizon_power: bool = True
    log_interval: int = 100


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
    env_id: str = ""
    library: str = ""
    env_config_path: str = ""
    action_dim: int = 7
    proprio_dim: int = 25
    server_mode : bool = False

    obs_horizon : int = 2
    pred_horizon: int = 16
    act_horizon: int = 8

    num_cameras: int = 2
    obs_mode: str = "rgb"
    env_control_mode: str = "delta_pose"
    controller_backend: str = ""
    control_mode: str = "pd_ee_delta_pose"
    reward_mode: str = "sparse"
    render_mode: str = "rgb_array"
    max_episode_steps: int = 150

    gamma: Any = 0.99
    penalty: float = -150.0
    reward_scale: int = 100
    reward_shape: bool = False
    flatten_obs_obj: Any = field(default_factory=lambda: ["all"])
    flatten_action: bool = True


@dataclass
class TrainConfig:
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 0
    save_interval: int = 5000
    train_object: str = "critic_then_actor"
    dataset_key: str = "expert_dataset"
    critic_iters: int = 200
    actor_iters: int = 200
    ckpt_path: str = ""
    finetune: bool = False
    save_root: str = "run_results"
    exp_name: str = ""

@dataclass
class ExpertDatasetConfig:
    demo_path: str = "data/demos/expert_data.hdf5"
    num_traj: Optional[int] = None  # None 表示加载全部轨迹
    format: str = "flat"  # flat | structured | auto
    success_only: bool = False

@dataclass
class ReplayBufferConfig:
    max_traj_num: int = 100
    folder_path: str = ""
    replaybuffer_path: str = ""

@dataclass
class RunnerConfig:
    type: str = "base"
    control_hz: int = 10
    buffer_capacity: int = 100
    redundancy_margin: int = 100
    save_dir: str = "data"
    no_safe_action_gap: bool = False
    planning_wait_sleep: float = 0.002
    config_path: str = "agent_factory/runner/config/base.yaml"
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetConfig:
    dataset_type: str = "cpiql"
    include_rgb: bool = True
    include_depth: bool = False
    config_path: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    expert: ExpertDatasetConfig = field(default_factory=ExpertDatasetConfig)
    replaybuffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
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
    resolved: bool = False
    agent_control_mode: str = "delta_pose"
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
