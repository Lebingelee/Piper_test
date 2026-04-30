import gymnasium as gym
import logging
from agent_factory.env.wrappers import UnifiedFrameStackWrapper

logger = logging.getLogger(__name__)


def _to_plain_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    try:
        return dict(obj)
    except Exception:
        return {}


def _get_library_kwargs(env_kwargs, library: str):
    """
    兼容 DictConfig / dict / dataclass 风格的 env_kwargs 容器。
    """
    if env_kwargs is None:
        return {}

    # Dict / DictConfig-like
    if hasattr(env_kwargs, "get"):
        branch = env_kwargs.get(library, None)
        if branch is not None:
            return _to_plain_dict(branch)

    # Attribute style fallback
    branch = getattr(env_kwargs, library, None)
    return _to_plain_dict(branch)


def create_env(env_cfg, env_kwargs):
    """
    Factory function to create environments based on configuration.

    Args:
        env_cfg (object): Common env config (e.g., env_id, control_mode).
        env_kwargs (object): Library-specific arguments (OmegaConf object).
    """
    library = getattr(env_cfg, 'library', 'gymnasium')
    env_id = env_cfg.env_id

    logger.info(f"Creating Environment: {env_id} via {library}...")

    # --- 1. Base Environment Creation ---
    if library == 'mani_skill':
        from agent_factory.env.wrappers import ManiSkillAdapterWrapper

        specific_kwargs = _get_library_kwargs(env_kwargs, "mani_skill")

        env = gym.make(
            env_id,
            obs_mode=getattr(env_cfg, 'obs_mode', 'rgbd'),
            control_mode=getattr(env_cfg, 'control_mode', 'pd_ee_delta_pose'),
            max_episode_steps=getattr(env_cfg, 'max_episode_steps', 100),
            **specific_kwargs
        )
        env = ManiSkillAdapterWrapper(env)

    elif library == 'realman':
        specific_kwargs = _get_library_kwargs(env_kwargs, "realman")
        is_dual = bool(specific_kwargs.get('is_dual', False))
        
        if getattr(env_cfg, 'server_mode', False):
            # 服务器模式：导入并使用 OfflineRealManEnv
            from agent_infra.Realman_Env.Env.offline_realman_env import OfflineRealManEnv
            env = OfflineRealManEnv(
                control_mode=getattr(env_cfg, 'control_mode', 'delta_ee_pose'),
                **specific_kwargs
            )
        else:
            if is_dual:
                from agent_infra.Realman_Env.Env.dual_realman_env import DualRealManEnv
                env = DualRealManEnv(
                    control_mode=getattr(env_cfg, 'control_mode', 'delta_ee_pose'),
                    **specific_kwargs
                )
            else:
                # 本地模式：使用真实的 RealManEnv
                from agent_infra.Realman_Env.Env.realman_env import RealManEnv
                env = RealManEnv(
                    control_mode=getattr(env_cfg, 'control_mode', 'delta_ee_pose'),
                    **specific_kwargs
                )

        from agent_factory.env.wrappers import MetadataAdapterWrapper
        env = MetadataAdapterWrapper(env)

    elif library == 'piper':
        from agent_infra.Piper_Env.Env.single_piper_env import SinglePiperEnv
        from agent_infra.Piper_Env.Env.dual_piper_env import DualPiperEnv
        from agent_factory.env.wrappers import MetadataAdapterWrapper

        specific_kwargs = _get_library_kwargs(env_kwargs, "piper")
        specific_kwargs = _to_plain_dict(specific_kwargs)

        is_dual = bool(specific_kwargs.pop("is_dual", False))
        # defaults 仅用于配置解析，不作为环境构造参数透传
        specific_kwargs.pop("defaults", None)

        env_cls = DualPiperEnv if is_dual else SinglePiperEnv
        env = env_cls(
            control_mode=getattr(env_cfg, 'control_mode', None),
            hz=getattr(env_cfg, 'hz', None),
            **specific_kwargs
        )
        env = MetadataAdapterWrapper(env)

    elif library == 'gymnasium':
        # 标准 Gym 环境
        env = gym.make(env_id, render_mode='rgb_array')
        # TODO: 这里未来需要添加 GymAdaptorWrapper
        pass
    else:
        raise ValueError(f"Unknown library: {library}")

    # 3. 通用 Wrapper (所有环境通用)
    if hasattr(env_cfg, 'num_stack') and env_cfg.num_stack > 1:
        env = UnifiedFrameStackWrapper(env, num_stack=env_cfg.num_stack)
    elif hasattr(env_cfg, 'obs_horizon') and env_cfg.obs_horizon > 1:
        env = UnifiedFrameStackWrapper(env, num_stack=env_cfg.obs_horizon)

    return env
