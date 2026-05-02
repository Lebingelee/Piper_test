import os
import yaml
import gymnasium as gym
import logging
from agent_factory.env.wrappers import UnifiedFrameStackWrapper

logger = logging.getLogger(__name__)


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_env_config_path(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.join(_project_root(), path)


def _load_env_config(env_cfg):
    config_path = _resolve_env_config_path(getattr(env_cfg, "env_config_path", "") or "")
    if not config_path:
        return {}, ""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"env_config_path not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}, config_path


def create_env(env_cfg):
    """
    Factory function to create environments based on configuration.

    Args:
        env_cfg (object): Common env config (e.g., env_id, control_mode).
    """
    library = getattr(env_cfg, 'library', 'gymnasium')
    env_id = env_cfg.env_id
    env_file_cfg, env_config_path = _load_env_config(env_cfg)

    logger.info(f"Creating Environment: {env_id} via {library}...")

    # --- 1. Base Environment Creation ---
    if library == 'mani_skill':
        from agent_factory.env.wrappers import ManiSkillAdapterWrapper

        env = gym.make(
            env_id,
            obs_mode=getattr(env_cfg, 'obs_mode', 'rgbd'),
            control_mode=getattr(env_cfg, 'control_mode', 'pd_ee_delta_pose'),
            max_episode_steps=getattr(env_cfg, 'max_episode_steps', 100),
            **env_file_cfg
        )
        env = ManiSkillAdapterWrapper(env)

    elif library == 'realman':
        robot_cfg = env_file_cfg.get("robot", {}) if isinstance(env_file_cfg, dict) else {}
        robots_cfg = env_file_cfg.get("robots", {}) if isinstance(env_file_cfg, dict) else {}
        cameras_cfg = env_file_cfg.get("cameras", {}) if isinstance(env_file_cfg, dict) else {}
        camera_nodes = cameras_cfg.get("nodes", []) if isinstance(cameras_cfg, dict) else []
        camera_sns = [
            node.get("serial_number")
            for node in camera_nodes
            if isinstance(node, dict) and node.get("serial_number")
        ]
        is_dual = bool(robots_cfg)
        base_kwargs = {}
        if camera_sns:
            base_kwargs["camera_sns"] = camera_sns
        
        if getattr(env_cfg, 'server_mode', False):
            # 服务器模式：导入并使用 OfflineRealManEnv
            from agent_infra.Realman_Env.Env.offline_realman_env import OfflineRealManEnv
            env = OfflineRealManEnv(
                control_mode=getattr(env_cfg, 'control_mode', 'delta_ee_pose'),
                **base_kwargs
            )
        else:
            if is_dual:
                from agent_infra.Realman_Env.Env.dual_realman_env import DualRealManEnv
                env = DualRealManEnv(
                    config_path=env_config_path or "dual_env_config.yaml",
                    control_mode=getattr(env_cfg, 'control_mode', 'delta_ee_pose'),
                    **base_kwargs
                )
            else:
                # 本地模式：使用真实的 RealManEnv
                from agent_infra.Realman_Env.Env.realman_env import RealManEnv
                env = RealManEnv(
                    robot_ip=robot_cfg.get("ip") if isinstance(robot_cfg, dict) else None,
                    hz=robot_cfg.get("default_hz") if isinstance(robot_cfg, dict) else None,
                    control_mode=getattr(env_cfg, 'control_mode', 'delta_ee_pose'),
                    **base_kwargs
                )

        from agent_factory.env.wrappers import MetadataAdapterWrapper
        env = MetadataAdapterWrapper(env)

    elif library == 'piper':
        from agent_infra.Piper_Env.Env.single_piper_env import SinglePiperEnv
        from agent_infra.Piper_Env.Env.dual_piper_env import DualPiperEnv
        from agent_factory.env.wrappers import MetadataAdapterWrapper

        robots = env_file_cfg.get("robots", {}) if isinstance(env_file_cfg, dict) else {}
        common_cfg = env_file_cfg.get("common", {}) if isinstance(env_file_cfg, dict) else {}
        is_dual = isinstance(robots, dict) and len(robots) == 2
        hz = common_cfg.get("default_hz", getattr(env_cfg, 'hz', None))

        env_cls = DualPiperEnv if is_dual else SinglePiperEnv
        env = env_cls(
            config_path=env_config_path or None,
            control_mode=getattr(env_cfg, 'control_mode', None),
            hz=hz,
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
