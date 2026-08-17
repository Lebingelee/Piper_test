import os
import yaml
import gymnasium as gym
import logging
import time
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


def _get_wrapper_attr(env, attr: str):
    try:
        if hasattr(env, "get_wrapper_attr"):
            return env.get_wrapper_attr(attr)
    except AttributeError:
        pass
    return getattr(env, attr, None)


def _maybe_start_piper_cameras(env, env_cfg):
    if not bool(getattr(env_cfg, "auto_start_cameras", True)):
        return

    start_cameras = _get_wrapper_attr(env, "start_cameras")
    if start_cameras is None:
        print("[EnvFactory] Piper camera startup hook not found; visual obs may be dummy frames.")
        return

    print("[EnvFactory] Starting Piper camera threads...")
    start_cameras()

    warmup_sec = float(
        getattr(
            env_cfg,
            "camera_warmup_sec",
            getattr(env_cfg, "camera_warmup", 0.0),
        )
    )
    if warmup_sec > 0.0:
        print(f"[EnvFactory] Piper camera warmup: {warmup_sec:.2f}s")
        time.sleep(warmup_sec)


def _get_flatten_obs_obj(env_cfg):
    return getattr(env_cfg, "flatten_obs_obj", ["all"])


def _get_flatten_action(env_cfg):
    return bool(getattr(env_cfg, "flatten_action", True))


def _get_robosuite_env_control_mode(env_cfg):
    mode = str(getattr(env_cfg, "env_control_mode", "") or "").strip().lower()
    if mode in {"absolute_joint", "absolute_pose", "delta_pose"}:
        return mode
    return None


def _socket_setting(env_cfg, file_cfg, name, default=None):
    """Prefer explicit runtime endpoint settings over the optional YAML file."""
    value = getattr(env_cfg, name, None)
    if value is not None and value != "":
        return value
    socket_cfg = file_cfg.get("socket", file_cfg) if isinstance(file_cfg, dict) else {}
    return socket_cfg.get(name, default) if isinstance(socket_cfg, dict) else default


def create_env(env_cfg):
    """
    Factory function to create environments based on configuration.

    Args:
        env_cfg (object): Common env config (e.g., env_id, control_mode).
    """
    library = str(getattr(env_cfg, "library", "") or "").strip().lower()
    env_id = str(getattr(env_cfg, "env_id", "") or "").strip()
    if not library:
        raise ValueError(
            "cfg.env.library is required before creating an environment. "
            "Set it explicitly to one of: mani_skill, robosuite, piper, realman, socket, gymnasium."
        )
    if library == "gymnasium" and not env_id:
        raise ValueError(f"cfg.env.env_id is required when cfg.env.library='{library}'.")
    if library in {"mani_skill", "robosuite", "piper"} and not str(getattr(env_cfg, "env_config_path", "") or "").strip():
        raise ValueError(f"cfg.env.env_config_path is required when cfg.env.library='{library}'.")
    env_file_cfg, env_config_path = _load_env_config(env_cfg)

    logger.info(f"Creating Environment: {env_id} via {library}...")

    # --- 1. Base Environment Creation ---
    if library == 'mani_skill':
        # ManiSkill-specific normalization is owned by agent_infra.  The
        # algorithm boundary is identical to every other project environment:
        # MetadataAdapterWrapper consumes the raw nested contract + meta_keys.
        from agent_factory.env.wrappers import MetadataAdapterWrapper
        from agent_infra.maniskill_env import ManiSkillEnv

        env = ManiSkillEnv.from_config_path(env_config_path)
        env = MetadataAdapterWrapper(
            env,
            flatten_obs_obj=_get_flatten_obs_obj(env_cfg),
            flatten_action=_get_flatten_action(env_cfg),
        )

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
        env = MetadataAdapterWrapper(
            env,
            flatten_obs_obj=_get_flatten_obs_obj(env_cfg),
            flatten_action=_get_flatten_action(env_cfg),
        )

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
        env = MetadataAdapterWrapper(
            env,
            flatten_obs_obj=_get_flatten_obs_obj(env_cfg),
            flatten_action=_get_flatten_action(env_cfg),
        )

    elif library == 'robosuite':
        from agent_factory.env.wrappers import MetadataAdapterWrapper
        from agent_infra.robosuite_env.Env.robosuite_env import RobosuiteEnv

        common_cfg = env_file_cfg.get("common", {}) if isinstance(env_file_cfg, dict) else {}
        hz = common_cfg.get("default_hz", getattr(env_cfg, "hz", None))
        env = RobosuiteEnv(
            config_path=env_config_path or None,
            hz=hz,
            env_control_mode=_get_robosuite_env_control_mode(env_cfg),
            controller_backend=getattr(env_cfg, "controller_backend", None),
        )
        env = MetadataAdapterWrapper(
            env,
            flatten_obs_obj=_get_flatten_obs_obj(env_cfg),
            flatten_action=_get_flatten_action(env_cfg),
        )

    elif library == 'socket':
        from agent_infra.socket_env import LegacySocketEnv, SocketEnv
        from agent_factory.env.wrappers import MetadataAdapterWrapper

        host = str(_socket_setting(env_cfg, env_file_cfg, "host", "") or "")
        port = _socket_setting(env_cfg, env_file_cfg, "port")
        if not host or port is None:
            raise ValueError("cfg.env.host and cfg.env.port are required when cfg.env.library='socket'.")
        protocol_version = int(_socket_setting(env_cfg, env_file_cfg, "protocol_version", 1))
        common_socket_kwargs = dict(
            source_host=_socket_setting(env_cfg, env_file_cfg, "source_host"),
            source_port=_socket_setting(env_cfg, env_file_cfg, "source_port"),
            tcp_nodelay=bool(_socket_setting(env_cfg, env_file_cfg, "tcp_nodelay", True)),
            keepalive=bool(_socket_setting(env_cfg, env_file_cfg, "keepalive", False)),
        )
        if protocol_version == 2:
            env = SocketEnv(
                host, int(port),
                connect_timeout_s=float(_socket_setting(env_cfg, env_file_cfg, "connect_timeout_s", 10.0)),
                request_timeout_s=float(_socket_setting(env_cfg, env_file_cfg, "request_timeout_s", 10.0)),
                **common_socket_kwargs,
            )
        elif protocol_version == 1:
            env = LegacySocketEnv(
                host, int(port),
                timeout=float(_socket_setting(env_cfg, env_file_cfg, "timeout", _socket_setting(env_cfg, env_file_cfg, "socket_timeout", 10.0))),
                **common_socket_kwargs,
            )
        else:
            raise ValueError(f"Unsupported socket protocol_version={protocol_version}; use explicit 1 or 2.")
        # v2 connect only DESCRIBEs; legacy v1 retains its push-first handshake.
        env.connect()
        env = MetadataAdapterWrapper(
            env,
            flatten_obs_obj=_get_flatten_obs_obj(env_cfg),
            flatten_action=_get_flatten_action(env_cfg),
        )

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

    if library == "piper":
        _maybe_start_piper_cameras(env, env_cfg)

    return env
