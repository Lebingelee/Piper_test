import yaml
import os
import logging
from typing import Dict, Any
from omegaconf import OmegaConf
from agent_factory.control import canonicalize_control_mode, to_legacy_control_mode

logger = logging.getLogger(__name__)
_SEMANTIC_CONTROL_MODES = {
    "absolute_joint",
    "joint",
    "joint_pos",
    "absolute_pose",
    "pose",
    "delta_pose",
    "delta_ee_pose",
    "pd_ee_delta_pose",
    "delta_joint",
    "relative_pose_chunk",
}
_RUNNER_SHARED_KEYS = {
    "type",
    "control_hz",
    "buffer_capacity",
    "redundancy_margin",
    "save_dir",
    "no_safe_action_gap",
    "planning_wait_sleep",
    "config_path",
    "config",
}
_RUNNER_LEGACY_CONFIG_KEYS = {
    "hitl_enabled",
    "hitl_override_key",
    "hitl_teleop_key",
    "hitl_source",
    "hitl_finalize_intervention",
    "risk_check_hz",
    "risk_use_safe_action",
    "deploy_start_key",
    "deploy_stop_key",
    "deploy_continue_key",
    "deploy_init_key",
    "deploy_quit_key",
    "deploy_teleop_key",
    "deploy_save_key",
}
_DATASET_SHARED_KEYS = {
    "dataset_type",
    "include_rgb",
    "include_depth",
    "config_path",
    "config",
    "expert",
    "replaybuffer",
    "replay",
}
_DEFAULT_RUNNER_CONFIG_PATHS = {
    "base": "agent_factory/runner/config/base.yaml",
    "sim": "agent_factory/runner/config/sim.yaml",
    "pi05": "agent_factory/runner/config/base.yaml",
    "hitl": "agent_factory/runner/config/hitl.yaml",
    "hitl_deploy": "agent_factory/runner/config/hitl_deploy.yaml",
}

class ConfigManager:
    """
    配置管理中心：负责配置保存、读取和旧格式默认值归一化。

    参数一致性检查由 agent_factory.config.resolution.general_resolve()
    结合 module / mixin 声明规则负责。
    """

    @staticmethod
    def save_config(cfg: Any, save_dir: str, filename: str = "config.yaml"):
        """将配置保存为 YAML (兼容 OmegaConf, Dataclass, Dict)"""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        
        try:
            # OmegaConf.save 自动处理了 Dataclass 和 DictConfig 的转换
            # resolve=True 会将配置中的变量引用（如 ${env.lr}）解析为具体数值
            OmegaConf.save(config=cfg, f=path, resolve=True)
            # logger.info(f"Config saved successfully to: {path}") # 确保 logger 已定义
        except Exception as e:
            print(f"Failed to save config: {e}")

    @staticmethod
    def _dict_has_path(d: Dict[str, Any], path: list) -> bool:
        cur = d
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return False
            cur = cur[key]
        return True

    @staticmethod
    def _project_root() -> str:
        # agent_factory/config/manager.py -> repo root
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    @staticmethod
    def _resolve_config_path(path: str) -> str:
        if not path:
            return path
        if os.path.isabs(path):
            return path
        return os.path.join(ConfigManager._project_root(), path)

    @staticmethod
    def _load_yaml_dict(path: str) -> Dict[str, Any]:
        if not path:
            return {}
        abs_path = ConfigManager._resolve_config_path(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Config path not found: {abs_path}")
        with open(abs_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def _resolve_runner_defaults(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize runner configuration before structured merge.

        Public runner fields stay small and shared. Runner-specific knobs live in
        runner.config and are populated from runner.config_path, with inline
        runner.config overriding file defaults.
        """
        runner_cfg = cfg_dict.setdefault("runner", {})
        if not isinstance(runner_cfg, dict):
            cfg_dict["runner"] = {}
            runner_cfg = cfg_dict["runner"]

        runner_type = str(runner_cfg.get("type") or "base").strip() or "base"
        runner_cfg["type"] = runner_type
        config_path = runner_cfg.get("config_path") or _DEFAULT_RUNNER_CONFIG_PATHS.get(
            runner_type,
            _DEFAULT_RUNNER_CONFIG_PATHS["base"],
        )
        runner_cfg["config_path"] = ConfigManager._resolve_config_path(str(config_path))

        inline_config = runner_cfg.get("config") or {}
        if not isinstance(inline_config, dict):
            inline_config = {}

        try:
            file_config = ConfigManager._load_yaml_dict(runner_cfg["config_path"])
        except FileNotFoundError:
            if inline_config:
                logger.warning(
                    "Runner config_path not found, using inline runner.config only: %s",
                    runner_cfg["config_path"],
                )
                file_config = {}
            else:
                raise

        merged_config = dict(file_config)
        merged_config.update(inline_config)

        for key in list(runner_cfg.keys()):
            if key in _RUNNER_LEGACY_CONFIG_KEYS:
                merged_config.setdefault(key, runner_cfg.pop(key))
            elif key not in _RUNNER_SHARED_KEYS:
                merged_config.setdefault(key, runner_cfg.pop(key))

        runner_cfg["config"] = merged_config
        return cfg_dict

    @staticmethod
    def _resolve_dataset_defaults(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize dataset-specific configuration before structured merge.

        Public dataset fields remain shared across dataset types. Dataset-type
        specific knobs live in dataset.config and can be populated from
        dataset.config_path, with inline dataset.config overriding file defaults.
        """
        dataset_cfg = cfg_dict.setdefault("dataset", {})
        if not isinstance(dataset_cfg, dict):
            cfg_dict["dataset"] = {}
            dataset_cfg = cfg_dict["dataset"]

        config_path = str(dataset_cfg.get("config_path") or "").strip()
        inline_config = dataset_cfg.get("config") or {}
        if not isinstance(inline_config, dict):
            inline_config = {}

        file_config: Dict[str, Any] = {}
        if config_path:
            resolved_path = ConfigManager._resolve_config_path(config_path)
            dataset_cfg["config_path"] = resolved_path
            loaded = ConfigManager._load_yaml_dict(resolved_path)
            loaded_dataset = loaded.get("dataset") if isinstance(loaded.get("dataset"), dict) else None
            if loaded_dataset is not None:
                if isinstance(loaded_dataset.get("config"), dict):
                    file_config.update(loaded_dataset["config"])
                for key, value in loaded_dataset.items():
                    if key not in _DATASET_SHARED_KEYS:
                        file_config[key] = value
            else:
                file_config.update(loaded)

        for key in list(dataset_cfg.keys()):
            if key not in _DATASET_SHARED_KEYS:
                inline_config.setdefault(key, dataset_cfg.pop(key))

        merged_config = dict(file_config)
        merged_config.update(inline_config)
        dataset_cfg["config"] = merged_config
        return cfg_dict

    @staticmethod
    def _infer_piper_dims(env_cfg: Dict[str, Any], piper_cfg: Dict[str, Any]) -> Dict[str, int]:
        robots = piper_cfg.get("robots", {}) if isinstance(piper_cfg, dict) else {}
        common = piper_cfg.get("common", {}) if isinstance(piper_cfg, dict) else {}
        cameras = piper_cfg.get("cameras", {}) if isinstance(piper_cfg, dict) else {}

        arm_count = max(len(robots), 1)
        control_mode = str(
            env_cfg.get("control_mode")
            or env_cfg.get("env_control_mode")
            or common.get("default_control_mode")
            or "joint"
        )
        chunk_size = int(common.get("relative_pose_chunk_size", 8))
        arm_action_dim = (
            chunk_size * 6
            if canonicalize_control_mode(control_mode, default="joint") == "relative_pose_chunk"
            else 6
        )

        action_dim = arm_count * (arm_action_dim + 1)
        proprio_dim = arm_count * 19  # joint_pos(6)+joint_vel(6)+ee_pose(6)+gripper_pos(1)
        num_cameras = len(cameras.get("nodes", []) or [])
        if num_cameras <= 0:
            num_cameras = 1

        return {
            "action_dim": int(action_dim),
            "proprio_dim": int(proprio_dim),
            "num_cameras": int(num_cameras),
        }

    @staticmethod
    def _sync_control_mode_fields(
        env_cfg: Dict[str, Any],
        cfg_dict: Dict[str, Any],
        user_dict: Dict[str, Any],
        library: str,
    ):
        legacy_mode = env_cfg.get("control_mode")
        env_mode = env_cfg.get("env_control_mode")

        if env_mode is None and legacy_mode is not None:
            raw_legacy = str(legacy_mode).strip().lower()
            if raw_legacy in _SEMANTIC_CONTROL_MODES:
                env_mode = canonicalize_control_mode(legacy_mode)
                env_cfg["env_control_mode"] = env_mode
        elif env_mode is not None:
            env_cfg["env_control_mode"] = canonicalize_control_mode(env_mode)

        if legacy_mode is None and env_mode is not None:
            guessed_legacy = None
            if library == "realman":
                realman_legacy = {
                    "absolute_joint": "joint_pos",
                    "delta_pose": "delta_ee_pose",
                }
                guessed_legacy = realman_legacy.get(env_cfg["env_control_mode"])
            elif library == "mani_skill":
                maniskill_legacy = {
                    "delta_pose": "pd_ee_delta_pose",
                }
                guessed_legacy = maniskill_legacy.get(env_cfg["env_control_mode"])
            else:
                guessed_legacy = to_legacy_control_mode(env_mode, default=str(env_mode))
            if guessed_legacy is not None:
                env_cfg["control_mode"] = guessed_legacy

        if (
            not ConfigManager._dict_has_path(user_dict, ["agent_control_mode"])
            and env_cfg.get("env_control_mode") is not None
        ):
            cfg_dict["agent_control_mode"] = env_cfg["env_control_mode"]

    @staticmethod
    def _resolve_env_defaults(cfg_dict: Dict[str, Any], user_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        从 agent_infra 环境配置中解析默认参数，并与用户参数合并。
        约束：
        - env.env_config_path 是环境特定 YAML 的唯一入口。
        - 主配置的 env.* 字段优先；环境 YAML 只用于补默认值和推断维度。
        - 不再兼容旧 env_kwargs / env.env_config 写法。
        """
        env_cfg = cfg_dict.setdefault("env", {})
        if not isinstance(env_cfg, dict):
            return cfg_dict

        library = str(env_cfg.get("library", "") or "").strip().lower()
        if not library:
            return cfg_dict

        config_path = env_cfg.get("env_config_path")

        if library == "piper" and not config_path:
            config_path = "agent_infra/Piper_Env/Config/piper_config.yaml"
        elif library == "robosuite" and not config_path:
            config_path = "agent_infra/robosuite_env/Config/lift_panda_state.yaml"

        loaded_cfg = {}
        if isinstance(config_path, str) and config_path.strip():
            abs_path = ConfigManager._resolve_config_path(config_path.strip())
            env_cfg["env_config_path"] = abs_path
            if os.path.exists(abs_path):
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        loaded_cfg = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Failed to load env config from {abs_path}: {e}")
            else:
                logger.warning(f"Env config path not found: {abs_path}")

        if library == "realman" and isinstance(loaded_cfg, dict):
            robot_cfg = loaded_cfg.get("robot", {})
            common_cfg = loaded_cfg.get("common", {})
            defaults = common_cfg if isinstance(common_cfg, dict) and common_cfg else robot_cfg
            if isinstance(defaults, dict):
                if (
                    "default_control_mode" in defaults
                    and not ConfigManager._dict_has_path(user_dict, ["env", "control_mode"])
                ):
                    env_cfg["control_mode"] = defaults["default_control_mode"]
                if (
                    "default_hz" in defaults
                    and not ConfigManager._dict_has_path(user_dict, ["runner", "control_hz"])
                ):
                    cfg_dict.setdefault("runner", {})["control_hz"] = defaults["default_hz"]
            if not ConfigManager._dict_has_path(user_dict, ["env", "env_control_mode"]):
                env_cfg["env_control_mode"] = canonicalize_control_mode(
                    env_cfg.get("control_mode") or defaults.get("default_control_mode") or "joint_pos"
                )
            if not ConfigManager._dict_has_path(user_dict, ["env", "controller_backend"]):
                env_cfg["controller_backend"] = env_cfg.get("control_mode", defaults.get("default_control_mode", "joint_pos"))

        # 对特定库做自动推导（当前先支持 piper）
        if library == "piper" and isinstance(loaded_cfg, dict):
            common_cfg = loaded_cfg.get("common", {})
            if isinstance(common_cfg, dict):
                if (
                    "default_control_mode" in common_cfg
                    and not ConfigManager._dict_has_path(user_dict, ["env", "control_mode"])
                ):
                    env_cfg["control_mode"] = common_cfg["default_control_mode"]
                if (
                    "default_hz" in common_cfg
                    and not ConfigManager._dict_has_path(user_dict, ["runner", "control_hz"])
                ):
                    cfg_dict.setdefault("runner", {})["control_hz"] = common_cfg["default_hz"]
                if not ConfigManager._dict_has_path(user_dict, ["env", "env_control_mode"]):
                    env_cfg["env_control_mode"] = canonicalize_control_mode(
                        env_cfg.get("control_mode") or common_cfg.get("default_control_mode") or "joint"
                    )
                if not ConfigManager._dict_has_path(user_dict, ["env", "controller_backend"]):
                    env_cfg["controller_backend"] = env_cfg.get("control_mode", common_cfg.get("default_control_mode", "joint"))

            inferred = ConfigManager._infer_piper_dims(env_cfg, loaded_cfg)

            # 用户显式设置优先；否则使用推导值覆盖全局默认
            for key, val in inferred.items():
                if not ConfigManager._dict_has_path(user_dict, ["env", key]):
                    env_cfg[key] = val

        if library == "robosuite" and isinstance(loaded_cfg, dict):
            common_cfg = loaded_cfg.get("common", {}) or {}
            task_cfg = loaded_cfg.get("task", {}) or {}
            action_cfg = loaded_cfg.get("action", {}) or {}
            controller_cfg = loaded_cfg.get("controller", {}) or {}
            robot_cfg = loaded_cfg.get("robot", {}) or {}
            contract_cfg = loaded_cfg.get("contract", {}) or {}
            rollout_cfg = loaded_cfg.get("rollout", {}) or {}

            if not ConfigManager._dict_has_path(user_dict, ["env", "env_control_mode"]):
                env_cfg["env_control_mode"] = canonicalize_control_mode(
                    action_cfg.get("env_control_mode") or "absolute_pose"
                )
            if not ConfigManager._dict_has_path(user_dict, ["env", "controller_backend"]):
                env_cfg["controller_backend"] = str(
                    action_cfg.get("controller_backend")
                    or controller_cfg.get("name")
                    or robot_cfg.get("default_controller")
                    or ""
                )
            if not ConfigManager._dict_has_path(user_dict, ["runner", "control_hz"]):
                control_hz = common_cfg.get("default_hz", task_cfg.get("control_freq"))
                if control_hz is not None:
                    cfg_dict.setdefault("runner", {})["control_hz"] = int(control_hz)
            if not ConfigManager._dict_has_path(user_dict, ["env", "max_episode_steps"]):
                horizon = rollout_cfg.get("benchmark_horizon", task_cfg.get("horizon"))
                if horizon is not None:
                    env_cfg["max_episode_steps"] = int(horizon)

            for key in ("action_dim", "proprio_dim", "num_cameras", "obs_mode"):
                if key in contract_cfg and not ConfigManager._dict_has_path(user_dict, ["env", key]):
                    env_cfg[key] = contract_cfg[key]

        if library == "mani_skill" and isinstance(loaded_cfg, dict):
            task_cfg = loaded_cfg.get("task", {}) or {}
            action_cfg = loaded_cfg.get("action", {}) or {}
            contract_cfg = loaded_cfg.get("contract", {}) or {}
            if not ConfigManager._dict_has_path(user_dict, ["env", "env_id"]):
                env_id = task_cfg.get("env_id")
                if env_id:
                    env_cfg["env_id"] = str(env_id)
            if not ConfigManager._dict_has_path(user_dict, ["env", "control_mode"]):
                control_mode = action_cfg.get("control_mode")
                if control_mode:
                    env_cfg["control_mode"] = str(control_mode)
            if not ConfigManager._dict_has_path(user_dict, ["env", "env_control_mode"]):
                env_mode = action_cfg.get("env_control_mode")
                if env_mode:
                    env_cfg["env_control_mode"] = canonicalize_control_mode(env_mode)
            if not ConfigManager._dict_has_path(user_dict, ["env", "controller_backend"]):
                backend = action_cfg.get("control_mode")
                if backend:
                    env_cfg["controller_backend"] = str(backend)
            if not ConfigManager._dict_has_path(user_dict, ["env", "max_episode_steps"]):
                horizon = task_cfg.get("max_episode_steps")
                if horizon is not None:
                    env_cfg["max_episode_steps"] = int(horizon)
            for key in ("action_dim", "proprio_dim", "num_cameras", "obs_mode"):
                if key in contract_cfg and not ConfigManager._dict_has_path(user_dict, ["env", key]):
                    env_cfg[key] = contract_cfg[key]

        has_maniskill_config_mode = (
            library == "mani_skill"
            and isinstance(loaded_cfg, dict)
            and bool((loaded_cfg.get("action", {}) or {}).get("env_control_mode"))
        )
        if (
            library == "mani_skill"
            and not has_maniskill_config_mode
            and not ConfigManager._dict_has_path(user_dict, ["env", "env_control_mode"])
        ):
            env_cfg["env_control_mode"] = canonicalize_control_mode(
                env_cfg.get("control_mode") or "pd_ee_delta_pose"
            )
        if library == "mani_skill" and not ConfigManager._dict_has_path(user_dict, ["env", "controller_backend"]):
            env_cfg["controller_backend"] = env_cfg.get("control_mode", "pd_ee_delta_pose")

        ConfigManager._sync_control_mode_fields(env_cfg, cfg_dict, user_dict, library)

        return cfg_dict

    @staticmethod
    def merge_configs(base_cfg, user_cfg):
        """将用户配置合并到基础配置上"""
        user_cfg = OmegaConf.create(user_cfg)

        def _normalize_base_policy_cfg(bp: Any) -> Dict[str, Any]:
            bp = bp or {}
            if not isinstance(bp, dict):
                return {}
            out = {
                "type": bp.get("type", bp.get("agent_type", "")),
                "ckpt_path": bp.get("ckpt_path", bp.get("checkpoint_path", "")),
                "base_config_path": bp.get("base_config_path", ""),
            }
            return out

        # DSRL compatibility migration:
        # move deprecated actor.base_policy -> agent_sp.base_policy
        user_dict = OmegaConf.to_container(user_cfg, resolve=False)
        if isinstance(user_dict, dict):
            agent_sp_cfg = user_dict.setdefault("agent_sp", {})
            if not isinstance(agent_sp_cfg, dict):
                agent_sp_cfg = {}
                user_dict["agent_sp"] = agent_sp_cfg

            actor_cfg = user_dict.get("actor", {})
            if isinstance(actor_cfg, dict) and "base_policy" in actor_cfg:
                if "base_policy" not in agent_sp_cfg:
                    agent_sp_cfg["base_policy"] = _normalize_base_policy_cfg(actor_cfg.get("base_policy"))
                actor_cfg.pop("base_policy", None)

            if "base_policy" in agent_sp_cfg:
                agent_sp_cfg["base_policy"] = _normalize_base_policy_cfg(agent_sp_cfg.get("base_policy"))

            user_dict = ConfigManager._resolve_runner_defaults(user_dict)
            user_dict = ConfigManager._resolve_dataset_defaults(user_dict)
            user_cfg = OmegaConf.create(user_dict)
        else:
            user_dict = {}

        merged = OmegaConf.merge(base_cfg, user_cfg)
        merged_dict = OmegaConf.to_container(merged, resolve=False)
        if isinstance(merged_dict, dict):
            merged_dict = ConfigManager._resolve_env_defaults(merged_dict, user_dict)
            return OmegaConf.create(merged_dict)
        return merged

    @staticmethod
    def load_config(path: str) -> Any:
        """读取 YAML 配置并返回 DictConfig 对象"""
        # OmegaConf.load 返回的对象既可以像字典一样 cfg['key'] 访问，
        # 也可以像对象一样 cfg.key 访问，非常方便
        cfg_dict = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        if isinstance(cfg_dict, dict):
            cfg_dict = ConfigManager._resolve_runner_defaults(cfg_dict)
            cfg_dict = ConfigManager._resolve_dataset_defaults(cfg_dict)
            return OmegaConf.create(cfg_dict)
        return OmegaConf.create(cfg_dict)
