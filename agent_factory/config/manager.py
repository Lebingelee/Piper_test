import yaml
import os
import logging
from typing import Dict, Any
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    配置管理中心：负责配置的 校验、保存、读取。
    """

    @staticmethod
    def check_consistency(cfg) -> bool:
        """
        核心校验逻辑：检测 Env, Dataset, Actor 之间的参数是否冲突。
        在实例化 Agent 之前必须调用此函数。
        """
        logger.info("Starting Configuration Consistency Check...")

        def _get(path: str, default=None):
            """Safely read config values from DictConfig/Dataclass/Dict."""
            try:
                return OmegaConf.select(cfg, path, default=default)
            except Exception:
                return default

        actor_cfg = _get("actor", None)
        critic_cfg = _get("critic", None)
        
        # --- 1. Horizon Consistency (时间步一致性) ---
        # Obs Horizon: Env vs Dataset vs Actor
        env_oh = _get("env.obs_horizon", None)
        dataset_oh = _get("dataset.obs_horizon", None)
        actor_oh = _get("actor.obs_horizon", None)
        if dataset_oh is not None and env_oh is not None and dataset_oh != env_oh:
            raise ValueError(f"[Horizon Mismatch] Env obs_horizon({env_oh}) != Dataset({dataset_oh})")
        if actor_oh is not None and env_oh is not None and actor_oh != env_oh:
            raise ValueError(f"[Horizon Mismatch] Env obs_horizon({env_oh}) != Actor({actor_oh})")

        # Pred Horizon: Dataset vs Actor
        dataset_ph = _get("dataset.pred_horizon", None)
        actor_ph = _get("actor.pred_horizon", None)
        if actor_ph is not None and dataset_ph is not None and dataset_ph != actor_ph:
            raise ValueError(f"[Horizon Mismatch] Dataset pred_horizon({dataset_ph}) != Actor({actor_ph})")

        # --- 2. Dimension Consistency (维度一致性) ---
        # Action Dim
        env_action_dim = _get("env.action_dim", None)
        actor_action_dim = _get("actor.action_dim", None)
        require_match = bool(_get("actor.require_env_action_dim_match", True))
        if actor_cfg is not None and require_match and env_action_dim is not None and actor_action_dim is not None:
            if env_action_dim != actor_action_dim:
                raise ValueError(f"[Dim Mismatch] Env action_dim({env_action_dim}) != Actor({actor_action_dim})")

        # Proprio Dim (本体感知维度)
        # 注意：通常由 Dataset 动态计算后回填，这里检查回填后的结果
        env_pd = _get("env.proprio_dim", None)
        
        if env_pd is not None:
            if actor_cfg is not None:
                actor_pd = _get("actor.encoder.proprio_dim", None)
                if actor_pd != env_pd:
                    raise ValueError(f"[Dim Mismatch] Env proprio_dim({env_pd}) != Actor Encoder({actor_pd})")
            if critic_cfg is not None:
                critic_pd = _get("critic.encoder.proprio_dim", None)
                if critic_pd != env_pd:
                    raise ValueError(f"[Dim Mismatch] Env proprio_dim({env_pd}) != Critic Encoder({critic_pd})")

        # --- 3. Observation Mode Logic (模式一致性) ---
        mode = _get("env.obs_mode", "rgb")
        inc_rgb = bool(_get("dataset.include_rgb", True))
        inc_depth = bool(_get("dataset.include_depth", False))

        if mode == 'rgb' and not inc_rgb:
            raise ValueError(f"[Mode Conflict] Env obs_mode='rgb' but Dataset include_rgb=False")
        if mode == 'depth' and not inc_depth:
            raise ValueError(f"[Mode Conflict] Env obs_mode='depth' but Dataset include_depth=False")
        if mode == 'rgbd' and not (inc_rgb and inc_depth):
            raise ValueError(f"[Mode Conflict] Env obs_mode='rgbd' but Dataset missing rgb/depth")
        if mode == 'state' and (inc_rgb or inc_depth):
            logger.warning(f"[Mode Warning] Env obs_mode='state' but Dataset includes visual keys. Check efficiency.")

        logger.info("Configuration Check Passed. All systems go.")
        return True

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
    def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base or {})
        for k, v in (update or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = ConfigManager._deep_update(out[k], v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _infer_piper_dims(env_cfg: Dict[str, Any], piper_cfg: Dict[str, Any]) -> Dict[str, int]:
        robots = piper_cfg.get("robots", {}) if isinstance(piper_cfg, dict) else {}
        common = piper_cfg.get("common", {}) if isinstance(piper_cfg, dict) else {}
        cameras = piper_cfg.get("cameras", {}) if isinstance(piper_cfg, dict) else {}

        arm_count = max(len(robots), 1)
        control_mode = str(env_cfg.get("control_mode") or common.get("default_control_mode") or "joint")
        chunk_size = int(common.get("relative_pose_chunk_size", 8))
        arm_action_dim = chunk_size * 6 if control_mode == "relative_pose_chunk" else 6

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
    def _resolve_env_defaults(cfg_dict: Dict[str, Any], user_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        从 agent_infra 环境配置中解析默认参数，并与用户参数合并。
        约束：
        - 不改变用户现有 YAML 格式（仍支持 env_kwargs.realman / env_kwargs.mani_skill）。
        - 新环境可直接使用 env_kwargs.<library>，无需再改 structure.py。
        """
        env_cfg = cfg_dict.setdefault("env", {})
        env_kwargs = cfg_dict.setdefault("env_kwargs", {})
        if not isinstance(env_cfg, dict) or not isinstance(env_kwargs, dict):
            return cfg_dict

        library = str(env_cfg.get("library", "") or "").strip().lower()
        if not library:
            return cfg_dict

        lib_kwargs = env_kwargs.get(library)
        if lib_kwargs is None or not isinstance(lib_kwargs, dict):
            lib_kwargs = {}
            env_kwargs[library] = lib_kwargs

        # Generic env-specific YAML loading via config_path
        config_path = lib_kwargs.get("config_path") or env_cfg.get("config_path")

        # Piper default config fallback (keeps user YAML unchanged)
        if library == "piper" and not config_path:
            is_dual = bool(lib_kwargs.get("is_dual", False))
            default_rel = (
                "agent_infra/Piper_Env/Config/dual_piper_config.yaml"
                if is_dual else
                "agent_infra/Piper_Env/Config/piper_config.yaml"
            )
            config_path = default_rel

        loaded_cfg = {}
        if isinstance(config_path, str) and config_path.strip():
            abs_path = ConfigManager._resolve_config_path(config_path.strip())
            lib_kwargs["config_path"] = abs_path
            if os.path.exists(abs_path):
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        loaded_cfg = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Failed to load env config from {abs_path}: {e}")
            else:
                logger.warning(f"Env config path not found: {abs_path}")

        # 对特定库做自动推导（当前先支持 piper）
        if library == "piper" and isinstance(loaded_cfg, dict):
            inferred = ConfigManager._infer_piper_dims(env_cfg, loaded_cfg)

            # 用户显式设置优先；否则使用推导值覆盖全局默认
            for key, val in inferred.items():
                if not ConfigManager._dict_has_path(user_dict, ["env", key]):
                    env_cfg[key] = val

            # 将详细配置合并进 env_kwargs.piper.defaults，供后续环境构建使用
            defaults_slot = lib_kwargs.get("defaults", {})
            if not isinstance(defaults_slot, dict):
                defaults_slot = {}
            lib_kwargs["defaults"] = ConfigManager._deep_update(defaults_slot, loaded_cfg)

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

        def _normalize_camera_sns(cfg_dict: Dict[str, Any]):
            env_kwargs = cfg_dict.get("env_kwargs", {})
            if not isinstance(env_kwargs, dict):
                return
            realman = env_kwargs.get("realman", {})
            if not isinstance(realman, dict) or "camera_sns" not in realman:
                return
            camera_sns = realman.get("camera_sns")
            if camera_sns is None:
                realman["camera_sns"] = []
            elif isinstance(camera_sns, str):
                token = camera_sns.strip()
                if token == "" or token.lower() in {"none", "null"}:
                    realman["camera_sns"] = []
                else:
                    realman["camera_sns"] = [token]
            elif not isinstance(camera_sns, list):
                try:
                    realman["camera_sns"] = list(camera_sns)
                except TypeError:
                    realman["camera_sns"] = []

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

            _normalize_camera_sns(user_dict)
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
        return OmegaConf.load(path)
