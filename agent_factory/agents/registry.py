from typing import Dict, Type, TYPE_CHECKING, Any
from omegaconf import OmegaConf
from agent_factory.config.structure import GlobalConfig

if TYPE_CHECKING:
    from agent_factory.agents.base_agent import BaseAgent

_AGENT_REGISTRY: Dict[str, Type['BaseAgent']] = {}


def _sanitize_common_cfg_fields(cfg_obj: Any):
    """
    Normalize weakly-typed YAML values before structured merge.
    Centralized here so all models benefit, not only DSRL.
    """
    cfg_dict = OmegaConf.to_container(OmegaConf.create(cfg_obj), resolve=False)
    if not isinstance(cfg_dict, dict):
        return OmegaConf.create(cfg_obj)

    from agent_factory.config.manager import ConfigManager
    cfg_dict = ConfigManager._resolve_runner_defaults(cfg_dict)
    cfg_dict = ConfigManager._resolve_dataset_defaults(cfg_dict)
    return OmegaConf.create(cfg_dict)

def register_agent(name: str):
    def decorator(cls):
        if name in _AGENT_REGISTRY:
            raise ValueError(f"Agent '{name}' is already registered!")
        _AGENT_REGISTRY[name] = cls
        return cls
    return decorator


def infer_default_exp_name_from_agent_type(agent_type: str) -> str:
    agent_name = str(agent_type or "").strip().lower().replace("_", "-")
    if agent_name == "pi05":
        return "pi05"
    if agent_name in {"smolvla", "smolvla-vanilla"}:
        return "smolvla"
    if agent_name == "tdqc-rnn":
        return "tdqc_rnn"
    if agent_name == "cpiql-rnn":
        return "cpiql_rnn"
    if agent_name == "tdqc-mlp":
        return "tdqc_mlp"
    if "tdqc-rnn" in agent_name:
        return "tdqc_rnn"
    if "tdqc" in agent_name:
        return "tdqc_mlp"
    return ""


def iter_agent_config_mixins(agent_type: str):
    """
    Yield config-bearing mixins/classes for a registered agent in assembly order.

    This is the single public way for config resolution to discover which mixins
    an impl actually uses; callers should not duplicate MRO traversal logic.
    """
    if agent_type not in _AGENT_REGISTRY:
        raise ValueError(f"Agent '{agent_type}' not found. Available: {list(_AGENT_REGISTRY.keys())}")
    agent_cls = _AGENT_REGISTRY[agent_type]
    for cls in reversed(agent_cls.mro()):
        if hasattr(cls, "CONFIG_CLASS") and hasattr(cls, "CONFIG_KEY"):
            yield cls


def get_default_config(agent_type: str) -> OmegaConf:
    """
    Core Logic: Auto-Assembly
    根据 agent_type 获取类，扫描其 MRO，提取 Mixin 绑定的 CONFIG_CLASS，
    并自动挂载到 GlobalConfig 的对应插槽（actor, critic 等）。
    """
    # 1. 实例化基础 GlobalConfig
    base_cfg = GlobalConfig(agent_type=agent_type)
    
    # 2. 扫描继承链 (MRO)
    # 我们倒序遍历 (reversed)，这样子类的配置定义（如果有）会覆盖父类
    # 但通常 Mixin 是正交的
    for cls in iter_agent_config_mixins(agent_type):
        config_cls = getattr(cls, "CONFIG_CLASS")
        config_key = getattr(cls, "CONFIG_KEY")
        
        # 3. 实例化默认配置并挂载
        # 例如: base_cfg.actor = DiffusionActorConfig()
        if hasattr(base_cfg, config_key):
            setattr(base_cfg, config_key, config_cls())
        else:
            # 如果 GlobalConfig 没有预定义这个 slot，可以选择动态添加或报错
            # 这里我们假设 GlobalConfig 涵盖了所有标准 slot
            print(f"[Registry] Warning: Dynamic config key '{config_key}' added to GlobalConfig.")
            setattr(base_cfg, config_key, config_cls())

    agent_name = str(agent_type or "").strip().lower()
    if "tdqc" in agent_name:
        base_cfg.dataset.dataset_type = "tdqc"
        if not str(base_cfg.train.exp_name or "").strip():
            base_cfg.train.exp_name = infer_default_exp_name_from_agent_type(agent_type) or "tdqc_mlp"
    elif "itqc" in agent_name:
        base_cfg.dataset.dataset_type = "diffusion_itqc"
    elif agent_name == "cpiql-rnn":
        base_cfg.dataset.dataset_type = "vla_feature_cpiql_rnn"
        base_cfg.train.exp_name = infer_default_exp_name_from_agent_type(agent_type) or "cpiql_rnn"
    elif "cpiql" in agent_name:
        base_cfg.dataset.dataset_type = "cpiql"
    elif agent_name == "pi05":
        base_cfg.train.exp_name = infer_default_exp_name_from_agent_type(agent_type) or "pi05"
    elif agent_name in {"smolvla", "smolvla_vanilla", "smolvla-vanilla"}:
        base_cfg.env.action_dim = 7
        base_cfg.env.proprio_dim = 23
        base_cfg.env.num_cameras = 2
        base_cfg.env.pred_horizon = 50
        base_cfg.env.flatten_obs_obj = ["state"]
        base_cfg.dataset.dataset_type = "smolvla_h5"
        base_cfg.dataset.expert.demo_path = "data/hugging_face/delta_all/Robosuite_3_task.h5"
        if getattr(base_cfg, "actor", None) is not None:
            base_cfg.actor.action_dim = base_cfg.env.action_dim
            base_cfg.actor.pred_horizon = base_cfg.env.pred_horizon
        base_cfg.train.exp_name = infer_default_exp_name_from_agent_type(agent_type) or "smolvla"

    # 4. 转换为 OmegaConf 对象，支持后续的 YAML Merge
    return OmegaConf.structured(base_cfg)

def make_agent(agent_type: str, cfg: Any = None) -> 'BaseAgent':
    """
    Build an agent from a resolved config.

    Config resolution owns defaults, file config, command-line overrides, and
    conflict checks. This factory only instantiates the registered agent class.
    """
    if cfg is None:
        raise ValueError("make_agent requires a resolved config. Call general_resolve() first.")
    if agent_type not in _AGENT_REGISTRY:
        raise ValueError(f"Agent '{agent_type}' not found. Available: {list(_AGENT_REGISTRY.keys())}")

    final_cfg = OmegaConf.create(cfg)
    if not bool(OmegaConf.select(final_cfg, "resolved", default=False)):
        raise ValueError("make_agent requires cfg.resolved=True. Call general_resolve() before make_agent().")
    if OmegaConf.select(final_cfg, "agent_type", default=None) in {None, ""}:
        final_cfg.agent_type = agent_type
    return _AGENT_REGISTRY[agent_type](final_cfg)
