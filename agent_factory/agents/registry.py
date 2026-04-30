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

    env_kwargs = cfg_dict.get("env_kwargs", {})
    if isinstance(env_kwargs, dict):
        realman = env_kwargs.get("realman", {})
        if isinstance(realman, dict) and "camera_sns" in realman:
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

    return OmegaConf.create(cfg_dict)

def register_agent(name: str):
    def decorator(cls):
        if name in _AGENT_REGISTRY:
            raise ValueError(f"Agent '{name}' is already registered!")
        _AGENT_REGISTRY[name] = cls
        return cls
    return decorator

def get_default_config(agent_type: str) -> OmegaConf:
    """
    Core Logic: Auto-Assembly
    根据 agent_type 获取类，扫描其 MRO，提取 Mixin 绑定的 CONFIG_CLASS，
    并自动挂载到 GlobalConfig 的对应插槽（actor, critic 等）。
    """
    if agent_type not in _AGENT_REGISTRY:
        raise ValueError(f"Agent '{agent_type}' not found. Available: {list(_AGENT_REGISTRY.keys())}")
    
    agent_cls = _AGENT_REGISTRY[agent_type]
    
    # 1. 实例化基础 GlobalConfig
    base_cfg = GlobalConfig(agent_type=agent_type)
    
    # 2. 扫描继承链 (MRO)
    # 我们倒序遍历 (reversed)，这样子类的配置定义（如果有）会覆盖父类
    # 但通常 Mixin 是正交的
    for cls in reversed(agent_cls.mro()):
        if hasattr(cls, "CONFIG_CLASS") and hasattr(cls, "CONFIG_KEY"):
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

    # 4. 转换为 OmegaConf 对象，支持后续的 YAML Merge
    return OmegaConf.structured(base_cfg)

def make_agent(agent_type: str, cfg: Any = None) -> 'BaseAgent':
    """
    工厂方法：
    如果未提供 cfg，则自动生成默认配置。
    如果提供了 cfg (Dict 或 DictConfig)，则与默认配置 Merge。
    """
    default_cfg = get_default_config(agent_type)
    
    if cfg is not None:
        user_cfg = _sanitize_common_cfg_fields(cfg)
        # Merge: Default -> User (User overrides Default)
        final_cfg = OmegaConf.merge(default_cfg, user_cfg)
    else:
        final_cfg = default_cfg
        
    return _AGENT_REGISTRY[agent_type](final_cfg)
