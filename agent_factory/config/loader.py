import os
import yaml
from omegaconf import OmegaConf
from agent_factory.config.structure import GlobalConfig
from agent_factory.config.manager import ConfigManager

def load_config(
    config_path: str = None, 
    cli_args: list = None
) -> GlobalConfig:
    """
    加载配置的通用入口。
    
    Args:
        config_path: 具体的 yaml 配置文件路径 (可以是外部路径)。
        cli_args: 命令行参数列表 (e.g. ['actor.lr=1e-3', 'seed=1'])
    
    Returns:
        GlobalConfig 对象 (支持点号访问 cfg.actor.lr)
    """
    # 1. 获取基础 Schema (默认值)
    base_cfg = OmegaConf.structured(GlobalConfig)
    override_cfg = OmegaConf.create({})
    
    # 2. 如果提供了 YAML 文件，则合并 (覆盖默认值)
    if config_path and os.path.exists(config_path):
        print(f"[Config] Loading from file: {config_path}")
        file_cfg = OmegaConf.load(config_path)
        base_cfg = OmegaConf.merge(base_cfg, file_cfg)
        override_cfg = OmegaConf.merge(override_cfg, file_cfg)
    elif config_path:
         print(f"[Config] Warning: Config file {config_path} not found. Using defaults.")

    # 3. 如果有 CLI 参数，最后合并 (优先级最高)
    if cli_args:
        cli_cfg = OmegaConf.from_dotlist(cli_args)
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)
        override_cfg = OmegaConf.merge(override_cfg, cli_cfg)
    
    # 4. 解析环境默认配置（例如 piper config），并保持用户覆盖优先
    merged_dict = OmegaConf.to_container(base_cfg, resolve=False)
    override_dict = OmegaConf.to_container(override_cfg, resolve=False)
    if isinstance(merged_dict, dict) and isinstance(override_dict, dict):
        merged_dict = ConfigManager._resolve_env_defaults(merged_dict, override_dict)
        base_cfg = OmegaConf.create(merged_dict)
        
    # 5. 转换为原生 Dataclass (可选，转为 DictConfig 更灵活)
    # return OmegaConf.to_object(base_cfg) 
    return base_cfg # 返回 DictConfig，支持 .key 访问和 OmegaConf.save

def save_config(cfg, save_dir):
    """将当前配置保存到 checkpoint 文件夹，确保实验可复现"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "config_snapshot.yaml")
    OmegaConf.save(cfg, path)
    print(f"[Config] Saved snapshot to {path}")
