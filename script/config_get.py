import os
import sys
import torch
import argparse

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.registry import get_default_config, make_agent
from agent_factory.env.env_factories import create_env
from agent_factory.data.dataset import ExpertDataset
from agent_factory.config.manager import ConfigManager

def main():
    parser = argparse.ArgumentParser(description="Train Dual-Arm Diffusion-ITQC on Towel Folding.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file to load.")
    args = parser.parse_args()
    final_save_dir = "run_results"
    # 1. 获取默认配置
    print("[Init] Loading default Diffusion_ITQC configuration...")
    cfg = get_default_config("Diffusion_ITQC")

    print(f"[Config] Saving final configuration to: {final_save_dir}")
    ConfigManager.save_config(cfg, save_dir=final_save_dir)

if __name__ == "__main__":
    main()
