"""
Manual training launcher for users who prefer editing Python variables directly.

Typical usage:
1. Edit CONFIG_PATH and OVERRIDES below.
2. Run: python3 script/train_universal_manual.py
"""

import os
import sys
from typing import Any, Dict


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent_factory.script.train_universal import train_universal


#"run_results/piper_dual_merged_cpiql_dac/config.yaml"
CONFIG_PATH = "run_results/piper_dual_merged_cpiql_dac/model_config.yaml"
DRY_RUN = False
OVERRIDES: Dict[str, Any] = {
    "device": None,
    "train_object": None,
    "dataset_mode": None,
    "dataset_key": None,
    "critic_iters": None,
    "actor_iters": None,
    "batch_size": None,
    "num_workers": None,
    "save_interval": None,
    "save_root": None,
    "exp_name": None,
    "critic_ckpt_path": None,
}


def main():
    return train_universal(config_path=CONFIG_PATH, overrides=OVERRIDES, dry_run=DRY_RUN)


if __name__ == "__main__":
    main()
