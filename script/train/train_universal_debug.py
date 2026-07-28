import os
import sys
from typing import Dict, Optional

if __package__ in {None, ""}:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from agent_factory.script.train_universal import resolve_config_path, train_universal


# Manual entrypoint for local experiments.
# Edit the constants below, then run:
#   python agent_factory/script/train_universal_manual.py

CONFIG_PATH: Optional[str] = "train_setting/debug/cpiql_finetune.yaml"
FINETUNE: bool = True
DRY_RUN: bool = False

TRAIN_OVERRIDES: Dict[str, object] = {
    "critic_iters": 3000,
    "actor_iters": 20000,
    #"ckpt_path": "run_results/piper_dual_merged_cpiql_dac/actor_final.pth",
    # Common finetune defaults. Remove or edit as needed.
    "dataset_key": "replaybuffer",
    "exp_name": "piper_dual_merged_cpiql_dac_finetune",
    "finetune": FINETUNE,
}


def main() -> None:
    config_path = resolve_config_path(
        config_path=CONFIG_PATH,
        finetune=FINETUNE,
        ckpt_path=str(TRAIN_OVERRIDES.get("ckpt_path", "") or ""),
    )
    result = train_universal(
        config_path=config_path,
        overrides=TRAIN_OVERRIDES,
        dry_run=DRY_RUN,
        finetune=FINETUNE,
    )
    print(f"[ManualTrain] Config snapshot: {os.path.join(result['run_root'], result['snapshot_filename'])}")


if __name__ == "__main__":
    main()
