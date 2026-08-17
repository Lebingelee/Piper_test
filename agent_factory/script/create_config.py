import argparse
import os
import sys
from typing import Any, Dict

from omegaconf import OmegaConf


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import agent implementations so @register_agent decorators populate registry.
import agent_factory.agents  # noqa: F401,E402
from agent_factory.agents.registry import get_default_config  # noqa: E402
from agent_factory.data.registry import infer_dataset_type_from_agent_type  # noqa: E402


def _canonical_training_config(cfg: Any) -> Dict[str, Any]:
    """
    Save an editable training config rather than the raw internal config.

    The runtime still keeps a top-level cfg.device compatibility alias and a
    dataset.replay legacy alias, but user-facing train configs should keep
    device under cfg.train and replay data under cfg.dataset.replaybuffer.
    """
    snapshot = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(snapshot, dict):
        return {}

    snapshot.pop("device", None)
    dataset_cfg = snapshot.get("dataset")
    if isinstance(dataset_cfg, dict):
        dataset_cfg.pop("replay", None)
    return snapshot


def create_default_config(
    agent_type: str,
    save_dir: str,
    filename: str = "default_config.yaml",
) -> str:
    cfg = get_default_config(agent_type)
    cfg.dataset.dataset_type = infer_dataset_type_from_agent_type(agent_type)
    snapshot = _canonical_training_config(cfg)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, filename)
    OmegaConf.save(OmegaConf.create(snapshot), output_path, resolve=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a default editable config for an agent.")
    parser.add_argument(
        "--agent-type",
        type=str,
        default="Diffusion_CPIQL_DAC",
        help="Registered agent type, e.g. Diffusion_CPIQL_DAC, Diffusion_ITQC, Diffusion_Vanilla, dsrl.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory where default_config.yaml will be saved.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="default_config.yaml",
        help="Config filename to save under --save-dir.",
    )
    args = parser.parse_args()

    output_path = create_default_config(
        agent_type=args.agent_type,
        save_dir=args.save_dir,
        filename=args.filename,
    )
    print(f"[CreateConfig] Saved {args.agent_type} default config to {output_path}")


if __name__ == "__main__":
    main()
