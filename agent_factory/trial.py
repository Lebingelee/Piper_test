"""
Minimal smoke script for agent_factory.

Usage example:
    python3 -m agent_factory.trial --agent-type Diffusion_ITQC --steps 10
"""

from __future__ import annotations

import argparse

from agent_factory.agents.registry import get_default_config, make_agent
from agent_factory.config.resolution import general_resolve
from agent_factory.data.impl.diffusion_itqc.dataset import ExpertDataset
from agent_factory.env.env_factories import create_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="agent_factory smoke trial")
    parser.add_argument("--agent-type", default="Diffusion_ITQC", help="registered agent type")
    parser.add_argument("--steps", type=int, default=0, help="quick run steps for agents that support num_steps")
    parser.add_argument("--demo-path", default=None, help="optional override for dataset.expert.demo_path")
    return parser


def main():
    args = build_parser().parse_args()

    cfg = get_default_config(args.agent_type)
    cfg.train.batch_size = 2
    cfg.train.num_workers = 0

    if args.demo_path:
        cfg.dataset.expert.demo_path = args.demo_path

    cfg, _runtime_spec = general_resolve(file_config=cfg)

    # Keep this script lightweight: only align config and build components.
    env = create_env(cfg.env)
    env.reset()

    dataset = ExpertDataset(
        cfg=cfg,
        obs_space=getattr(env, "observation_space", None),
        device=cfg.device,
    )

    agent = make_agent(args.agent_type, cfg)
    required = agent.required_keys
    print(f"[Trial] agent={args.agent_type}, required_keys={required}, dataset_len={len(dataset)}")

    # Optional tiny training entry for quick plumbing checks.
    # Note: some agents ignore num_steps in additional_args and use cfg iters instead.
    if args.steps > 0 and hasattr(agent, "start_train"):
        agent.start_train({"offline": dataset}, additional_args={"num_steps": args.steps})


if __name__ == "__main__":
    main()
