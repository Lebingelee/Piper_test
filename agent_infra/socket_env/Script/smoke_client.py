"""Connect to a configured generic socket environment and print its raw reset schema."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import yaml

from agent_factory.env.env_factories import create_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="socket_env YAML configuration")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    socket_cfg = document.get("socket", document)
    if not isinstance(socket_cfg, dict):
        raise TypeError("socket configuration must be a mapping")
    cfg = SimpleNamespace(library="socket", flatten_obs_obj=["all"], flatten_action=True, **socket_cfg)
    env = create_env(cfg)
    try:
        observation, info = env.reset()
        print({"observation_space": env.observation_space, "action_space": env.action_space, "info": info})
    finally:
        env.close()


if __name__ == "__main__":
    main()
