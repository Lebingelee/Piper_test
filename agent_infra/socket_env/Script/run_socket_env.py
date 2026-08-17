"""Run v2 DESCRIBE → RESET → one sampled STEP → CLOSE from a YAML endpoint."""

from __future__ import annotations

import argparse
import json

import yaml

from agent_infra.socket_env import SocketEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="v2 socket client YAML")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        cfg = (yaml.safe_load(stream) or {}).get("socket", {})
    if int(cfg.get("protocol_version", 2)) != 2:
        raise ValueError("run_socket_env.py is intentionally v2-only")
    env = SocketEnv(
        cfg["host"], int(cfg["port"]),
        connect_timeout_s=float(cfg.get("connect_timeout_s", 10.0)),
        request_timeout_s=float(cfg.get("request_timeout_s", 10.0)),
        source_host=cfg.get("source_host"), source_port=cfg.get("source_port"),
        tcp_nodelay=bool(cfg.get("tcp_nodelay", True)), keepalive=bool(cfg.get("keepalive", False)),
    )
    try:
        env.connect()
        observation, info = env.reset()
        _, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
        print(json.dumps({"descriptor_hash": env.descriptor.descriptor_hash, "reset_info": info, "reward": reward, "terminated": terminated, "truncated": truncated, "step_info": step_info}, default=str))
    finally:
        env.close()


if __name__ == "__main__":
    main()
