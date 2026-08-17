"""Start the generic v2 reporter around an application-owned local env factory."""

from __future__ import annotations

import argparse
import importlib

import yaml

from agent_infra.socket_env.Report import EnvReporter


def _load_factory(target: str):
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory must be MODULE:CALLABLE")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("factory target is not callable")
    return factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    reporter_cfg = document.get("reporter", {})
    factory = _load_factory(str(document["local_env_factory"]))
    reporter = EnvReporter(
        factory(), bind_host=reporter_cfg.get("bind_host", "127.0.0.1"),
        port=int(reporter_cfg.get("port", 0)),
        request_cache_size=int(reporter_cfg.get("request_cache_size", 128)),
        max_message_bytes=int(reporter_cfg.get("max_message_bytes", 512 * 1024 * 1024)),
    )
    print({"address": reporter.address, "protocol_version": 2}, flush=True)
    try:
        while not reporter._closed:
            reporter.serve_once()
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
