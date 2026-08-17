"""Typed, auditable v2 client/reporter configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ClientConfig:
    host: str
    port: int
    protocol_version: int = 2
    connect_timeout_s: float = 10.0
    request_timeout_s: float = 10.0
    source_host: Optional[str] = None
    source_port: Optional[int] = None
    tcp_nodelay: bool = True
    keepalive: bool = False


@dataclass(frozen=True)
class ReporterConfig:
    bind_host: str = "127.0.0.1"
    port: int = 0
    protocol_version: int = 2
    request_timeout_s: float = 10.0
    max_message_bytes: int = 512 * 1024 * 1024
    backlog: int = 1
    request_cache_size: int = 128
