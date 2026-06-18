"""Host-specific MCP configuration builders for AIKA."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_HOST = "claude-code"
DEFAULT_SERVER_NAME = "aika"
DEFAULT_TIMEOUT_MS = 600_000
DEFAULT_UV_CACHE_DIR = "/tmp/uv-cache"
SUPPORTED_HOSTS = {"claude-code"}
RESERVED_HOSTS = {"claude-desktop", "codex"}


class AikaMcpConfigError(RuntimeError):
    """Raised when AIKA cannot build a usable MCP host config."""


def build_mcp_config(
    host: str = DEFAULT_HOST,
    *,
    project_root: str | Path | None = None,
    uv_path: str | Path | None = None,
    uv_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the server JSON passed to host-specific MCP config commands."""
    normalized_host = _normalize_host(host)
    if normalized_host not in SUPPORTED_HOSTS:
        _raise_unsupported_host(normalized_host)

    root = _resolve_project_root(project_root)
    uv = _resolve_uv_path(uv_path)
    cache_dir = str(uv_cache_dir or os.getenv("UV_CACHE_DIR") or DEFAULT_UV_CACHE_DIR)
    return {
        "type": "stdio",
        "command": str(uv),
        "args": ["--directory", str(root), "run", DEFAULT_SERVER_NAME, "mcp"],
        "env": {
            "UV_CACHE_DIR": cache_dir,
        },
        "timeout": DEFAULT_TIMEOUT_MS,
    }


def _normalize_host(host: str | None) -> str:
    return str(host or DEFAULT_HOST).strip().lower()


def _resolve_project_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _resolve_uv_path(uv_path: str | Path | None) -> Path:
    if uv_path:
        return Path(uv_path).expanduser().resolve()
    discovered = shutil.which("uv")
    if not discovered:
        raise AikaMcpConfigError(
            "Could not find 'uv' on PATH. Install uv or run AIKA from an environment where uv is available."
        )
    return Path(discovered).expanduser().resolve()


def _raise_unsupported_host(host: str) -> None:
    if host in RESERVED_HOSTS:
        raise AikaMcpConfigError(
            f"MCP host '{host}' is reserved for a future AIKA release. "
            "Phase 3.1 currently supports --host claude-code."
        )
    raise AikaMcpConfigError(
        f"Unsupported MCP host '{host}'. Supported hosts: {', '.join(sorted(SUPPORTED_HOSTS))}."
    )
