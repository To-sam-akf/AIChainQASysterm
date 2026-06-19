"""Host-specific MCP configuration builders for AIKA."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


DEFAULT_HOST = "claude-code"
DEFAULT_SERVER_NAME = "aika"
DEFAULT_TIMEOUT_MS = 600_000
DEFAULT_CODEX_STARTUP_TIMEOUT_SEC = 30
DEFAULT_CODEX_TOOL_TIMEOUT_SEC = 600
DEFAULT_UV_CACHE_DIR = "/tmp/uv-cache"
SUPPORTED_HOSTS = {"claude-code", "codex"}
RESERVED_HOSTS = {"claude-desktop"}


class AikaMcpConfigError(RuntimeError):
    """Raised when AIKA cannot build a usable MCP host config."""


def build_mcp_config(
    host: str = DEFAULT_HOST,
    *,
    aika_path: str | Path | None = None,
    project_root: str | Path | None = None,
    uv_path: str | Path | None = None,
    uv_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the server config passed to host-specific MCP config commands."""
    normalized_host = _normalize_host(host)
    if normalized_host not in SUPPORTED_HOSTS:
        _raise_unsupported_host(normalized_host)

    launch = _build_stdio_launch(
        aika_path=aika_path,
        project_root=project_root,
        uv_path=uv_path,
        uv_cache_dir=uv_cache_dir,
    )
    if normalized_host == "codex":
        return {
            **launch,
            "startup_timeout_sec": DEFAULT_CODEX_STARTUP_TIMEOUT_SEC,
            "tool_timeout_sec": DEFAULT_CODEX_TOOL_TIMEOUT_SEC,
        }
    return {
        "type": "stdio",
        **launch,
        "timeout": DEFAULT_TIMEOUT_MS,
    }


def format_mcp_config(
    host: str = DEFAULT_HOST,
    *,
    server_name: str = DEFAULT_SERVER_NAME,
    aika_path: str | Path | None = None,
    project_root: str | Path | None = None,
    uv_path: str | Path | None = None,
    uv_cache_dir: str | Path | None = None,
) -> str:
    """Format the generated MCP config in the host's native config syntax."""
    normalized_host = _normalize_host(host)
    config = build_mcp_config(
        host=normalized_host,
        aika_path=aika_path,
        project_root=project_root,
        uv_path=uv_path,
        uv_cache_dir=uv_cache_dir,
    )
    if normalized_host == "codex":
        return format_codex_server_toml(config, server_name=server_name)
    return json.dumps(config, ensure_ascii=False, indent=2) + "\n"


def format_codex_server_toml(config: dict[str, Any], *, server_name: str = DEFAULT_SERVER_NAME) -> str:
    """Format one Codex stdio MCP server table as TOML."""
    lines = [f"[mcp_servers.{_toml_key(server_name)}]"]
    lines.append(f"command = {_toml_string(str(config['command']))}")
    lines.append("args = [" + ", ".join(_toml_string(str(arg)) for arg in config.get("args", [])) + "]")
    lines.append(f"startup_timeout_sec = {int(config.get('startup_timeout_sec', DEFAULT_CODEX_STARTUP_TIMEOUT_SEC))}")
    lines.append(f"tool_timeout_sec = {int(config.get('tool_timeout_sec', DEFAULT_CODEX_TOOL_TIMEOUT_SEC))}")
    env = {str(key): str(value) for key, value in dict(config.get("env") or {}).items()}
    if env:
        lines.append("")
        lines.append(f"[mcp_servers.{_toml_key(server_name)}.env]")
        for key in sorted(env):
            lines.append(f"{_toml_key(key)} = {_toml_string(env[key])}")
    return "\n".join(lines) + "\n"


def _build_stdio_launch(
    *,
    aika_path: str | Path | None = None,
    project_root: str | Path | None = None,
    uv_path: str | Path | None = None,
    uv_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    aika = _resolve_aika_path(aika_path) if (aika_path or (project_root is None and uv_path is None)) else None
    if aika is not None:
        return {
            "command": str(aika),
            "args": ["mcp"],
            "env": {},
        }

    root = _resolve_project_root(project_root)
    uv = _resolve_uv_path(uv_path)
    cache_dir = str(uv_cache_dir or os.getenv("UV_CACHE_DIR") or DEFAULT_UV_CACHE_DIR)
    return {
        "command": str(uv),
        "args": ["--directory", str(root), "run", DEFAULT_SERVER_NAME, "mcp"],
        "env": {
            "UV_CACHE_DIR": cache_dir,
        },
    }


def _normalize_host(host: str | None) -> str:
    return str(host or DEFAULT_HOST).strip().lower()


def _resolve_project_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _resolve_aika_path(aika_path: str | Path | None) -> Path | None:
    if aika_path:
        return Path(aika_path).expanduser().resolve()
    current = Path(sys.argv[0])
    if current.name == DEFAULT_SERVER_NAME and current.exists():
        return current.expanduser().resolve()
    discovered = shutil.which(DEFAULT_SERVER_NAME)
    if discovered:
        return Path(discovered).expanduser().resolve()
    return None


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
            "Current automatic installation supports --host claude-code and --host codex."
        )
    raise AikaMcpConfigError(
        f"Unsupported MCP host '{host}'. Supported hosts: {', '.join(sorted(SUPPORTED_HOSTS))}."
    )


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_key(value: str) -> str:
    if value and all(char.isalnum() or char in "_-" for char in value):
        return value
    return _toml_string(value)
