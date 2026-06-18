"""Installer for registering AIKA with MCP host agents."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

from src.aika_mcp.host_configs import DEFAULT_HOST, DEFAULT_SERVER_NAME, build_mcp_config


RunCommand = Callable[[list[str]], subprocess.CompletedProcess[str]]
VALID_SCOPES = {"user", "project"}
COMMAND_TIMEOUT_SECONDS = 15
COMMAND_TIMEOUT_RETURN_CODE = 124


class AikaMcpInstallError(RuntimeError):
    """Raised when MCP host installation cannot continue."""


@dataclass(frozen=True)
class InstallResult:
    status: str
    exit_code: int
    config: dict[str, Any]
    commands: list[list[str]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def install_mcp_server(
    *,
    host: str = DEFAULT_HOST,
    scope: str = "user",
    force: bool = False,
    dry_run: bool = False,
    server_name: str = DEFAULT_SERVER_NAME,
    runner: RunCommand | None = None,
) -> InstallResult:
    """Install AIKA into a supported MCP host."""
    normalized_scope = _normalize_scope(scope)
    config = build_mcp_config(host)
    config_json = json.dumps(config, ensure_ascii=False, separators=(",", ":"))

    add_command = ["claude", "mcp", "add-json", server_name, config_json, "--scope", normalized_scope]
    remove_command = ["claude", "mcp", "remove", server_name, "--scope", normalized_scope]
    if dry_run:
        commands = [remove_command, add_command] if force else [add_command]
        return InstallResult(
            status="dry-run",
            exit_code=0,
            config=config,
            commands=commands,
            messages=[
                "Dry run only. No host configuration was changed.",
                f"Would register MCP server '{server_name}' for host '{host}' with scope '{normalized_scope}'.",
            ],
        )

    claude = shutil.which("claude")
    if not claude:
        raise AikaMcpInstallError(
            "Claude Code CLI was not found on PATH. Install Claude Code, or use 'aika mcp config --host claude-code' "
            "and copy the JSON manually."
        )

    run = runner or _run_command
    get_command = [claude, "mcp", "get", server_name]
    existing = run(get_command)
    if existing.returncode == COMMAND_TIMEOUT_RETURN_CODE:
        raise AikaMcpInstallError(_command_error("Timed out while checking existing AIKA MCP server", existing))
    if existing.returncode == 0 and not force:
        return InstallResult(
            status="conflict",
            exit_code=2,
            config=config,
            commands=[get_command],
            messages=[
                f"Claude Code already has an MCP server named '{server_name}'.",
                f"Re-run with --force to replace only the '{server_name}' server.",
            ],
        )

    commands: list[list[str]] = [get_command]
    if existing.returncode == 0 and force:
        scoped_remove = [claude, *remove_command[1:]]
        removed = run(scoped_remove)
        commands.append(scoped_remove)
        if removed.returncode != 0:
            raise AikaMcpInstallError(_command_error("Failed to remove existing AIKA MCP server", removed))

    scoped_add = [claude, *add_command[1:]]
    added = run(scoped_add)
    commands.append(scoped_add)
    if added.returncode != 0:
        raise AikaMcpInstallError(_command_error("Failed to add AIKA MCP server", added))

    return InstallResult(
        status="installed",
        exit_code=0,
        config=config,
        commands=commands,
        messages=[
            f"Registered AIKA MCP server '{server_name}' for Claude Code with scope '{normalized_scope}'.",
            "Open Claude Code and run /mcp to confirm the server is available.",
        ],
    )


def format_install_result(result: InstallResult) -> str:
    lines: list[str] = []
    lines.extend(result.messages)
    if result.status == "dry-run":
        lines.append("MCP server JSON:")
        lines.append(json.dumps(result.config, ensure_ascii=False, indent=2))
    if result.commands:
        lines.append("Commands:")
        lines.extend(f"  {format_command(command)}" for command in result.commands)
    return "\n".join(lines)


def format_command(command: list[str]) -> str:
    return " ".join(_shell_quote(part) for part in command)


def _normalize_scope(scope: str | None) -> str:
    normalized = str(scope or "user").strip().lower()
    if normalized not in VALID_SCOPES:
        raise AikaMcpInstallError("Unsupported scope. Use --scope user or --scope project.")
    return normalized


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            command,
            COMMAND_TIMEOUT_RETURN_CODE,
            "",
            f"command timed out after {COMMAND_TIMEOUT_SECONDS}s",
        )


def _command_error(prefix: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout or "").strip()
    if detail:
        return f"{prefix}: {detail}"
    return f"{prefix}: command exited with status {result.returncode}"


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
