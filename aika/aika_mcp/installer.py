"""Installer for registering AIKA with MCP host agents."""

from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from aika.aika_mcp.host_configs import (
    DEFAULT_HOST,
    DEFAULT_SERVER_NAME,
    build_mcp_config,
    format_codex_server_toml,
)


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
    config_text: str = ""


def install_mcp_server(
    *,
    host: str = DEFAULT_HOST,
    scope: str = "user",
    force: bool = False,
    dry_run: bool = False,
    server_name: str = DEFAULT_SERVER_NAME,
    runner: RunCommand | None = None,
    codex_user_config_path: str | Path | None = None,
    codex_project_root: str | Path | None = None,
) -> InstallResult:
    """Install AIKA into a supported MCP host."""
    normalized_host = _normalize_host(host)
    normalized_scope = _normalize_scope(scope)
    if normalized_host == "claude-code":
        return _install_claude_code(
            host=normalized_host,
            scope=normalized_scope,
            force=force,
            dry_run=dry_run,
            server_name=server_name,
            runner=runner,
        )
    if normalized_host == "codex":
        return _install_codex(
            host=normalized_host,
            scope=normalized_scope,
            force=force,
            dry_run=dry_run,
            server_name=server_name,
            runner=runner,
            user_config_path=codex_user_config_path,
            project_root=codex_project_root,
        )
    raise AikaMcpInstallError("Unsupported MCP host. Use --host claude-code or --host codex.")


def _install_claude_code(
    *,
    host: str,
    scope: str,
    force: bool,
    dry_run: bool,
    server_name: str,
    runner: RunCommand | None,
) -> InstallResult:
    config = build_mcp_config(host)
    config_json = json.dumps(config, ensure_ascii=False, separators=(",", ":"))

    add_command = ["claude", "mcp", "add-json", server_name, config_json, "--scope", scope]
    remove_command = ["claude", "mcp", "remove", server_name, "--scope", scope]
    if dry_run:
        commands = [remove_command, add_command] if force else [add_command]
        return InstallResult(
            status="dry-run",
            exit_code=0,
            config=config,
            commands=commands,
            messages=[
                "Dry run only. No host configuration was changed.",
                f"Would register MCP server '{server_name}' for host '{host}' with scope '{scope}'.",
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
            f"Registered AIKA MCP server '{server_name}' for Claude Code with scope '{scope}'.",
            "Open Claude Code and run /mcp to confirm the server is available.",
        ],
    )


def _install_codex(
    *,
    host: str,
    scope: str,
    force: bool,
    dry_run: bool,
    server_name: str,
    runner: RunCommand | None,
    user_config_path: str | Path | None,
    project_root: str | Path | None,
) -> InstallResult:
    config = build_mcp_config(host)
    config_text = format_codex_server_toml(config, server_name=server_name)
    if scope == "project":
        return _install_codex_project(
            host=host,
            config=config,
            config_text=config_text,
            force=force,
            dry_run=dry_run,
            server_name=server_name,
            project_root=project_root,
        )
    return _install_codex_user(
        host=host,
        config=config,
        config_text=config_text,
        force=force,
        dry_run=dry_run,
        server_name=server_name,
        runner=runner,
        user_config_path=user_config_path,
    )


def _install_codex_user(
    *,
    host: str,
    config: dict[str, Any],
    config_text: str,
    force: bool,
    dry_run: bool,
    server_name: str,
    runner: RunCommand | None,
    user_config_path: str | Path | None,
) -> InstallResult:
    add_command = _codex_add_command("codex", config, server_name=server_name)
    remove_command = ["codex", "mcp", "remove", server_name]
    if dry_run:
        commands = [remove_command, add_command] if force else [add_command]
        return InstallResult(
            status="dry-run",
            exit_code=0,
            config=config,
            commands=commands,
            messages=[
                "Dry run only. No host configuration was changed.",
                f"Would register MCP server '{server_name}' for host '{host}' with scope 'user'.",
                "The TOML below is the Codex config block that would be present after installation.",
            ],
            config_text=config_text,
        )

    codex = shutil.which("codex")
    if not codex:
        raise AikaMcpInstallError(
            "Codex CLI was not found on PATH. Install Codex, or use 'aika mcp config --host codex' "
            "and copy the TOML into ~/.codex/config.toml manually."
        )

    run = runner or _run_command
    get_command = [codex, "mcp", "get", server_name, "--json"]
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
                f"Codex already has an MCP server named '{server_name}'.",
                f"Re-run with --force to replace only the '{server_name}' server.",
            ],
            config_text=config_text,
        )

    commands: list[list[str]] = [get_command]
    if existing.returncode == 0 and force:
        resolved_remove = [codex, *remove_command[1:]]
        removed = run(resolved_remove)
        commands.append(resolved_remove)
        if removed.returncode != 0:
            raise AikaMcpInstallError(_command_error("Failed to remove existing AIKA MCP server", removed))

    resolved_add = _codex_add_command(codex, config, server_name=server_name)
    added = run(resolved_add)
    commands.append(resolved_add)
    if added.returncode != 0:
        raise AikaMcpInstallError(_command_error("Failed to add AIKA MCP server", added))

    config_path = _codex_user_config_path(user_config_path)
    write_codex_server_config(config_path, config, server_name=server_name)
    return InstallResult(
        status="installed",
        exit_code=0,
        config=config,
        commands=commands,
        messages=[
            f"Registered AIKA MCP server '{server_name}' for Codex with scope 'user'.",
            f"Updated Codex config block with AIKA timeouts: {config_path}",
            "Open Codex and run /mcp to confirm the server is available.",
        ],
        config_text=config_text,
    )


def _install_codex_project(
    *,
    host: str,
    config: dict[str, Any],
    config_text: str,
    force: bool,
    dry_run: bool,
    server_name: str,
    project_root: str | Path | None,
) -> InstallResult:
    config_path = _codex_project_config_path(project_root)
    existing = _codex_server_exists(config_path, server_name=server_name)
    if existing and not force:
        return InstallResult(
            status="conflict",
            exit_code=2,
            config=config,
            messages=[
                f"Project Codex config already has an MCP server named '{server_name}': {config_path}",
                f"Re-run with --force to replace only the '{server_name}' server.",
            ],
            config_text=config_text,
        )
    if dry_run:
        return InstallResult(
            status="dry-run",
            exit_code=0,
            config=config,
            messages=[
                "Dry run only. No host configuration was changed.",
                f"Would write project Codex MCP server '{server_name}' to {config_path}.",
                "Codex loads project config only after the project is trusted.",
            ],
            config_text=config_text,
        )

    write_codex_server_config(config_path, config, server_name=server_name)
    return InstallResult(
        status="installed",
        exit_code=0,
        config=config,
        messages=[
            f"Registered AIKA MCP server '{server_name}' in project Codex config: {config_path}",
            "Codex loads project config only after the project is trusted.",
            "Open Codex in this project and run /mcp to confirm the server is available.",
        ],
        config_text=config_text,
    )


def format_install_result(result: InstallResult) -> str:
    lines: list[str] = []
    lines.extend(result.messages)
    if result.status == "dry-run":
        if result.config_text:
            lines.append("MCP server TOML:")
            lines.append(result.config_text.rstrip())
        else:
            lines.append("MCP server JSON:")
            lines.append(json.dumps(result.config, ensure_ascii=False, indent=2))
    if result.commands:
        lines.append("Commands:")
        lines.extend(f"  {format_command(command)}" for command in result.commands)
    return "\n".join(lines)


def format_command(command: list[str]) -> str:
    return " ".join(_shell_quote(part) for part in command)


def write_codex_server_config(path: str | Path, config: dict[str, Any], *, server_name: str = DEFAULT_SERVER_NAME) -> None:
    target = Path(path).expanduser().resolve()
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    if existing.strip():
        _parse_toml(existing, source=str(target))
    updated = _replace_codex_server_block(
        existing,
        format_codex_server_toml(config, server_name=server_name),
        server_name=server_name,
    )
    _parse_toml(updated, source=str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(updated, encoding="utf-8")


def _normalize_host(host: str | None) -> str:
    return str(host or DEFAULT_HOST).strip().lower()


def _normalize_scope(scope: str | None) -> str:
    normalized = str(scope or "user").strip().lower()
    if normalized not in VALID_SCOPES:
        raise AikaMcpInstallError("Unsupported scope. Use --scope user or --scope project.")
    return normalized


def _codex_add_command(codex: str, config: dict[str, Any], *, server_name: str) -> list[str]:
    command = [codex, "mcp", "add", server_name]
    env = {str(key): str(value) for key, value in dict(config.get("env") or {}).items()}
    for key in sorted(env):
        command.extend(["--env", f"{key}={env[key]}"])
    command.append("--")
    command.append(str(config["command"]))
    command.extend(str(arg) for arg in config.get("args", []))
    return command


def _codex_user_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    return (Path.home() / ".codex" / "config.toml").expanduser().resolve()


def _codex_project_config_path(project_root: str | Path | None = None) -> Path:
    root = Path(project_root).expanduser().resolve() if project_root is not None else Path.cwd().resolve()
    return root / ".codex" / "config.toml"


def _codex_server_exists(path: Path, *, server_name: str) -> bool:
    if not path.exists():
        return False
    data = _load_toml_file(path)
    servers = data.get("mcp_servers", {})
    return isinstance(servers, dict) and server_name in servers


def _replace_codex_server_block(existing: str, new_block: str, *, server_name: str) -> str:
    kept: list[str] = []
    skip = False
    for line in existing.splitlines():
        table_name = _toml_table_name(line)
        if table_name is not None:
            skip = _is_codex_server_table(table_name, server_name=server_name)
        if not skip:
            kept.append(line)

    prefix = "\n".join(kept).rstrip()
    block = new_block.rstrip()
    if prefix:
        return f"{prefix}\n\n{block}\n"
    return f"{block}\n"


def _is_codex_server_table(table_name: str, *, server_name: str) -> bool:
    prefix = f"mcp_servers.{server_name}"
    return table_name == prefix or table_name.startswith(f"{prefix}.")


def _toml_table_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("[") or stripped.startswith("[["):
        return None
    close_index = stripped.find("]")
    if close_index <= 1:
        return None
    return stripped[1:close_index].strip()


def _load_toml_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise AikaMcpInstallError(f"Codex config is not valid TOML: {path}: {exc}") from exc
    if not isinstance(data, dict):
        return {}
    return data


def _parse_toml(text: str, *, source: str) -> dict[str, Any]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise AikaMcpInstallError(f"Codex config is not valid TOML: {source}: {exc}") from exc
    if not isinstance(data, dict):
        return {}
    return data


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
