"""End-to-end diagnostics for AIKA MCP installation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aika.aika_core.backends.sqlite_backend import (
    DEFAULT_PROFILE,
    inspect_sqlite_index,
    profile_index_path,
    resolve_aika_home,
)
from aika.aika_mcp.host_configs import (
    DEFAULT_HOST,
    DEFAULT_SERVER_NAME,
    AikaMcpConfigError,
    build_mcp_config,
)
from aika.aika_mcp.tools import tool_names


STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass(frozen=True)
class DoctorReport:
    checks: list[DoctorCheck]

    @property
    def exit_code(self) -> int:
        if any(check.status == STATUS_FAIL for check in self.checks):
            return 2
        if any(check.status == STATUS_WARN for check in self.checks):
            return 1
        return 0


def run_mcp_doctor(
    *,
    host: str = DEFAULT_HOST,
    scope: str = "user",
    home: str | Path | None = None,
    profile: str = DEFAULT_PROFILE,
    server_name: str = DEFAULT_SERVER_NAME,
    timeout_seconds: float = 8.0,
) -> DoctorReport:
    checks: list[DoctorCheck] = []
    config: dict[str, Any] | None = None

    try:
        config = build_mcp_config(host)
        checks.append(DoctorCheck("mcp_command", STATUS_PASS, f"Using MCP command: {config['command']}"))
    except AikaMcpConfigError as exc:
        checks.append(
            DoctorCheck(
                "mcp_command",
                STATUS_FAIL,
                str(exc),
                "Install aika-research-mcp or run AIKA from a source environment where uv is on PATH.",
            )
        )

    if config is not None:
        checks.append(_check_mcp_server(config, timeout_seconds=timeout_seconds))
    else:
        checks.append(
            DoctorCheck(
                "mcp_server",
                STATUS_FAIL,
                "Skipped because MCP command configuration could not be generated.",
                "Fix the uv/config error above, then retry aika mcp doctor.",
            )
        )

    checks.append(_check_sqlite_index(home=home, profile=profile))
    checks.append(_check_host_config(host=host, server_name=server_name, timeout_seconds=timeout_seconds))
    checks.extend(_check_skill_install(host=host, scope=scope, server_name=server_name, timeout_seconds=timeout_seconds))
    return DoctorReport(checks)


def format_doctor_report(report: DoctorReport) -> str:
    lines: list[str] = []
    for check in report.checks:
        lines.append(f"[{check.status.upper()}] {check.name}: {check.detail}")
        if check.fix:
            lines.append(f"  Fix: {check.fix}")
    return "\n".join(lines)


def _check_mcp_server(config: dict[str, Any], *, timeout_seconds: float) -> DoctorCheck:
    command = [str(config["command"]), *[str(arg) for arg in config.get("args", [])]]
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "aika-doctor", "version": "0.1.0"},
            },
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    input_text = "\n".join(json.dumps(message, ensure_ascii=False) for message in messages) + "\n"
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in dict(config.get("env") or {}).items()})
    try:
        result = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
        )
    except FileNotFoundError as exc:
        return DoctorCheck(
            "mcp_server",
            STATUS_FAIL,
            f"Command not found: {exc.filename or command[0]}",
            "Run aika mcp config --host claude-code or aika mcp config --host codex and verify the command path exists.",
        )
    except subprocess.TimeoutExpired:
        return DoctorCheck(
            "mcp_server",
            STATUS_FAIL,
            f"Timed out after {timeout_seconds:g}s while listing MCP tools.",
            "Run aika mcp config for your host, then try the generated command manually.",
        )

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or f"exit status {result.returncode}"
        return DoctorCheck(
            "mcp_server",
            STATUS_FAIL,
            f"Could not start AIKA MCP server: {detail}",
            "Run UV_CACHE_DIR=/tmp/uv-cache uv run aika mcp config --host claude-code or --host codex, then retry aika mcp doctor.",
        )

    response = _jsonrpc_response(result.stdout, request_id=2)
    if not response:
        return DoctorCheck(
            "mcp_server",
            STATUS_FAIL,
            "AIKA MCP server started but did not return a tools/list response.",
            "Retry with AIKA_MCP_TRACE=/tmp/aika-mcp.trace aika mcp doctor for protocol details.",
        )
    if "error" in response:
        return DoctorCheck(
            "mcp_server",
            STATUS_FAIL,
            f"tools/list returned error: {response['error']}",
            "Run the generated command from aika mcp config manually and inspect stderr.",
        )

    tools = response.get("result", {}).get("tools", [])
    names = {str(item.get("name") or "") for item in tools if isinstance(item, dict)}
    missing = set(tool_names()) - names
    if missing:
        return DoctorCheck(
            "mcp_server",
            STATUS_FAIL,
            f"Listed {len(names)} tools but missing: {', '.join(sorted(missing))}",
            "Re-run tests/test_aika_mcp_tools.py to verify MCP tool registration.",
        )
    return DoctorCheck("mcp_server", STATUS_PASS, f"Started and listed {len(names)} MCP tools.")


def _check_sqlite_index(*, home: str | Path | None, profile: str) -> DoctorCheck:
    resolved_home = resolve_aika_home(home)
    index_path = profile_index_path(resolved_home, profile=profile)
    index = inspect_sqlite_index(index_path)
    if not index["exists"]:
        return DoctorCheck(
            "sqlite_index",
            STATUS_FAIL,
            f"SQLite index not found: {index_path}",
            "Run aika init --sample && aika build-index, or pass --home/AIKA_HOME for an initialized AIKA home.",
        )
    if index["error"]:
        return DoctorCheck(
            "sqlite_index",
            STATUS_FAIL,
            f"SQLite index is not queryable: {index['error']}",
            "Rebuild the index with aika build-index.",
        )
    counts = index.get("counts") or {}
    if counts:
        count_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        return DoctorCheck("sqlite_index", STATUS_PASS, f"{index_path} ({count_text})")
    return DoctorCheck("sqlite_index", STATUS_PASS, str(index_path))


def _check_host_config(*, host: str, server_name: str, timeout_seconds: float) -> DoctorCheck:
    normalized = str(host or DEFAULT_HOST).strip().lower()
    if normalized == "claude-code":
        return _check_claude_code(server_name=server_name, timeout_seconds=timeout_seconds)
    if normalized == "codex":
        return _check_codex(server_name=server_name, timeout_seconds=timeout_seconds)
    return DoctorCheck(
        "host_config",
        STATUS_WARN,
        f"Host '{host}' is not implemented for automatic diagnostics.",
        "Use --host claude-code or --host codex for automatic installation checks.",
    )


def _check_claude_code(*, server_name: str, timeout_seconds: float) -> DoctorCheck:
    claude = shutil.which("claude")
    if not claude:
        return DoctorCheck(
            "claude_code",
            STATUS_WARN,
            "Claude Code CLI not found on PATH.",
            "Install Claude Code CLI, or copy the output of aika mcp config --host claude-code manually.",
        )

    try:
        result = subprocess.run(
            [claude, "mcp", "get", server_name],
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return DoctorCheck(
            "claude_code",
            STATUS_WARN,
            f"Claude Code MCP lookup timed out after {timeout_seconds:g}s.",
            "Run claude mcp get aika manually, or re-run aika mcp install --host claude-code --scope user.",
        )
    if result.returncode == 0:
        return DoctorCheck("claude_code", STATUS_PASS, f"Claude Code has MCP server '{server_name}' configured.")
    return DoctorCheck(
        "claude_code",
        STATUS_WARN,
        f"Claude Code does not have MCP server '{server_name}' configured.",
        "Run aika mcp install --host claude-code --scope user.",
    )


def _check_codex(*, server_name: str, timeout_seconds: float) -> DoctorCheck:
    codex = shutil.which("codex")
    if not codex:
        return DoctorCheck(
            "codex",
            STATUS_WARN,
            "Codex CLI not found on PATH.",
            "Install Codex, or copy the output of aika mcp config --host codex into ~/.codex/config.toml.",
        )

    try:
        result = subprocess.run(
            [codex, "mcp", "get", server_name, "--json"],
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return DoctorCheck(
            "codex",
            STATUS_WARN,
            f"Codex MCP lookup timed out after {timeout_seconds:g}s.",
            "Run codex mcp get aika --json manually, or re-run aika mcp install --host codex --scope user.",
        )
    if result.returncode == 0:
        return DoctorCheck("codex", STATUS_PASS, f"Codex has MCP server '{server_name}' configured.")
    return DoctorCheck(
        "codex",
        STATUS_WARN,
        f"Codex does not have MCP server '{server_name}' configured.",
        "Run aika mcp install --host codex --scope user. For project scope, run it from the project and trust the project in Codex.",
    )


def _check_skill_install(*, host: str, scope: str, server_name: str, timeout_seconds: float) -> list[DoctorCheck]:
    from aika.aika_skills import AikaSkillError, run_skill_doctor

    try:
        report = run_skill_doctor(host=host, scope=scope, server_name=server_name, timeout_seconds=timeout_seconds)
    except AikaSkillError as exc:
        return [
            DoctorCheck(
                "skill",
                STATUS_FAIL,
                str(exc),
                f"Run aika skill install --host {host} --scope {scope}.",
            )
        ]
    checks: list[DoctorCheck] = []
    for check in report.checks:
        status = check.status
        if status == STATUS_FAIL and check.name != "bundled_skill":
            status = STATUS_WARN
        checks.append(DoctorCheck(f"skill_{check.name}", status, check.detail, check.fix))
    return checks


def _jsonrpc_response(stdout: str, *, request_id: int) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("id") == request_id:
            return payload
    return None
