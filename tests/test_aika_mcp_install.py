from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.aika_mcp import doctor, host_configs, installer
from src.aika_mcp.host_configs import AikaMcpConfigError, build_mcp_config
from src.aika_mcp.tools import tool_names


def completed(command: list[str], returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def mcp_tools_stdout() -> str:
    tools = [{"name": name} for name in tool_names()]
    return "\n".join(
        [
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}, ensure_ascii=False),
            json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"tools": tools}}, ensure_ascii=False),
        ]
    )


def assert_check(report: doctor.DoctorReport, name: str, status: str) -> doctor.DoctorCheck:
    matches = [check for check in report.checks if check.name == name]
    assert matches, f"missing doctor check: {name}"
    assert matches[0].status == status
    return matches[0]


def stable_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "type": "stdio",
        "command": "/usr/bin/uv",
        "args": ["--directory", str(tmp_path), "run", "aika", "mcp"],
        "env": {"UV_CACHE_DIR": "/tmp/uv-cache"},
        "timeout": 600000,
    }


def test_build_mcp_config_for_claude_code_is_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uv_path = tmp_path / "bin" / "uv"
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/uv-cache")
    monkeypatch.setattr(host_configs.shutil, "which", lambda name: str(uv_path) if name == "uv" else None)

    config = build_mcp_config(host="claude-code", project_root=tmp_path)

    assert config == {
        "type": "stdio",
        "command": str(uv_path.resolve()),
        "args": ["--directory", str(tmp_path.resolve()), "run", "aika", "mcp"],
        "env": {"UV_CACHE_DIR": "/tmp/uv-cache"},
        "timeout": 600000,
    }


def test_build_mcp_config_reports_missing_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_configs.shutil, "which", lambda name: None)

    with pytest.raises(AikaMcpConfigError, match="uv"):
        build_mcp_config(host="claude-code")


def test_build_mcp_config_reports_reserved_hosts() -> None:
    with pytest.raises(AikaMcpConfigError, match="future AIKA release"):
        build_mcp_config(host="codex", uv_path="/usr/bin/uv")


def test_install_refuses_existing_config_without_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(installer, "build_mcp_config", lambda host: stable_config(tmp_path))
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/bin/claude" if name == "claude" else None)

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return completed(command, 0, stdout="aika configured")

    result = installer.install_mcp_server(host="claude-code", scope="user", runner=run)

    assert result.status == "conflict"
    assert result.exit_code == 2
    assert calls == [["/bin/claude", "mcp", "get", "aika"]]
    assert "--force" in "\n".join(result.messages)


def test_install_force_removes_then_adds_existing_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(installer, "build_mcp_config", lambda host: stable_config(tmp_path))
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/bin/claude" if name == "claude" else None)

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return completed(command, 0)

    result = installer.install_mcp_server(host="claude-code", scope="project", force=True, runner=run)

    assert result.status == "installed"
    assert calls[0] == ["/bin/claude", "mcp", "get", "aika"]
    assert calls[1] == ["/bin/claude", "mcp", "remove", "aika", "--scope", "project"]
    assert calls[2][:4] == ["/bin/claude", "mcp", "add-json", "aika"]
    assert calls[2][-2:] == ["--scope", "project"]
    assert json.loads(calls[2][4])["args"][-1] == "mcp"


def test_install_dry_run_does_not_require_or_call_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "build_mcp_config", lambda host: stable_config(tmp_path))
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)

    def fail_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"dry-run should not call host CLI: {command}")

    result = installer.install_mcp_server(host="claude-code", scope="user", dry_run=True, runner=fail_run)

    assert result.status == "dry-run"
    assert result.exit_code == 0
    assert result.commands[0][:3] == ["claude", "mcp", "add-json"]


def test_install_treats_claude_conflict_check_timeout_as_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(installer, "build_mcp_config", lambda host: stable_config(tmp_path))
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/bin/claude" if name == "claude" else None)

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return completed(command, installer.COMMAND_TIMEOUT_RETURN_CODE, stderr="command timed out")

    with pytest.raises(installer.AikaMcpInstallError, match="Timed out"):
        installer.install_mcp_server(host="claude-code", scope="user", runner=run)


def test_doctor_warns_when_claude_code_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "build_mcp_config", lambda host: stable_config(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        doctor,
        "inspect_sqlite_index",
        lambda path: {"exists": True, "path": str(path), "metadata": {}, "counts": {"claims": 1}, "error": ""},
    )
    monkeypatch.setattr(doctor.subprocess, "run", lambda command, **kwargs: completed(command, 0, stdout=mcp_tools_stdout()))

    report = doctor.run_mcp_doctor(home=tmp_path)

    assert report.exit_code == 1
    assert_check(report, "mcp_server", doctor.STATUS_PASS)
    assert_check(report, "sqlite_index", doctor.STATUS_PASS)
    assert "config --host claude-code" in assert_check(report, "claude_code", doctor.STATUS_WARN).fix


def test_doctor_fails_when_sqlite_index_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "build_mcp_config", lambda host: stable_config(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        doctor,
        "inspect_sqlite_index",
        lambda path: {"exists": False, "path": str(path), "metadata": {}, "counts": {}, "error": ""},
    )
    monkeypatch.setattr(doctor.subprocess, "run", lambda command, **kwargs: completed(command, 0, stdout=mcp_tools_stdout()))

    report = doctor.run_mcp_doctor(home=tmp_path)

    assert report.exit_code == 2
    check = assert_check(report, "sqlite_index", doctor.STATUS_FAIL)
    assert "build-index" in check.fix


def test_doctor_fails_when_mcp_server_cannot_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "build_mcp_config", lambda host: stable_config(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        doctor,
        "inspect_sqlite_index",
        lambda path: {"exists": True, "path": str(path), "metadata": {}, "counts": {"claims": 1}, "error": ""},
    )
    monkeypatch.setattr(doctor.subprocess, "run", lambda command, **kwargs: completed(command, 1, stderr="boom"))

    report = doctor.run_mcp_doctor(home=tmp_path)

    assert report.exit_code == 2
    assert "boom" in assert_check(report, "mcp_server", doctor.STATUS_FAIL).detail


def test_doctor_warns_when_claude_code_has_no_aika_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "build_mcp_config", lambda host: stable_config(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(
        doctor,
        "inspect_sqlite_index",
        lambda path: {"exists": True, "path": str(path), "metadata": {}, "counts": {"claims": 1}, "error": ""},
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["/bin/claude", "mcp", "get"]:
            return completed(command, 1, stderr="not found")
        return completed(command, 0, stdout=mcp_tools_stdout())

    monkeypatch.setattr(doctor.subprocess, "run", run)

    report = doctor.run_mcp_doctor(home=tmp_path)

    assert report.exit_code == 1
    check = assert_check(report, "claude_code", doctor.STATUS_WARN)
    assert "aika mcp install --host claude-code --scope user" in check.fix


def test_doctor_warns_when_claude_code_lookup_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "build_mcp_config", lambda host: stable_config(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(
        doctor,
        "inspect_sqlite_index",
        lambda path: {"exists": True, "path": str(path), "metadata": {}, "counts": {"claims": 1}, "error": ""},
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["/bin/claude", "mcp", "get"]:
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout") or 0)
        return completed(command, 0, stdout=mcp_tools_stdout())

    monkeypatch.setattr(doctor.subprocess, "run", run)

    report = doctor.run_mcp_doctor(home=tmp_path, timeout_seconds=0.01)

    assert report.exit_code == 1
    assert "timed out" in assert_check(report, "claude_code", doctor.STATUS_WARN).detail
