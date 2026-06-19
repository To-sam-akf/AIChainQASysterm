from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

from aika.aika_mcp import doctor, host_configs, installer
from aika.aika_mcp.host_configs import AikaMcpConfigError, build_mcp_config, format_mcp_config
from aika.aika_mcp.tools import tool_names
from aika import aika_cli
from aika.aika_skills import SkillInstallResult


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


def stable_codex_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "command": "/usr/bin/uv",
        "args": ["--directory", str(tmp_path), "run", "aika", "mcp"],
        "env": {"UV_CACHE_DIR": "/tmp/uv-cache"},
        "startup_timeout_sec": 30,
        "tool_timeout_sec": 600,
    }


def test_build_mcp_config_prefers_installed_aika(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    aika_path = tmp_path / "bin" / "aika"
    monkeypatch.setattr(host_configs.shutil, "which", lambda name: str(aika_path) if name == "aika" else None)

    config = build_mcp_config(host="claude-code")

    assert config == {
        "type": "stdio",
        "command": str(aika_path.resolve()),
        "args": ["mcp"],
        "env": {},
        "timeout": 600000,
    }


def test_build_mcp_config_uses_current_aika_executable_when_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    aika_path = tmp_path / "bin" / "aika"
    aika_path.parent.mkdir(parents=True)
    aika_path.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(host_configs.shutil, "which", lambda name: None)
    monkeypatch.setattr(host_configs.sys, "argv", [str(aika_path), "mcp", "config"])

    config = build_mcp_config(host="claude-code")

    assert config["command"] == str(aika_path.resolve())
    assert config["args"] == ["mcp"]
    assert config["env"] == {}


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
        build_mcp_config(host="claude-desktop", uv_path="/usr/bin/uv")


def test_build_mcp_config_for_codex_is_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uv_path = tmp_path / "bin" / "uv"
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/uv-cache")
    monkeypatch.setattr(host_configs.shutil, "which", lambda name: str(uv_path) if name == "uv" else None)

    config = build_mcp_config(host="codex", project_root=tmp_path)

    assert config == {
        "command": str(uv_path.resolve()),
        "args": ["--directory", str(tmp_path.resolve()), "run", "aika", "mcp"],
        "env": {"UV_CACHE_DIR": "/tmp/uv-cache"},
        "startup_timeout_sec": 30,
        "tool_timeout_sec": 600,
    }


def test_format_mcp_config_for_codex_is_valid_toml(tmp_path: Path) -> None:
    text = format_mcp_config(host="codex", project_root=tmp_path, uv_path="/usr/bin/uv")
    parsed = tomllib.loads(text)

    server = parsed["mcp_servers"]["aika"]
    assert server["command"] == "/usr/bin/uv"
    assert server["args"] == ["--directory", str(tmp_path.resolve()), "run", "aika", "mcp"]
    assert server["startup_timeout_sec"] == 30
    assert server["tool_timeout_sec"] == 600
    assert server["env"]["UV_CACHE_DIR"] == "/tmp/uv-cache"


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


def test_install_codex_dry_run_does_not_require_or_call_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(installer, "build_mcp_config", lambda host: stable_codex_config(tmp_path))
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)

    def fail_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"dry-run should not call host CLI: {command}")

    result = installer.install_mcp_server(host="codex", scope="user", dry_run=True, runner=fail_run)

    assert result.status == "dry-run"
    assert result.exit_code == 0
    assert result.commands[0][:4] == ["codex", "mcp", "add", "aika"]
    assert "--env" in result.commands[0]
    assert "[mcp_servers.aika]" in result.config_text


def test_install_codex_refuses_existing_config_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(installer, "build_mcp_config", lambda host: stable_codex_config(tmp_path))
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/bin/codex" if name == "codex" else None)

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return completed(command, 0, stdout=json.dumps({"name": "aika"}))

    result = installer.install_mcp_server(host="codex", scope="user", runner=run)

    assert result.status == "conflict"
    assert result.exit_code == 2
    assert calls == [["/bin/codex", "mcp", "get", "aika", "--json"]]
    assert "--force" in "\n".join(result.messages)


def test_install_codex_force_removes_then_adds_existing_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(installer, "build_mcp_config", lambda host: stable_codex_config(tmp_path))
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/bin/codex" if name == "codex" else None)

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return completed(command, 0)

    result = installer.install_mcp_server(
        host="codex",
        scope="user",
        force=True,
        runner=run,
        codex_user_config_path=config_path,
    )

    assert result.status == "installed"
    assert calls[0] == ["/bin/codex", "mcp", "get", "aika", "--json"]
    assert calls[1] == ["/bin/codex", "mcp", "remove", "aika"]
    assert calls[2] == [
        "/bin/codex",
        "mcp",
        "add",
        "aika",
        "--env",
        "UV_CACHE_DIR=/tmp/uv-cache",
        "--",
        "/usr/bin/uv",
        "--directory",
        str(tmp_path),
        "run",
        "aika",
        "mcp",
    ]
    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["mcp_servers"]["aika"]["startup_timeout_sec"] == 30
    assert parsed["mcp_servers"]["aika"]["tool_timeout_sec"] == 600


def test_install_codex_project_replaces_only_aika_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        "\n".join(
            [
                'model = "gpt-5"',
                "",
                "[mcp_servers.other]",
                'command = "npx"',
                'args = ["server"]',
                "",
                "[mcp_servers.aika]",
                'command = "old-aika"',
                'args = ["old"]',
                "startup_timeout_sec = 1",
                "",
                "[mcp_servers.aika.env]",
                'OLD = "1"',
                "",
                "[projects.\"/tmp/demo\"]",
                "trust_level = \"trusted\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(installer, "build_mcp_config", lambda host: stable_codex_config(tmp_path))

    result = installer.install_mcp_server(
        host="codex",
        scope="project",
        force=True,
        codex_project_root=tmp_path,
    )

    assert result.status == "installed"
    content = config_path.read_text(encoding="utf-8")
    assert 'model = "gpt-5"' in content
    assert "[mcp_servers.other]" in content
    assert "[projects.\"/tmp/demo\"]" in content
    assert "old-aika" not in content
    assert "OLD" not in content
    parsed = tomllib.loads(content)
    assert parsed["mcp_servers"]["other"]["command"] == "npx"
    assert parsed["mcp_servers"]["aika"]["command"] == "/usr/bin/uv"
    assert parsed["mcp_servers"]["aika"]["env"]["UV_CACHE_DIR"] == "/tmp/uv-cache"


def test_install_codex_project_conflict_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir()
    config_path.write_text("[mcp_servers.aika]\ncommand = \"old\"\nargs = []\n", encoding="utf-8")
    monkeypatch.setattr(installer, "build_mcp_config", lambda host: stable_codex_config(tmp_path))

    result = installer.install_mcp_server(host="codex", scope="project", codex_project_root=tmp_path)

    assert result.status == "conflict"
    assert result.exit_code == 2
    assert "--force" in "\n".join(result.messages)


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


def test_doctor_warns_when_codex_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "build_mcp_config", lambda host: stable_codex_config(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        doctor,
        "inspect_sqlite_index",
        lambda path: {"exists": True, "path": str(path), "metadata": {}, "counts": {"claims": 1}, "error": ""},
    )
    monkeypatch.setattr(doctor.subprocess, "run", lambda command, **kwargs: completed(command, 0, stdout=mcp_tools_stdout()))

    report = doctor.run_mcp_doctor(host="codex", home=tmp_path)

    assert report.exit_code == 1
    assert_check(report, "mcp_server", doctor.STATUS_PASS)
    assert "config --host codex" in assert_check(report, "codex", doctor.STATUS_WARN).fix


def test_doctor_warns_when_codex_has_no_aika_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "build_mcp_config", lambda host: stable_codex_config(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(
        doctor,
        "inspect_sqlite_index",
        lambda path: {"exists": True, "path": str(path), "metadata": {}, "counts": {"claims": 1}, "error": ""},
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["/bin/codex", "mcp", "get", "aika"]:
            return completed(command, 1, stderr="not found")
        return completed(command, 0, stdout=mcp_tools_stdout())

    monkeypatch.setattr(doctor.subprocess, "run", run)

    report = doctor.run_mcp_doctor(host="codex", home=tmp_path)

    assert report.exit_code == 1
    check = assert_check(report, "codex", doctor.STATUS_WARN)
    assert "aika mcp install --host codex --scope user" in check.fix


def test_doctor_passes_when_codex_has_aika_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "build_mcp_config", lambda host: stable_codex_config(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(
        doctor,
        "inspect_sqlite_index",
        lambda path: {"exists": True, "path": str(path), "metadata": {}, "counts": {"claims": 1}, "error": ""},
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["/bin/codex", "mcp", "get", "aika"]:
            return completed(command, 0, stdout=json.dumps({"name": "aika"}))
        return completed(command, 0, stdout=mcp_tools_stdout())

    monkeypatch.setattr(doctor.subprocess, "run", run)

    report = doctor.run_mcp_doctor(host="codex", home=tmp_path)

    assert report.exit_code == 1
    assert_check(report, "codex", doctor.STATUS_PASS)
    assert_check(report, "skill_installed_skill", doctor.STATUS_WARN)


def test_mcp_install_with_skill_dry_run_reports_both_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(installer, "build_mcp_config", lambda host: stable_codex_config(tmp_path))
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)

    exit_code = aika_cli.main(["mcp", "install", "--host", "codex", "--scope", "project", "--with-skill", "--dry-run"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "MCP server TOML" in output
    assert "AIKA skill install" in output
    assert "Would install skill 'aika-research'" in output


def test_mcp_install_with_skill_calls_skill_after_mcp_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, str, bool, bool]] = []

    def fake_install_mcp_server(**kwargs: Any) -> installer.InstallResult:
        return installer.InstallResult(
            status="installed",
            exit_code=0,
            config=stable_codex_config(tmp_path),
            messages=["mcp ok"],
        )

    def fake_install_skill(**kwargs: Any) -> SkillInstallResult:
        calls.append((kwargs["host"], kwargs["scope"], kwargs["force"], kwargs["dry_run"]))
        return SkillInstallResult(
            status="installed",
            exit_code=0,
            skill_name="aika-research",
            source_path=tmp_path / "source",
            target_path=tmp_path / "target",
            files=["SKILL.md"],
            messages=["skill ok"],
        )

    monkeypatch.setattr(installer, "install_mcp_server", fake_install_mcp_server)
    import aika.aika_skills as aika_skills

    monkeypatch.setattr(aika_skills, "install_skill", fake_install_skill)

    exit_code = aika_cli.main(["mcp", "install", "--host", "codex", "--scope", "user", "--with-skill", "--force"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls == [("codex", "user", True, False)]
    assert "mcp ok" in output
    assert "skill ok" in output


def test_mcp_install_with_skill_skips_skill_when_mcp_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_install_mcp_server(**kwargs: Any) -> installer.InstallResult:
        return installer.InstallResult(
            status="conflict",
            exit_code=2,
            config=stable_codex_config(tmp_path),
            messages=["mcp conflict"],
        )

    def fail_install_skill(**kwargs: Any) -> SkillInstallResult:
        raise AssertionError("skill install should not run when MCP install fails")

    monkeypatch.setattr(installer, "install_mcp_server", fake_install_mcp_server)
    import aika.aika_skills as aika_skills

    monkeypatch.setattr(aika_skills, "install_skill", fail_install_skill)

    exit_code = aika_cli.main(["mcp", "install", "--host", "codex", "--scope", "user", "--with-skill"])

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "mcp conflict" in output
    assert "AIKA skill install" not in output


def test_mcp_doctor_reports_project_skill_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import aika.aika_skills as aika_skills

    aika_skills.install_skill(host="codex", scope="project", project_root=tmp_path)
    (tmp_path / ".codex" / "config.toml").write_text(
        "[mcp_servers.aika]\ncommand = \"aika\"\nargs = [\"mcp\"]\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(doctor, "build_mcp_config", lambda host: stable_codex_config(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(
        doctor,
        "inspect_sqlite_index",
        lambda path: {"exists": True, "path": str(path), "metadata": {}, "counts": {"claims": 1}, "error": ""},
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["/bin/codex", "mcp", "get", "aika"]:
            return completed(command, 0, stdout=json.dumps({"name": "aika"}))
        return completed(command, 0, stdout=mcp_tools_stdout())

    monkeypatch.setattr(doctor.subprocess, "run", run)

    report = doctor.run_mcp_doctor(host="codex", scope="project", home=tmp_path)

    assert report.exit_code == 0
    assert_check(report, "skill_installed_skill", doctor.STATUS_PASS)
    assert_check(report, "skill_skill_mcp_config", doctor.STATUS_PASS)
