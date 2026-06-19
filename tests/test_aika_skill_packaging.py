from __future__ import annotations

from pathlib import Path

import pytest

from aika import aika_cli
from aika import aika_skills


def test_resolve_bundled_skill_finds_source_checkout() -> None:
    source = aika_skills.resolve_bundled_skill()

    assert source.name == "aika-research"
    assert (source / "SKILL.md").is_file()
    assert (source / "agents" / "openai.yaml").is_file()


def test_skill_list_cli_lists_aika_research(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = aika_cli.main(["skill", "list"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "aika-research" in output


def test_skill_install_dry_run_does_not_write_project_files(tmp_path: Path) -> None:
    result = aika_skills.install_skill(host="codex", scope="project", dry_run=True, project_root=tmp_path)

    assert result.status == "dry-run"
    assert result.exit_code == 0
    assert not (tmp_path / ".codex" / "skills" / "aika-research").exists()
    assert "SKILL.md" in result.files


def test_skill_install_codex_project_writes_skill_files(tmp_path: Path) -> None:
    result = aika_skills.install_skill(host="codex", scope="project", project_root=tmp_path)

    target = tmp_path / ".codex" / "skills" / "aika-research"
    assert result.status == "installed"
    assert (target / "SKILL.md").is_file()
    assert (target / "agents" / "openai.yaml").is_file()


def test_skill_install_codex_user_uses_codex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    codex_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    result = aika_skills.install_skill(host="codex", scope="user")

    assert result.status == "installed"
    assert (codex_home / "skills" / "aika-research" / "SKILL.md").is_file()


def test_skill_install_claude_user_uses_claude_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    claude_home = tmp_path / "claude-home"
    monkeypatch.setenv("CLAUDE_HOME", str(claude_home))

    result = aika_skills.install_skill(host="claude-code", scope="user")

    assert result.status == "installed"
    assert (claude_home / "skills" / "aika-research" / "SKILL.md").is_file()


def test_skill_install_claude_project_scope_is_unsupported(tmp_path: Path) -> None:
    with pytest.raises(aika_skills.AikaSkillError, match="project-scope"):
        aika_skills.install_skill(host="claude-code", scope="project", project_root=tmp_path)


def test_skill_install_refuses_different_existing_content_without_force(tmp_path: Path) -> None:
    target = tmp_path / ".codex" / "skills" / "aika-research"
    (target / "agents").mkdir(parents=True)
    (target / "SKILL.md").write_text("custom skill\n", encoding="utf-8")
    (target / "agents" / "openai.yaml").write_text("custom agent\n", encoding="utf-8")

    result = aika_skills.install_skill(host="codex", scope="project", project_root=tmp_path)

    assert result.status == "conflict"
    assert result.exit_code == 2
    assert "custom skill" in (target / "SKILL.md").read_text(encoding="utf-8")


def test_skill_install_force_overwrites_aika_files_only(tmp_path: Path) -> None:
    target = tmp_path / ".codex" / "skills" / "aika-research"
    (target / "agents").mkdir(parents=True)
    (target / "SKILL.md").write_text("custom skill\n", encoding="utf-8")
    (target / "agents" / "openai.yaml").write_text("custom agent\n", encoding="utf-8")
    (target / "user-notes.md").write_text("keep me\n", encoding="utf-8")

    result = aika_skills.install_skill(host="codex", scope="project", project_root=tmp_path, force=True)

    assert result.status == "installed"
    assert "name: aika-research" in (target / "SKILL.md").read_text(encoding="utf-8")
    assert 'value: "aika"' in (target / "agents" / "openai.yaml").read_text(encoding="utf-8")
    assert (target / "user-notes.md").read_text(encoding="utf-8") == "keep me\n"


def test_skill_doctor_passes_for_project_skill_and_mcp_config(tmp_path: Path) -> None:
    aika_skills.install_skill(host="codex", scope="project", project_root=tmp_path)
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.write_text("[mcp_servers.aika]\ncommand = \"aika\"\nargs = [\"mcp\"]\n", encoding="utf-8")

    report = aika_skills.run_skill_doctor(host="codex", scope="project", project_root=tmp_path)

    assert report.exit_code == 0
    assert {check.name: check.status for check in report.checks}["installed_skill"] == aika_skills.STATUS_PASS
    assert {check.name: check.status for check in report.checks}["skill_mcp_config"] == aika_skills.STATUS_PASS
