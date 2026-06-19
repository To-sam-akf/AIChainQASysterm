"""Install and diagnose bundled AIKA Agent skills."""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any


DEFAULT_SKILL_NAME = "aika-research"
DEFAULT_MCP_SERVER_NAME = "aika"
SUPPORTED_SKILL_HOSTS = {"codex", "claude-code"}
VALID_SCOPES = {"user", "project"}
STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
SKILL_FILES = ("SKILL.md", "agents/openai.yaml")


class AikaSkillError(RuntimeError):
    """Raised when AIKA Skill installation cannot continue."""


@dataclass(frozen=True)
class SkillInfo:
    name: str
    source_path: Path
    description: str = ""


@dataclass(frozen=True)
class SkillInstallResult:
    status: str
    exit_code: int
    skill_name: str
    source_path: Path
    target_path: Path
    files: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SkillDoctorCheck:
    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass(frozen=True)
class SkillDoctorReport:
    checks: list[SkillDoctorCheck]

    @property
    def exit_code(self) -> int:
        if any(check.status == STATUS_FAIL for check in self.checks):
            return 2
        if any(check.status == STATUS_WARN for check in self.checks):
            return 1
        return 0


def list_bundled_skills() -> list[SkillInfo]:
    """List AIKA skills shipped with this package or source checkout."""
    try:
        source = resolve_bundled_skill(DEFAULT_SKILL_NAME)
    except AikaSkillError:
        return []
    description = _skill_description(source / "SKILL.md")
    return [SkillInfo(name=DEFAULT_SKILL_NAME, source_path=source, description=description)]


def resolve_bundled_skill(name: str = DEFAULT_SKILL_NAME) -> Path:
    """Resolve a bundled skill directory in wheel installs or source checkouts."""
    if name != DEFAULT_SKILL_NAME:
        raise AikaSkillError(f"Unknown AIKA skill '{name}'. Available skill: {DEFAULT_SKILL_NAME}.")

    try:
        package_resource = resources.files("aika").joinpath("bundled_skills", name)
        package_path = Path(str(package_resource))
        if package_path.is_dir() and _skill_files_exist(package_path):
            return package_path
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        pass

    source_path = Path(__file__).resolve().parents[1] / "skills" / name
    if source_path.is_dir() and _skill_files_exist(source_path):
        return source_path
    raise AikaSkillError(
        f"Bundled skill '{name}' was not found. Reinstall aika-research-mcp or run from a source checkout "
        "that contains skills/aika-research."
    )


def install_skill(
    *,
    host: str,
    scope: str,
    force: bool = False,
    dry_run: bool = False,
    project_root: str | Path | None = None,
    skill_name: str = DEFAULT_SKILL_NAME,
) -> SkillInstallResult:
    """Install the bundled AIKA skill into a supported host skill directory."""
    normalized_host = _normalize_host(host)
    normalized_scope = _normalize_scope(scope)
    source = resolve_bundled_skill(skill_name)
    target = skill_target_path(host=normalized_host, scope=normalized_scope, project_root=project_root, skill_name=skill_name)
    files = _relative_skill_files(source)

    if dry_run:
        status = "up-to-date" if _target_matches(source, target, files) else "dry-run"
        return SkillInstallResult(
            status=status,
            exit_code=0,
            skill_name=skill_name,
            source_path=source,
            target_path=target,
            files=files,
            messages=[
                "Dry run only. No skill files were changed.",
                f"Would install skill '{skill_name}' for host '{normalized_host}' with scope '{normalized_scope}'.",
            ],
        )

    if target.exists() and not _target_matches(source, target, files) and not force:
        return SkillInstallResult(
            status="conflict",
            exit_code=2,
            skill_name=skill_name,
            source_path=source,
            target_path=target,
            files=files,
            messages=[
                f"Skill '{skill_name}' already exists with different content: {target}",
                "Re-run with --force to overwrite only AIKA-managed files for this skill.",
            ],
        )

    if _target_matches(source, target, files):
        return SkillInstallResult(
            status="up-to-date",
            exit_code=0,
            skill_name=skill_name,
            source_path=source,
            target_path=target,
            files=files,
            messages=[f"Skill '{skill_name}' is already installed and up to date: {target}"],
        )

    for relative in files:
        src = source / relative
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    return SkillInstallResult(
        status="installed",
        exit_code=0,
        skill_name=skill_name,
        source_path=source,
        target_path=target,
        files=files,
        messages=[f"Installed skill '{skill_name}' to {target}."],
    )


def run_skill_doctor(
    *,
    host: str,
    scope: str,
    project_root: str | Path | None = None,
    skill_name: str = DEFAULT_SKILL_NAME,
    server_name: str = DEFAULT_MCP_SERVER_NAME,
    timeout_seconds: float = 8.0,
) -> SkillDoctorReport:
    """Check bundled and installed AIKA skill state."""
    checks: list[SkillDoctorCheck] = []
    normalized_host = _normalize_host(host)
    normalized_scope = _normalize_scope(scope)

    try:
        source = resolve_bundled_skill(skill_name)
        checks.append(SkillDoctorCheck("bundled_skill", STATUS_PASS, str(source)))
    except AikaSkillError as exc:
        checks.append(
            SkillDoctorCheck(
                "bundled_skill",
                STATUS_FAIL,
                str(exc),
                "Reinstall aika-research-mcp or rebuild the package with bundled skills.",
            )
        )
        source = None

    try:
        target = skill_target_path(
            host=normalized_host,
            scope=normalized_scope,
            project_root=project_root,
            skill_name=skill_name,
        )
        checks.append(_check_installed_skill(target, source=source))
        checks.append(_check_skill_frontmatter(target))
        checks.append(_check_skill_mcp_dependency(target))
        checks.append(
            _check_host_mcp_config(
                host=normalized_host,
                scope=normalized_scope,
                project_root=project_root,
                server_name=server_name,
                timeout_seconds=timeout_seconds,
            )
        )
    except AikaSkillError as exc:
        checks.append(SkillDoctorCheck("skill_target", STATUS_FAIL, str(exc), "Use --scope user for Claude Code."))

    return SkillDoctorReport(checks)


def skill_target_path(
    *,
    host: str,
    scope: str,
    project_root: str | Path | None = None,
    skill_name: str = DEFAULT_SKILL_NAME,
) -> Path:
    normalized_host = _normalize_host(host)
    normalized_scope = _normalize_scope(scope)
    if normalized_host == "codex":
        if normalized_scope == "project":
            root = Path(project_root).expanduser().resolve() if project_root is not None else Path.cwd().resolve()
            return root / ".codex" / "skills" / skill_name
        codex_home = Path(os.getenv("CODEX_HOME") or Path.home() / ".codex").expanduser().resolve()
        return codex_home / "skills" / skill_name
    if normalized_host == "claude-code":
        if normalized_scope == "project":
            raise AikaSkillError("Claude Code project-scope skill install is not supported in this AIKA release.")
        claude_home = Path(os.getenv("CLAUDE_HOME") or Path.home() / ".claude").expanduser().resolve()
        return claude_home / "skills" / skill_name
    raise AikaSkillError("Unsupported skill host. Use --host claude-code or --host codex.")


def format_skill_install_result(result: SkillInstallResult) -> str:
    lines = list(result.messages)
    lines.append(f"Source: {result.source_path}")
    lines.append(f"Target: {result.target_path}")
    if result.files:
        lines.append("Files:")
        lines.extend(f"  {item}" for item in result.files)
    return "\n".join(lines)


def format_skill_doctor_report(report: SkillDoctorReport) -> str:
    lines: list[str] = []
    for check in report.checks:
        lines.append(f"[{check.status.upper()}] {check.name}: {check.detail}")
        if check.fix:
            lines.append(f"  Fix: {check.fix}")
    return "\n".join(lines)


def _normalize_host(host: str | None) -> str:
    normalized = str(host or "").strip().lower()
    if normalized not in SUPPORTED_SKILL_HOSTS:
        raise AikaSkillError("Unsupported skill host. Use --host claude-code or --host codex.")
    return normalized


def _normalize_scope(scope: str | None) -> str:
    normalized = str(scope or "user").strip().lower()
    if normalized not in VALID_SCOPES:
        raise AikaSkillError("Unsupported scope. Use --scope user or --scope project.")
    return normalized


def _skill_files_exist(path: Path) -> bool:
    return all((path / relative).is_file() for relative in SKILL_FILES)


def _relative_skill_files(source: Path) -> list[str]:
    return sorted(
        str(path.relative_to(source))
        for path in source.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(source).parts
    )


def _target_matches(source: Path, target: Path, files: list[str]) -> bool:
    if not target.exists():
        return False
    for relative in files:
        src = source / relative
        dst = target / relative
        if not dst.is_file() or src.read_bytes() != dst.read_bytes():
            return False
    return True


def _skill_description(skill_file: Path) -> str:
    if not skill_file.is_file():
        return ""
    for line in skill_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("description:"):
            return line.partition(":")[2].strip().strip('"')
    return ""


def _check_installed_skill(target: Path, *, source: Path | None) -> SkillDoctorCheck:
    if not target.is_dir():
        return SkillDoctorCheck(
            "installed_skill",
            STATUS_FAIL,
            f"Skill is not installed: {target}",
            "Run aika skill install --host <host> --scope <scope>.",
        )
    if source is not None and _target_matches(source, target, _relative_skill_files(source)):
        return SkillDoctorCheck("installed_skill", STATUS_PASS, f"Installed and up to date: {target}")
    return SkillDoctorCheck(
        "installed_skill",
        STATUS_WARN,
        f"Skill exists but differs from bundled AIKA skill: {target}",
        "Run aika skill install --host <host> --scope <scope> --force to refresh AIKA-managed files.",
    )


def _check_skill_frontmatter(target: Path) -> SkillDoctorCheck:
    path = target / "SKILL.md"
    if not path.is_file():
        return SkillDoctorCheck(
            "skill_frontmatter",
            STATUS_FAIL,
            f"Missing SKILL.md: {path}",
            "Run aika skill install --host <host> --scope <scope> --force.",
        )
    text = path.read_text(encoding="utf-8")
    if text.startswith("---") and "name: aika-research" in text.split("---", 2)[1]:
        return SkillDoctorCheck("skill_frontmatter", STATUS_PASS, "name: aika-research")
    return SkillDoctorCheck(
        "skill_frontmatter",
        STATUS_FAIL,
        "SKILL.md frontmatter does not declare name: aika-research.",
        "Run aika skill install --host <host> --scope <scope> --force.",
    )


def _check_skill_mcp_dependency(target: Path) -> SkillDoctorCheck:
    path = target / "agents" / "openai.yaml"
    if not path.is_file():
        return SkillDoctorCheck(
            "skill_mcp_dependency",
            STATUS_FAIL,
            f"Missing agents/openai.yaml: {path}",
            "Run aika skill install --host <host> --scope <scope> --force.",
        )
    text = path.read_text(encoding="utf-8")
    if 'type: "mcp"' in text and 'value: "aika"' in text:
        return SkillDoctorCheck("skill_mcp_dependency", STATUS_PASS, "Requires MCP server: aika")
    return SkillDoctorCheck(
        "skill_mcp_dependency",
        STATUS_FAIL,
        "agents/openai.yaml does not declare mcp:aika.",
        "Run aika skill install --host <host> --scope <scope> --force.",
    )


def _check_host_mcp_config(
    *,
    host: str,
    scope: str,
    project_root: str | Path | None,
    server_name: str,
    timeout_seconds: float,
) -> SkillDoctorCheck:
    if host == "codex":
        config_path = _codex_config_path(scope=scope, project_root=project_root)
        if not config_path.is_file():
            return SkillDoctorCheck(
                "skill_mcp_config",
                STATUS_WARN,
                f"Codex config not found: {config_path}",
                f"Run aika mcp install --host codex --scope {scope}.",
            )
        try:
            data = _load_toml_file(config_path)
        except AikaSkillError as exc:
            return SkillDoctorCheck("skill_mcp_config", STATUS_WARN, str(exc), "Fix Codex config TOML.")
        servers = data.get("mcp_servers", {})
        if isinstance(servers, dict) and server_name in servers:
            return SkillDoctorCheck("skill_mcp_config", STATUS_PASS, f"Codex MCP server '{server_name}' is configured.")
        return SkillDoctorCheck(
            "skill_mcp_config",
            STATUS_WARN,
            f"Codex MCP server '{server_name}' is not configured.",
            f"Run aika mcp install --host codex --scope {scope}.",
        )

    claude = shutil.which("claude")
    if not claude:
        return SkillDoctorCheck(
            "skill_mcp_config",
            STATUS_WARN,
            "Claude Code CLI not found on PATH.",
            "Run aika mcp config --host claude-code and configure the MCP server manually.",
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
        return SkillDoctorCheck(
            "skill_mcp_config",
            STATUS_WARN,
            f"Claude Code MCP lookup timed out after {timeout_seconds:g}s.",
            "Run claude mcp get aika manually.",
        )
    if result.returncode == 0:
        return SkillDoctorCheck("skill_mcp_config", STATUS_PASS, f"Claude Code MCP server '{server_name}' is configured.")
    return SkillDoctorCheck(
        "skill_mcp_config",
        STATUS_WARN,
        f"Claude Code MCP server '{server_name}' is not configured.",
        "Run aika mcp install --host claude-code --scope user.",
    )


def _codex_config_path(*, scope: str, project_root: str | Path | None) -> Path:
    if scope == "project":
        root = Path(project_root).expanduser().resolve() if project_root is not None else Path.cwd().resolve()
        return root / ".codex" / "config.toml"
    codex_home = Path(os.getenv("CODEX_HOME") or Path.home() / ".codex").expanduser().resolve()
    return codex_home / "config.toml"


def _load_toml_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise AikaSkillError(f"Config is not valid TOML: {path}: {exc}") from exc
    if not isinstance(data, dict):
        return {}
    return data
