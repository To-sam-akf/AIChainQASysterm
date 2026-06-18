"""Public lightweight AIKA CLI for local sample data and SQLite search."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aika.aika_core.backends.sqlite_backend import (
    DEFAULT_PROFILE,
    SQLiteResearchBackend,
    build_sqlite_index,
    inspect_sqlite_index,
    profile_index_path,
    profile_knowledge_dir,
    resolve_aika_home,
    sqlite_fts_status,
)
from aika.aika_core.knowledge_pack import format_validation_result, validate_knowledge_pack
from aika.aika_core.sample_data import (
    SAMPLE_FILES,
    copy_sample_files,
    resolve_sample_source,
    sample_source_status,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_SOURCE_DIR = ROOT_DIR / "data" / "knowledge_packs" / "sample"
DEFAULT_DEMO_QUERY = "液冷产业链"
EXPECTED_MCP_TOOLS = [
    "search_evidence",
    "search_claims",
    "get_company_profile",
    "compare_companies",
    "query_industry_graph",
    "build_research_brief",
    "audit_evidence_gaps",
    "run_research_task",
]
STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aika", description="AIKA lightweight local research CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize local AIKA data directories.")
    init_parser.add_argument("--sample", action="store_true", help="Copy bundled sample knowledge files.")
    init_parser.add_argument("--home", default="", help="AIKA home directory; defaults to AIKA_HOME or ~/.aika.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing sample files.")
    init_parser.set_defaults(func=run_init)

    build_parser_ = subparsers.add_parser("build-index", help="Build the local SQLite FTS5 index.")
    build_parser_.add_argument("--home", default="", help="AIKA home directory; defaults to AIKA_HOME or ~/.aika.")
    build_parser_.add_argument("--profile", default=DEFAULT_PROFILE, help="Knowledge profile name.")
    build_parser_.set_defaults(func=run_build_index)

    doctor_parser = subparsers.add_parser("doctor", help="Check local AIKA data and index status.")
    doctor_parser.add_argument("--home", default="", help="AIKA home directory; defaults to AIKA_HOME or ~/.aika.")
    doctor_parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Knowledge profile name.")
    doctor_parser.set_defaults(func=run_doctor)

    demo_parser = subparsers.add_parser("demo", help="Run a deterministic local SQLite demo.")
    demo_parser.add_argument("--home", default="", help="AIKA home directory; defaults to AIKA_HOME or ~/.aika.")
    demo_parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Knowledge profile name.")
    demo_parser.add_argument("--query", default=DEFAULT_DEMO_QUERY, help="Demo query.")
    demo_parser.set_defaults(func=run_demo)

    evidence_parser = subparsers.add_parser("search-evidence", help="Search evidence spans in the local SQLite index.")
    add_search_args(evidence_parser)
    evidence_parser.set_defaults(func=run_search_evidence)

    claims_parser = subparsers.add_parser("search-claims", help="Search claims in the local SQLite index.")
    add_search_args(claims_parser)
    claims_parser.add_argument("--claim-type", action="append", default=[], help="Filter by claim type.")
    claims_parser.set_defaults(func=run_search_claims)

    validate_parser = subparsers.add_parser("validate-data", help="Validate a local AIKA knowledge pack.")
    validate_parser.add_argument("--path", required=True, help="Knowledge pack directory to validate.")
    validate_parser.add_argument("--sample-size", type=int, default=20, help="Evidence rows to include in sample counts.")
    validate_parser.set_defaults(func=run_validate_data)

    mcp_parser = subparsers.add_parser("mcp", help="Start or configure the AIKA MCP server.")
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport; stdio is the default for desktop Agent hosts.",
    )
    mcp_parser.set_defaults(func=run_mcp)
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")

    mcp_config_parser = mcp_subparsers.add_parser("config", help="Print host MCP server JSON.")
    mcp_config_parser.add_argument("--host", default="claude-code", help="MCP host; Phase 3.1 supports claude-code.")
    mcp_config_parser.set_defaults(func=run_mcp_config)

    mcp_install_parser = mcp_subparsers.add_parser("install", help="Register AIKA with an MCP host.")
    mcp_install_parser.add_argument("--host", default="claude-code", help="MCP host; Phase 3.1 supports claude-code.")
    mcp_install_parser.add_argument("--scope", choices=["user", "project"], default="user", help="Host config scope.")
    mcp_install_parser.add_argument("--force", action="store_true", help="Replace an existing AIKA MCP server config.")
    mcp_install_parser.add_argument("--dry-run", action="store_true", help="Print config and commands without changing host config.")
    mcp_install_parser.set_defaults(func=run_mcp_install)

    mcp_doctor_parser = mcp_subparsers.add_parser("doctor", help="Diagnose AIKA MCP and host configuration.")
    mcp_doctor_parser.add_argument("--host", default="claude-code", help="MCP host; Phase 3.1 supports claude-code.")
    mcp_doctor_parser.add_argument("--home", default="", help="AIKA home directory; defaults to AIKA_HOME or ~/.aika.")
    mcp_doctor_parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Knowledge profile name.")
    mcp_doctor_parser.set_defaults(func=run_mcp_doctor)

    return parser


def add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query", help="Search query.")
    parser.add_argument("--home", default="", help="AIKA home directory; defaults to AIKA_HOME or ~/.aika.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Knowledge profile name.")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum results to return.")
    parser.add_argument("--company", action="append", default=[], help="Filter by company. May be repeated.")
    parser.add_argument("--topic", action="append", default=[], help="Filter by topic. May be repeated.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def run_init(args: argparse.Namespace) -> int:
    home = resolve_aika_home(args.home or None)
    for dirname in ("knowledge", "indexes", "logs"):
        (home / dirname).mkdir(parents=True, exist_ok=True)
    if args.sample:
        target_dir = profile_knowledge_dir(home, profile=DEFAULT_PROFILE)
        source = resolve_sample_source(SAMPLE_SOURCE_DIR)
        status = copy_sample_files(target_dir, source=source, force=bool(args.force))
        if not status.available:
            print(f"Sample source is incomplete: {', '.join(status.missing)}", file=sys.stderr)
            return 2
    write_config(home, profile=DEFAULT_PROFILE)
    print(f"AIKA home: {home}")
    if args.sample:
        print(f"Sample knowledge: {profile_knowledge_dir(home, profile=DEFAULT_PROFILE)}")
    return 0


def run_build_index(args: argparse.Namespace) -> int:
    home = resolve_aika_home(args.home or None)
    profile = str(args.profile or DEFAULT_PROFILE)
    knowledge_dir = profile_knowledge_dir(home, profile=profile)
    index_path = profile_index_path(home, profile=profile)
    if not knowledge_dir.exists():
        print(f"Knowledge directory not found: {knowledge_dir}", file=sys.stderr)
        return 2
    result = build_sqlite_index(knowledge_dir, index_path)
    counts = ", ".join(f"{key}={value}" for key, value in result["counts"].items())
    print(f"Built SQLite index: {index_path}")
    print(f"Counts: {counts}")
    return 0


def run_doctor(args: argparse.Namespace) -> int:
    report = run_cli_doctor(home=args.home or None, profile=args.profile)
    print(format_cli_doctor_report(report))
    return report.exit_code


def run_demo(args: argparse.Namespace) -> int:
    home = resolve_aika_home(args.home or None)
    profile = str(args.profile or DEFAULT_PROFILE)
    index_path = profile_index_path(home, profile=profile)
    index = inspect_sqlite_index(index_path)
    if not index["exists"] or index["error"]:
        print(f"SQLite index is not ready: {index_path}", file=sys.stderr)
        print("Run: aika init --sample && aika build-index", file=sys.stderr)
        return 2

    backend = SQLiteResearchBackend(index_path)
    query = str(args.query or DEFAULT_DEMO_QUERY).strip() or DEFAULT_DEMO_QUERY
    evidence_cards = backend.search_evidence(query, top_k=3)
    claims = backend.search_claims("液冷", top_k=3)
    brief = backend.build_research_brief(query, topic="液冷")

    if not evidence_cards or not claims or not brief.markdown:
        print("AIKA demo could not retrieve enough local sample evidence.", file=sys.stderr)
        return 1

    counts = index.get("counts") or {}
    print("AIKA local demo")
    print(f"Home: {home}")
    print(f"Index: {index_path}")
    if counts:
        print("Counts: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    print(f"Query: {query}")
    print(f"Evidence cards: {len(evidence_cards)}")
    print(f"Claims: {len(claims)}")
    first_card = evidence_cards[0]
    print(f"Top evidence: [{first_card.citation_id}] {first_card.title} p.{first_card.page}")
    print(f"Top claim: {claims[0].claim_type} / {claims[0].topic}")
    print(f"Brief title: {brief.title}")
    return 0


def run_search_evidence(args: argparse.Namespace) -> int:
    backend = SQLiteResearchBackend.from_home(args.home or None, profile=args.profile)
    filters = search_filters(args)
    rows = [card.to_dict() for card in backend.search_evidence(args.query, top_k=args.top_k, **filters)]
    print_json(rows)
    return 0


def run_search_claims(args: argparse.Namespace) -> int:
    backend = SQLiteResearchBackend.from_home(args.home or None, profile=args.profile)
    filters = search_filters(args)
    if args.claim_type:
        filters["claim_type"] = args.claim_type
    rows = [claim.to_dict() for claim in backend.search_claims(args.query, top_k=args.top_k, **filters)]
    print_json(rows)
    return 0


def run_validate_data(args: argparse.Namespace) -> int:
    result = validate_knowledge_pack(args.path, sample_size=args.sample_size)
    print(format_validation_result(result))
    return result.exit_code


def run_mcp(args: argparse.Namespace) -> int:
    from aika.aika_mcp.server import run_server

    run_server(transport=args.transport)
    return 0


def run_mcp_config(args: argparse.Namespace) -> int:
    from aika.aika_mcp.host_configs import AikaMcpConfigError, build_mcp_config

    try:
        print_json(build_mcp_config(host=args.host))
    except AikaMcpConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def run_mcp_install(args: argparse.Namespace) -> int:
    from aika.aika_mcp.host_configs import AikaMcpConfigError
    from aika.aika_mcp.installer import AikaMcpInstallError, format_install_result, install_mcp_server

    try:
        result = install_mcp_server(
            host=args.host,
            scope=args.scope,
            force=bool(args.force),
            dry_run=bool(args.dry_run),
        )
    except (AikaMcpConfigError, AikaMcpInstallError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(format_install_result(result))
    return result.exit_code


def run_mcp_doctor(args: argparse.Namespace) -> int:
    from aika.aika_mcp.doctor import format_doctor_report, run_mcp_doctor as diagnose_mcp

    report = diagnose_mcp(host=args.host, home=args.home or None, profile=args.profile)
    print(format_doctor_report(report))
    return report.exit_code


@dataclass(frozen=True)
class CliDoctorCheck:
    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass(frozen=True)
class CliDoctorReport:
    checks: list[CliDoctorCheck]

    @property
    def exit_code(self) -> int:
        if any(check.status == STATUS_FAIL for check in self.checks):
            return 2
        if any(check.status == STATUS_WARN for check in self.checks):
            return 1
        return 0


def run_cli_doctor(
    *,
    home: str | Path | None = None,
    profile: str = DEFAULT_PROFILE,
    timeout_seconds: float = 8.0,
) -> CliDoctorReport:
    resolved_home = resolve_aika_home(home)
    profile_name = str(profile or DEFAULT_PROFILE)
    knowledge_dir = profile_knowledge_dir(resolved_home, profile=profile_name)
    index_path = profile_index_path(resolved_home, profile=profile_name)
    checks: list[CliDoctorCheck] = [
        _check_python_version(),
        _check_directory("home", resolved_home, "Run aika init --sample."),
        _check_file("config", resolved_home / "config.toml", "Run aika init."),
        _check_directory("knowledge_dir", resolved_home / "knowledge", "Run aika init."),
        _check_directory("index_dir", resolved_home / "indexes", "Run aika init."),
        _check_directory("logs_dir", resolved_home / "logs", "Run aika init."),
        _check_bundled_sample(),
        _check_sample_files(knowledge_dir),
        _check_sqlite_fts(),
        _check_sqlite_index(index_path),
    ]
    checks.extend(_check_mcp_server_and_tools(timeout_seconds=timeout_seconds))
    checks.append(_check_postgres_not_required())
    checks.append(_check_sample_query(index_path))
    return CliDoctorReport(checks)


def format_cli_doctor_report(report: CliDoctorReport) -> str:
    lines: list[str] = []
    for check in report.checks:
        lines.append(f"[{check.status.upper()}] {check.name}: {check.detail}")
        if check.fix:
            lines.append(f"  Fix: {check.fix}")
    return "\n".join(lines)


def _check_python_version() -> CliDoctorCheck:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11):
        return CliDoctorCheck("python", STATUS_PASS, version)
    return CliDoctorCheck(
        "python",
        STATUS_FAIL,
        f"{version}; AIKA requires Python 3.11+.",
        "Install Python 3.11 or newer, then reinstall aika-research-mcp.",
    )


def _check_directory(name: str, path: Path, fix: str) -> CliDoctorCheck:
    if path.is_dir():
        return CliDoctorCheck(name, STATUS_PASS, str(path))
    return CliDoctorCheck(name, STATUS_FAIL, f"Missing directory: {path}", fix)


def _check_file(name: str, path: Path, fix: str) -> CliDoctorCheck:
    if path.is_file():
        return CliDoctorCheck(name, STATUS_PASS, str(path))
    return CliDoctorCheck(name, STATUS_FAIL, f"Missing file: {path}", fix)


def _check_bundled_sample() -> CliDoctorCheck:
    status = sample_source_status(SAMPLE_SOURCE_DIR)
    if status.available:
        return CliDoctorCheck("bundled_sample", STATUS_PASS, status.source)
    return CliDoctorCheck(
        "bundled_sample",
        STATUS_FAIL,
        f"Sample source is incomplete: {', '.join(status.missing)}",
        "Reinstall aika-research-mcp or run from a source checkout that contains data/knowledge_packs/sample.",
    )


def _check_sample_files(knowledge_dir: Path) -> CliDoctorCheck:
    missing = [name for name in SAMPLE_FILES if not (knowledge_dir / name).is_file()]
    if not missing:
        return CliDoctorCheck("sample_files", STATUS_PASS, str(knowledge_dir))
    return CliDoctorCheck(
        "sample_files",
        STATUS_FAIL,
        f"Missing sample files in {knowledge_dir}: {', '.join(missing)}",
        "Run aika init --sample.",
    )


def _check_sqlite_fts() -> CliDoctorCheck:
    fts = sqlite_fts_status()
    detail = f"sqlite={fts['sqlite_version']} tokenizer={fts.get('tokenizer') or '-'}"
    if fts["fts5"]:
        return CliDoctorCheck("sqlite_fts5", STATUS_PASS, detail)
    return CliDoctorCheck(
        "sqlite_fts5",
        STATUS_FAIL,
        f"{detail}; {fts.get('error') or 'FTS5 unavailable'}",
        "Use a Python build whose sqlite3 module includes FTS5.",
    )


def _check_sqlite_index(index_path: Path) -> CliDoctorCheck:
    index = inspect_sqlite_index(index_path)
    if not index["exists"]:
        return CliDoctorCheck(
            "sqlite_index",
            STATUS_FAIL,
            f"SQLite index not found: {index_path}",
            "Run aika build-index.",
        )
    if index["error"]:
        return CliDoctorCheck(
            "sqlite_index",
            STATUS_FAIL,
            f"SQLite index is not queryable: {index['error']}",
            "Rebuild the index with aika build-index.",
        )
    counts = index.get("counts") or {}
    count_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    metadata = index.get("metadata") or {}
    schema = metadata.get("schema_version", "?")
    tokenizer = metadata.get("tokenizer", "?")
    detail = f"{index_path} schema={schema} tokenizer={tokenizer}"
    if count_text:
        detail = f"{detail} ({count_text})"
    return CliDoctorCheck("sqlite_index", STATUS_PASS, detail)


def _check_mcp_server_and_tools(*, timeout_seconds: float) -> list[CliDoctorCheck]:
    command = [sys.executable, "-m", "aika.aika_cli", "mcp"]
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "aika-doctor", "version": _package_version()},
            },
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    input_text = "\n".join(json.dumps(message, ensure_ascii=False) for message in messages) + "\n"
    try:
        result = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return [
            CliDoctorCheck(
                "mcp_server",
                STATUS_FAIL,
                f"Command not found: {exc.filename or command[0]}",
                "Reinstall aika-research-mcp.",
            ),
            CliDoctorCheck("mcp_tools", STATUS_FAIL, "Skipped because MCP server could not start."),
        ]
    except subprocess.TimeoutExpired:
        return [
            CliDoctorCheck(
                "mcp_server",
                STATUS_FAIL,
                f"Timed out after {timeout_seconds:g}s while listing MCP tools.",
                "Run aika mcp manually to inspect the server startup.",
            ),
            CliDoctorCheck("mcp_tools", STATUS_FAIL, "Skipped because MCP server timed out."),
        ]

    if result.returncode != 0:
        return [
            CliDoctorCheck(
                "mcp_server",
                STATUS_FAIL,
                f"Could not start AIKA MCP server: {_compact_detail(result.stderr or result.stdout)}",
                "Reinstall with MCP dependencies, then retry aika doctor.",
            ),
            CliDoctorCheck("mcp_tools", STATUS_FAIL, "Skipped because MCP server failed."),
        ]

    response = _jsonrpc_response(result.stdout, request_id=2)
    if not response or "error" in response:
        detail = response.get("error") if isinstance(response, dict) else "missing tools/list response"
        return [
            CliDoctorCheck(
                "mcp_server",
                STATUS_FAIL,
                f"AIKA MCP server did not return tools/list successfully: {detail}",
                "Run AIKA_MCP_TRACE=/tmp/aika-mcp.trace aika mcp for protocol details.",
            ),
            CliDoctorCheck("mcp_tools", STATUS_FAIL, "Skipped because tools/list failed."),
        ]

    tools = response.get("result", {}).get("tools", [])
    names = {str(item.get("name") or "") for item in tools if isinstance(item, dict)}
    missing = set(EXPECTED_MCP_TOOLS) - names
    server_check = CliDoctorCheck("mcp_server", STATUS_PASS, "Started and returned tools/list.")
    if missing:
        tools_check = CliDoctorCheck(
            "mcp_tools",
            STATUS_FAIL,
            f"Listed {len(names)} tools but missing: {', '.join(sorted(missing))}",
            "Re-run tests/test_aika_mcp_tools.py to verify tool registration.",
        )
    else:
        tools_check = CliDoctorCheck("mcp_tools", STATUS_PASS, f"Registered {len(names)} tools.")
    return [server_check, tools_check]


def _check_postgres_not_required() -> CliDoctorCheck:
    if "psycopg" in sys.modules:
        return CliDoctorCheck(
            "postgres_not_required",
            STATUS_WARN,
            "psycopg was imported during lightweight CLI diagnostics.",
            "Keep PostgreSQL-only imports behind optional/full install paths.",
        )
    return CliDoctorCheck("postgres_not_required", STATUS_PASS, "psycopg was not imported.")


def _check_sample_query(index_path: Path) -> CliDoctorCheck:
    if not index_path.exists():
        return CliDoctorCheck(
            "sample_query",
            STATUS_FAIL,
            f"Skipped because SQLite index is missing: {index_path}",
            "Run aika build-index.",
        )
    try:
        backend = SQLiteResearchBackend(index_path)
        evidence_cards = backend.search_evidence(DEFAULT_DEMO_QUERY, top_k=1)
        claims = backend.search_claims("液冷", top_k=1)
    except Exception as exc:
        return CliDoctorCheck(
            "sample_query",
            STATUS_FAIL,
            f"Sample query failed: {exc}",
            "Rebuild the index with aika build-index.",
        )
    if evidence_cards and claims:
        return CliDoctorCheck(
            "sample_query",
            STATUS_PASS,
            f"{DEFAULT_DEMO_QUERY}: evidence={len(evidence_cards)} claims={len(claims)}",
        )
    return CliDoctorCheck(
        "sample_query",
        STATUS_FAIL,
        f"{DEFAULT_DEMO_QUERY}: evidence={len(evidence_cards)} claims={len(claims)}",
        "Run aika init --sample --force && aika build-index.",
    )


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


def _compact_detail(value: str, *, limit: int = 360) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text or "no error output"
    return text[: limit - 3] + "..."


def _package_version() -> str:
    for distribution_name in ("aika-research-mcp", "aiqasys"):
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "0.1.0"


def search_filters(args: argparse.Namespace) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if args.company:
        filters["company"] = args.company
    if args.topic:
        filters["topic"] = args.topic
    return filters


def write_config(home: Path, *, profile: str) -> None:
    config_path = home / "config.toml"
    content = "\n".join(
        [
            f'profile = "{profile}"',
            f'aika_home = "{home}"',
            f'knowledge_dir = "knowledge/{profile}"',
            f'index_path = "indexes/{profile}.sqlite"',
            "",
        ]
    )
    config_path.write_text(content, encoding="utf-8")


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
