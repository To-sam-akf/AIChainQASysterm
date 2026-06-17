"""Public lightweight AIKA CLI for local sample data and SQLite search."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from src.aika_core.backends.sqlite_backend import (
    DEFAULT_PROFILE,
    SQLiteResearchBackend,
    build_sqlite_index,
    inspect_sqlite_index,
    profile_index_path,
    profile_knowledge_dir,
    resolve_aika_home,
    sqlite_fts_status,
)
from src.aika_core.data_paths import (
    CLAIMS_FILE,
    ENTITIES_FILE,
    EVIDENCE_SPANS_FILE,
    RELATIONS_FILE,
    SEGMENT_DOSSIERS_FILE,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_SOURCE_DIR = ROOT_DIR / "data" / "curated"
SAMPLE_FILES = [
    ENTITIES_FILE,
    RELATIONS_FILE,
    CLAIMS_FILE,
    EVIDENCE_SPANS_FILE,
    SEGMENT_DOSSIERS_FILE,
]


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

    evidence_parser = subparsers.add_parser("search-evidence", help="Search evidence spans in the local SQLite index.")
    add_search_args(evidence_parser)
    evidence_parser.set_defaults(func=run_search_evidence)

    claims_parser = subparsers.add_parser("search-claims", help="Search claims in the local SQLite index.")
    add_search_args(claims_parser)
    claims_parser.add_argument("--claim-type", action="append", default=[], help="Filter by claim type.")
    claims_parser.set_defaults(func=run_search_claims)

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
    (home / "knowledge").mkdir(parents=True, exist_ok=True)
    (home / "indexes").mkdir(parents=True, exist_ok=True)
    if args.sample:
        missing = [name for name in SAMPLE_FILES if not (SAMPLE_SOURCE_DIR / name).exists()]
        if missing:
            print(f"Sample source is incomplete: {', '.join(missing)}", file=sys.stderr)
            return 2
        target_dir = profile_knowledge_dir(home, profile=DEFAULT_PROFILE)
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in SAMPLE_FILES:
            source = SAMPLE_SOURCE_DIR / name
            target = target_dir / name
            if target.exists() and not args.force:
                continue
            shutil.copy2(source, target)
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
    home = resolve_aika_home(args.home or None)
    profile = str(args.profile or DEFAULT_PROFILE)
    knowledge_dir = profile_knowledge_dir(home, profile=profile)
    index_path = profile_index_path(home, profile=profile)
    fts = sqlite_fts_status()
    index = inspect_sqlite_index(index_path)

    checks = [
        ("home", home.exists(), str(home)),
        ("config", (home / "config.toml").exists(), str(home / "config.toml")),
        ("knowledge", knowledge_dir.exists(), str(knowledge_dir)),
        ("sample_files", all((knowledge_dir / name).exists() for name in SAMPLE_FILES), str(knowledge_dir)),
        ("sqlite_fts5", bool(fts["fts5"]), f"sqlite={fts['sqlite_version']} tokenizer={fts.get('tokenizer') or '-'}"),
        ("index", bool(index["exists"]) and not index["error"], str(index_path)),
    ]
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
    if index["counts"]:
        count_text = ", ".join(f"{key}={value}" for key, value in sorted(index["counts"].items()))
        print(f"Index counts: {count_text}")
    if index["metadata"]:
        print(f"Index tokenizer: {index['metadata'].get('tokenizer', '')}")
    if fts["error"]:
        print(f"SQLite error: {fts['error']}", file=sys.stderr)
    if index["error"]:
        print(f"Index error: {index['error']}", file=sys.stderr)
    return 0 if all(ok for _, ok, _ in checks) else 1


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
