#!/usr/bin/env python3
"""Build research claims, evidence spans, and segment dossiers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.curated_graph import CURATED_RELATIONS_CSV, DEFAULT_CURATED_DIR
from src.llm_client import load_dotenv
from src.postgres_retrieval import PostgresRetrievalStore
from src.research_claims import build_research_artifacts
from src.text_cleaner import CHUNKS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build professional research artifacts from curated KG relations.")
    parser.add_argument("--relations", type=Path, default=CURATED_RELATIONS_CSV)
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--no-direct-claims", action="store_true", help="Only derive claims from curated KG relations.")
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    claims, evidence_spans, dossiers = build_research_artifacts(
        relations_csv=args.relations,
        output_dir=DEFAULT_CURATED_DIR,
        chunks_dir=None if args.no_direct_claims else args.chunks_dir,
        include_direct_claims=not args.no_direct_claims,
        write_outputs=False,
    )
    store = PostgresRetrievalStore.from_env()
    try:
        store.ensure_ready()
        result = store.sync_research(claims, dossiers)
    finally:
        store.close()
    print(
        f"Synced claims={result['claims']} evidence_spans={len(evidence_spans)} "
        f"dossiers={result['dossiers']} -> PostgreSQL"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
