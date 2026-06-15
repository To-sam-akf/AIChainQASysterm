#!/usr/bin/env python3
"""One-time import of the existing committed retrieval artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.extraction_schema import load_jsonl, read_csv
from src.llm_client import load_dotenv
from src.postgres_retrieval import PostgresRetrievalStore
from src.rag_index import DEFAULT_RAG_DIR, LocalRagIndex
from src.research_claims import (
    CLAIMS_FILE,
    DEFAULT_RESEARCH_DIR,
    SEGMENT_DOSSIERS_FILE,
    load_claim_reviews,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap PostgreSQL from legacy retrieval artifacts.")
    parser.add_argument("--rag-dir", type=Path, default=DEFAULT_RAG_DIR)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    rag = LocalRagIndex.load(args.rag_dir)
    claims = read_csv(args.research_dir / CLAIMS_FILE)
    reviews = list(load_claim_reviews(args.research_dir).values())
    dossiers = load_jsonl(args.research_dir / SEGMENT_DOSSIERS_FILE)
    store = PostgresRetrievalStore.from_env()
    try:
        store.ensure_ready()
        rag_result = store.sync_rag_documents(rag.documents)
        research_result = store.sync_research(claims, dossiers)
        imported_reviews = store.import_claim_reviews(reviews)
        if reviews:
            research_result = store.sync_research(claims, dossiers)
    finally:
        store.close()
    print(
        f"Bootstrapped rag={rag_result['record_count']} claims={research_result['claims']} "
        f"dossiers={research_result['dossiers']} reviews={imported_reviews}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
