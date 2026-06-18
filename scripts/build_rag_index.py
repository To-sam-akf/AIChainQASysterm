#!/usr/bin/env python3
"""Build the lightweight local RAG index from chunk JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from aika.llm_client import load_dotenv
from aika.postgres_retrieval import PostgresRetrievalStore
from aika.rag_index import build_rag_documents
from aika.text_cleaner import CHUNKS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the PostgreSQL RAG corpus from data/chunks JSONL files.")
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    documents = build_rag_documents(args.chunks_dir)
    store = PostgresRetrievalStore.from_env()
    try:
        store.ensure_ready()
        result = store.sync_rag_documents(documents)
    finally:
        store.close()
    print(
        f"Synced RAG chunks={result['record_count']} deleted={result['deleted']} "
        f"corpus_hash={result['corpus_hash']} -> PostgreSQL"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
