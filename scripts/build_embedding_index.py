"""Build or incrementally refresh PostgreSQL embeddings."""

from __future__ import annotations

import argparse
import os
import sys

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.embedding_client import OpenAICompatibleEmbeddingClient
from src.llm_client import load_dotenv
from src.postgres_retrieval import EMBEDDING_DIMENSIONS, PostgresRetrievalStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PostgreSQL semantic embeddings.")
    parser.add_argument("--limit", type=int, default=None, help="Embed at most this many records.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override EMBEDDING_BATCH_SIZE.")
    parser.add_argument("--force", action="store_true", help="Rebuild every embedding.")
    parser.add_argument("--dry-run", action="store_true", help="Only count records that would be embedded.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    configured_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", str(EMBEDDING_DIMENSIONS)))
    if configured_dimensions != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"EMBEDDING_DIMENSIONS must be {EMBEDDING_DIMENSIONS} for PostgreSQL retrieval"
        )
    store = PostgresRetrievalStore.from_env()
    try:
        store.ensure_ready()
        rows = store.pending_embeddings(force=args.force, limit=args.limit)
        print(f"semantic_documents={len(rows)}")
        print(f"embedding_dimensions={EMBEDDING_DIMENSIONS}")
        if args.dry_run or not rows:
            return
        client = OpenAICompatibleEmbeddingClient(
            batch_size=args.batch_size,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        completed = 0
        batch_size = int(args.batch_size or client.batch_size)
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            try:
                vectors = client.embed_texts([str(row["semantic_text"]) for row in batch])
                store.update_embeddings(batch, vectors, model=client.model)
            except Exception:
                for row in batch:
                    store.mark_embedding_failed(row)
                store.record_embedding_build(model=client.model, count=completed, status="failed")
                raise
            completed += len(batch)
            print(f"embedded_texts={completed}/{len(rows)}", flush=True)
        store.record_embedding_build(model=client.model, count=completed)
        print(f"vector_count={completed}")
        print(f"embedding_model={client.model}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
