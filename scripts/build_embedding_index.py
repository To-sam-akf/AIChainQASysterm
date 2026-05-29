"""Build the local JSONL embedding index for semantic recall."""

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
from src.rag_index import DEFAULT_RAG_DIR
from src.research_claims import DEFAULT_RESEARCH_DIR
from src.semantic_index import DEFAULT_SEMANTIC_DIR, build_semantic_documents, build_semantic_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local semantic embedding index.")
    parser.add_argument("--rag-dir", type=Path, default=None, help="RAG index directory containing documents.jsonl.")
    parser.add_argument("--research-dir", type=Path, default=None, help="Research artifacts directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Semantic index output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Index at most this many semantic documents.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override EMBEDDING_BATCH_SIZE.")
    parser.add_argument("--dry-run", action="store_true", help="Only count semantic documents; do not call embeddings or write files.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    rag_dir = args.rag_dir or Path(os.getenv("RAG_INDEX_DIR", str(DEFAULT_RAG_DIR)))
    research_dir = args.research_dir or Path(os.getenv("RESEARCH_ARTIFACT_DIR", str(DEFAULT_RESEARCH_DIR)))
    output_dir = args.output_dir or Path(os.getenv("EMBEDDING_INDEX_DIR", str(DEFAULT_SEMANTIC_DIR)))
    documents = build_semantic_documents(rag_dir=rag_dir, research_dir=research_dir, limit=args.limit)

    print(f"semantic_documents={len(documents)}")
    print(f"rag_dir={rag_dir}")
    print(f"research_dir={research_dir}")
    print(f"output_dir={output_dir}")
    if args.dry_run:
        return

    def show_progress(done: int, total: int) -> None:
        print(f"embedded_texts={done}/{total}", flush=True)

    client = OpenAICompatibleEmbeddingClient(batch_size=args.batch_size)
    metadata = build_semantic_index(
        documents=documents,
        embedding_client=client,
        output_dir=output_dir,
        rag_dir=rag_dir,
        research_dir=research_dir,
        progress_callback=show_progress,
    )
    print(f"vector_count={metadata.vector_count}")
    print(f"dimension={metadata.dimension}")
    print(f"embedding_model={metadata.embedding_model}")


if __name__ == "__main__":
    main()
