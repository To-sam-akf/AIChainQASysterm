#!/usr/bin/env python3
"""Validate PostgreSQL corpus counts, embeddings, and retrieval quality gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from aika.embedding_client import OpenAICompatibleEmbeddingClient
from aika.eval.rag_dataset import DEFAULT_RAG_RETRIEVAL_BENCHMARK
from aika.eval.rag_runner import run_rag_retrieval_benchmark
from aika.eval.store import EvalRunStore
from aika.llm_client import load_dotenv
from aika.postgres_retrieval import (
    EMBEDDING_DIMENSIONS,
    PostgresRagIndex,
    PostgresRetrievalStore,
    PostgresSemanticIndex,
)


BASELINE_RUN_ID = "rag_eval_20260614_160354_be36b613"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-rag", type=int, default=6189)
    parser.add_argument("--expected-claims", type=int, default=568)
    parser.add_argument("--expected-dossiers", type=int, default=9)
    parser.add_argument("--expected-vectors", type=int, default=6766)
    parser.add_argument("--run-eval", action="store_true", help="Run BM25, semantic, and RRF quality gates.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_RAG_RETRIEVAL_BENCHMARK)
    parser.add_argument("--report-dir", type=Path, default=ROOT_DIR / "data" / "eval_runs")
    parser.add_argument("--bm25-min", type=float, default=0.51)
    parser.add_argument("--semantic-min", type=float, default=0.85)
    parser.add_argument("--rrf-min", type=float, default=0.85)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    store = PostgresRetrievalStore.from_env()
    try:
        store.ensure_ready()
        counts = retrieval_counts(store)
        expected = {
            "rag": args.expected_rag,
            "claims": args.expected_claims,
            "dossiers": args.expected_dossiers,
            "vectors": args.expected_vectors,
        }
        failures = [
            f"{key}: expected {expected[key]}, got {counts[key]}"
            for key in expected
            if counts[key] != expected[key]
        ]
        report: dict[str, Any] = {
            "baseline_run_id": BASELINE_RUN_ID,
            "counts": counts,
            "expected": expected,
            "count_failures": failures,
            "quality_gates": {},
        }
        if not failures and args.run_eval:
            report["quality_gates"] = run_quality_gates(store, args)
            failures.extend(report["quality_gates"]["failures"])
        report["passed"] = not failures
        report["failures"] = failures
    finally:
        store.close()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"PostgreSQL corpus: rag={counts['rag']} claims={counts['claims']} "
            f"dossiers={counts['dossiers']} vectors={counts['vectors']}"
        )
        quality = report.get("quality_gates") or {}
        for retriever, recall in quality.get("recall_at_6", {}).items():
            print(f"{retriever} recall@6={recall:.1%}")
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print("Cutover validation passed." if not failures else "Cutover validation failed.")
    return 0 if not failures else 1


def retrieval_counts(store: PostgresRetrievalStore) -> dict[str, int]:
    with store.pool.connection() as connection:
        row = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM rag_chunks) AS rag,
                (SELECT count(*) FROM research_claims) AS claims,
                (SELECT count(*) FROM segment_dossiers) AS dossiers,
                (SELECT count(*) FROM rag_chunks
                 WHERE embedding_status = 'ready' AND embedding IS NOT NULL) +
                (SELECT count(*) FROM research_claims
                 WHERE embedding_status = 'ready' AND embedding IS NOT NULL) +
                (SELECT count(*) FROM segment_dossiers
                 WHERE embedding_status = 'ready' AND embedding IS NOT NULL) AS vectors
            """
        ).fetchone()
    assert row is not None
    return {key: int(row[key] or 0) for key in ("rag", "claims", "dossiers", "vectors")}


def run_quality_gates(store: PostgresRetrievalStore, args: argparse.Namespace) -> dict[str, Any]:
    embedding_client = OpenAICompatibleEmbeddingClient(dimensions=EMBEDDING_DIMENSIONS)
    report = run_rag_retrieval_benchmark(
        PostgresRagIndex(store),
        benchmark_path=args.benchmark,
        semantic_index=PostgresSemanticIndex(store, embedding_client=embedding_client),
        retrievers=("bm25", "semantic", "rrf"),
        store=EvalRunStore(args.report_dir),
        save=not args.no_save,
        review_dir=args.report_dir if not args.no_save else None,
    )
    recall_at_6 = {
        retriever: float(row.get("primary", {}).get("recall") or 0.0)
        for retriever, row in report["summary"]["by_retriever"].items()
    }
    thresholds = {
        "bm25": float(args.bm25_min),
        "semantic": float(args.semantic_min),
        "rrf": float(args.rrf_min),
    }
    failures = [
        f"{retriever} recall@6 below gate: {recall_at_6.get(retriever, 0.0):.4f} < {minimum:.4f}"
        for retriever, minimum in thresholds.items()
        if recall_at_6.get(retriever, 0.0) < minimum
    ]
    return {
        "run_id": report["run_id"],
        "recall_at_6": recall_at_6,
        "thresholds": thresholds,
        "failures": failures,
    }


if __name__ == "__main__":
    raise SystemExit(main())
