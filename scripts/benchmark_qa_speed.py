#!/usr/bin/env python3
"""Benchmark local QA retrieval speed without remote LLM latency."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.curated_graph import DEFAULT_CURATED_DIR
from src.frontend_data import LocalKnowledgeGraph
from src.qa_engine import QAEngine
from src.rag_index import DEFAULT_RAG_DIR, LocalRagIndex


QUESTIONS = [
    "液冷产业链有哪些上市公司，各自处于什么环节？",
    "中际旭创和新易盛在光模块业务上的差异是什么？",
    "AI算力产业链当前最大的瓶颈是什么？",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local KG/RAG QA speed.")
    parser.add_argument("--kg-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--rag-dir", type=Path, default=DEFAULT_RAG_DIR)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print JSON lines instead of a compact table.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    graph = LocalKnowledgeGraph.from_dir(args.kg_dir)
    rag = LocalRagIndex.load(args.rag_dir)
    engine = QAEngine(
        csv_graph=graph,
        rag_index=rag,
        llm_client=None,
        enable_llm_cypher=False,
        enable_llm_planner=False,
    )

    rows = []
    for question in QUESTIONS:
        for run in range(1, max(args.repeat, 1) + 1):
            started_at = time.perf_counter()
            result = engine.answer_question(question)
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
            timings = result["diagnostics"].get("timings_ms", {})
            rows.append(
                {
                    "run": run,
                    "question": question,
                    "elapsed_ms": elapsed_ms,
                    "graph_ms": timings.get("graph", 0),
                    "rag_ms": timings.get("rag", 0),
                    "total_ms": timings.get("total", 0),
                    "graph_records": len(result["graph_records"]),
                    "rag_hits": len(result["rag_hits"]),
                    "llm_calls": result["diagnostics"].get("llm_calls", {}).get("total", 0),
                }
            )

    if args.json:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        return 0

    for row in rows:
        print(
            f"run={row['run']} total={row['total_ms']:>7}ms graph={row['graph_ms']:>6}ms "
            f"rag={row['rag_ms']:>6}ms llm={row['llm_calls']} q={row['question']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
