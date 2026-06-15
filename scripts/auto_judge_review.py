#!/usr/bin/env python3
"""Run LLM auto-judgment on an unjudged review queue produced by rag_runner.

Usage:
    python scripts/auto_judge_review.py \\
        --input data/eval_runs/rag_eval_xxx_unjudged.jsonl \\
        --dataset data/eval/rag_retrieval_v1.jsonl \\
        --output data/eval_runs/auto_judgments.jsonl \\
        --batch-size 8

    # Dry-run (no LLM calls) to inspect what would be judged:
    python scripts/auto_judge_review.py \\
        --input data/eval_runs/rag_eval_xxx_unjudged.jsonl \\
        --dataset data/eval/rag_retrieval_v1.jsonl \\
        --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.eval.auto_judge import (  # noqa: E402
    AutoJudgeConfig,
    auto_judge_review_queue,
    dedupe_review_rows,
    load_review_rows,
    load_judgments,
    referenced_chunk_ids_from_cases,
    save_judgments,
)
from src.eval.rag_dataset import (  # noqa: E402
    DEFAULT_RAG_RETRIEVAL_BENCHMARK,
    load_rag_retrieval_cases,
)
from src.llm_client import OpenAICompatibleClient, load_dotenv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-judge unjudged retrieval chunks using an LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the unjudged review queue JSONL (from rag_runner)",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=str(DEFAULT_RAG_RETRIEVAL_BENCHMARK),
        help="Path to the evaluation dataset JSONL (default: data/eval/rag_retrieval_v1.jsonl)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/eval_runs/auto_judgments.jsonl",
        help="Path to write the auto judgments JSONL",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Max chunks per LLM batch call (default: 8)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence below which needs_review is set (default: 0.7)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds delay between batches (default: 0.5)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max LLM call retries on failure (default: 2)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="LLM model override (default: from env LLM_MODEL)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show deduplication stats and exit without calling LLM",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-batch progress output",
    )
    return parser


def progress_print(batch: int, total: int, case_id: str, *, quiet: bool = False) -> None:
    if quiet:
        return
    print(f"  [{batch}/{total}] judged case {case_id}")


def main() -> None:
    load_dotenv()

    args = build_parser().parse_args()
    input_path = Path(args.input)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    # -- Load dataset cases --------------------------------------------------
    print(f"Loading dataset: {dataset_path}")
    cases = load_rag_retrieval_cases(dataset_path)
    print(f"  cases: {len(cases)}")

    # -- Load review rows ----------------------------------------------------
    print(f"Loading review queue: {input_path}")
    raw_rows = load_review_rows(input_path)
    print(f"  raw unjudged rows: {len(raw_rows)}")

    known_ids = referenced_chunk_ids_from_cases(cases)
    deduped = dedupe_review_rows(raw_rows, known_chunk_ids=known_ids)
    print(f"  after deduplication: {len(deduped)} unique (case, chunk) pairs")
    print(f"  already-judged chunk_ids in dataset: {len(known_ids)}")

    # Summary by grade expectation
    case_counts: dict[str, int] = {}
    for row in deduped:
        case_counts[row.case_id] = case_counts.get(row.case_id, 0) + 1
    print(f"  cases with unjudged chunks: {len(case_counts)}")
    if case_counts:
        print(f"  avg unjudged per case: {sum(case_counts.values()) / len(case_counts):.1f}")

    if not deduped:
        print("No unjudged chunks to process. Exiting.")
        return

    if args.dry_run:
        print("\n[Dry-run mode] No LLM calls made.")
        print(f"Would judge {len(deduped)} chunks in approx "
              f"{(len(deduped) + args.batch_size - 1) // args.batch_size} batches.")
        return

    # -- Configure LLM client ------------------------------------------------
    print("\nInitializing LLM client...")
    try:
        llm_client = OpenAICompatibleClient(model=args.model or None)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print("Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL in .env")
        sys.exit(1)

    print(f"  model: {llm_client.model}")
    print(f"  base_url: {llm_client.base_url}")

    # -- Run auto-judgment ---------------------------------------------------
    config = AutoJudgeConfig(
        model=args.model or llm_client.model,
        temperature=0.0,
        max_chunks_per_batch=args.batch_size,
        batch_delay_seconds=args.delay,
        low_confidence_threshold=args.confidence_threshold,
        max_retries=args.max_retries,
    )

    print(f"\nRunning auto-judgment...")
    print(f"  batches: ~{(len(deduped) + config.max_chunks_per_batch - 1) // config.max_chunks_per_batch}")
    print(f"  batch size: {config.max_chunks_per_batch}")
    print(f"  confidence threshold: {config.low_confidence_threshold}")

    def cb(batch: int, total: int, case_id: str) -> None:
        progress_print(batch, total, case_id, quiet=args.quiet)

    judgments = auto_judge_review_queue(
        llm_client,
        deduped,
        cases,
        config=config,
        progress_callback=cb,
    )

    # -- Save results ---------------------------------------------------------
    save_judgments(judgments, output_path)
    print(f"\nSaved {len(judgments)} judgments to {output_path}")

    # -- Summary stats --------------------------------------------------------
    grade_counts = {0: 0, 1: 0, 2: 0}
    needs_review_count = 0
    confidence_sum = 0.0
    for j in judgments:
        grade_counts[j.grade] = grade_counts.get(j.grade, 0) + 1
        if j.needs_review:
            needs_review_count += 1
        confidence_sum += j.confidence

    total = len(judgments)
    print(f"\nJudgment summary ({total} chunks):")
    print(f"  grade=0 (irrelevant):     {grade_counts[0]:4d}  ({grade_counts[0]/total*100:5.1f}%)")
    print(f"  grade=1 (partial):        {grade_counts[1]:4d}  ({grade_counts[1]/total*100:5.1f}%)")
    print(f"  grade=2 (direct support): {grade_counts[2]:4d}  ({grade_counts[2]/total*100:5.1f}%)")
    print(f"  needs human review:       {needs_review_count:4d}  ({needs_review_count/total*100:5.1f}%)")
    print(f"  avg confidence:           {confidence_sum/total:.3f}" if total else "  avg confidence: N/A")

    # -- Next steps hint ------------------------------------------------------
    print(f"\nNext steps:")
    print(f"  1. Review low-confidence judgments marked 'needs_review'")
    print(f"  2. Run: python scripts/apply_judgments.py \\")
    print(f"       --judgments {output_path} \\")
    print(f"       --dataset {dataset_path} \\")
    print(f"       --output data/eval/rag_retrieval_v2.jsonl")
    print(f"  3. Re-run evaluation with the augmented dataset")


if __name__ == "__main__":
    main()