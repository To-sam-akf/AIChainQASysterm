#!/usr/bin/env python3
"""Run LLM extraction over chunk JSONL files."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.extraction_schema import append_jsonl, load_jsonl
from src.llm_client import MockLLMClient, OpenAICompatibleClient
from src.llm_extractor import extract_from_chunk
from src.text_cleaner import CHUNKS_DIR


DEFAULT_OUTPUT = ROOT_DIR / "data" / "extracted" / "extractions.jsonl"
DEFAULT_ERRORS = ROOT_DIR / "data" / "extracted" / "extraction_errors.csv"


def iter_chunks(input_dir: Path) -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        chunks.extend(load_jsonl(path))
    return chunks


def filter_chunks(chunks: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.kind:
        chunks = [chunk for chunk in chunks if chunk.get("kind") == args.kind]
    if args.report_id:
        wanted = set(args.report_id)
        chunks = [chunk for chunk in chunks if chunk.get("report_id") in wanted]
    if args.contains:
        terms = args.contains
        chunks = [chunk for chunk in chunks if any(term in chunk.get("text", "") for term in terms)]
    return chunks


def load_done_chunk_ids(output_path: Path) -> set[str]:
    return {record.get("chunk_id", "") for record in load_jsonl(output_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract KG entities and relations from chunks.")
    parser.add_argument("--input-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--limit-chunks", type=int, default=0)
    parser.add_argument("--kind", choices=("annual", "research"))
    parser.add_argument("--report-id", action="append", help="Only extract chunks from this report_id; can be repeated.")
    parser.add_argument("--contains", action="append", help="Only extract chunks containing this text; can be repeated.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between LLM calls.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock extraction instead of an API call.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        args.output.unlink(missing_ok=True)
        args.errors.unlink(missing_ok=True)
    client = MockLLMClient() if args.mock else OpenAICompatibleClient()
    chunks = filter_chunks(iter_chunks(args.input_dir), args)
    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]
    done = load_done_chunk_ids(args.output) if args.resume else set()
    errors = []
    extracted = 0
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id in done:
            continue
        try:
            result = extract_from_chunk(chunk, client)
            append_jsonl(args.output, result)
            extracted += 1
            print(f"OK   {chunk_id} {chunk.get('source_title', '')} p{chunk.get('page', '')}")
            if args.sleep:
                time.sleep(args.sleep)
        except Exception as exc:
            errors.append({"chunk_id": chunk_id, "report_id": chunk.get("report_id", ""), "error": str(exc)})
            print(f"FAIL {chunk_id}: {exc}")
    if errors:
        with args.errors.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["chunk_id", "report_id", "error"])
            writer.writeheader()
            writer.writerows(errors)
    else:
        args.errors.parent.mkdir(parents=True, exist_ok=True)
        with args.errors.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["chunk_id", "report_id", "error"])
            writer.writeheader()
    print(f"Extracted chunks: {extracted}; errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
