#!/usr/bin/env python3
"""Build the lightweight local RAG index from chunk JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.rag_index import DEFAULT_RAG_DIR, build_rag_index
from src.text_cleaner import CHUNKS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local RAG index from data/chunks JSONL files.")
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RAG_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = build_rag_index(args.chunks_dir, args.output_dir)
    print(f"Wrote RAG index with {metadata.chunk_count} chunks -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
