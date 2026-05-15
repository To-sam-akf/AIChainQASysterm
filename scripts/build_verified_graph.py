#!/usr/bin/env python3
"""Build reviewable entity and relation CSVs from LLM extraction JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.graph_builder import build_verified_graph


DEFAULT_MANIFEST = ROOT_DIR / "data" / "metadata" / "reports_manifest.csv"
DEFAULT_EXTRACTIONS = ROOT_DIR / "data" / "extracted" / "extractions.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "verified"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build verified entity/relation CSVs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entities_csv = args.output_dir / "entities.csv"
    relations_csv = args.output_dir / "relations.csv"
    entity_rows, relation_rows = build_verified_graph(
        extraction_paths=[args.extracted],
        manifest_path=args.manifest,
        entities_csv=entities_csv,
        relations_csv=relations_csv,
    )
    print(f"Wrote {len(entity_rows)} entities -> {entities_csv}")
    print(f"Wrote {len(relation_rows)} relations -> {relations_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
