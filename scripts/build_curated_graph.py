#!/usr/bin/env python3
"""Build the professional curated graph CSVs used by the QA system."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.curated_graph import DEFAULT_CURATED_DIR, build_curated_graph
from src.frontend_data import DEFAULT_ENTITIES_CSV, DEFAULT_RELATIONS_CSV
from src.research_claims import build_research_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build curated professional KG CSVs from verified graph CSVs.")
    parser.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES_CSV)
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CURATED_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    entities, relations = build_curated_graph(
        entities_csv=args.entities,
        relations_csv=args.relations,
        output_dir=args.output_dir,
    )
    claims, evidence_spans, dossiers = build_research_artifacts(
        relations_csv=args.output_dir / "relations.csv",
        output_dir=args.output_dir,
    )
    print(
        f"Wrote curated entities={len(entities)} relations={len(relations)} "
        f"claims={len(claims)} evidence_spans={len(evidence_spans)} dossiers={len(dossiers)} -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
