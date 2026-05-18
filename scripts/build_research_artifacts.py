#!/usr/bin/env python3
"""Build research claims, evidence spans, and segment dossiers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.curated_graph import CURATED_RELATIONS_CSV, DEFAULT_CURATED_DIR
from src.research_claims import build_research_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build professional research artifacts from curated KG relations.")
    parser.add_argument("--relations", type=Path, default=CURATED_RELATIONS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CURATED_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    claims, evidence_spans, dossiers = build_research_artifacts(
        relations_csv=args.relations,
        output_dir=args.output_dir,
    )
    print(
        f"Wrote claims={len(claims)} evidence_spans={len(evidence_spans)} "
        f"dossiers={len(dossiers)} -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
