#!/usr/bin/env python3
"""Load verified graph CSVs into Neo4j."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.kg_loader import Neo4jGraphLoader, validate_graph_csvs


DEFAULT_ENTITIES = ROOT_DIR / "data" / "verified" / "entities.csv"
DEFAULT_RELATIONS = ROOT_DIR / "data" / "verified" / "relations.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load verified KG CSVs into Neo4j.")
    parser.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS)
    parser.add_argument("--clear", action="store_true", help="Clear project graph labels before loading.")
    parser.add_argument("--dry-run", action="store_true", help="Validate CSVs and print counts without connecting to Neo4j.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        entity_count, relation_count = validate_graph_csvs(args.entities, args.relations)
        print(f"Validated entities={entity_count}, relations={relation_count}")
        return 0
    loader = Neo4jGraphLoader()
    try:
        entity_count, relation_count = loader.load_graph(args.entities, args.relations, clear=args.clear)
    finally:
        loader.close()
    print(f"Loaded entities={entity_count}, relations={relation_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
