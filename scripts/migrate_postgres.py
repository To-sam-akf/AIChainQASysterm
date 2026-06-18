#!/usr/bin/env python3
"""Apply the PostgreSQL retrieval schema migrations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from aika.llm_client import load_dotenv
from aika.postgres_retrieval import migrate_database


def main() -> int:
    load_dotenv()
    applied = migrate_database()
    print("Applied migrations: " + (", ".join(applied) if applied else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
