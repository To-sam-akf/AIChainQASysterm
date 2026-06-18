"""Filesystem defaults for AIKA Core."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DEFAULT_CURATED_DIR = DATA_DIR / "curated"
DEFAULT_VERIFIED_DIR = DATA_DIR / "verified"

CLAIMS_FILE = "claims.csv"
EVIDENCE_SPANS_FILE = "evidence_spans.csv"
SEGMENT_DOSSIERS_FILE = "segment_dossiers.jsonl"
ENTITIES_FILE = "entities.csv"
RELATIONS_FILE = "relations.csv"
MANIFEST_FILE = "manifest.csv"
EXAMPLES_FILE = "examples.jsonl"

DEFAULT_CLAIMS_CSV = DEFAULT_CURATED_DIR / CLAIMS_FILE
DEFAULT_EVIDENCE_SPANS_CSV = DEFAULT_CURATED_DIR / EVIDENCE_SPANS_FILE
DEFAULT_SEGMENT_DOSSIERS_JSONL = DEFAULT_CURATED_DIR / SEGMENT_DOSSIERS_FILE
DEFAULT_ENTITIES_CSV = DEFAULT_CURATED_DIR / ENTITIES_FILE
DEFAULT_RELATIONS_CSV = DEFAULT_CURATED_DIR / RELATIONS_FILE


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    return Path(data_dir).expanduser().resolve() if data_dir else DEFAULT_CURATED_DIR
