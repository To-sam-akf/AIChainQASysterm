"""Bundled sample-data helpers for the public AIKA install path."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from aika.aika_core.data_paths import (
    CLAIMS_FILE,
    ENTITIES_FILE,
    EVIDENCE_SPANS_FILE,
    RELATIONS_FILE,
    SEGMENT_DOSSIERS_FILE,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
REPO_SAMPLE_DIR = ROOT_DIR / "data" / "curated"
RESOURCE_PACKAGE = "aika.aika_core"
RESOURCE_SAMPLE_DIR = "bundled_sample"
SAMPLE_FILES = [
    ENTITIES_FILE,
    RELATIONS_FILE,
    CLAIMS_FILE,
    EVIDENCE_SPANS_FILE,
    SEGMENT_DOSSIERS_FILE,
]


@dataclass(frozen=True)
class SampleSourceStatus:
    available: bool
    source: str
    missing: list[str]


def resolve_sample_source(preferred: str | Path | None = None) -> Path | Traversable | None:
    """Return the first complete sample-data source for this install."""
    for candidate in _candidate_sources(preferred):
        if not missing_sample_files(candidate):
            return candidate
    return None


def sample_source_status(preferred: str | Path | None = None) -> SampleSourceStatus:
    """Describe bundled sample availability without creating any files."""
    best_source = ""
    best_missing = list(SAMPLE_FILES)
    for candidate in _candidate_sources(preferred):
        missing = missing_sample_files(candidate)
        if not missing:
            return SampleSourceStatus(True, _source_label(candidate), [])
        if len(missing) < len(best_missing):
            best_source = _source_label(candidate)
            best_missing = missing
    return SampleSourceStatus(False, best_source or "not found", best_missing)


def copy_sample_files(
    target_dir: str | Path,
    *,
    source: str | Path | Traversable | None = None,
    force: bool = False,
) -> SampleSourceStatus:
    """Copy bundled sample files into a profile knowledge directory."""
    resolved_source = source or resolve_sample_source()
    if resolved_source is None:
        return sample_source_status()
    missing = missing_sample_files(resolved_source)
    if missing:
        return SampleSourceStatus(False, _source_label(resolved_source), missing)

    target = Path(target_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    for name in SAMPLE_FILES:
        output_path = target / name
        if output_path.exists() and not force:
            continue
        _copy_file(resolved_source.joinpath(name), output_path)
    return SampleSourceStatus(True, _source_label(resolved_source), [])


def missing_sample_files(source: str | Path | Traversable) -> list[str]:
    resolved = Path(source).expanduser().resolve() if isinstance(source, (str, Path)) else source
    return [name for name in SAMPLE_FILES if not resolved.joinpath(name).is_file()]


def _candidate_sources(preferred: str | Path | None = None) -> list[Path | Traversable]:
    candidates: list[Path | Traversable] = []
    if preferred:
        candidates.append(Path(preferred).expanduser().resolve())
    candidates.append(resources.files(RESOURCE_PACKAGE).joinpath(RESOURCE_SAMPLE_DIR))
    candidates.append(REPO_SAMPLE_DIR)
    return candidates


def _copy_file(source: Traversable, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, target.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle)


def _source_label(source: str | Path | Traversable) -> str:
    if isinstance(source, (str, Path)):
        return str(Path(source).expanduser().resolve())
    return str(source)
