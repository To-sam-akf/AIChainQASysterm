"""Company comparison entrypoints for AIKA Core."""

from __future__ import annotations

from collections.abc import Iterable

from src.aika_core.backends.csv_backend import CSVResearchBackend
from src.aika_core.models import CompanyComparison, ResearchBackend


def compare_companies(
    companies: Iterable[str],
    *,
    topic: str = "",
    backend: ResearchBackend | None = None,
) -> CompanyComparison:
    active_backend = backend or CSVResearchBackend.from_env()
    if not hasattr(active_backend, "compare_companies"):
        raise NotImplementedError("Backend does not expose compare_companies.")
    return active_backend.compare_companies(companies, topic=topic)  # type: ignore[attr-defined]
