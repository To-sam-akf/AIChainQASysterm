"""Research brief entrypoints for AIKA Core."""

from __future__ import annotations

from aika.aika_core.backends.csv_backend import CSVResearchBackend
from aika.aika_core.models import ResearchBackend, ResearchBrief


def build_research_brief(
    query: str,
    *,
    topic: str = "",
    backend: ResearchBackend | None = None,
) -> ResearchBrief:
    active_backend = backend or CSVResearchBackend.from_env()
    if not hasattr(active_backend, "build_research_brief"):
        raise NotImplementedError("Backend does not expose build_research_brief.")
    return active_backend.build_research_brief(query, topic=topic)  # type: ignore[attr-defined]
