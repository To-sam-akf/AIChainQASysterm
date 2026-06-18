"""Evidence-gap audit entrypoints for AIKA Core."""

from __future__ import annotations

from collections.abc import Iterable

from aika.aika_core.backends.csv_backend import CSVResearchBackend
from aika.aika_core.models import EvidenceGap, ResearchBackend


def audit_evidence_gaps(
    query: str,
    *,
    companies: Iterable[str] | None = None,
    topic: str = "",
    backend: ResearchBackend | None = None,
) -> list[EvidenceGap]:
    active_backend = backend or CSVResearchBackend.from_env()
    if not hasattr(active_backend, "audit_evidence_gaps"):
        raise NotImplementedError("Backend does not expose audit_evidence_gaps.")
    return active_backend.audit_evidence_gaps(query, companies=companies, topic=topic)  # type: ignore[attr-defined]
