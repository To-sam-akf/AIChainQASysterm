"""Company profile entrypoints for AIKA Core."""

from __future__ import annotations

from aika.aika_core.backends.csv_backend import CSVResearchBackend
from aika.aika_core.models import CompanyProfile, ResearchBackend


def get_company_profile(
    company: str,
    *,
    topic: str = "",
    backend: ResearchBackend | None = None,
) -> CompanyProfile:
    active_backend = backend or CSVResearchBackend.from_env()
    return active_backend.get_company_profile(company, topic=topic)
