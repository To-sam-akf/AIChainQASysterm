"""PostgreSQL backend placeholder for professional AIKA Core deployments."""

from __future__ import annotations

from typing import Any

from aika.aika_core.models import ClaimRecord, CompanyProfile, EvidenceCard, GraphEdge, ResearchBackend


class PostgresResearchBackend(ResearchBackend):
    """Interface stub for adapting the existing professional PostgreSQL store."""

    def __init__(self, retrieval_store: Any | None = None) -> None:
        self.retrieval_store = retrieval_store

    def search_evidence(self, query: str, *, top_k: int = 8, **filters: Any) -> list[EvidenceCard]:
        raise NotImplementedError("PostgreSQL backend adapter will wrap PostgresRetrievalStore in a later phase.")

    def search_claims(self, query: str, *, top_k: int = 8, **filters: Any) -> list[ClaimRecord]:
        raise NotImplementedError("PostgreSQL backend adapter will wrap PostgresRetrievalStore in a later phase.")

    def query_graph(
        self,
        *,
        company: str = "",
        technology: str = "",
        relation_type: str = "",
        limit: int = 80,
    ) -> list[GraphEdge]:
        raise NotImplementedError("PostgreSQL backend adapter will wrap PostgresRetrievalStore in a later phase.")

    def get_company_profile(self, company: str, *, topic: str = "") -> CompanyProfile:
        raise NotImplementedError("PostgreSQL backend adapter will wrap PostgresRetrievalStore in a later phase.")
