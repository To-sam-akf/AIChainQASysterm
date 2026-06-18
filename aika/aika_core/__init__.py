"""AIKA Core lightweight research library.

This package exposes the service-free core used by CLI, MCP tools, and the web
app. The default backend reads local CSV/JSONL artifacts from data/curated.
"""

from __future__ import annotations

from typing import Any

from aika.aika_core.backends.csv_backend import CSVResearchBackend
from aika.aika_core.backends.sqlite_backend import SQLiteResearchBackend
from aika.aika_core.brief import build_research_brief
from aika.aika_core.claims import load_claims, search_claim_records
from aika.aika_core.compare import compare_companies
from aika.aika_core.config import AikaCoreConfig
from aika.aika_core.gaps import audit_evidence_gaps
from aika.aika_core.graph import LocalKnowledgeGraph, load_graph, subgraph_edges
from aika.aika_core.models import (
    ClaimRecord,
    CompanyComparison,
    CompanyProfile,
    EvidenceCard,
    EvidenceGap,
    GraphEdge,
    GraphNode,
    ResearchBackend,
    ResearchBrief,
)
from aika.aika_core.profiles import get_company_profile


def default_backend() -> CSVResearchBackend:
    return CSVResearchBackend.from_env()


def search_evidence(query: str, *, top_k: int = 8, **filters: Any) -> list[EvidenceCard]:
    return default_backend().search_evidence(query, top_k=top_k, **filters)


def search_claims(query: str, *, top_k: int = 8, **filters: Any) -> list[ClaimRecord]:
    return default_backend().search_claims(query, top_k=top_k, **filters)


def query_graph(
    *,
    company: str = "",
    technology: str = "",
    relation_type: str = "",
    limit: int = 80,
) -> list[GraphEdge]:
    return default_backend().query_graph(
        company=company,
        technology=technology,
        relation_type=relation_type,
        limit=limit,
    )


__all__ = [
    "AikaCoreConfig",
    "CSVResearchBackend",
    "ClaimRecord",
    "CompanyComparison",
    "CompanyProfile",
    "EvidenceCard",
    "EvidenceGap",
    "GraphEdge",
    "GraphNode",
    "LocalKnowledgeGraph",
    "ResearchBackend",
    "ResearchBrief",
    "SQLiteResearchBackend",
    "audit_evidence_gaps",
    "build_research_brief",
    "compare_companies",
    "default_backend",
    "get_company_profile",
    "load_claims",
    "load_graph",
    "query_graph",
    "search_claim_records",
    "search_claims",
    "search_evidence",
    "subgraph_edges",
]
