"""AIKA MCP server package."""

from __future__ import annotations

from aika.aika_mcp.server import create_server, registered_tool_names, run_server
from aika.aika_mcp.tools import (
    audit_evidence_gaps,
    build_research_brief,
    compare_companies,
    get_company_profile,
    query_industry_graph,
    run_research_task,
    search_claims,
    search_evidence,
    tool_names,
)


__all__ = [
    "audit_evidence_gaps",
    "build_research_brief",
    "compare_companies",
    "create_server",
    "get_company_profile",
    "query_industry_graph",
    "registered_tool_names",
    "run_research_task",
    "run_server",
    "search_claims",
    "search_evidence",
    "tool_names",
]

