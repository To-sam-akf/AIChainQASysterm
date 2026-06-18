"""Pure Python handlers behind the AIKA MCP server."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from src.aika_core.backends.csv_backend import CSVResearchBackend
from src.aika_core.backends.sqlite_backend import (
    DEFAULT_PROFILE,
    SQLiteResearchBackend,
    profile_index_path,
    resolve_aika_home,
)
from src.aika_core.models import ClaimRecord, EvidenceCard, ResearchBackend
from src.aika_mcp.schemas import (
    AikaMcpRequest,
    AuditEvidenceGapsRequest,
    BuildResearchBriefRequest,
    CompareCompaniesRequest,
    CompanyProfileRequest,
    FilteredSearchRequest,
    QueryIndustryGraphRequest,
    RunResearchTaskRequest,
    SearchClaimsRequest,
    SearchEvidenceRequest,
)


RequestT = TypeVar("RequestT", bound=AikaMcpRequest)

DEPTH_LIMITS = {
    "light": {"claims": 4, "evidence": 6, "graph": 20},
    "standard": {"claims": 8, "evidence": 12, "graph": 40},
    "deep": {"claims": 12, "evidence": 18, "graph": 80},
}


@dataclass(frozen=True)
class BackendContext:
    backend: ResearchBackend
    meta: dict[str, Any]
    warnings: list[str]


def search_evidence(payload: Mapping[str, Any] | SearchEvidenceRequest | None = None, **kwargs: Any) -> dict[str, Any]:
    return _run_tool("search_evidence", SearchEvidenceRequest, payload, kwargs, _search_evidence)


def search_claims(payload: Mapping[str, Any] | SearchClaimsRequest | None = None, **kwargs: Any) -> dict[str, Any]:
    return _run_tool("search_claims", SearchClaimsRequest, payload, kwargs, _search_claims)


def get_company_profile(
    payload: Mapping[str, Any] | CompanyProfileRequest | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _run_tool("get_company_profile", CompanyProfileRequest, payload, kwargs, _get_company_profile)


def compare_companies(
    payload: Mapping[str, Any] | CompareCompaniesRequest | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _run_tool("compare_companies", CompareCompaniesRequest, payload, kwargs, _compare_companies)


def query_industry_graph(
    payload: Mapping[str, Any] | QueryIndustryGraphRequest | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _run_tool("query_industry_graph", QueryIndustryGraphRequest, payload, kwargs, _query_industry_graph)


def build_research_brief(
    payload: Mapping[str, Any] | BuildResearchBriefRequest | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _run_tool("build_research_brief", BuildResearchBriefRequest, payload, kwargs, _build_research_brief)


def audit_evidence_gaps(
    payload: Mapping[str, Any] | AuditEvidenceGapsRequest | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _run_tool("audit_evidence_gaps", AuditEvidenceGapsRequest, payload, kwargs, _audit_evidence_gaps)


def run_research_task(
    payload: Mapping[str, Any] | RunResearchTaskRequest | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _run_tool("run_research_task", RunResearchTaskRequest, payload, kwargs, _run_research_task)


def resolve_backend(request: AikaMcpRequest) -> BackendContext:
    warnings: list[str] = []
    mode = request.backend or "auto"
    profile = request.profile or DEFAULT_PROFILE
    if mode in {"auto", "sqlite"}:
        home = resolve_aika_home(request.home)
        index_path = profile_index_path(home, profile=profile)
        if index_path.exists() or mode == "sqlite":
            if not index_path.exists():
                warnings.append(f"SQLite index not found: {index_path}")
            return BackendContext(
                backend=SQLiteResearchBackend(index_path),
                meta={
                    "backend": "sqlite",
                    "backend_mode": mode,
                    "home": str(home),
                    "profile": profile,
                    "index_path": str(index_path),
                    "index_exists": index_path.exists(),
                },
                warnings=warnings,
            )
        warnings.append(f"SQLite index not found, falling back to bundled CSV backend: {index_path}")

    backend = CSVResearchBackend.from_env()
    return BackendContext(
        backend=backend,
        meta={
            "backend": "csv",
            "backend_mode": mode,
            "data_dir": str(backend.config.data_dir),
            "graph_dir": str(backend.config.graph_dir),
            "research_dir": str(backend.config.research_dir),
            "profile": profile,
        },
        warnings=warnings,
    )


def tool_names() -> list[str]:
    return [
        "search_evidence",
        "search_claims",
        "get_company_profile",
        "compare_companies",
        "query_industry_graph",
        "build_research_brief",
        "audit_evidence_gaps",
        "run_research_task",
    ]


def _run_tool(
    tool: str,
    model: type[RequestT],
    payload: Mapping[str, Any] | RequestT | None,
    kwargs: dict[str, Any],
    handler: Callable[[RequestT, BackendContext], dict[str, Any]],
) -> dict[str, Any]:
    try:
        request = _parse_request(model, payload, kwargs)
    except ValidationError as exc:
        return _error_envelope(
            tool,
            meta={"backend": "unresolved", "warnings": []},
            error_type="validation_error",
            message="Invalid MCP tool arguments.",
            details=_jsonable(exc.errors()),
        )
    except ValueError as exc:
        return _error_envelope(
            tool,
            meta={"backend": "unresolved", "warnings": []},
            error_type="validation_error",
            message=str(exc),
            details={},
        )

    context: BackendContext | None = None
    try:
        context = resolve_backend(request)
        return handler(request, context)
    except Exception as exc:  # pragma: no cover - defensive MCP boundary.
        meta = _meta(context) if context is not None else {"backend": "unresolved", "warnings": []}
        return _error_envelope(tool, meta=meta, error_type=exc.__class__.__name__, message=str(exc), details={})


def _parse_request(
    model: type[RequestT],
    payload: Mapping[str, Any] | RequestT | None,
    kwargs: dict[str, Any],
) -> RequestT:
    if isinstance(payload, model):
        data = payload.model_dump()
    elif payload is None:
        data = {}
    elif isinstance(payload, Mapping):
        data = dict(payload)
    else:
        raise ValueError(f"payload must be a mapping or {model.__name__}")
    data.update(kwargs)
    return model.model_validate(data)


def _search_evidence(request: SearchEvidenceRequest, context: BackendContext) -> dict[str, Any]:
    cards = context.backend.search_evidence(request.query, top_k=request.top_k, **_filters(request))
    return _success_envelope(
        "search_evidence",
        context,
        query=request.query,
        evidence_cards=_normalize_cards(cards),
    )


def _search_claims(request: SearchClaimsRequest, context: BackendContext) -> dict[str, Any]:
    claims = context.backend.search_claims(request.query, top_k=request.top_k, **_filters(request))
    return _success_envelope(
        "search_claims",
        context,
        query=request.query,
        claims=[_to_dict(claim) for claim in claims],
        evidence_cards=_normalize_cards(_cards_from_claims(claims)),
    )


def _get_company_profile(request: CompanyProfileRequest, context: BackendContext) -> dict[str, Any]:
    profile = context.backend.get_company_profile(request.company, topic=request.topic)
    payload = _to_dict(profile)
    return _success_envelope(
        "get_company_profile",
        context,
        company_profile=payload,
        evidence_cards=_normalize_cards(payload.get("evidence_cards", [])),
        evidence_gaps=_normalize_gaps(payload.get("evidence_gaps", [])),
    )


def _compare_companies(request: CompareCompaniesRequest, context: BackendContext) -> dict[str, Any]:
    comparison = _require_backend_method(context.backend, "compare_companies")(request.companies, topic=request.topic)
    payload = _to_dict(comparison)
    return _success_envelope(
        "compare_companies",
        context,
        comparison=payload,
        evidence_cards=_normalize_cards(payload.get("evidence_cards", [])),
        evidence_gaps=_normalize_gaps(payload.get("evidence_gaps", [])),
    )


def _query_industry_graph(request: QueryIndustryGraphRequest, context: BackendContext) -> dict[str, Any]:
    edges = context.backend.query_graph(
        company=request.company,
        technology=request.technology,
        relation_type=request.relation_type,
        limit=request.limit,
    )
    return _success_envelope(
        "query_industry_graph",
        context,
        graph_edges=[_to_dict(edge) for edge in edges],
    )


def _build_research_brief(request: BuildResearchBriefRequest, context: BackendContext) -> dict[str, Any]:
    query = request.query or f"{request.topic}投研简报"
    brief = _require_backend_method(context.backend, "build_research_brief")(query, topic=request.topic)
    payload = _to_dict(brief)
    return _success_envelope(
        "build_research_brief",
        context,
        title=payload.get("title", ""),
        report_markdown=payload.get("markdown", ""),
        sections=payload.get("sections", []),
        evidence_cards=_normalize_cards(payload.get("evidence_cards", [])),
        evidence_gaps=_normalize_gaps(payload.get("evidence_gaps", [])),
        research_outputs=payload.get("research_outputs", {}),
    )


def _audit_evidence_gaps(request: AuditEvidenceGapsRequest, context: BackendContext) -> dict[str, Any]:
    gaps = _require_backend_method(context.backend, "audit_evidence_gaps")(
        request.query,
        companies=request.companies,
        topic=request.topic,
    )
    return _success_envelope(
        "audit_evidence_gaps",
        context,
        evidence_gaps=_normalize_gaps(gaps),
    )


def _run_research_task(request: RunResearchTaskRequest, context: BackendContext) -> dict[str, Any]:
    limits = DEPTH_LIMITS[request.depth]
    subject = _subject(request.topic, request.companies)
    filters = {"company": request.companies, "topic": request.topic}
    trace: list[dict[str, Any]] = []

    trace.append(_trace_step("Planner", "planned", {"topic": request.topic, "companies": request.companies, "depth": request.depth}))

    claims = context.backend.search_claims(subject, top_k=limits["claims"], **filters)
    trace.append(_trace_step("Claim Retrieval", "completed", {"result_count": len(claims)}))

    evidence_cards = context.backend.search_evidence(subject, top_k=limits["evidence"], **filters)
    trace.append(_trace_step("Evidence Retrieval", "completed", {"result_count": len(evidence_cards)}))

    graph_edges = _collect_graph_edges(context.backend, topic=request.topic, companies=request.companies, limit=limits["graph"])
    trace.append(_trace_step("Graph Retrieval", "completed", {"result_count": len(graph_edges)}))

    gaps = _require_backend_method(context.backend, "audit_evidence_gaps")(subject, companies=request.companies, topic=request.topic)
    trace.append(_trace_step("Risk/Gaps Audit", "completed", {"result_count": len(gaps)}))

    brief = _require_backend_method(context.backend, "build_research_brief")(subject, topic=request.topic)
    brief_payload = _to_dict(brief)
    trace.append(
        _trace_step(
            "Brief Builder",
            "completed",
            {
                "report_present": bool(brief_payload.get("markdown")),
                "evidence_count": len(brief_payload.get("evidence_cards", []) or []),
            },
        )
    )

    final_cards = _renumber_cards(
        _dedupe_cards(
            [
                *_normalize_cards(brief_payload.get("evidence_cards", [])),
                *_normalize_cards(evidence_cards),
                *_normalize_cards(_cards_from_claims(claims)),
            ]
        )
    )
    final_gaps = _normalize_gaps([*list(brief_payload.get("evidence_gaps", []) or []), *list(gaps or [])])
    verification = _verify_result(
        report_markdown=str(brief_payload.get("markdown") or ""),
        evidence_cards=final_cards,
        require_citations=request.require_citations,
    )
    trace.append(_trace_step("Verification", verification["status"], verification["checks"]))

    return _success_envelope(
        "run_research_task",
        context,
        task_type=request.task_type,
        topic=request.topic,
        report_markdown=str(brief_payload.get("markdown") or ""),
        evidence_cards=final_cards,
        agent_trace=trace,
        verification=verification,
        evidence_gaps=final_gaps,
        graph_edges=[_to_dict(edge) for edge in graph_edges],
    )


def _collect_graph_edges(
    backend: ResearchBackend,
    *,
    topic: str,
    companies: list[str],
    limit: int,
) -> list[Any]:
    edges: list[Any] = []
    if companies:
        per_company = max(1, limit // max(len(companies), 1))
        for company in companies:
            edges.extend(backend.query_graph(company=company, technology=topic, limit=per_company))
    else:
        edges.extend(backend.query_graph(technology=topic, limit=limit))
    return edges[:limit]


def _filters(request: FilteredSearchRequest) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if request.companies:
        filters["company"] = request.companies
    if request.topics:
        filters["topic"] = request.topics
    if request.claim_types:
        filters["claim_type"] = request.claim_types
    return filters


def _require_backend_method(backend: ResearchBackend, name: str) -> Callable[..., Any]:
    method = getattr(backend, name, None)
    if method is None or not callable(method):
        raise NotImplementedError(f"Backend does not expose {name}.")
    return method


def _success_envelope(tool: str, context: BackendContext, **fields: Any) -> dict[str, Any]:
    return {"status": "completed", "tool": tool, "meta": _meta(context), **_jsonable(fields)}


def _error_envelope(
    tool: str,
    *,
    meta: dict[str, Any],
    error_type: str,
    message: str,
    details: Any,
) -> dict[str, Any]:
    return {
        "status": "error",
        "tool": tool,
        "meta": _jsonable(meta),
        "error": {"type": error_type, "message": message, "details": _jsonable(details)},
    }


def _meta(context: BackendContext) -> dict[str, Any]:
    return {**context.meta, "warnings": list(context.warnings)}


def _cards_from_claims(claims: list[ClaimRecord]) -> list[EvidenceCard]:
    return [claim.to_evidence_card(citation_id=f"E{index}") for index, claim in enumerate(claims, start=1)]


def _normalize_cards(cards: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in list(cards or []):
        card = _to_dict(item)
        citation_id = str(card.get("citation_id") or "").strip()
        card["citation_status"] = "cited" if citation_id else "uncited"
        normalized.append(card)
    return normalized


def _normalize_gaps(gaps: Any) -> list[dict[str, Any]]:
    return [_to_dict(gap) for gap in list(gaps or [])]


def _dedupe_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for card in cards:
        key = (
            str(card.get("claim_id") or ""),
            str(card.get("source") or card.get("source_title") or ""),
            str(card.get("page") or ""),
            str(card.get("evidence") or card.get("evidence_span") or "")[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(card)
    return output


def _renumber_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, card in enumerate(cards, start=1):
        row = dict(card)
        if row.get("citation_status") != "uncited":
            row["citation_id"] = f"E{index}"
            row["citation_status"] = "cited"
        output.append(row)
    return output


def _verify_result(
    *,
    report_markdown: str,
    evidence_cards: list[dict[str, Any]],
    require_citations: bool,
) -> dict[str, Any]:
    evidence_count = len(evidence_cards)
    cited_count = sum(1 for card in evidence_cards if card.get("citation_id"))
    uncited_count = sum(1 for card in evidence_cards if not card.get("citation_id"))
    checks = {
        "report_present": bool(report_markdown.strip()),
        "evidence_count": evidence_count,
        "cited_count": cited_count,
        "uncited_count": uncited_count,
        "require_citations": require_citations,
        "citation_coverage": round(cited_count / evidence_count, 4) if evidence_count else 0.0,
    }
    gaps: list[dict[str, str]] = []
    if not checks["report_present"]:
        gaps.append({"gap": "Research brief markdown is empty.", "priority": "high", "suggested_source": "Build a brief from retrieved evidence."})
    if require_citations and not evidence_count:
        gaps.append({"gap": "No evidence cards were retrieved.", "priority": "high", "suggested_source": "Initialize or rebuild the AIKA knowledge index."})
    if require_citations and uncited_count:
        gaps.append({"gap": "Some evidence cards are uncited.", "priority": "medium", "suggested_source": "Attach citation ids to all evidence cards."})
    if not checks["report_present"] or (require_citations and not evidence_count):
        status = "fail"
    elif gaps:
        status = "warning"
    else:
        status = "pass"
    return {"status": status, "checks": checks, "evidence_gaps": gaps}


def _trace_step(phase: str, status: str, observation: dict[str, Any]) -> dict[str, Any]:
    return {"phase": phase, "status": status, "observation": _jsonable(observation)}


def _subject(topic: str, companies: list[str]) -> str:
    if companies:
        return f"{topic} {' '.join(companies)}".strip()
    return topic


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return dict(value.to_dict())
    return {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    return str(value)
