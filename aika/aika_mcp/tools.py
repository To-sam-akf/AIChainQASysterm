"""Pure Python handlers behind the AIKA MCP server."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from aika.aika_core.evidence import build_evidence_ux_bundle, freshness_status
from aika.aika_core.backends.csv_backend import CSVResearchBackend
from aika.aika_core.backends.sqlite_backend import (
    DEFAULT_PROFILE,
    SQLiteResearchBackend,
    profile_index_path,
    resolve_aika_home,
)
from aika.aika_core.models import ClaimRecord, EvidenceCard, ResearchBackend
from aika.aika_mcp.schemas import (
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
LLM_COUNTER_MODE_ENV = "AIKA_EVIDENCE_UX_LLM_MODE"

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


class AikaCounterAuditError(RuntimeError):
    """Raised when LLM counter-evidence audit is explicitly required."""


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
    evidence_cards = _normalize_cards(cards)
    ux = _evidence_ux_bundle(request.query, context, evidence_cards=evidence_cards, claims=[])
    return _success_envelope(
        "search_evidence",
        context,
        query=request.query,
        evidence_cards=ux["evidence_cards"],
        conclusions=ux["conclusions"],
        evidence_links=ux["evidence_links"],
        counter_evidence=ux["counter_evidence"],
        evidence_ux_meta=ux["evidence_ux_meta"],
    )


def _search_claims(request: SearchClaimsRequest, context: BackendContext) -> dict[str, Any]:
    claims = context.backend.search_claims(request.query, top_k=request.top_k, **_filters(request))
    evidence_cards = _normalize_cards(_cards_from_claims(claims))
    ux = _evidence_ux_bundle(request.query, context, evidence_cards=evidence_cards, claims=claims)
    return _success_envelope(
        "search_claims",
        context,
        query=request.query,
        claims=[_to_dict(claim) for claim in claims],
        evidence_cards=ux["evidence_cards"],
        conclusions=ux["conclusions"],
        evidence_links=ux["evidence_links"],
        counter_evidence=ux["counter_evidence"],
        evidence_ux_meta=ux["evidence_ux_meta"],
    )


def _get_company_profile(request: CompanyProfileRequest, context: BackendContext) -> dict[str, Any]:
    profile = context.backend.get_company_profile(request.company, topic=request.topic)
    payload = _to_dict(profile)
    evidence_cards = _normalize_cards(payload.get("evidence_cards", []))
    claims = _claims_from_payload(payload)
    subject = _subject(request.topic or request.company, [request.company])
    ux = _evidence_ux_bundle(
        subject,
        context,
        evidence_cards=evidence_cards,
        claims=claims,
        gaps=payload.get("evidence_gaps", []),
    )
    return _success_envelope(
        "get_company_profile",
        context,
        company_profile=payload,
        evidence_cards=ux["evidence_cards"],
        evidence_gaps=_normalize_gaps(payload.get("evidence_gaps", [])),
        conclusions=ux["conclusions"],
        evidence_links=ux["evidence_links"],
        counter_evidence=ux["counter_evidence"],
        evidence_ux_meta=ux["evidence_ux_meta"],
    )


def _compare_companies(request: CompareCompaniesRequest, context: BackendContext) -> dict[str, Any]:
    comparison = _require_backend_method(context.backend, "compare_companies")(request.companies, topic=request.topic)
    payload = _to_dict(comparison)
    evidence_cards = _normalize_cards(payload.get("evidence_cards", []))
    ux = _evidence_ux_bundle(
        _subject(request.topic, request.companies),
        context,
        evidence_cards=evidence_cards,
        claims=_claims_from_payload(payload),
        gaps=payload.get("evidence_gaps", []),
    )
    return _success_envelope(
        "compare_companies",
        context,
        comparison=payload,
        evidence_cards=ux["evidence_cards"],
        evidence_gaps=_normalize_gaps(payload.get("evidence_gaps", [])),
        conclusions=ux["conclusions"],
        evidence_links=ux["evidence_links"],
        counter_evidence=ux["counter_evidence"],
        evidence_ux_meta=ux["evidence_ux_meta"],
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
        conclusions=[],
        evidence_links=[],
        counter_evidence=[],
    )


def _build_research_brief(request: BuildResearchBriefRequest, context: BackendContext) -> dict[str, Any]:
    query = request.query or f"{request.topic}投研简报"
    brief = _require_backend_method(context.backend, "build_research_brief")(query, topic=request.topic)
    payload = _to_dict(brief)
    evidence_cards = _normalize_cards(payload.get("evidence_cards", []))
    claims = _claims_from_payload(payload)
    gaps = _normalize_gaps(payload.get("evidence_gaps", []))
    ux = _evidence_ux_bundle(query, context, evidence_cards=evidence_cards, claims=claims, gaps=gaps)
    report_markdown = _prepend_evidence_ux(str(payload.get("markdown") or ""), ux, gaps=gaps)
    return _success_envelope(
        "build_research_brief",
        context,
        title=payload.get("title", ""),
        report_markdown=report_markdown,
        sections=payload.get("sections", []),
        evidence_cards=ux["evidence_cards"],
        evidence_gaps=gaps,
        research_outputs=payload.get("research_outputs", {}),
        conclusions=ux["conclusions"],
        evidence_links=ux["evidence_links"],
        counter_evidence=ux["counter_evidence"],
        evidence_ux_meta=ux["evidence_ux_meta"],
    )


def _audit_evidence_gaps(request: AuditEvidenceGapsRequest, context: BackendContext) -> dict[str, Any]:
    gaps = _require_backend_method(context.backend, "audit_evidence_gaps")(
        request.query,
        companies=request.companies,
        topic=request.topic,
    )
    ux = build_evidence_ux_bundle(
        request.query or _subject(request.topic, request.companies),
        [],
        claims=[],
        gaps=gaps,
    )
    return _success_envelope(
        "audit_evidence_gaps",
        context,
        evidence_gaps=_normalize_gaps(gaps),
        conclusions=ux["conclusions"],
        evidence_links=ux["evidence_links"],
        counter_evidence=ux["counter_evidence"],
        evidence_ux_meta=ux["evidence_ux_meta"],
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
    ux = _evidence_ux_bundle(subject, context, evidence_cards=final_cards, claims=claims, gaps=final_gaps)
    report_markdown = _prepend_evidence_ux(str(brief_payload.get("markdown") or ""), ux, gaps=final_gaps)
    verification = _verify_result(
        report_markdown=report_markdown,
        evidence_cards=ux["evidence_cards"],
        require_citations=request.require_citations,
    )
    trace.append(_trace_step("Verification", verification["status"], verification["checks"]))

    return _success_envelope(
        "run_research_task",
        context,
        task_type=request.task_type,
        topic=request.topic,
        report_markdown=report_markdown,
        evidence_cards=ux["evidence_cards"],
        agent_trace=trace,
        verification=verification,
        evidence_gaps=final_gaps,
        graph_edges=[_to_dict(edge) for edge in graph_edges],
        conclusions=ux["conclusions"],
        evidence_links=ux["evidence_links"],
        counter_evidence=ux["counter_evidence"],
        evidence_ux_meta=ux["evidence_ux_meta"],
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


def _evidence_ux_bundle(
    subject: str,
    context: BackendContext,
    *,
    evidence_cards: list[dict[str, Any]],
    claims: list[ClaimRecord],
    gaps: list[Any] | None = None,
) -> dict[str, Any]:
    audit = _llm_counter_audit(context, subject=subject, evidence_cards=evidence_cards, claims=claims)
    return build_evidence_ux_bundle(
        subject,
        evidence_cards,
        claims=claims,
        gaps=gaps or [],
        llm_counter_audit=audit,
    )


def _llm_counter_audit(
    context: BackendContext,
    *,
    subject: str,
    evidence_cards: list[dict[str, Any]],
    claims: list[ClaimRecord],
) -> dict[str, Any]:
    mode = os.getenv(LLM_COUNTER_MODE_ENV, "auto").strip().casefold() or "auto"
    if mode not in {"auto", "off", "required"}:
        context.warnings.append(f"{LLM_COUNTER_MODE_ENV}={mode!r} is invalid; using auto.")
        mode = "auto"
    if mode == "off" or not evidence_cards:
        return {"status": "disabled" if mode == "off" else "not_run", "source": "rules"}
    try:
        from aika.llm_client import OpenAICompatibleClient

        client = OpenAICompatibleClient()
        payload = {
            "subject": subject,
            "evidence_cards": [
                {
                    "evidence_id": card.get("evidence_id") or card.get("citation_id") or card.get("claim_id"),
                    "citation_id": card.get("citation_id"),
                    "claim_type": card.get("claim_type"),
                    "evidence": _short_text(str(card.get("evidence") or card.get("evidence_span") or ""), 260),
                    "source_title": card.get("source_title") or card.get("source"),
                    "published_at": card.get("published_at") or card.get("as_of_date"),
                }
                for card in evidence_cards[:12]
            ],
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "claim_type": claim.claim_type,
                    "claim_text": _short_text(claim.claim_text, 260),
                }
                for claim in claims[:12]
            ],
        }
        response = client.chat_json(
            system_prompt=(
                "You audit evidence for counter-evidence. Return JSON only with keys: "
                "status, summary, counter_evidence_ids. Use only provided evidence_id values. "
                "If no counter-evidence is visible, return an empty counter_evidence_ids list and say the sample did not reveal counter-evidence."
            ),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            temperature=0.0,
            thinking_enabled=False,
        )
        response = dict(response)
        response.setdefault("status", "completed")
        response["source"] = "llm"
        return response
    except Exception as exc:
        if mode == "required":
            raise AikaCounterAuditError(f"LLM counter-evidence audit failed: {exc}") from exc
        context.warnings.append(f"LLM counter-evidence audit unavailable; used rule fallback: {exc}")
        return {"status": "fallback", "source": "rules", "error": str(exc)}


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
    for index, item in enumerate(list(cards or []), start=1):
        card = _to_dict(item)
        citation_id = str(card.get("citation_id") or "").strip()
        card["citation_status"] = "cited" if citation_id else "uncited"
        card["source_title"] = str(card.get("source_title") or card.get("source") or card.get("title") or "").strip()
        card["published_at"] = str(card.get("published_at") or card.get("as_of_date") or "").strip()
        card["freshness_status"] = str(card.get("freshness_status") or freshness_status(card["published_at"]))
        card["evidence_id"] = str(citation_id or card.get("claim_id") or f"UXE{index}").strip()
        card["counter_evidence_status"] = str(card.get("counter_evidence_status") or "none")
        card["counter_evidence_summary"] = str(card.get("counter_evidence_summary") or "当前样本未检出反证。")
        card["supported_conclusion_ids"] = list(card.get("supported_conclusion_ids") or [])
        card["contradicted_conclusion_ids"] = list(card.get("contradicted_conclusion_ids") or [])
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
            row["evidence_id"] = row["citation_id"]
            row["citation_status"] = "cited"
        output.append(row)
    return output


def _claims_from_payload(payload: dict[str, Any]) -> list[ClaimRecord]:
    rows: list[Any] = []
    if isinstance(payload.get("claims"), list):
        rows.extend(payload.get("claims") or [])
    outputs = payload.get("research_outputs") or {}
    if isinstance(outputs, dict):
        for key in ("claims", "claim_records"):
            if isinstance(outputs.get(key), list):
                rows.extend(outputs.get(key) or [])
    claims: list[ClaimRecord] = []
    seen: set[str] = set()
    for row in rows:
        data = _to_dict(row)
        if not data:
            continue
        claim = ClaimRecord.from_row(data, score=float(data.get("score") or 0.0))
        key = claim.claim_id or claim.claim_text
        if key and key not in seen:
            seen.add(key)
            claims.append(claim)
    return claims


def _prepend_evidence_ux(report_markdown: str, ux: dict[str, Any], gaps: list[Any] | None = None) -> str:
    return _compose_research_markdown(report_markdown, ux, gaps=gaps)


def _compose_research_markdown(report_markdown: str, ux: dict[str, Any], *, gaps: list[Any] | None = None) -> str:
    conclusions = list(ux.get("conclusions") or [])
    evidence_cards = list(ux.get("evidence_cards") or [])
    cards = {str(card.get("evidence_id") or ""): card for card in evidence_cards}
    links = list(ux.get("evidence_links") or [])
    by_conclusion: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        if not isinstance(link, dict):
            continue
        conclusion_id = str(link.get("conclusion_id") or "")
        card = cards.get(str(link.get("evidence_id") or ""))
        if conclusion_id and card:
            by_conclusion.setdefault(conclusion_id, []).append(card)

    gap_rows = _dedupe_report_gaps(_normalize_gaps(gaps or []))
    supported = [row for row in conclusions if str(row.get("evidence_status") or "") == "supported"]
    unsupported = [row for row in conclusions if str(row.get("evidence_status") or "") != "supported"]
    meta = dict(ux.get("evidence_ux_meta") or {})
    coverage_score = _coverage_score(conclusions, gap_rows, evidence_cards)
    report_type = _coverage_report_type(coverage_score)
    usability = _coverage_usability(coverage_score)

    lines: list[str] = [
        "## 一页结论",
        "",
        "### 本次报告能回答什么",
    ]
    if supported:
        for conclusion in supported[:4]:
            citations = _conclusion_citations(str(conclusion.get("conclusion_id") or ""), by_conclusion)
            suffix = f"（支撑证据：{citations}）" if citations else "（支撑证据：当前证据不足）"
            lines.append(f"- {_short_text(str(conclusion.get('conclusion_text') or ''), 160)}{suffix}")
    else:
        lines.append("- 当前证据不足，无法形成可验证结论。")

    lines.extend(["", "### 本次报告不能回答什么"])
    if gap_rows:
        for gap in gap_rows[:5]:
            lines.append(f"- {_short_text(str(gap.get('gap') or gap.get('reason') or ''), 160)}")
    elif unsupported:
        for conclusion in unsupported[:3]:
            lines.append(f"- {_short_text(str(conclusion.get('conclusion_text') or '当前证据不足。'), 160)}")
    else:
        lines.append("- 当前未识别出明确证据缺口；仍需回到证据附录核验来源和适用边界。")

    lines.extend(
        [
            "",
            "### 当前报告适合的使用方式",
            f"- Report Type: `{report_type}`",
            f"- Conclusion Usability: {usability}",
            "- 适合作为证据覆盖审计和初步研究简报使用；需要完整投研结论时，应补充缺口证据后再升级报告类型。",
            "",
            "## 证据覆盖评级",
            "",
            f"- Coverage: {coverage_score:.0%}",
            f"- Evidence Cards: {len(evidence_cards)}",
            f"- Supported Conclusions: {len(supported)}/{len(conclusions)}",
            f"- Evidence Gaps: {len(gap_rows) if gap_rows else int(meta.get('gap_count') or 0)}",
            f"- Counter Evidence Flags: {int(meta.get('counter_evidence_count') or 0)}",
            "",
            "## 关键发现",
            "",
        ]
    )
    if conclusions:
        for conclusion in conclusions[:3]:
            status = str(conclusion.get("evidence_status") or "insufficient")
            citations = _conclusion_citations(str(conclusion.get("conclusion_id") or ""), by_conclusion)
            evidence_note = citations if citations else "当前证据不足"
            lines.append(
                f"- {_short_text(str(conclusion.get('conclusion_text') or '当前证据不足。'), 180)}"
                f"（证据状态：{status}；证据：{evidence_note}）"
            )
    else:
        lines.append("- 当前证据不足，无法生成关键发现。")

    body = _strip_composed_sections(str(report_markdown or "").strip())
    if body:
        lines.extend(["", "## 正文分析", "", body])

    lines.extend(["", "## Evidence Review"])
    if not conclusions:
        lines.append("当前证据不足。")
    for conclusion in conclusions[:6]:
        conclusion_id = str(conclusion.get("conclusion_id") or "")
        text = _short_text(str(conclusion.get("conclusion_text") or "当前证据不足。"), 180)
        status = str(conclusion.get("evidence_status") or "insufficient")
        lines.append(f"- 结论 {conclusion_id}: {text}（证据状态：{status}）")
        linked_cards = by_conclusion.get(conclusion_id, [])[:3]
        if not linked_cards:
            lines.append("  - 证据摘要：当前证据不足。")
            continue
        for card in linked_cards:
            lines.append("  - " + _evidence_summary_line(card))

    lines.extend(["", "## 证据附录"])
    if not evidence_cards:
        lines.append("当前证据不足。")
    for card in evidence_cards:
        lines.extend(["", *_evidence_appendix_lines(card)])

    return "\n".join(lines).strip()


def _coverage_score(conclusions: list[dict[str, Any]], gaps: list[dict[str, Any]], evidence_cards: list[dict[str, Any]]) -> float:
    if not evidence_cards:
        return 0.0
    supported = sum(1 for conclusion in conclusions if str(conclusion.get("evidence_status") or "") == "supported")
    denominator = max(len(conclusions) + len(gaps), 1)
    return max(0.0, min(1.0, supported / denominator))


def _dedupe_report_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for gap in gaps:
        text = " ".join(str(gap.get("gap") or gap.get("reason") or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(gap)
    return output


def _strip_composed_sections(markdown: str) -> str:
    if not markdown:
        return ""
    drop_titles = {"一页结论", "证据覆盖审计", "证据摘要", "证据附录"}
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    preface: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if current_title:
                sections.append((current_title, current_lines))
            elif current_lines:
                preface.extend(current_lines)
            current_title = line[3:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_title:
        sections.append((current_title, current_lines))
    elif current_lines:
        preface.extend(current_lines)

    kept: list[str] = [line for line in preface if line.strip()]
    for title, lines in sections:
        if title in drop_titles:
            continue
        kept.extend(lines)
    return "\n".join(kept).strip()


def _coverage_report_type(score: float) -> str:
    if score < 0.3:
        return "evidence_coverage_audit"
    if score < 0.6:
        return "preliminary_research_brief"
    if score < 0.8:
        return "industry_research_report"
    return "deep_research_report"


def _coverage_usability(score: float) -> str:
    if score < 0.3:
        return "Limited"
    if score < 0.6:
        return "Preliminary"
    if score < 0.8:
        return "Usable with caveats"
    return "High"


def _conclusion_citations(conclusion_id: str, by_conclusion: dict[str, list[dict[str, Any]]]) -> str:
    citations = []
    for card in by_conclusion.get(conclusion_id, []):
        citation = str(card.get("citation_id") or card.get("evidence_id") or "").strip()
        if citation and citation not in citations:
            citations.append(f"[{citation}]")
    return " ".join(citations[:5])


def _evidence_summary_line(card: dict[str, Any]) -> str:
    citation = str(card.get("citation_id") or card.get("evidence_id") or "未编号")
    page_or_section = str(card.get("page") or card.get("section") or card.get("paragraph_id") or "未标注")
    return (
        f"[{citation}] 来源：{card.get('source_title') or card.get('source') or '未标注'}；"
        f"日期：{card.get('published_at') or 'unknown'}；"
        f"页码/段落：{page_or_section}；"
        f"claim 类型：{card.get('claim_type') or 'unknown'}；"
        f"置信度：{card.get('confidence') or 'unknown'}；"
        f"时效：{card.get('freshness_status') or 'unknown'}；"
        f"反证：{card.get('counter_evidence_status') or 'none'}"
    )


def _evidence_appendix_lines(card: dict[str, Any]) -> list[str]:
    citation = str(card.get("citation_id") or card.get("evidence_id") or "未编号")
    source = str(card.get("source_title") or card.get("source") or card.get("title") or "未标注")
    page_or_section = str(card.get("page") or card.get("section") or card.get("paragraph_id") or "未标注")
    evidence_text = _short_text(str(card.get("evidence") or card.get("evidence_span") or ""), 420) or "未提供。"
    return [
        f"### [{citation}] {source}",
        f"- 来源：{source}",
        f"- 日期：{card.get('published_at') or 'unknown'}",
        f"- 页码/段落：{page_or_section}",
        f"- Claim 类型：{card.get('claim_type') or 'unknown'}",
        f"- 置信度：{card.get('confidence') or 'unknown'}",
        f"- 时效：{card.get('freshness_status') or 'unknown'}",
        f"- 反证状态：{card.get('counter_evidence_status') or 'none'}",
        f"- 证据原文：{evidence_text}",
    ]


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


def _short_text(value: str, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


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
