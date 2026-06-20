"""Build structured ReportSpec objects from deterministic AIKA evidence data."""

from __future__ import annotations

import re
from typing import Any, Iterable

from aika.report.charts.common import as_dict, dedupe, evidence_id, freshness_status, text_attr
from aika.report.charts.evidence_bar import build_evidence_strength_bar
from aika.report.charts.flow_map import build_flow_map
from aika.report.charts.freshness import build_source_freshness_timeline
from aika.report.charts.heatmap import build_company_coverage_heatmap
from aika.report.spec import AppendixSpec, ChartsSpec, CoverageSpec, ExecutiveSummarySpec, ReportSpec
from aika.report_type import classify_report_type, report_title, report_type_label, report_usability


def build_report_spec(
    *,
    question: str,
    plan: Any,
    evidence_cards: Iterable[Any],
    graph_records: Iterable[Any],
    gaps: Iterable[Any],
    verification: dict[str, Any] | None = None,
    company_table: dict[str, Any] | None = None,
    risk_checklist: list[dict[str, Any]] | None = None,
) -> ReportSpec:
    cards = list(evidence_cards or [])
    records = list(graph_records or [])
    gap_rows = [_gap_dict(gap) for gap in gaps or [] if _gap_dict(gap).get("gap")]
    verification = verification or {}
    target_companies = _target_companies(plan)
    covered_companies = _covered_companies(cards, records)
    unsupported_claims = len(gap_rows) + _unsupported_terms_count(verification)
    direct_claims = sum(1 for card in cards if _coverage_level(card) == "direct")
    indirect_claims = sum(1 for card in cards if _coverage_level(card) == "indirect")
    mentioned_claims = sum(1 for card in cards if _coverage_level(card) == "mentioned")
    card_count = len(cards)
    coverage_score = card_count / max(card_count + unsupported_claims, 1) if card_count else 0.0
    direct_claim_ratio = direct_claims / max(card_count, 1) if card_count else 0.0
    company_coverage = 1.0 if not target_companies else len([company for company in target_companies if company in covered_companies]) / max(len(target_companies), 1)
    coverage = CoverageSpec(
        coverage_score=_ratio(coverage_score),
        direct_claims=direct_claims,
        indirect_claims=indirect_claims,
        mentioned_claims=mentioned_claims,
        unsupported_claims=unsupported_claims,
        covered_companies=len(covered_companies if not target_companies else [company for company in target_companies if company in covered_companies]),
        target_companies=len(target_companies),
        direct_claim_ratio=_ratio(direct_claim_ratio),
        company_coverage=_ratio(company_coverage),
        freshness_status=_aggregate_freshness(cards),
    )
    report_type = classify_report_type(coverage.coverage_score, coverage.company_coverage, coverage.direct_claim_ratio)
    topic = _report_topic(question, plan, cards)
    charts = ChartsSpec(
        flow_map=build_flow_map(records, evidence_cards=cards),
        company_coverage_heatmap=build_company_coverage_heatmap(cards, target_companies=target_companies, graph_records=records),
        evidence_strength_bar=build_evidence_strength_bar(
            cards,
            unsupported_claims=unsupported_claims,
            no_evidence=max(len(target_companies) - coverage.covered_companies, 0),
        ),
        source_freshness_timeline=build_source_freshness_timeline(cards),
    )
    return ReportSpec(
        report_type=report_type,
        topic=topic,
        title=report_title(topic, report_type),
        report_type_label=report_type_label(report_type),
        usability=report_usability(report_type),
        coverage=coverage,
        executive_summary=_executive_summary(cards, gap_rows, report_type),
        charts=charts,
        appendix=AppendixSpec(
            evidence_cards=[_card_dict(card, index) for index, card in enumerate(cards, start=1)],
            evidence_gaps=gap_rows,
            company_compare_table=company_table or {},
            risk_checklist=risk_checklist or [],
        ),
    )


def _executive_summary(cards: list[Any], gaps: list[dict[str, Any]], report_type: str) -> ExecutiveSummarySpec:
    findings = []
    for card in cards:
        text = _with_citation(_short_text(text_attr(card, "evidence"), 150), card)
        if text and text not in findings:
            findings.append(_qualify(text, report_type))
        if len(findings) >= 4:
            break
    cannot_answer = [_short_text(str(gap.get("gap") or ""), 180) for gap in gaps if gap.get("gap")]
    if not findings:
        findings = ["当前证据不足，无法形成可验证结论。"]
    if not cannot_answer:
        cannot_answer = ["当前未识别出明确证据缺口；仍需回到证据附录核验来源和适用边界。"]
    return ExecutiveSummarySpec(can_answer=findings[:4], cannot_answer=cannot_answer[:5], key_findings=findings[:4])


def _coverage_level(card: Any) -> str:
    exposure = text_attr(card, "exposure_level").casefold()
    claim_type = text_attr(card, "claim_type").casefold()
    if exposure in {"core", "direct"} or claim_type == "company_exposure":
        return "direct"
    if exposure == "indirect" or claim_type in {"mechanism", "indicator", "supply_chain", "trend", "policy"}:
        return "indirect"
    return "mentioned"


def _target_companies(plan: Any) -> list[str]:
    return dedupe(str(company).strip() for company in list(getattr(plan, "companies", []) or []) if str(company).strip())


def _covered_companies(cards: list[Any], records: list[Any]) -> set[str]:
    covered = {text_attr(card, "company") for card in cards if text_attr(card, "company")}
    covered.update(text_attr(record, "company") for record in records if text_attr(record, "company"))
    return {company for company in covered if company}


def _unsupported_terms_count(verification: dict[str, Any]) -> int:
    checks = verification.get("checks", {}) if isinstance(verification, dict) else {}
    terms = checks.get("unsupported_terms") or []
    return len(dedupe(str(term) for term in terms if str(term).strip())) if isinstance(terms, list) else 0


def _aggregate_freshness(cards: list[Any]) -> str:
    statuses = []
    for card in cards:
        statuses.append(text_attr(card, "freshness_status") or freshness_status(text_attr(card, "published_at") or text_attr(card, "as_of_date")))
    statuses = [status for status in statuses if status]
    if not statuses:
        return "unknown"
    for status in ("stale", "aging", "fresh", "unknown"):
        if status in statuses:
            return status
    return "unknown"


def _report_topic(question: str, plan: Any, cards: list[Any]) -> str:
    quoted = re.search(r"“([^”]{1,40})”", str(question or ""))
    if quoted:
        return quoted.group(1).strip()
    topics = [str(topic).strip() for topic in list(getattr(plan, "topics", []) or []) if str(topic).strip()]
    for topic in topics:
        if f"{topic}产业链" in str(question or ""):
            return f"{topic}产业链"
    if topics:
        return "、".join(topics[:3])
    card_topics = dedupe(text_attr(card, "topic") for card in cards)
    if card_topics:
        return "、".join(card_topics[:3])
    return _short_text(question, 36) or "AI 算力产业链"


def _card_dict(card: Any, index: int) -> dict[str, Any]:
    row = as_dict(card)
    if not row:
        row = {
            "citation_id": evidence_id(card, f"E{index}"),
            "kind": text_attr(card, "kind"),
            "title": text_attr(card, "title"),
            "evidence": text_attr(card, "evidence"),
            "source": text_attr(card, "source"),
            "page": text_attr(card, "page"),
            "company": text_attr(card, "company"),
            "topic": text_attr(card, "topic"),
            "claim_type": text_attr(card, "claim_type"),
            "exposure_level": text_attr(card, "exposure_level"),
            "published_at": text_attr(card, "published_at"),
        }
    row["citation_id"] = str(row.get("citation_id") or evidence_id(card, f"E{index}"))
    row["evidence_id"] = str(row.get("evidence_id") or row.get("citation_id") or f"E{index}")
    row["evidence"] = str(row.get("evidence") or row.get("text") or row.get("evidence_span") or "")
    row["source"] = str(row.get("source") or row.get("source_title") or "")
    return row


def _gap_dict(gap: Any) -> dict[str, Any]:
    row = as_dict(gap)
    if not row and isinstance(gap, dict):
        row = dict(gap)
    text = str(row.get("gap") or row.get("reason") or "").strip()
    if not text:
        return {}
    return {
        "gap": text,
        "priority": str(row.get("priority") or "中"),
        "suggested_source": str(row.get("suggested_source") or ""),
    }


def _with_citation(text: str, card: Any) -> str:
    citation = evidence_id(card)
    return f"{text} [{citation}]" if text and citation else text


def _qualify(text: str, report_type: str) -> str:
    if report_type == "evidence_coverage_audit":
        return f"当前证据只能支持以下有限判断：{text}"
    if report_type == "preliminary_research_brief":
        return f"当前证据初步显示：{text}"
    return text


def _short_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _ratio(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
