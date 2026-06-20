"""Structured investment-research outputs built from QA evidence cards."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from aika.domain_lexicon import company_segment
from aika.report_type import (
    CoverageMetrics,
    aggregate_freshness_status,
    classify_report_type,
    report_title,
    report_type_label,
    report_usability,
)
from aika.report import build_report_spec, render_html, render_markdown, render_markdown_sections


EXPOSURE_LABELS = {
    "core": "核心敞口",
    "direct": "直接敞口",
    "indirect": "间接敞口",
    "mentioned": "仅提及",
    "": "未分级",
}

CLAIM_TYPE_LABELS = {
    "company_exposure": "公司敞口",
    "mechanism": "技术机理",
    "bottleneck": "瓶颈",
    "indicator": "领先指标",
    "risk": "风险",
    "supply_chain": "产业传导",
    "trend": "趋势",
    "policy": "政策",
}


def build_research_outputs(
    *,
    question: str,
    plan: Any,
    evidence_cards: list[Any],
    graph_records: list[dict[str, Any]],
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create report/table/checklist/gap artifacts for the frontend.

    The function is deterministic and only uses the evidence pack already
    selected by QA. It is deliberately conservative: missing sections become
    evidence gaps instead of being filled with unsupported market knowledge.
    """
    cards = list(evidence_cards or [])
    verification = verification or {}
    gaps = merge_gap_rows(
        evidence_gap_list(question, plan, cards, graph_records, verification),
        list(verification.get("evidence_gaps") or []),
    )
    coverage = report_coverage_metrics(question, plan, cards, graph_records, gaps, verification)
    report_type = classify_report_type(coverage.coverage_score, coverage.company_coverage, coverage.direct_claim_ratio)
    company_table = company_compare_table(plan, cards, graph_records)
    risks = risk_checklist(plan, cards, graph_records)
    report = research_report(
        question,
        plan,
        cards,
        graph_records,
        gaps,
        coverage=coverage,
        report_type=report_type,
        company_table=company_table,
        risk_checklist_rows=risks,
        verification=verification,
    )
    return {
        "report": report,
        "company_compare_table": company_table,
        "risk_checklist": risks,
        "evidence_gaps": gaps,
        "verification": verification,
        "meta": {
            "question": question,
            "answer_type": str(getattr(plan, "answer_type", "")),
            "companies": list(getattr(plan, "companies", []) or []),
            "topics": list(getattr(plan, "topics", []) or []),
            "evidence_cards": len(cards),
            "verification_status": str(verification.get("status") or ""),
            "conflict_group_count": len(verification.get("conflict_groups") or []),
            "coverage": coverage.to_dict(),
            "report_type": report_type,
            "report_type_label": report_type_label(report_type),
        },
    }


def research_report(
    question: str,
    plan: Any,
    cards: list[Any],
    graph_records: list[dict[str, Any]],
    gaps: list[dict[str, str]],
    *,
    coverage: CoverageMetrics | None = None,
    report_type: str = "",
    company_table: dict[str, Any] | None = None,
    risk_checklist_rows: list[dict[str, Any]] | None = None,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del coverage, report_type
    spec = build_report_spec(
        question=question,
        plan=plan,
        evidence_cards=cards,
        graph_records=graph_records,
        gaps=gaps,
        verification=verification or {},
        company_table=company_table if company_table is not None else company_compare_table(plan, cards, graph_records),
        risk_checklist=risk_checklist_rows if risk_checklist_rows is not None else risk_checklist(plan, cards, graph_records),
    )
    sections = render_markdown_sections(spec)
    markdown = render_markdown(spec)
    return {
        "title": spec.title,
        "markdown": markdown,
        "html": render_html(spec),
        "sections": sections,
        "report_type": spec.report_type,
        "report_type_label": spec.report_type_label,
        "coverage": spec.coverage.model_dump(),
        "spec": spec.model_dump(),
    }


def company_compare_table(plan: Any, cards: list[Any], graph_records: list[dict[str, Any]]) -> dict[str, Any]:
    companies = ordered_companies(plan, cards, graph_records)
    rows: list[dict[str, str]] = []
    for company in companies:
        company_cards = [card for card in cards if card_text_attr(card, "company") == company]
        company_records = [row for row in graph_records if str(row.get("company") or "") == company]
        exposure = strongest_exposure(company_cards)
        rows.append(
            {
                "company": company,
                "chain_segment": company_segment(company) or infer_record_segments(company_records),
                "exposure_level": EXPOSURE_LABELS.get(exposure, exposure or "未分级"),
                "business_evidence": summarize_company_evidence(company_cards, company_records, {"company_exposure", "mechanism", "supply_chain", ""}),
                "leading_indicators": summarize_company_evidence(company_cards, company_records, {"indicator"}),
                "risks": summarize_company_evidence(company_cards, company_records, {"risk", "bottleneck"}),
                "citations": "、".join(unique(card_text_attr(card, "citation_id") for card in company_cards if card_text_attr(card, "citation_id"))[:6]),
            }
        )
    return {
        "columns": ["company", "chain_segment", "exposure_level", "business_evidence", "leading_indicators", "risks", "citations"],
        "rows": rows,
    }


def risk_checklist(plan: Any, cards: list[Any], graph_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    del plan
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for card in cards:
        claim_type = card_text_attr(card, "claim_type")
        relation = card_text_attr(card, "relation")
        if claim_type not in {"risk", "bottleneck"} and relation != "DISCLOSES_RISK":
            continue
        risk = short_text(card_text_attr(card, "evidence"), 180)
        key = normalize_key(risk)
        if not risk or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "scope": card_text_attr(card, "company") or card_text_attr(card, "topic") or "主题",
                "risk": risk,
                "priority": "高" if claim_type == "risk" or relation == "DISCLOSES_RISK" else "中",
                "follow_up": follow_up_for_risk(risk),
                "citation_id": card_text_attr(card, "citation_id"),
                "source": format_source(card),
            }
        )
    for row in graph_records:
        if str(row.get("relation") or "") != "DISCLOSES_RISK":
            continue
        risk = short_text(str(row.get("evidence") or row.get("target") or ""), 180)
        key = normalize_key(risk)
        if not risk or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "scope": str(row.get("company") or "公司"),
                "risk": risk,
                "priority": "高",
                "follow_up": follow_up_for_risk(risk),
                "citation_id": "",
                "source": format_record_source(row),
            }
        )
    return rows[:12]


def evidence_gap_list(
    question: str,
    plan: Any,
    cards: list[Any],
    graph_records: list[dict[str, Any]],
    verification: dict[str, Any],
) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    if not cards:
        gaps.append(
            {
                "gap": "当前问题没有召回可用证据卡片。",
                "priority": "高",
                "suggested_source": "补充年报、研报原文或行业白皮书后重建 Claim/RAG 索引。",
            }
        )
        return gaps

    for line in dossier_gap_lines(cards):
        gaps.append({"gap": line, "priority": "中", "suggested_source": suggested_source_for_gap(line)})

    answer_type = str(getattr(plan, "answer_type", ""))
    companies = list(getattr(plan, "companies", []) or [])
    if answer_type == "company_compare":
        covered = {card_text_attr(card, "company") for card in cards if card_text_attr(card, "company")}
        covered.update(str(row.get("company") or "") for row in graph_records if row.get("company"))
        for company in companies:
            if company not in covered:
                gaps.append(
                    {
                        "gap": f"{company} 缺少入选证据，当前对比表可能不完整。",
                        "priority": "高",
                        "suggested_source": "补充该公司年报、业务公告、研报或投资者关系记录。",
                    }
                )

    if answer_type in {"topic_to_company", "thematic_research", "industry_bottleneck"} and not has_direct_exposure(cards):
        gaps.append(
            {
                "gap": "缺少可稳定排序的核心/直接公司敞口证据。",
                "priority": "高",
                "suggested_source": "补充公司产品、客户导入、订单、产能或产业链环节证据。",
            }
        )

    if getattr(plan, "needs_metrics", False) or any(term in question for term in ("指标", "订单", "业绩", "验证", "跟踪")):
        if not cards_by_type(cards, {"indicator"}):
            gaps.append(
                {
                    "gap": "缺少订单、收入、毛利率、产能、客户导入或渗透率等领先指标证据。",
                    "priority": "高",
                    "suggested_source": "补充年报财务表、经营数据、公告和研报指标表。",
                }
            )

    if getattr(plan, "needs_risk", False) or "风险" in question:
        if not cards_by_type(cards, {"risk", "bottleneck"}) and not any(str(row.get("relation") or "") == "DISCLOSES_RISK" for row in graph_records):
            gaps.append(
                {
                    "gap": "缺少明确风险、反证或不确定性证据。",
                    "priority": "中",
                    "suggested_source": "补充年报风险披露、行业竞争格局和技术路线替代证据。",
                }
            )

    checks = verification.get("checks", {}) if isinstance(verification, dict) else {}
    unsupported = checks.get("unsupported_terms") or []
    if unsupported:
        gaps.append(
            {
                "gap": "答案中存在证据包未直接支撑的术语或数值：" + "、".join(str(item) for item in unsupported[:8]),
                "priority": "高",
                "suggested_source": "回到原文证据或结构化指标表中核验后再引用。",
            }
        )

    return dedupe_gap_rows(gaps)[:12]


def subject_label(plan: Any) -> str:
    companies = list(getattr(plan, "companies", []) or [])
    topics = list(getattr(plan, "topics", []) or [])
    if companies and topics:
        return "、".join(companies[:2]) + " " + "、".join(topics[:2])
    if companies:
        return "、".join(companies[:3])
    if topics:
        return "、".join(topics[:3])
    return ""


def report_subject_label(question: str, plan: Any) -> str:
    quoted = re.search(r"“([^”]{1,40})”", str(question or ""))
    if quoted:
        return quoted.group(1).strip()
    topics = list(getattr(plan, "topics", []) or [])
    for topic in topics:
        topic = str(topic or "").strip()
        if topic and f"{topic}产业链" in str(question or ""):
            return f"{topic}产业链"
    return subject_label(plan) or short_text(question, 36) or "AI 算力产业链"


def report_coverage_metrics(
    question: str,
    plan: Any,
    cards: list[Any],
    graph_records: list[dict[str, Any]],
    gaps: list[dict[str, str]],
    verification: dict[str, Any],
) -> CoverageMetrics:
    del question
    coverage_score = agent_coverage_score(cards, gaps)
    direct_cards = direct_exposure_cards(cards)
    direct_claim_ratio = len(direct_cards) / max(len(cards), 1) if cards else 0.0
    target_companies = [str(company).strip() for company in list(getattr(plan, "companies", []) or []) if str(company).strip()]
    covered_companies = covered_company_names(cards, graph_records)
    company_coverage = 1.0 if not target_companies else len([company for company in target_companies if company in covered_companies]) / max(len(target_companies), 1)
    checks = verification.get("checks", {}) if isinstance(verification, dict) else {}
    unsupported_terms = unique(str(item) for item in list(checks.get("unsupported_terms") or []) if str(item).strip())
    freshness_values = []
    for card in cards:
        freshness_values.extend(
            [
                card_text_attr(card, "freshness_status"),
                card_text_attr(card, "published_at"),
                card_text_attr(card, "as_of_date"),
            ]
        )
    return CoverageMetrics(
        coverage_score=coverage_score,
        company_coverage=max(0.0, min(1.0, company_coverage)),
        direct_claim_ratio=max(0.0, min(1.0, direct_claim_ratio)),
        unsupported_claims=len(gaps) + len(unsupported_terms),
        freshness_status=aggregate_freshness_status(freshness_values),
    )


def covered_company_names(cards: list[Any], graph_records: list[dict[str, Any]]) -> set[str]:
    covered = {card_text_attr(card, "company") for card in cards if card_text_attr(card, "company")}
    covered.update(str(row.get("company") or "").strip() for row in graph_records if str(row.get("company") or "").strip())
    return covered


def direct_exposure_cards(cards: list[Any]) -> list[Any]:
    return [
        card
        for card in cards
        if card_text_attr(card, "claim_type") == "company_exposure" and card_text_attr(card, "exposure_level") in {"core", "direct"}
    ]


def core_judgment(cards: list[Any], plan: Any, *, report_type: str = "industry_research_report") -> str:
    dossier = next((card for card in cards if card_text_attr(card, "kind") == "dossier"), None)
    if dossier:
        first_line = card_text_attr(dossier, "evidence").splitlines()[0:1]
        if first_line:
            return qualify_conclusion(with_citation(short_text(first_line[0], 220), dossier), report_type)
    claim = next((card for card in cards if card_text_attr(card, "kind") == "claim"), None)
    if claim:
        return qualify_conclusion(with_citation(short_text(card_text_attr(claim, "evidence"), 220), claim), report_type)
    topic = subject_label(plan) or "该主题"
    return f"{topic} 的研究结论需要以证据包中的公司敞口、技术机理、指标和风险共同验证。"


def qualify_conclusion(text: str, report_type: str) -> str:
    text = str(text or "").strip()
    if not text:
        return text
    if report_type == "evidence_coverage_audit":
        return f"当前证据只能支持以下有限判断：{text}"
    if report_type == "preliminary_research_brief":
        return f"当前证据初步显示：{text}"
    return text


def industry_transmission(cards: list[Any], graph_records: list[dict[str, Any]]) -> str:
    lines = card_lines(cards_by_type(cards, {"supply_chain", "company_exposure", "policy"}), 4)
    if lines:
        return "；".join(lines)
    facts = []
    for row in graph_records[:6]:
        company = str(row.get("company") or "")
        target = str(row.get("target") or "")
        relation = str(row.get("relation") or "")
        if company and target:
            facts.append(f"{company}-{relation}-{target}")
    return "；".join(facts) if facts else "当前证据不足。"


def markdown_compare_table(table: dict[str, Any]) -> str:
    rows = table.get("rows") or []
    if not rows:
        return "当前证据不足以生成公司对比表。"
    lines = ["| 公司 | 环节 | 敞口 | 业务证据 | 指标 | 风险 | 证据 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows[:12]:
        lines.append(
            "| {company} | {chain_segment} | {exposure_level} | {business_evidence} | {leading_indicators} | {risks} | {citations} |".format(
                **{key: escape_table_cell(str(row.get(key, "") or "当前证据不足")) for key in table["columns"]}
            )
        )
    return "\n".join(lines)


def markdown_risks(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前证据不足以生成风险清单。"
    return "\n".join(
        f"- {row['scope']}：{row['risk']}（优先级：{row['priority']}，跟踪：{row['follow_up']}）"
        + (f" [{row['citation_id']}]" if row.get("citation_id") else "")
        for row in rows[:8]
    )


def markdown_gaps(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "当前证据包未识别出关键缺口。"
    return "\n".join(f"- {row['gap']} 建议：{row['suggested_source']}" for row in rows[:8])


def markdown_executive_page(
    question: str,
    plan: Any,
    cards: list[Any],
    gaps: list[dict[str, str]],
    coverage: CoverageMetrics,
    report_type: str,
) -> str:
    can_answer = executive_answerable_items(cards, plan, report_type=report_type)
    cannot_answer = [str(row.get("gap") or "").strip() for row in gaps if str(row.get("gap") or "").strip()]
    lines = ["### 本次报告能回答什么"]
    if can_answer:
        lines.extend(f"- {item}" for item in can_answer[:4])
    else:
        lines.append("- 当前证据不足，无法形成可验证结论。")
    lines.append("")
    lines.append("### 本次报告不能回答什么")
    if cannot_answer:
        lines.extend(f"- {short_text(item, 180)}" for item in cannot_answer[:5])
    else:
        lines.append("- 当前未识别出明确证据缺口；仍需回到证据附录核验来源和适用边界。")
    lines.append("")
    lines.append("### 当前报告适合的使用方式")
    lines.append(f"- 研究问题：{short_text(question, 120)}")
    lines.append(f"- Report Type: `{report_type}`（{report_type_label(report_type)}）")
    lines.append(f"- Coverage: {coverage.coverage_score:.0%}；Direct Claim Ratio: {coverage.direct_claim_ratio:.0%}")
    lines.append(f"- Conclusion Usability: {report_usability(report_type)}")
    if report_type == "evidence_coverage_audit":
        lines.append("- 覆盖审计提醒：当前证据覆盖不足，本报告只适合识别证据边界和补数方向，不应当作为完整深度结论使用。")
    else:
        lines.append(f"- {report_usage_sentence(report_type)}")
    return "\n".join(lines)


def markdown_coverage_audit(
    cards: list[Any],
    gaps: list[dict[str, str]],
    coverage: CoverageMetrics,
    report_type: str,
) -> str:
    direct_cards = direct_exposure_cards(cards)
    counter_cards = [card for card in cards if card_text_attr(card, "claim_type") in {"risk", "bottleneck"}]
    lines = [
        f"- Report Type: `{report_type}`（{report_type_label(report_type)}）",
        f"- Coverage: {coverage.coverage_score:.0%}",
        f"- Company Coverage: {coverage.company_coverage:.0%}",
        f"- Direct Claim Ratio: {coverage.direct_claim_ratio:.0%}",
        f"- Evidence Cards: {len(cards)}",
        f"- Direct/Core Exposure Evidence: {len(direct_cards)}",
        f"- Risk/Bottleneck Evidence: {len(counter_cards)}",
        f"- Unsupported Claims: {coverage.unsupported_claims}",
        f"- Source Freshness: {coverage.freshness_status}",
        f"- Evidence Gaps: {len(gaps)}",
        f"- Conclusion Usability: {report_usability(report_type)}",
    ]
    if report_type == "evidence_coverage_audit":
        lines.append("- 覆盖审计提醒：证据覆盖低或直接证据占比不足，系统已自动降级为覆盖审计报告。")
    return "\n".join(lines)


def executive_answerable_items(cards: list[Any], plan: Any, *, report_type: str = "industry_research_report") -> list[str]:
    items: list[str] = []
    core = core_judgment(cards, plan, report_type=report_type)
    if core and "需要以证据包" not in core:
        items.append(core)
    for card in cards:
        text = qualify_conclusion(with_citation(short_text(card_text_attr(card, "evidence"), 150), card), report_type)
        if text and text not in items:
            items.append(text)
        if len(items) >= 4:
            break
    return items


def agent_coverage_score(cards: list[Any], gaps: list[dict[str, str]]) -> float:
    if not cards:
        return 0.0
    denominator = max(len(cards) + len(gaps), 1)
    return max(0.0, min(1.0, len(cards) / denominator))


def agent_usability_label(score: float) -> str:
    if score < 0.3:
        return "Limited"
    if score < 0.6:
        return "Preliminary"
    if score < 0.8:
        return "Usable with caveats"
    return "High"


def report_usage_sentence(report_type: str) -> str:
    if report_type == "preliminary_research_brief":
        return "适合作为初步研究简报使用；需要完整投研结论时，应补充缺口证据后再升级报告类型。"
    if report_type == "industry_research_report":
        return "适合作为带证据边界的产业分析报告使用；关键投资判断仍需回到证据附录核验。"
    if report_type == "deep_research_report":
        return "适合作为深度研究报告使用；仍需保留证据附录用于审计和复核。"
    return "适合作为证据覆盖审计使用；需要完整投研结论时，应补充缺口证据后再升级报告类型。"


def markdown_evidence_summary(cards: list[Any], limit: int) -> str:
    if not cards:
        return "当前证据不足。"
    return "\n".join(
        "- "
        f"[{card_text_attr(card, 'citation_id') or 'E?'}] "
        f"来源：{format_source(card) or '未标注'}；"
        f"claim 类型：{CLAIM_TYPE_LABELS.get(card_text_attr(card, 'claim_type'), card_text_attr(card, 'claim_type') or 'unknown')}；"
        f"敞口：{EXPOSURE_LABELS.get(card_text_attr(card, 'exposure_level'), card_text_attr(card, 'exposure_level') or '未分级')}；"
        f"证据：{short_text(card_text_attr(card, 'evidence'), 120)}"
        for card in cards[:limit]
    )


def markdown_evidence_appendix(cards: list[Any], limit: int) -> str:
    if not cards:
        return "当前证据不足。"
    lines: list[str] = []
    for card in cards[:limit]:
        citation = card_text_attr(card, "citation_id") or "E?"
        source = format_source(card) or "未标注"
        lines.extend(
            [
                f"### [{citation}] {source}",
                f"- 来源：{source}",
                f"- Claim 类型：{CLAIM_TYPE_LABELS.get(card_text_attr(card, 'claim_type'), card_text_attr(card, 'claim_type') or 'unknown')}",
                f"- 敞口级别：{EXPOSURE_LABELS.get(card_text_attr(card, 'exposure_level'), card_text_attr(card, 'exposure_level') or '未分级')}",
                f"- 证据原文：{short_text(card_text_attr(card, 'evidence'), 360) or '未提供。'}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def ordered_companies(plan: Any, cards: list[Any], graph_records: list[dict[str, Any]]) -> list[str]:
    companies = list(getattr(plan, "companies", []) or [])
    for card in cards:
        company = card_text_attr(card, "company")
        if company and company not in companies:
            companies.append(company)
    for line_company in companies_from_dossiers(cards):
        if line_company not in companies:
            companies.append(line_company)
    for row in graph_records:
        company = str(row.get("company") or "")
        if company and company not in companies:
            companies.append(company)
    return companies[:18]


def companies_from_dossiers(cards: list[Any]) -> list[str]:
    values: list[str] = []
    for card in cards:
        if card_text_attr(card, "kind") != "dossier":
            continue
        for line in card_text_attr(card, "evidence").splitlines():
            if not line.startswith("公司敞口："):
                continue
            for part in line.split("：", 1)[-1].split("；"):
                if ":" in part:
                    part = part.split(":", 1)[-1]
                for name in part.split("、"):
                    name = name.strip()
                    if name and name not in values:
                        values.append(name)
    return values


def strongest_exposure(cards: list[Any]) -> str:
    order = {"core": 0, "direct": 1, "indirect": 2, "mentioned": 3, "": 4}
    best = ""
    best_rank = 99
    for card in cards:
        level = card_text_attr(card, "exposure_level")
        rank = order.get(level, 4)
        if rank < best_rank:
            best = level
            best_rank = rank
    return best


def summarize_company_evidence(
    cards: list[Any],
    records: list[dict[str, Any]],
    claim_types: set[str],
) -> str:
    lines = card_lines(
        [card for card in cards if card_text_attr(card, "claim_type") in claim_types or ("" in claim_types and not card_text_attr(card, "claim_type"))],
        2,
    )
    if lines:
        return "；".join(lines)
    if "" in claim_types:
        facts = []
        for row in records[:2]:
            relation = str(row.get("relation") or "")
            target = str(row.get("target") or "")
            if target:
                facts.append(f"{relation}{target}")
        if facts:
            return "；".join(facts)
    return "当前证据不足"


def infer_record_segments(records: list[dict[str, Any]]) -> str:
    segments = unique(str(row.get("chain_segment") or "") for row in records if row.get("chain_segment"))
    return "、".join(segments[:2]) if segments else "未识别"


def cards_by_type(cards: list[Any], claim_types: set[str]) -> list[Any]:
    return [card for card in cards if card_text_attr(card, "claim_type") in claim_types]


def card_lines(cards: list[Any], limit: int) -> list[str]:
    lines = []
    for card in cards:
        text = with_citation(short_text(card_text_attr(card, "evidence"), 150), card)
        if text and text not in lines:
            lines.append(text)
        if len(lines) >= limit:
            break
    return lines


def bullet_lines(cards: list[Any], limit: int) -> str:
    lines = card_lines(cards, limit)
    if not lines:
        return "当前证据不足。"
    return "\n".join(f"- {line}" for line in lines)


def has_direct_exposure(cards: list[Any]) -> bool:
    return any(card_text_attr(card, "claim_type") == "company_exposure" and card_text_attr(card, "exposure_level") in {"core", "direct"} for card in cards)


def dossier_gap_lines(cards: list[Any]) -> list[str]:
    rows: list[str] = []
    for card in cards:
        if card_text_attr(card, "kind") != "dossier":
            continue
        for line in card_text_attr(card, "evidence").splitlines():
            if line.startswith("证据缺口："):
                values = line.split("：", 1)[-1].split("；")
                rows.extend(short_text(value.strip(), 180) for value in values if value.strip())
    return unique(rows)


def dedupe_gap_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    seen = set()
    for row in rows:
        key = normalize_key(row.get("gap", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def merge_gap_rows(*groups: list[Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            gap = str(item.get("gap") or item.get("reason") or "").strip()
            if not gap:
                continue
            rows.append(
                {
                    "gap": gap,
                    "priority": str(item.get("priority") or "中"),
                    "suggested_source": str(item.get("suggested_source") or ""),
                }
            )
    return dedupe_gap_rows(rows)[:12]


def follow_up_for_risk(risk: str) -> str:
    if any(term in risk for term in ("订单", "需求", "客户")):
        return "订单、客户导入、合同负债"
    if any(term in risk for term in ("价格", "毛利", "竞争")):
        return "ASP、毛利率、竞争格局"
    if any(term in risk for term in ("供应", "产能", "良率")):
        return "产能、良率、交付周期"
    if any(term in risk for term in ("技术", "路线", "替代")):
        return "技术路线、产品代际、客户验证"
    return "公告、年报风险披露、行业数据"


def suggested_source_for_gap(gap: str) -> str:
    if any(term in gap for term in ("订单", "收入", "毛利", "产能", "指标")):
        return "年报财务表、经营数据、公告和研报指标表。"
    if any(term in gap for term in ("风险", "反证")):
        return "年报风险披露、竞争格局和技术路线对比材料。"
    if any(term in gap for term in ("敞口", "公司")):
        return "公司产品、客户导入、订单或产业链位置证据。"
    return "补充原文报告或人工校验 Claim。"


def format_source(card: Any) -> str:
    source = card_text_attr(card, "source") or card_text_attr(card, "title")
    page = card_text_attr(card, "page")
    return f"{source} p.{page}" if source and page else source


def format_record_source(row: dict[str, Any]) -> str:
    source = str(row.get("source") or "")
    page = str(row.get("page") or "")
    return f"{source} p.{page}" if source and page else source


def with_citation(text: str, card: Any) -> str:
    citation_id = card_text_attr(card, "citation_id")
    return f"{text} [{citation_id}]" if text and citation_id else text


def card_text_attr(card: Any, name: str) -> str:
    if isinstance(card, dict):
        value = card.get(name, "")
        if name == "source" and not value:
            value = card.get("source_title", "")
        if name == "evidence" and not value:
            value = card.get("text", "")
        if name == "published_at" and not value:
            value = card.get("as_of_date", "")
        return str(value or "")
    value = getattr(card, name, "")
    if value is None:
        return ""
    return str(value)


def short_text(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def normalize_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").casefold())[:120]


def unique(values: Iterable[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
