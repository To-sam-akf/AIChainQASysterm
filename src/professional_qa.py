"""Professional graph retrieval, evidence ranking, and fallback answers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from src.domain_lexicon import (
    BOTTLENECK_TERMS,
    TECHNICAL_SOURCE_TYPES,
    company_groups_by_segment,
    company_segment,
    is_core_company,
    is_disclaimer_text,
    normalize_topic,
    text_matches_terms,
)
from src.frontend_data import LocalKnowledgeGraph, RELATION_LABELS
from src.question_planner import QuestionPlan
from src.rag_index import RagHit
from src.research_claims import ResearchHit, query_focus_terms


@dataclass(frozen=True)
class EvidenceCard:
    kind: str
    title: str
    evidence: str
    source: str = ""
    page: str = ""
    section: str = ""
    company: str = ""
    relation: str = ""
    target: str = ""
    source_tier: str = ""
    score: float = 0.0
    reason: str = ""
    topic: str = ""
    claim_type: str = ""
    exposure_level: str = ""
    confidence: str = ""
    as_of_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["relation_label"] = RELATION_LABELS.get(self.relation, self.relation)
        return row


def search_csv_graph(
    graph: LocalKnowledgeGraph,
    plan: QuestionPlan,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    rows = graph.relation_candidates(exclude_relations={"MENTIONED_IN"})
    if plan.answer_type == "topic_to_company":
        rows = graph.relation_candidates(
            relations=plan.relations,
            head_type="Company",
            exclude_relations={"MENTIONED_IN"},
        )
        rows = apply_core_filter(rows, plan)
        rows = [row for row in rows if row_matches_plan(row, plan)]
        return normalize_graph_records(top_rows_by_company(rows, plan, max_per_company=2))[:limit]
    if plan.answer_type == "company_compare":
        rows = graph.relation_candidates(
            companies=plan.companies,
            relations=plan.relations,
            head_type="Company",
            exclude_relations={"MENTIONED_IN"},
        )
        if plan.expanded_topics:
            matched = [row for row in rows if row_matches_plan(row, plan)]
            rows = matched or rows
        return normalize_graph_records(top_rows_by_company(rows, plan, max_per_company=8))[:limit]
    if plan.answer_type == "risk_analysis":
        rows = graph.relation_candidates(
            companies=plan.companies,
            head_type="Company",
            exclude_relations={"MENTIONED_IN"},
        )
        risk_rows = [row for row in rows if row.get("relation") == "DISCLOSES_RISK"]
        business_rows = [row for row in rows if row.get("relation") in {"USES_TECHNOLOGY", "HAS_PRODUCT", "BELONGS_TO_CHAIN"} and row_matches_plan(row, plan)]
        return normalize_graph_records(top_rows(business_rows, plan, 12) + top_rows(risk_rows, plan, 14))[:limit]
    if plan.answer_type == "industry_bottleneck":
        rows = [
            row
            for row in rows
            if row.get("relation") in {"CONSTRAINS", "DISCLOSES_RISK", "SUPPORTED_BY_POLICY", "ENABLES"}
            or any(term in row_text(row) for term in BOTTLENECK_TERMS)
        ]
        if plan.expanded_topics:
            matched = [row for row in rows if row_matches_plan(row, plan)]
            rows = matched or rows
        return normalize_graph_records(top_rows(rows, plan, limit))[:limit]
    if plan.companies:
        rows = graph.relation_candidates(
            companies=plan.companies,
            relations=plan.relations,
            head_type="Company",
            exclude_relations={"MENTIONED_IN"},
        )
        if plan.expanded_topics:
            matched = [row for row in rows if row_matches_plan(row, plan)]
            rows = matched or rows
        return normalize_graph_records(top_rows(rows, plan, limit))[:limit]
    rows = apply_core_filter([row for row in rows if row.get("head_type") == "Company"], plan)
    if plan.expanded_topics:
        rows = [row for row in rows if row_matches_plan(row, plan)]
    return normalize_graph_records(top_rows(rows, plan, limit))[:limit]


def apply_core_filter(rows: list[dict[str, str]], plan: QuestionPlan) -> list[dict[str, str]]:
    if not plan.core_companies_only:
        return rows
    return [row for row in rows if is_core_company(row.get("head_name", ""))]


def row_text(row: dict[str, str]) -> str:
    return " ".join(
        str(row.get(key, "") or "")
        for key in ("head_name", "tail_name", "relation", "evidence", "source_title", "section")
    )


def row_matches_plan(row: dict[str, str], plan: QuestionPlan) -> bool:
    if not plan.expanded_topics:
        return True
    return text_matches_terms(row_text(row), plan.expanded_topics)


def score_row(row: dict[str, str], plan: QuestionPlan) -> float:
    score = 0.0
    relation = row.get("relation", "")
    score += {
        "HAS_PRODUCT": 5.0,
        "USES_TECHNOLOGY": 4.5,
        "BELONGS_TO_CHAIN": 4.0,
        "DISCLOSES_RISK": 4.0,
        "HAS_METRIC": 3.2,
        "CONSTRAINS": 4.2,
        "SUPPORTED_BY_POLICY": 2.5,
    }.get(relation, 1.0)
    if row_matches_plan(row, plan):
        score += 5.0
    if row.get("source_tier") == "1":
        score += 1.0
    if row.get("head_name") in plan.companies:
        score += 2.0
    if is_disclaimer_text(row.get("evidence", "")):
        score -= 8.0
    evidence_len = len(row.get("evidence", ""))
    if evidence_len < 12:
        score -= 1.5
    elif evidence_len > 40:
        score += 0.8
    return score


def top_rows(rows: list[dict[str, str]], plan: QuestionPlan, limit: int) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (-score_row(row, plan), row.get("head_name", ""), row.get("tail_name", "")))[:limit]


def top_rows_by_company(rows: list[dict[str, str]], plan: QuestionPlan, *, max_per_company: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in top_rows(rows, plan, max(len(rows), 1)):
        company = row.get("head_name", "")
        if len(grouped[company]) < max_per_company:
            grouped[company].append(row)
    output: list[dict[str, str]] = []
    for company in sorted(grouped, key=lambda name: (company_segment(name), name)):
        output.extend(grouped[company])
    return output


def normalize_graph_records(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        if row.get("head_type") == "Company":
            company = row.get("head_name", "")
            target = row.get("tail_name", "")
            target_labels = [row.get("tail_type", "")]
        else:
            company = row.get("head_name", "")
            target = row.get("tail_name", "")
            target_labels = [row.get("tail_type", "")]
        records.append(
            {
                "company": company,
                "company_labels": [row.get("head_type", "")],
                "relation": row.get("relation", ""),
                "target": target,
                "target_labels": target_labels,
                "evidence": row.get("evidence", ""),
                "source": row.get("source_title", ""),
                "source_tier": row.get("source_tier", ""),
                "page": row.get("page", ""),
                "section": row.get("section", ""),
                "report_id": row.get("source_report_id", ""),
                "chain_segment": company_segment(company),
                "head_type": row.get("head_type", ""),
                "head_name": row.get("head_name", ""),
                "tail_type": row.get("tail_type", ""),
                "tail_name": row.get("tail_name", ""),
            }
        )
    return records


def cards_from_graph_records(records: list[dict[str, Any]], plan: QuestionPlan) -> list[EvidenceCard]:
    cards = []
    for record in records:
        evidence = str(record.get("evidence") or "").strip()
        if not evidence:
            continue
        title = f"{record.get('company', '')} {RELATION_LABELS.get(str(record.get('relation', '')), record.get('relation', ''))} {record.get('target', '')}".strip()
        row = {
            "head_name": str(record.get("company", "")),
            "tail_name": str(record.get("target", "")),
            "relation": str(record.get("relation", "")),
            "evidence": evidence,
            "source_tier": str(record.get("source_tier", "")),
            "section": str(record.get("section", "")),
        }
        cards.append(
            EvidenceCard(
                kind="graph",
                title=title,
                evidence=evidence,
                source=str(record.get("source", "")),
                page=str(record.get("page", "")),
                section=str(record.get("section", "")),
                company=str(record.get("company", "")),
                relation=str(record.get("relation", "")),
                target=str(record.get("target", "")),
                source_tier=str(record.get("source_tier", "")),
                score=round(score_row(row, plan), 4),
                reason="图谱结构化关系",
            )
        )
    return cards


def cards_from_rag_hits(hits: list[RagHit], plan: QuestionPlan) -> list[EvidenceCard]:
    cards = []
    for hit in hits:
        score = float(hit.score)
        if hit.source_tier == "1":
            score += 1.0
        if hit.source_type == "authority_whitepaper":
            score += 1.0
        if hit.source_type in TECHNICAL_SOURCE_TYPES and plan.answer_type in {"industry_bottleneck", "thematic_research"}:
            score += 1.0
        if hit.company and hit.company in plan.companies:
            score += 1.2
        if plan.expanded_topics and any(normalize_topic(term) in normalize_topic(hit.snippet) for term in plan.expanded_topics):
            score += 1.5
        if is_disclaimer_text(hit.snippet):
            score -= 5.0
        cards.append(
            EvidenceCard(
                kind="rag",
                title=f"{hit.company or hit.source_type} 原文片段".strip(),
                evidence=hit.snippet,
                source=hit.source_title,
                page=str(hit.page),
                section=hit.section,
                company=hit.company,
                source_tier=hit.source_tier,
                score=round(score, 4),
                reason="本地文档检索",
            )
        )
    return cards


def cards_from_research_hits(hits: list[ResearchHit], plan: QuestionPlan) -> list[EvidenceCard]:
    cards = []
    focus_terms = query_focus_terms(plan.question)
    for hit in hits:
        score = hit.score + (60.0 if hit.kind == "dossier" else 45.0)
        if focus_terms:
            text = f"{hit.title} {hit.text} {hit.source} {hit.section}"
            if text_matches_terms(text, focus_terms):
                score += 8.0 if hit.kind == "claim" else 2.0
            elif hit.kind == "dossier":
                score -= 18.0
            else:
                score -= 4.0
        cards.append(
            EvidenceCard(
                kind=hit.kind,
                title=hit.title,
                evidence=hit.text,
                source=hit.source,
                page=hit.page,
                section=hit.section,
                company=hit.company,
                source_tier=hit.source_tier,
                score=round(score, 4),
                reason="投研 Claim/Dossier 证据包" if hit.kind == "claim" else "产业链社区摘要",
                topic=hit.topic,
                claim_type=hit.claim_type,
                exposure_level=hit.exposure_level,
                confidence=hit.confidence,
                as_of_date=hit.as_of_date,
            )
        )
    return cards


def rank_evidence_cards(cards: list[EvidenceCard], *, limit: int = 10) -> list[EvidenceCard]:
    deduped = []
    seen = set()
    for card in sorted(cards, key=lambda item: (-item.score, item.kind, item.source, item.page)):
        key = re.sub(r"\s+", "", card.evidence)[:80]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)
    return deduped[:limit]


def legacy_evidence_rows(cards: list[EvidenceCard]) -> list[dict[str, Any]]:
    return [
        {
            "kind": card.kind,
            "source": card.source,
            "source_tier": card.source_tier,
            "page": card.page,
            "section": card.section,
            "evidence": card.evidence,
            "score": card.score,
            "company": card.company,
            "relation": RELATION_LABELS.get(card.relation, card.relation),
            "target": card.target,
            "reason": card.reason,
        }
        for card in cards
    ]


def build_professional_answer_prompt(
    question: str,
    plan: QuestionPlan,
    graph_records: list[dict[str, Any]],
    cards: list[EvidenceCard],
) -> str:
    payload = {
        "question": question,
        "question_plan": plan.to_dict(),
        "neo4j_or_csv_graph_records": graph_records[:30],
        "evidence_cards": [card.to_dict() for card in cards[:12]],
    }
    return (
        "请基于以下证据包回答，不要使用证据外信息。优先使用 claim/dossier 形成投研逻辑，"
        "再用 graph/rag 作为原文支撑。每个关键判断都要能对应证据。\n"
        "输出结构固定为：核心判断、技术机理、产业传导、公司排序、领先指标、反证/边界、证据。\n"
        "如果某一项没有证据，明确写“当前证据不足”，不要编造。\n"
    ) + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def fallback_professional_answer(plan: QuestionPlan, cards: list[EvidenceCard], graph_records: list[dict[str, Any]]) -> str:
    if not cards:
        return "当前知识库未找到足够证据支持回答。建议补充更精确的公司、产业链环节或报告范围后再检索。"
    research_cards = [card for card in cards if card.kind in {"claim", "dossier"}]
    if research_cards and plan.answer_type in {"thematic_research", "industry_bottleneck"}:
        return fallback_research_answer(plan, research_cards, cards)
    if research_cards and plan.answer_type == "topic_to_company":
        exposure = format_exposure_ranking(research_cards, focus_topic(plan))
        if exposure:
            topic = "、".join(plan.topics or plan.expanded_topics[:2]) or "该主题"
            return (
                f"核心判断：{topic}相关公司需要按敞口强弱分层，而不是简单罗列。\n\n"
                f"公司排序：{exposure}\n\n"
                f"技术机理：{format_claims_by_type(research_cards, {'mechanism', 'supply_chain'}, 3)}\n\n"
                f"领先指标：{format_claims_by_type(research_cards, {'indicator'}, 3)}\n\n"
                f"反证/边界：{format_claims_by_type(research_cards, {'risk', 'bottleneck'}, 3)}\n\n"
                f"证据：{format_evidence(cards, 5)}"
            )
    if plan.answer_type == "topic_to_company":
        companies = unique([record.get("company", "") for record in graph_records if record.get("company")])
        groups = company_groups_by_segment(companies)
        group_text = "；".join(f"{segment}：{'、'.join(names)}" for segment, names in groups.items())
        topic = "、".join(plan.topics or plan.expanded_topics[:2]) or "该主题"
        evidence = format_evidence(cards, 3)
        return (
            f"结论：当前专业知识库中，{topic}相关的核心 A 股公司主要包括：{'、'.join(companies[:18])}。\n\n"
            f"产业链位置：{group_text or '证据不足以稳定分组'}。\n\n"
            f"证据：{evidence}\n\n"
            "研究要点：这类问题应继续跟踪订单兑现、产品代际升级、客户导入节奏和产能释放。"
            "风险与边界：以上为基于已入库报告的事实归纳，不构成投资建议。"
        )
    if plan.answer_type == "company_compare":
        sections = []
        for company in plan.companies:
            targets = unique(
                [
                    f"{RELATION_LABELS.get(str(record.get('relation', '')), record.get('relation', ''))}{record.get('target', '')}"
                    for record in graph_records
                    if record.get("company") == company and record.get("target")
                ]
            )
            sections.append(f"{company}：{'、'.join(targets[:8]) or '当前证据不足'}")
        return (
            "结论：两家公司可从产品代际、客户结构、产业链位置和风险暴露四个维度比较。\n\n"
            + "\n".join(sections)
            + f"\n\n证据：{format_evidence(cards, 4)}\n\n"
            "研究要点：重点看高端产品放量节奏、海外云厂商资本开支、价格压力和供应链约束。"
        )
    if plan.answer_type == "risk_analysis":
        company = "、".join(plan.companies) or "该公司"
        risks = unique([card.target for card in cards if card.relation == "DISCLOSES_RISK" and card.target])
        return (
            f"结论：{company}的业务进展和风险需要分开看，当前证据中可确认的风险包括：{'、'.join(risks[:10]) or '见下方证据'}。\n\n"
            f"业务证据：{format_evidence([card for card in cards if card.relation != 'DISCLOSES_RISK'], 3)}\n\n"
            f"风险证据：{format_evidence([card for card in cards if card.relation == 'DISCLOSES_RISK'] or cards, 4)}\n\n"
            "跟踪指标：订单交付、毛利率、应收账款、客户集中度、海外业务和政策变化。"
        )
    if plan.answer_type == "industry_bottleneck":
        return (
            "结论：AI 算力产业链瓶颈通常不只在单一环节，需要同时观察芯片/加速卡供给、功耗与电力、散热、网络互联和数据中心交付能力。\n\n"
            f"证据：{format_evidence(cards, 5)}\n\n"
            "研究要点：若需求持续上行，受益顺序取决于瓶颈是否从芯片供给外溢到液冷、光模块、交换机、电源和数据中心基础设施。"
            "以上仅为证据归纳，不构成投资建议。"
        )
    return (
        f"结论：当前知识库找到 {len(cards)} 条相关证据，可支持对问题做事实归纳。\n\n"
        f"证据：{format_evidence(cards, 5)}\n\n"
        "研究要点：建议结合公司所处产业链环节、产品代际、客户验证、财务兑现和风险披露继续跟踪。"
    )


def fallback_research_answer(plan: QuestionPlan, research_cards: list[EvidenceCard], all_cards: list[EvidenceCard]) -> str:
    topic = "、".join(plan.topics or unique([card.topic for card in research_cards])[:2]) or "该主题"
    focus_terms = query_focus_terms(plan.question)
    focused_claims = focus_cards(research_cards, focus_terms)
    primary_cards = focused_claims or research_cards
    core_judgment = ""
    if focused_claims:
        core_judgment = focused_claims[0].evidence.replace("\n", "；").strip()
    if not core_judgment:
        core_judgment = next((card.evidence.splitlines()[0] for card in research_cards if card.kind == "dossier"), "")
    if not core_judgment:
        core_judgment = f"{topic} 的判断需要同时看技术瓶颈、产业传导、公司直接敞口和可验证指标。"
    technical = format_claims_by_type(primary_cards, {"mechanism", "supply_chain", "bottleneck", "trend"}, 4)
    if technical == "当前证据不足。":
        technical = format_dossier_lines(research_cards, ("技术机理", "瓶颈"))
    transmission = format_dossier_lines(research_cards, ("公司敞口",)) or format_claims_by_type(research_cards, {"company_exposure", "policy"}, 4)
    indicators = format_claims_by_type(primary_cards, {"indicator"}, 4)
    if indicators == "当前证据不足。":
        indicators = format_dossier_lines(research_cards, ("领先指标",)) or indicators
    boundaries = format_dossier_lines(research_cards, ("风险与反证", "证据缺口")) or format_claims_by_type(primary_cards, {"risk"}, 4)
    return (
        f"核心判断：{core_judgment}\n\n"
        f"技术机理：{technical}\n\n"
        f"产业传导：{transmission}\n\n"
        f"公司排序：{format_exposure_ranking(research_cards, focus_topic(plan)) or '当前证据不足以稳定排序。'}\n\n"
        f"领先指标：{indicators}\n\n"
        f"反证/边界：{boundaries}\n\n"
        f"证据：{format_evidence(all_cards, 6)}"
    )


def format_claims_by_type(cards: list[EvidenceCard], claim_types: set[str], limit: int) -> str:
    values = []
    for card in cards:
        if card.kind == "dossier":
            continue
        if card.claim_type in claim_types or (card.kind == "dossier" and not card.claim_type):
            text = card.evidence.replace("\n", "；").strip()
            if text and text not in values:
                values.append(text[:180] + ("..." if len(text) > 180 else ""))
        if len(values) >= limit:
            break
    return "；".join(values) if values else "当前证据不足。"


def focus_cards(cards: list[EvidenceCard], focus_terms: list[str]) -> list[EvidenceCard]:
    if not focus_terms:
        return []
    focused = [
        card
        for card in cards
        if card.kind == "claim"
        and text_matches_terms(f"{card.title} {card.evidence} {card.source} {card.section}", focus_terms)
    ]
    return sorted(focused, key=lambda card: (-card.score, card.claim_type, card.source, card.page))


def format_dossier_lines(cards: list[EvidenceCard], labels: tuple[str, ...]) -> str:
    values = []
    for card in cards:
        if card.kind != "dossier":
            continue
        for line in card.evidence.splitlines():
            line = line.strip()
            if any(line.startswith(f"{label}：") for label in labels):
                values.append(line.split("：", 1)[-1][:220])
    return "；".join(values[:4])


def focus_topic(plan: QuestionPlan) -> str:
    return next((topic for topic in plan.topics if topic), "")


def format_exposure_ranking(cards: list[EvidenceCard], topic: str = "") -> str:
    grouped: dict[str, list[str]] = {"core": [], "direct": [], "indirect": [], "mentioned": []}
    labels = {"core": "核心敞口", "direct": "直接敞口", "indirect": "间接敞口", "mentioned": "仅提及"}
    for card in cards:
        if topic and card.topic and card.topic != topic:
            continue
        if card.kind == "dossier":
            for line in card.evidence.splitlines():
                if not line.startswith("公司敞口："):
                    continue
                for part in line.split("：", 1)[-1].split("；"):
                    if ":" not in part:
                        continue
                    level, names = part.split(":", 1)
                    if level not in grouped:
                        continue
                    for name in names.split("、"):
                        name = name.strip()
                        if name and name not in grouped[level]:
                            grouped[level].append(name)
        if card.claim_type != "company_exposure" or not card.company:
            continue
        level = card.exposure_level if card.exposure_level in grouped else "mentioned"
        if card.company not in grouped[level]:
            grouped[level].append(card.company)
    parts = [f"{labels[level]}：{'、'.join(names[:12])}" for level, names in grouped.items() if names]
    return "；".join(parts)


def format_evidence(cards: list[EvidenceCard], limit: int) -> str:
    if not cards:
        return "当前证据不足。"
    parts = []
    for card in cards[:limit]:
        source = f"{card.source} p.{card.page}" if card.page else card.source
        text = card.evidence.replace("\n", " ").strip()
        if len(text) > 120:
            text = text[:117] + "..."
        parts.append(f"{source}：{text}")
    return "；".join(parts) + "。"


def pseudo_cypher_for_plan(plan: QuestionPlan, limit: int = 50) -> str:
    rels = "|".join(plan.relations or ["USES_TECHNOLOGY", "HAS_PRODUCT", "BELONGS_TO_CHAIN"])
    where = []
    if plan.companies:
        where.append("c.name IN $companies")
    if plan.expanded_topics:
        where.append("(x.name CONTAINS $topic OR r.evidence CONTAINS $topic)")
    where_clause = "\nWHERE " + " AND ".join(where) if where else ""
    return (
        f"MATCH (c:Company)-[r:{rels}]->(x){where_clause}\n"
        "RETURN c.name AS company, type(r) AS relation, x.name AS target, "
        "r.evidence AS evidence, r.source_title AS source, r.page AS page\n"
        f"LIMIT {limit}"
    )


def unique(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
