"""CSV/JSONL backend for the lightweight public AIKA Core path."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from src.aika_core.claims import load_claims, search_claim_records
from src.aika_core.config import AikaCoreConfig
from src.aika_core.evidence import evidence_cards_from_claims, standardize_evidence_cards
from src.aika_core.graph import edge_from_graph_record, load_graph, query_graph_edges
from src.aika_core.models import (
    ClaimRecord,
    CompanyComparison,
    CompanyProfile,
    EvidenceCard,
    EvidenceGap,
    GraphEdge,
    ResearchBackend,
    ResearchBrief,
)
from src.domain_lexicon import expanded_terms
from src.professional_qa import (
    EvidenceCard as LegacyEvidenceCard,
    cards_from_graph_records,
    cards_from_research_hits,
    rank_evidence_cards,
    search_csv_graph,
)
from src.question_planner import QuestionPlan, heuristic_plan_question
from src.research_agent import build_research_outputs
from src.research_claims import ResearchMemory


class CSVResearchBackend(ResearchBackend):
    """Core backend backed only by local CSV and JSONL artifacts."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        config: AikaCoreConfig | None = None,
        graph: Any | None = None,
        research_memory: ResearchMemory | None = None,
    ) -> None:
        self.config = config or AikaCoreConfig.from_dir(data_dir)
        self.graph = graph if graph is not None else load_graph(self.config.graph_dir)
        self.claims = load_claims(self.config.claims_path)
        self.research_memory = research_memory if research_memory is not None else self._load_research_memory()

    @classmethod
    def from_env(cls) -> "CSVResearchBackend":
        return cls(config=AikaCoreConfig.from_env())

    @classmethod
    def from_dir(cls, data_dir: str | Path | None = None) -> "CSVResearchBackend":
        return cls(data_dir=data_dir)

    def search_claims(self, query: str, *, top_k: int = 8, **filters: Any) -> list[ClaimRecord]:
        limit = max(0, int(top_k))
        if limit == 0:
            return []
        records: list[ClaimRecord] = []
        plan = self._plan(query, **filters)
        if self.research_memory is not None:
            hits = self.research_memory.search_local_claims(
                query,
                plan,
                limit=max(limit * 3, limit),
                claim_types=filters.get("claim_types") or filters.get("claim_type"),
                companies=filters.get("companies") or filters.get("company"),
                topics=filters.get("topics") or filters.get("topic"),
            )
            records.extend(ClaimRecord.from_research_hit(hit) for hit in hits if hit.kind == "claim")
        if len(records) < limit:
            records.extend(search_claim_records(self.claims, query, top_k=max(limit * 3, limit), **filters))
        return _dedupe_claims(records)[:limit]

    def search_evidence(self, query: str, *, top_k: int = 8, **filters: Any) -> list[EvidenceCard]:
        limit = max(0, int(top_k))
        if limit == 0:
            return []
        plan = self._plan(query, **filters)
        legacy_cards, _ = self._collect_legacy_evidence(query, plan, limit=limit)
        if legacy_cards:
            return standardize_evidence_cards(legacy_cards)
        return evidence_cards_from_claims(self.search_claims(query, top_k=limit, **filters))[:limit]

    def query_graph(
        self,
        *,
        company: str = "",
        technology: str = "",
        relation_type: str = "",
        limit: int = 80,
    ) -> list[GraphEdge]:
        if self.graph is None:
            return []
        edges = query_graph_edges(
            self.graph,
            company=company,
            technology=technology,
            relation_type=relation_type,
            limit=limit,
        )
        if edges or not company or (relation_type and relation_type != "MENTIONED_IN"):
            return edges
        return self._chunk_edges_for_company(company, limit=limit)

    def get_company_profile(self, company: str, *, topic: str = "") -> CompanyProfile:
        question = f"{company}{topic}业务画像，包含证据、指标和风险" if topic else f"{company}公司画像，包含证据、指标和风险"
        plan = self._plan(question, company=company, topic=topic)
        legacy_cards, graph_records = self._collect_legacy_evidence(question, plan, limit=10)
        outputs = self._research_outputs(question, plan, legacy_cards, graph_records)
        table_rows = outputs.get("company_compare_table", {}).get("rows") or []
        row = next((item for item in table_rows if str(item.get("company") or "") == company), {})
        return CompanyProfile(
            company=company,
            topic=topic,
            summary=_section_content(outputs, "核心判断"),
            business_position=str(row.get("chain_segment") or ""),
            technology_products=_split_summary(str(row.get("business_evidence") or "")),
            indicators=_split_summary(str(row.get("leading_indicators") or "")),
            risks=list(outputs.get("risk_checklist") or []),
            evidence_cards=standardize_evidence_cards(legacy_cards),
            graph_edges=[edge_from_graph_record(record) for record in graph_records],
            evidence_gaps=[EvidenceGap.from_row(row) for row in outputs.get("evidence_gaps", [])],
            research_outputs=outputs,
            meta=dict(outputs.get("meta") or {}),
        )

    def compare_companies(self, companies: Iterable[str], *, topic: str = "") -> CompanyComparison:
        company_list = [str(company).strip() for company in companies if str(company).strip()]
        question = f"{'和'.join(company_list)}在{topic}业务上的差异、风险和跟踪指标是什么？" if topic else f"{'和'.join(company_list)}业务差异、风险和跟踪指标是什么？"
        plan = self._plan(question, companies=company_list, topic=topic)
        legacy_cards, graph_records = self._collect_legacy_evidence(question, plan, limit=12)
        outputs = self._research_outputs(question, plan, legacy_cards, graph_records)
        table = outputs.get("company_compare_table", {}) or {}
        return CompanyComparison(
            companies=company_list,
            topic=topic,
            columns=list(table.get("columns") or []),
            rows=list(table.get("rows") or []),
            evidence_cards=standardize_evidence_cards(legacy_cards),
            evidence_gaps=[EvidenceGap.from_row(row) for row in outputs.get("evidence_gaps", [])],
            research_outputs=outputs,
        )

    def audit_evidence_gaps(
        self,
        query: str,
        *,
        companies: Iterable[str] | None = None,
        topic: str = "",
    ) -> list[EvidenceGap]:
        company_list = [str(company).strip() for company in (companies or []) if str(company).strip()]
        subject = query or "、".join(company_list) or topic or "AI算力产业链"
        question = f"{subject} 证据缺口审计，关注公司、指标、风险和反证"
        plan = self._plan(question, companies=company_list, topic=topic)
        legacy_cards, graph_records = self._collect_legacy_evidence(question, plan, limit=10)
        outputs = self._research_outputs(question, plan, legacy_cards, graph_records)
        return [EvidenceGap.from_row(row) for row in outputs.get("evidence_gaps", [])]

    def build_research_brief(self, query: str, *, topic: str = "") -> ResearchBrief:
        question = query or (f"{topic}投研简报" if topic else "AI算力产业链投研简报")
        plan = self._plan(question, topic=topic)
        legacy_cards, graph_records = self._collect_legacy_evidence(question, plan, limit=12)
        outputs = self._research_outputs(question, plan, legacy_cards, graph_records)
        report = outputs.get("report", {}) or {}
        return ResearchBrief(
            title=str(report.get("title") or "投研简报"),
            markdown=str(report.get("markdown") or ""),
            sections=list(report.get("sections") or []),
            evidence_cards=standardize_evidence_cards(legacy_cards),
            evidence_gaps=[EvidenceGap.from_row(row) for row in outputs.get("evidence_gaps", [])],
            meta=dict(outputs.get("meta") or {}),
            research_outputs=outputs,
        )

    def _load_research_memory(self) -> ResearchMemory | None:
        try:
            return ResearchMemory.load(self.config.research_dir)
        except FileNotFoundError:
            return None

    def _plan(self, query: str, **filters: Any) -> QuestionPlan:
        query_text = _query_with_filters(query, filters)
        plan = heuristic_plan_question(query_text)
        companies = _values(filters.get("companies") or filters.get("company"))
        topics = _values(filters.get("topics") or filters.get("topic"))
        if companies:
            plan = replace(plan, companies=_dedupe([*plan.companies, *companies]))
        if topics:
            merged_topics = _dedupe([*plan.topics, *topics])
            plan = replace(plan, topics=merged_topics, expanded_topics=expanded_terms(merged_topics))
        return plan

    def _collect_legacy_evidence(
        self,
        query: str,
        plan: QuestionPlan,
        *,
        limit: int,
    ) -> tuple[list[LegacyEvidenceCard], list[dict[str, Any]]]:
        graph_records = search_csv_graph(self.graph, plan, limit=max(limit * 4, 20)) if self.graph is not None else []
        if not graph_records and plan.companies:
            graph_records = [
                _graph_record_from_edge(edge)
                for company in plan.companies
                for edge in self._chunk_edges_for_company(company, limit=max(limit, 4))
            ]
        cards: list[LegacyEvidenceCard] = []
        if self.research_memory is not None:
            hits = self.research_memory.search(query, plan, limit=max(limit * 3, limit))
            cards.extend(cards_from_research_hits(hits, plan))
        if graph_records:
            cards.extend(cards_from_graph_records(graph_records, plan))
        if not cards:
            cards.extend(_legacy_cards_from_claims(self.search_claims(query, top_k=limit)))
        ranked = rank_evidence_cards(cards, limit=limit, plan=plan) if cards else []
        return ranked, graph_records

    def _research_outputs(
        self,
        question: str,
        plan: QuestionPlan,
        legacy_cards: list[LegacyEvidenceCard],
        graph_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return build_research_outputs(
            question=question,
            plan=plan,
            evidence_cards=legacy_cards,
            graph_records=graph_records,
            verification={"status": "not_run", "checks": {}},
        )

    def _chunk_edges_for_company(self, company: str, *, limit: int = 8) -> list[GraphEdge]:
        chunks_dir = self.config.data_dir.parent / "chunks"
        if not chunks_dir.exists():
            return []
        edges: list[GraphEdge] = []
        seen_reports: set[str] = set()
        for path in sorted(chunks_dir.glob("*.jsonl")):
            if len(edges) >= max(0, int(limit)):
                break
            try:
                lines = path.open(encoding="utf-8")
            except OSError:
                continue
            with lines as file:
                for line in file:
                    if len(edges) >= max(0, int(limit)):
                        break
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk_company = str(chunk.get("company") or "")
                    text = str(chunk.get("text") or "")
                    if chunk_company != company and company not in text:
                        continue
                    report_id = str(chunk.get("report_id") or path.stem)
                    if report_id in seen_reports:
                        continue
                    seen_reports.add(report_id)
                    source_title = str(chunk.get("source_title") or report_id)
                    edges.append(
                        GraphEdge(
                            source=company,
                            target=source_title,
                            relation="MENTIONED_IN",
                            label="来源报告",
                            source_type="Company",
                            target_type="Report",
                            evidence=_short_text(text, 240),
                            source_title=source_title,
                            page=str(chunk.get("page") or ""),
                            section=str(chunk.get("section") or ""),
                            source_tier=str(chunk.get("source_tier") or ""),
                            report_id=report_id,
                            raw=chunk,
                        )
                    )
        return edges


def _legacy_cards_from_claims(claims: Iterable[ClaimRecord]) -> list[LegacyEvidenceCard]:
    cards = []
    for claim in claims:
        cards.append(
            LegacyEvidenceCard(
                citation_id="",
                kind="claim",
                title=f"{claim.topic} {claim.claim_type}".strip(),
                evidence=claim.claim_text,
                claim_id=claim.claim_id,
                source=claim.source_title,
                page=claim.page,
                section=claim.section,
                company=claim.companies[0] if claim.companies else "",
                source_tier=claim.source_tier,
                score=claim.score,
                reason="curated claim",
                topic=claim.topic,
                claim_type=claim.claim_type,
                exposure_level=claim.exposure_level,
                confidence=claim.confidence,
                as_of_date=claim.as_of_date,
                evidence_span=claim.evidence_span,
                review_status=claim.review_status,
                reviewer_note=claim.reviewer_note,
            )
        )
    return cards


def _graph_record_from_edge(edge: GraphEdge) -> dict[str, Any]:
    return {
        "company": edge.source,
        "company_labels": [edge.source_type],
        "relation": edge.relation,
        "target": edge.target,
        "target_labels": [edge.target_type],
        "evidence": edge.evidence,
        "source": edge.source_title,
        "source_tier": edge.source_tier,
        "page": edge.page,
        "section": edge.section,
        "report_id": edge.report_id,
        "chain_segment": "",
        "head_type": edge.source_type,
        "head_name": edge.source,
        "tail_type": edge.target_type,
        "tail_name": edge.target,
    }


def _query_with_filters(query: str, filters: dict[str, Any]) -> str:
    parts = [str(query or "").strip()]
    parts.extend(_values(filters.get("company") or filters.get("companies")))
    parts.extend(_values(filters.get("topic") or filters.get("topics")))
    return " ".join(part for part in parts if part)


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _dedupe(values: Iterable[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        key = str(value).strip()
        if key and key not in seen:
            seen.add(key)
            output.append(key)
    return output


def _dedupe_claims(records: Iterable[ClaimRecord]) -> list[ClaimRecord]:
    output = []
    seen = set()
    for record in records:
        key = record.claim_id or (record.topic, record.claim_text[:80])
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def _section_content(outputs: dict[str, Any], title: str) -> str:
    report = outputs.get("report", {}) or {}
    for section in report.get("sections") or []:
        if section.get("title") == title:
            return str(section.get("content") or "")
    return ""


def _split_summary(value: str) -> list[str]:
    if not value or value == "当前证据不足":
        return []
    return [item.strip() for item in value.split("；") if item.strip()]


def _short_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
