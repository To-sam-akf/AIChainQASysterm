"""GraphRAG / DRIFT orchestration helpers for the QA agent."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from aika.domain_lexicon import company_segment, infer_themes, normalize_topic
from aika.frontend_data import RELATION_LABELS
from aika.question_planner import QuestionPlan
from aika.research_claims import ResearchHit


EXPOSURE_SCORE = {"core": 40.0, "direct": 30.0, "indirect": 16.0, "mentioned": 6.0, "": 0.0}
CLAIM_TYPE_SCORE = {
    "company_exposure": 10.0,
    "indicator": 8.0,
    "risk": 5.0,
    "bottleneck": 4.0,
    "mechanism": 3.0,
    "supply_chain": 3.0,
    "trend": 2.0,
}
METRIC_TERMS = ("指标", "订单", "业绩", "营收", "收入", "利润", "毛利", "产能", "客户导入", "渗透率")
RISK_TERMS = ("风险", "反证", "不确定", "波动", "替代", "压力", "受限", "约束")
BROAD_TERMS = ("产业链", "传导", "瓶颈", "趋势", "为什么", "怎么看", "哪些公司", "谁受益", "当前最大")


@dataclass(frozen=True)
class QueryRoute:
    kind: str
    reason: str
    use_global: bool
    use_local: bool
    use_paths: bool
    use_drift: bool
    metric_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DriftSubquestion:
    id: str
    question: str
    focus: str
    topics: list[str] = field(default_factory=list)
    claim_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphRagPath:
    path_id: str
    topic: str
    company: str
    demand: str
    technology: str
    segment: str
    indicator: str
    risk: str
    evidence_ids: list[str]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"explanation": self.explanation}

    @property
    def explanation(self) -> str:
        parts = [
            f"需求:{self.demand}",
            f"技术:{self.technology}",
            f"环节:{self.segment}",
            f"公司:{self.company}",
            f"指标:{self.indicator}",
            f"风险:{self.risk}",
        ]
        return " -> ".join(part for part in parts if not part.endswith(":"))


@dataclass(frozen=True)
class CompanyRanking:
    company: str
    topic: str
    score: float
    exposure_level: str = ""
    segment: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    indicator_evidence: str = ""
    risk_evidence: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphRagResult:
    route: QueryRoute
    subquestions: list[DriftSubquestion]
    global_hits: list[ResearchHit]
    local_hits: list[ResearchHit]
    paths: list[GraphRagPath]
    company_rankings: list[CompanyRanking]
    edges: list[dict[str, str]]
    reranker: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.to_dict(),
            "subquestions": [item.to_dict() for item in self.subquestions],
            "global_hits": [hit.to_dict() for hit in self.global_hits],
            "local_hits": [hit.to_dict() for hit in self.local_hits],
            "paths": [path.to_dict() for path in self.paths],
            "company_rankings": [ranking.to_dict() for ranking in self.company_rankings],
            "edges": list(self.edges),
            "reranker": dict(self.reranker),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "route": self.route.kind,
            "global_hits": len(self.global_hits),
            "local_hits": len(self.local_hits),
            "subquestions": len(self.subquestions),
            "paths": len(self.paths),
            "company_rankings": len(self.company_rankings),
            "reranker_source": self.reranker.get("source", "heuristic"),
            "reranker_mode": self.reranker.get("mode", ""),
        }


class QueryRouter:
    """Map a question plan to GraphRAG retrieval behavior."""

    def route(self, question: str, plan: QuestionPlan) -> QueryRoute:
        question = question.strip()
        if plan.needs_metrics or any(term in question for term in METRIC_TERMS):
            return QueryRoute("metric_only", "问题要求订单、业绩或可验证指标。", False, True, True, False, True)
        if plan.answer_type == "company_compare" or plan.needs_comparison:
            return QueryRoute("company_compare", "问题要求公司对比。", True, True, True, False)
        if plan.answer_type == "risk_analysis" or plan.needs_risk:
            return QueryRoute("risk_review", "问题要求风险或反证证据。", False, True, True, False)
        if plan.answer_type == "company_profile":
            return QueryRoute("company_profile", "问题聚焦单家公司画像。", False, True, True, False)
        if plan.answer_type == "topic_to_company" or any(term in question for term in ("哪些公司", "谁受益", "受益公司", "上市公司")):
            return QueryRoute("company_exposure", "问题要求主题到公司敞口排序。", True, True, True, False)
        return QueryRoute("global_causal", "问题要求宽主题机理、瓶颈或产业传导。", True, True, True, True)


class DriftPlanner:
    def __init__(self, *, max_subquestions: int = 6) -> None:
        self.max_subquestions = max(1, int(max_subquestions or 6))

    def plan(self, question: str, plan: QuestionPlan, route: QueryRoute) -> list[DriftSubquestion]:
        if not route.use_drift and not is_broad_question(question, plan):
            return []
        topics = graph_topics(question, plan)
        subquestions: list[DriftSubquestion] = []
        for topic in topics:
            subquestions.append(
                DriftSubquestion(
                    id=f"d{len(subquestions) + 1}",
                    question=f"{topic} 技术机理 瓶颈 产业传导",
                    focus="global_mechanism",
                    topics=[topic],
                    claim_types=["mechanism", "bottleneck", "supply_chain", "trend"],
                )
            )
            if len(subquestions) >= self.max_subquestions:
                break
            subquestions.append(
                DriftSubquestion(
                    id=f"d{len(subquestions) + 1}",
                    question=f"{topic} 公司敞口 领先指标 风险 反证",
                    focus="local_company_metric_risk",
                    topics=[topic],
                    claim_types=["company_exposure", "indicator", "risk"],
                )
            )
            if len(subquestions) >= self.max_subquestions:
                break
        return subquestions[: self.max_subquestions]


class CompanyExposureRanker:
    def rank(
        self,
        *,
        plan: QuestionPlan,
        local_hits: list[ResearchHit],
        global_hits: list[ResearchHit],
        paths: list[GraphRagPath],
        limit: int = 12,
    ) -> list[CompanyRanking]:
        del global_hits
        rows: dict[str, dict[str, Any]] = defaultdict(lambda: {"score": 0.0, "evidence_ids": []})
        path_count_by_company = defaultdict(int)
        for path in paths:
            if path.company:
                path_count_by_company[path.company] += 1

        for hit in local_hits:
            if not hit.company:
                continue
            row = rows[hit.company]
            row["company"] = hit.company
            row["topic"] = row.get("topic") or hit.topic
            row["segment"] = company_segment(hit.company)
            row["score"] += CLAIM_TYPE_SCORE.get(hit.claim_type, 1.0)
            row["score"] += EXPOSURE_SCORE.get(hit.exposure_level, 0.0)
            row["score"] += safe_score(hit.confidence) * 2.0
            row["score"] += 1.5 if hit.source_tier == "1" else 0.0
            if hit.company in plan.companies:
                row["score"] += 3.0
            if hit.claim_type == "indicator":
                row["indicator_evidence"] = row.get("indicator_evidence") or hit.text
            if hit.claim_type == "risk":
                row["risk_evidence"] = row.get("risk_evidence") or hit.text
            if hit.claim_type == "company_exposure":
                current_level = str(row.get("exposure_level") or "")
                row["exposure_level"] = stronger_exposure(current_level, hit.exposure_level)
            if hit.claim_id and hit.claim_id not in row["evidence_ids"]:
                row["evidence_ids"].append(hit.claim_id)

        for company, count in path_count_by_company.items():
            row = rows[company]
            row["company"] = company
            row["segment"] = row.get("segment") or company_segment(company)
            row["score"] += min(10.0, count * 3.0)

        rankings: list[CompanyRanking] = []
        for row in rows.values():
            company = str(row.get("company") or "")
            if not company:
                continue
            level = str(row.get("exposure_level") or "")
            topic = str(row.get("topic") or (plan.topics[0] if plan.topics else ""))
            reason_parts = []
            if level:
                reason_parts.append(f"敞口等级={level}")
            if row.get("indicator_evidence"):
                reason_parts.append("有指标证据")
            if row.get("risk_evidence"):
                reason_parts.append("有风险/反证证据")
            if path_count_by_company.get(company):
                reason_parts.append("多跳路径可解释")
            rankings.append(
                CompanyRanking(
                    company=company,
                    topic=topic,
                    score=round(float(row.get("score") or 0.0), 4),
                    exposure_level=level,
                    segment=str(row.get("segment") or ""),
                    evidence_ids=list(row.get("evidence_ids") or [])[:8],
                    indicator_evidence=shorten(row.get("indicator_evidence") or "", 180),
                    risk_evidence=shorten(row.get("risk_evidence") or "", 180),
                    reason="；".join(reason_parts) or "基于局部 Claim 证据排序",
                )
            )
        rankings.sort(key=lambda item: (-item.score, exposure_rank(item.exposure_level), item.segment, item.company))
        return rankings[:limit]


class GraphRagPathBuilder:
    def build(
        self,
        *,
        plan: QuestionPlan,
        graph_records: list[dict[str, Any]],
        local_hits: list[ResearchHit],
        rankings: list[CompanyRanking],
        limit: int = 6,
    ) -> list[GraphRagPath]:
        topics = graph_topics(plan.question, plan)
        by_company: dict[str, list[ResearchHit]] = defaultdict(list)
        for hit in local_hits:
            if hit.company:
                by_company[hit.company].append(hit)
        graph_by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in graph_records:
            company = str(record.get("company") or "")
            if company:
                graph_by_company[company].append(record)

        paths: list[GraphRagPath] = []
        for ranking in rankings:
            hits = by_company.get(ranking.company, [])
            records = graph_by_company.get(ranking.company, [])
            topic = ranking.topic or (topics[0] if topics else "")
            demand = first_text(hits, {"mechanism", "trend"}) or f"{topic or 'AI算力'}需求"
            technology = first_target(records, {"USES_TECHNOLOGY", "HAS_PRODUCT"}) or topic or first_text(hits, {"mechanism", "bottleneck"})
            segment = company_segment(ranking.company) or first_target(records, {"BELONGS_TO_CHAIN"}) or topic
            indicator = first_text(hits, {"indicator"}) or first_target(records, {"HAS_METRIC", "HAS_INDICATOR"}) or "当前指标证据不足"
            risk = first_text(hits, {"risk", "bottleneck"}) or first_target(records, {"DISCLOSES_RISK", "CONSTRAINS"}) or "当前风险证据不足"
            evidence_ids = [hit.claim_id for hit in hits if hit.claim_id][:6]
            if not any((technology, segment, ranking.company)):
                continue
            paths.append(
                GraphRagPath(
                    path_id=f"path_{len(paths) + 1}",
                    topic=topic,
                    company=ranking.company,
                    demand=shorten(demand, 80),
                    technology=shorten(technology, 80),
                    segment=shorten(segment, 60),
                    indicator=shorten(indicator, 90),
                    risk=shorten(risk, 90),
                    evidence_ids=evidence_ids,
                    score=round(ranking.score + (4.0 if indicator else 0.0) + (2.0 if risk else 0.0), 4),
                )
            )
            if len(paths) >= limit:
                break
        return paths


def run_graphrag(
    *,
    question: str,
    plan: QuestionPlan,
    research_memory: Any | None,
    graph_records: list[dict[str, Any]],
    max_subquestions: int = 6,
    global_top_k: int = 3,
    local_top_k: int = 12,
    path_top_k: int = 6,
) -> GraphRagResult:
    partial = retrieve_graphrag(
        question=question,
        plan=plan,
        research_memory=research_memory,
        max_subquestions=max_subquestions,
        global_top_k=global_top_k,
        local_top_k=local_top_k,
        path_top_k=path_top_k,
    )
    return finalize_graphrag(
        partial,
        plan=plan,
        graph_records=graph_records,
        path_top_k=path_top_k,
    )


def retrieve_graphrag(
    *,
    question: str,
    plan: QuestionPlan,
    research_memory: Any | None,
    max_subquestions: int = 6,
    global_top_k: int = 3,
    local_top_k: int = 12,
    path_top_k: int = 6,
) -> GraphRagResult:
    route = QueryRouter().route(question, plan)
    subquestions = DriftPlanner(max_subquestions=max_subquestions).plan(question, plan, route)
    global_hits: list[ResearchHit] = []
    local_hits: list[ResearchHit] = []
    if research_memory is not None:
        if route.use_global and hasattr(research_memory, "search_global_dossiers"):
            global_hits.extend(research_memory.search_global_dossiers(question, plan, limit=global_top_k))
        if route.use_local and hasattr(research_memory, "search_local_claims"):
            local_hits.extend(
                research_memory.search_local_claims(
                    question,
                    plan,
                    limit=local_top_k,
                    claim_types=claim_types_for_route(route),
                )
            )
        for subquestion in subquestions:
            if route.use_global and subquestion.focus == "global_mechanism" and hasattr(research_memory, "search_global_dossiers"):
                global_hits.extend(
                    research_memory.search_global_dossiers(
                        subquestion.question,
                        plan,
                        limit=max(1, min(2, global_top_k)),
                        topics=subquestion.topics,
                    )
                )
            if route.use_local and hasattr(research_memory, "search_local_claims"):
                local_hits.extend(
                    research_memory.search_local_claims(
                        subquestion.question,
                        plan,
                        limit=max(2, min(6, local_top_k)),
                        claim_types=subquestion.claim_types,
                        topics=subquestion.topics,
                    )
                )
    global_hits = dedupe_hits(global_hits)[: max(global_top_k, len(subquestions))]
    local_hits = dedupe_hits(local_hits)[: max(local_top_k, len(subquestions) * 3)]
    rankings = CompanyExposureRanker().rank(
        plan=plan,
        local_hits=local_hits,
        global_hits=global_hits,
        paths=[],
        limit=max(8, path_top_k * 2),
    )
    return GraphRagResult(
        route=route,
        subquestions=subquestions,
        global_hits=global_hits,
        local_hits=local_hits,
        paths=[],
        company_rankings=rankings,
        edges=[],
    )


def finalize_graphrag(
    partial: GraphRagResult,
    *,
    plan: QuestionPlan,
    graph_records: list[dict[str, Any]],
    path_top_k: int = 6,
) -> GraphRagResult:
    rankings = partial.company_rankings or CompanyExposureRanker().rank(
        plan=plan,
        local_hits=partial.local_hits,
        global_hits=partial.global_hits,
        paths=[],
        limit=max(8, path_top_k * 2),
    )
    paths = GraphRagPathBuilder().build(
        plan=plan,
        graph_records=graph_records,
        local_hits=partial.local_hits,
        rankings=rankings,
        limit=path_top_k,
    )
    if paths:
        rankings = CompanyExposureRanker().rank(
            plan=plan,
            local_hits=partial.local_hits,
            global_hits=partial.global_hits,
            paths=paths,
            limit=max(8, path_top_k * 2),
        )
    return GraphRagResult(
        route=partial.route,
        subquestions=partial.subquestions,
        global_hits=partial.global_hits,
        local_hits=partial.local_hits,
        paths=paths,
        company_rankings=rankings,
        edges=edges_from_paths(paths),
        reranker=dict(partial.reranker),
    )


def claim_types_for_route(route: QueryRoute) -> list[str]:
    if route.kind == "metric_only":
        return ["indicator"]
    if route.kind == "risk_review":
        return ["risk", "bottleneck", "company_exposure"]
    if route.kind == "company_compare":
        return ["company_exposure", "indicator", "risk", "mechanism", "supply_chain"]
    if route.kind == "company_profile":
        return ["company_exposure", "indicator", "risk", "mechanism", "supply_chain"]
    if route.kind == "company_exposure":
        return ["company_exposure", "indicator", "risk", "mechanism", "supply_chain"]
    return ["mechanism", "bottleneck", "supply_chain", "trend", "indicator", "risk", "company_exposure"]


def graph_topics(question: str, plan: QuestionPlan) -> list[str]:
    topics = list(plan.topics or [])
    topics.extend(infer_themes(question))
    if not topics and "算力" in question:
        topics.extend(["AI芯片", "算力网络", "液冷"])
    output: list[str] = []
    for topic in topics:
        if topic and topic not in output:
            output.append(topic)
        if len(output) >= 3:
            break
    return output


def is_broad_question(question: str, plan: QuestionPlan) -> bool:
    if plan.answer_type in {"industry_bottleneck", "thematic_research"}:
        return True
    return any(term in question for term in BROAD_TERMS) and not plan.companies


def dedupe_hits(hits: list[ResearchHit]) -> list[ResearchHit]:
    output: list[ResearchHit] = []
    seen: set[tuple[str, str, str, str]] = set()
    for hit in sorted(hits, key=lambda item: (-item.score, item.kind, item.topic, item.company, item.title)):
        key = (hit.kind, hit.claim_id or hit.title, hit.topic, normalize_topic(hit.text)[:120])
        if key in seen:
            continue
        seen.add(key)
        output.append(hit)
    return output


def edges_from_paths(paths: list[GraphRagPath]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for path in paths:
        items = [
            (path.demand, path.technology, "需求驱动", "DemandDriver", "Technology"),
            (path.technology, path.segment, "技术落到环节", "Technology", "ValueChainSegment"),
            (path.segment, path.company, "产业环节对应公司", "ValueChainSegment", "Company"),
            (path.company, path.indicator, "指标验证", "Company", "Metric"),
            (path.company, path.risk, "风险反证", "Company", "Risk"),
        ]
        for source, target, label, source_type, target_type in items:
            if not source or not target or source == "当前指标证据不足" or target == "当前指标证据不足":
                continue
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "label": label,
                    "source_type": source_type,
                    "target_type": target_type,
                    "source_kind": "graphrag_path",
                    "citation_id": "",
                    "claim_type": "",
                    "exposure_level": "",
                }
            )
    return dedupe_edges(edges)[:80]


def dedupe_edges(edges: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    seen = set()
    for edge in edges:
        key = (edge.get("source", ""), edge.get("target", ""), edge.get("label", ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(edge)
    return output


def first_text(hits: list[ResearchHit], claim_types: set[str]) -> str:
    for hit in hits:
        if hit.claim_type in claim_types and hit.text:
            return hit.text
    return ""


def first_target(records: list[dict[str, Any]], relations: set[str]) -> str:
    for record in records:
        if str(record.get("relation") or "") in relations:
            return str(record.get("target") or record.get("tail_name") or "")
    return ""


def stronger_exposure(current: str, candidate: str) -> str:
    return candidate if exposure_rank(candidate) < exposure_rank(current) else current


def exposure_rank(level: str) -> int:
    return {"core": 0, "direct": 1, "indirect": 2, "mentioned": 3, "": 9}.get(level, 9)


def safe_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def shorten(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."
