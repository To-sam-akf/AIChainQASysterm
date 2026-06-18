"""Evidence coverage checks for dynamic supplement retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from aika.agents.planner import AgentTaskPlan, query_suffix_for_coverage
from aika.question_planner import QuestionPlan


@dataclass(frozen=True)
class EvidenceGap:
    coverage: str
    reason: str
    query_suffix: str
    companies: list[str] = field(default_factory=list)
    priority: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageReport:
    status: str
    required: list[str]
    satisfied: list[str]
    missing: list[str]
    gaps: list[EvidenceGap]
    stop_reason: str
    retrieval_round: int
    max_retrieval_rounds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "required": list(self.required),
            "satisfied": list(self.satisfied),
            "missing": list(self.missing),
            "gaps": [gap.to_dict() for gap in self.gaps],
            "stop_reason": self.stop_reason,
            "retrieval_round": self.retrieval_round,
            "max_retrieval_rounds": self.max_retrieval_rounds,
        }

    @property
    def sufficient(self) -> bool:
        return not self.missing

    @property
    def should_continue(self) -> bool:
        return bool(self.missing) and self.retrieval_round < self.max_retrieval_rounds


class EvidenceCoverageChecker:
    """Checks whether retrieved evidence satisfies a task plan."""

    def check(
        self,
        question_plan: QuestionPlan,
        task_plan: AgentTaskPlan,
        state: Any,
        *,
        retrieval_round: int,
    ) -> CoverageReport:
        required = list(task_plan.required_coverage)
        satisfied = [coverage for coverage in required if self._has_coverage(coverage, question_plan, state)]
        missing = [coverage for coverage in required if coverage not in satisfied]
        gaps = [self._gap_for(coverage, question_plan, state) for coverage in missing]
        max_rounds = task_plan.budgets.max_retrieval_rounds
        if not missing:
            stop_reason = "evidence_sufficient"
            status = "pass"
        elif retrieval_round >= max_rounds:
            stop_reason = "max_retrieval_rounds_reached"
            status = "fail" if "metric_evidence" in missing else "warn"
        else:
            stop_reason = "needs_supplement"
            status = "warn"
        return CoverageReport(
            status=status,
            required=required,
            satisfied=satisfied,
            missing=missing,
            gaps=gaps,
            stop_reason=stop_reason,
            retrieval_round=retrieval_round,
            max_retrieval_rounds=max_rounds,
        )

    def _has_coverage(self, coverage: str, plan: QuestionPlan, state: Any) -> bool:
        if coverage == "company_coverage":
            return not missing_companies(plan, state)
        if coverage == "risk_evidence":
            return has_risk_evidence(state)
        if coverage == "metric_evidence":
            return has_metric_evidence(state)
        if coverage == "company_exposure":
            return has_company_exposure(state)
        if coverage == "mechanism_evidence":
            return has_mechanism_evidence(state)
        return True

    def _gap_for(self, coverage: str, plan: QuestionPlan, state: Any) -> EvidenceGap:
        if coverage == "company_coverage":
            missing = missing_companies(plan, state)
            return EvidenceGap(
                coverage=coverage,
                reason=f"公司对比证据未覆盖：{'、'.join(missing)}，需要按缺失公司补检。",
                query_suffix=query_suffix_for_coverage(coverage),
                companies=missing,
            )
        if coverage == "risk_evidence":
            return EvidenceGap(
                coverage=coverage,
                reason="风险问题缺少明确风险、反证或不确定性证据，需要补检风险证据。",
                query_suffix=query_suffix_for_coverage(coverage),
            )
        if coverage == "metric_evidence":
            return EvidenceGap(
                coverage=coverage,
                reason="指标问题缺少 indicator/HAS_METRIC/HAS_INDICATOR 证据，需要补检指标证据。",
                query_suffix=query_suffix_for_coverage(coverage),
            )
        if coverage == "company_exposure":
            return EvidenceGap(
                coverage=coverage,
                reason="主题到公司问题缺少公司敞口证据，需要补检核心/直接敞口公司。",
                query_suffix=query_suffix_for_coverage(coverage),
            )
        if coverage == "mechanism_evidence":
            return EvidenceGap(
                coverage=coverage,
                reason="主题研究缺少技术机理、瓶颈或产业传导证据，需要补检机理证据。",
                query_suffix=query_suffix_for_coverage(coverage),
            )
        return EvidenceGap(coverage=coverage, reason="证据覆盖不足。", query_suffix=query_suffix_for_coverage(coverage))


def next_supplement_gap(report: CoverageReport) -> EvidenceGap | None:
    if not report.gaps:
        return None
    priority = {
        "company_coverage": 0,
        "risk_evidence": 1,
        "metric_evidence": 2,
        "company_exposure": 3,
        "mechanism_evidence": 4,
    }
    return sorted(report.gaps, key=lambda gap: priority.get(gap.coverage, 99))[0]


def metric_gap_answer(report: CoverageReport) -> str:
    gap_text = "；".join(gap.reason for gap in report.gaps if gap.coverage == "metric_evidence")
    if not gap_text:
        gap_text = "当前证据包缺少可引用的指标证据。"
    return (
        "当前知识库无法回答该指标问题。\n\n"
        f"证据缺口：{gap_text}\n\n"
        "需要补充订单、收入、毛利率、产能、客户导入、渗透率等原文或结构化指标证据后再回答。"
    )


def missing_companies(plan: QuestionPlan, state: Any) -> list[str]:
    if not plan.companies:
        return []
    covered = {
        str(record.get("company") or "")
        for record in getattr(state, "graph_records", [])
        if record.get("company")
    }
    covered.update(hit.company for hit in getattr(state, "research_hits", []) if getattr(hit, "company", ""))
    covered.update(hit.company for hit in getattr(state, "rag_hits", []) if getattr(hit, "company", ""))
    covered.update(hit.company for hit in getattr(state, "semantic_hits", []) if getattr(hit, "company", ""))
    return [company for company in plan.companies if company not in covered]


def has_risk_evidence(state: Any) -> bool:
    if any(record.get("relation") == "DISCLOSES_RISK" for record in getattr(state, "graph_records", [])):
        return True
    if any(getattr(hit, "claim_type", "") == "risk" for hit in getattr(state, "research_hits", [])):
        return True
    if any(getattr(hit, "claim_type", "") == "risk" for hit in getattr(state, "semantic_hits", [])):
        return True
    risk_terms = ("风险", "不确定", "波动", "不及预期", "反证")
    return any(any(term in getattr(hit, "snippet", "") for term in risk_terms) for hit in getattr(state, "rag_hits", [])) or any(
        any(term in getattr(hit, "text", "") for term in risk_terms) for hit in getattr(state, "semantic_hits", [])
    )


def has_metric_evidence(state: Any) -> bool:
    metric_relations = {"HAS_METRIC", "HAS_INDICATOR"}
    if any(record.get("relation") in metric_relations for record in getattr(state, "graph_records", [])):
        return True
    if any(getattr(hit, "claim_type", "") == "indicator" for hit in getattr(state, "research_hits", [])):
        return True
    return any(getattr(hit, "claim_type", "") == "indicator" for hit in getattr(state, "semantic_hits", []))


def has_company_exposure(state: Any) -> bool:
    if any(record.get("company") for record in getattr(state, "graph_records", [])):
        return True
    return any(getattr(hit, "claim_type", "") == "company_exposure" and getattr(hit, "company", "") for hit in getattr(state, "research_hits", [])) or any(
        getattr(hit, "claim_type", "") == "company_exposure" and getattr(hit, "company", "") for hit in getattr(state, "semantic_hits", [])
    )


def has_mechanism_evidence(state: Any) -> bool:
    mechanism_types = {"mechanism", "bottleneck", "supply_chain", "trend", "indicator"}
    if any(getattr(hit, "claim_type", "") in mechanism_types for hit in getattr(state, "research_hits", [])):
        return True
    if any(getattr(hit, "claim_type", "") in mechanism_types for hit in getattr(state, "semantic_hits", [])):
        return True
    mechanism_relations = {"CONSTRAINS", "ENABLES", "DRIVES", "DEPENDS_ON", "HAS_INDICATOR"}
    if any(record.get("relation") in mechanism_relations for record in getattr(state, "graph_records", [])):
        return True
    mechanism_terms = ("机理", "瓶颈", "传导", "指标", "带宽", "功耗", "散热")
    return any(any(term in getattr(hit, "snippet", "") for term in mechanism_terms) for hit in getattr(state, "rag_hits", [])) or any(
        any(term in getattr(hit, "text", "") for term in mechanism_terms) for hit in getattr(state, "semantic_hits", [])
    )
