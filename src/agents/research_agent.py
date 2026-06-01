"""Research task agent built on the existing evidence-constrained QA engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.agents.models import AgentStep, AgentTask
from src.agents.qa_agent import QAAgent
from src.agents.store import AgentTaskStore, InvalidAgentTaskError
from src.agents.tools import ToolRegistry, default_tool_registry


@dataclass(frozen=True)
class TaskSpec:
    task_type: str
    label: str
    schema_type: str
    normalizer: Callable[[str], str]
    output_schema: list[str]
    prepare_thought: str
    prepare_action: str


SUPPORTED_TASK_TYPES = {
    "research_brief",
    "company_compare",
    "company_profile",
    "risk_review",
    "evidence_gap_audit",
}


class ResearchAgent:
    def __init__(
        self,
        engine: Any,
        store: AgentTaskStore,
        *,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.qa_agent = QAAgent(engine)
        self.tool_registry = tool_registry or default_tool_registry()

    def run(
        self,
        *,
        task_type: str,
        goal: str,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        task_type = task_type.strip() or "research_brief"
        goal = goal.strip()
        if task_type not in SUPPORTED_TASK_TYPES:
            raise InvalidAgentTaskError(f"Unsupported agent task type: {task_type}")
        if not goal:
            raise InvalidAgentTaskError("Agent task goal cannot be empty")

        spec = TASK_SPECS[task_type]
        task = self.store.create_pending(task_type=task_type, goal=goal)
        question = spec.normalizer(goal)
        plan = {
            "agent": "ResearchAgent",
            "task_type": task_type,
            "workflow": task_type,
            "task_label": spec.label,
            "goal": goal,
            "question": question,
            "output_schema": {
                "schema_type": spec.schema_type,
                "fields": spec.output_schema,
            },
            "tool_registry": self.tool_registry.names(),
            "execution_model": "sync",
        }
        steps = [
            AgentStep(
                step=1,
                phase="prepare_task",
                thought=spec.prepare_thought,
                action=spec.prepare_action,
                observation=question,
            ).to_dict()
        ]
        task = task.with_updates(status="running", plan=plan, steps=steps)
        self.store.save(task)

        try:
            result = self.qa_agent.run(
                question,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            )
            steps = [*steps, *trace_steps_from_result(result, offset=len(steps))]
            research_outputs = dict(result.get("research_outputs") or {})
            research_outputs["task_outputs"] = build_task_outputs(
                task_type=task_type,
                task_label=spec.label,
                result=result,
                research_outputs=research_outputs,
            )
            meta = dict(research_outputs.get("meta") or {})
            meta.update(
                {
                    "task_type": task_type,
                    "task_label": spec.label,
                    "task_schema_type": spec.schema_type,
                }
            )
            research_outputs["meta"] = meta
            evidence_cards = list(result.get("evidence_cards") or [])
            diagnostics = dict(result.get("diagnostics") or {})
            errors = [str(error) for error in list(result.get("errors") or [])]
            final_outputs = final_outputs_from_result(
                result,
                research_outputs,
                evidence_cards,
                task_type=task_type,
                task_label=spec.label,
                task_schema_type=spec.schema_type,
            )
            tool_calls = tool_calls_from_steps(steps)
            task = task.with_updates(
                status="completed",
                steps=steps,
                tool_calls=tool_calls,
                evidence_cards=evidence_cards,
                research_outputs=research_outputs,
                diagnostics=diagnostics,
                errors=errors,
                final_outputs=final_outputs,
            )
        except Exception as exc:
            steps.append(
                AgentStep(
                    step=len(steps) + 1,
                    phase="failed",
                    thought="执行投研任务时发生异常。",
                    action="qa_agent.run",
                    observation=str(exc),
                    status="failed",
                ).to_dict()
            )
            task = task.with_updates(
                status="failed",
                steps=steps,
                tool_calls=tool_calls_from_steps(steps),
                errors=[str(exc)],
                final_outputs={
                    "report_markdown": "",
                    "report_title": task.title,
                    "task_type": task_type,
                    "task_label": spec.label,
                    "task_schema_type": spec.schema_type,
                    "evidence_gap_count": 0,
                    "evidence_card_count": 0,
                    "qa_answer": "",
                },
            )
        self.store.save(task)
        return task.to_dict()


def normalize_goal(goal: str) -> str:
    return " ".join(str(goal or "").split())


def normalize_research_brief_question(goal: str) -> str:
    goal = normalize_goal(goal)
    if not goal:
        return ""
    if "投研简报" in goal and any(term in goal for term in ("核心判断", "证据", "风险")):
        return goal
    return (
        f"请围绕“{goal}”生成投研简报，覆盖核心判断、技术机理、产业传导、"
        "公司排序、领先指标、风险反证、证据索引和证据缺口。"
    )


def normalize_company_compare_question(goal: str) -> str:
    goal = normalize_goal(goal)
    if not goal:
        return ""
    return (
        f"请围绕“{goal}”生成公司对比任务，覆盖业务卡位、共同驱动、"
        "差异点、领先指标、风险差异、证据索引和证据缺口。"
    )


def normalize_company_profile_question(goal: str) -> str:
    goal = normalize_goal(goal)
    if not goal:
        return ""
    return (
        f"请围绕“{goal}”生成公司产业链画像，覆盖公司业务卡位、产品/技术、"
        "产业链环节、指标证据、风险和证据缺口。"
    )


def normalize_risk_review_question(goal: str) -> str:
    goal = normalize_goal(goal)
    if not goal:
        return ""
    return (
        f"请围绕“{goal}”生成风险审查任务，覆盖主要风险、反证、不确定性、"
        "影响范围、跟踪指标、证据索引和证据缺口。"
    )


def normalize_evidence_gap_audit_question(goal: str) -> str:
    goal = normalize_goal(goal)
    if not goal:
        return ""
    return (
        f"请围绕“{goal}”执行证据缺口审查，聚焦当前知识库缺少哪些公司、"
        "指标、风险、来源和建议补充数据源。"
    )


TASK_SPECS: dict[str, TaskSpec] = {
    "research_brief": TaskSpec(
        task_type="research_brief",
        label="投研简报",
        schema_type="research_brief",
        normalizer=normalize_research_brief_question,
        output_schema=["report", "company_table", "risk_checklist", "evidence_gaps", "evidence_index"],
        prepare_thought="将用户目标规范化为可检索的投研简报任务。",
        prepare_action="normalize_research_brief_question",
    ),
    "company_compare": TaskSpec(
        task_type="company_compare",
        label="公司对比",
        schema_type="company_compare",
        normalizer=normalize_company_compare_question,
        output_schema=["compare_table", "common_drivers", "differences", "risk_differences", "evidence_gaps"],
        prepare_thought="将用户目标规范化为公司对比任务。",
        prepare_action="normalize_company_compare_question",
    ),
    "company_profile": TaskSpec(
        task_type="company_profile",
        label="公司画像",
        schema_type="company_profile",
        normalizer=normalize_company_profile_question,
        output_schema=["profile", "business_position", "technology_products", "indicators", "risks", "evidence_gaps"],
        prepare_thought="将用户目标规范化为公司产业链画像任务。",
        prepare_action="normalize_company_profile_question",
    ),
    "risk_review": TaskSpec(
        task_type="risk_review",
        label="风险审查",
        schema_type="risk_review",
        normalizer=normalize_risk_review_question,
        output_schema=["risk_checklist", "counter_evidence", "follow_up_indicators", "evidence_gaps"],
        prepare_thought="将用户目标规范化为风险审查任务。",
        prepare_action="normalize_risk_review_question",
    ),
    "evidence_gap_audit": TaskSpec(
        task_type="evidence_gap_audit",
        label="证据缺口审查",
        schema_type="evidence_gap_audit",
        normalizer=normalize_evidence_gap_audit_question,
        output_schema=["evidence_gaps", "missing_companies", "missing_metrics", "missing_risks", "suggested_sources"],
        prepare_thought="将用户目标规范化为证据缺口审查任务。",
        prepare_action="normalize_evidence_gap_audit_question",
    ),
}


def trace_steps_from_result(result: dict[str, Any], *, offset: int) -> list[dict[str, Any]]:
    diagnostics = result.get("diagnostics") if isinstance(result, dict) else {}
    trace = diagnostics.get("agent_trace") if isinstance(diagnostics, dict) else []
    steps: list[dict[str, Any]] = []
    if isinstance(trace, list) and trace:
        for index, item in enumerate(trace, start=1):
            if not isinstance(item, dict):
                continue
            steps.append(
                AgentStep(
                    step=offset + index,
                    phase=str(item.get("phase") or "qa_agent"),
                    thought=str(item.get("thought") or ""),
                    action=str(item.get("action") or ""),
                    tool_calls=list(item.get("tool_calls") or []),
                    observation=str(item.get("observation") or ""),
                ).to_dict()
            )
    else:
        steps.append(
            AgentStep(
                step=offset + 1,
                phase="qa_agent",
                thought="调用现有证据约束问答链路生成研究产物。",
                action="QAEngine.answer_question",
                observation=f"answer_type={result.get('answer_type', '')}",
            ).to_dict()
        )
    return steps


def final_outputs_from_result(
    result: dict[str, Any],
    research_outputs: dict[str, Any],
    evidence_cards: list[dict[str, Any]],
    *,
    task_type: str = "research_brief",
    task_label: str = "投研简报",
    task_schema_type: str = "research_brief",
) -> dict[str, Any]:
    report = research_outputs.get("report") if isinstance(research_outputs, dict) else {}
    if not isinstance(report, dict):
        report = {}
    gaps = research_outputs.get("evidence_gaps") if isinstance(research_outputs, dict) else []
    if not isinstance(gaps, list):
        gaps = []
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    if not verification and isinstance(research_outputs, dict):
        candidate = research_outputs.get("verification")
        verification = candidate if isinstance(candidate, dict) else {}
    conflict_groups = verification.get("conflict_groups") if isinstance(verification, dict) else []
    if not isinstance(conflict_groups, list):
        conflict_groups = []
    return {
        "report_markdown": str(report.get("markdown") or ""),
        "report_title": str(report.get("title") or "投研简报"),
        "task_type": task_type,
        "task_label": task_label,
        "task_schema_type": task_schema_type,
        "evidence_gap_count": len(gaps),
        "evidence_card_count": len(evidence_cards),
        "verification_status": str(verification.get("status") or ""),
        "conflict_group_count": len(conflict_groups),
        "qa_answer": str(result.get("answer") or ""),
        "contextual_question": str(result.get("contextual_question") or result.get("question") or ""),
        "answer_type": str(result.get("answer_type") or ""),
    }


def build_task_outputs(
    *,
    task_type: str,
    task_label: str,
    result: dict[str, Any],
    research_outputs: dict[str, Any],
) -> dict[str, Any]:
    evidence_cards = list(result.get("evidence_cards") or [])
    report = ensure_dict(research_outputs.get("report"))
    company_table = ensure_dict(research_outputs.get("company_compare_table"))
    risks = ensure_list(research_outputs.get("risk_checklist"))
    gaps = ensure_list(research_outputs.get("evidence_gaps"))
    evidence_index = evidence_index_from_cards(evidence_cards)

    if task_type == "company_compare":
        rows = ensure_list(company_table.get("rows"))
        return {
            "schema_type": "company_compare",
            "task_label": task_label,
            "compare_table": company_table,
            "common_drivers": evidence_items(evidence_cards, {"mechanism", "supply_chain", "trend", "indicator"}, limit=6),
            "differences": compare_differences(rows),
            "risk_differences": compare_risk_differences(rows, risks),
            "evidence_gaps": gaps,
        }
    if task_type == "company_profile":
        rows = ensure_list(company_table.get("rows"))
        first_row = rows[0] if rows and isinstance(rows[0], dict) else {}
        return {
            "schema_type": "company_profile",
            "task_label": task_label,
            "profile": {
                "title": report.get("title", "公司画像"),
                "contextual_question": str(result.get("contextual_question") or result.get("question") or ""),
                "answer_type": str(result.get("answer_type") or ""),
            },
            "business_position": first_row,
            "technology_products": evidence_items(evidence_cards, {"company_exposure", "mechanism", "supply_chain", ""}, limit=8),
            "indicators": evidence_items(evidence_cards, {"indicator"}, limit=8),
            "risks": risks,
            "evidence_gaps": gaps,
        }
    if task_type == "risk_review":
        return {
            "schema_type": "risk_review",
            "task_label": task_label,
            "risk_checklist": risks,
            "counter_evidence": evidence_items(evidence_cards, {"risk", "bottleneck"}, limit=8),
            "follow_up_indicators": follow_up_indicators(risks, gaps),
            "evidence_gaps": gaps,
        }
    if task_type == "evidence_gap_audit":
        return {
            "schema_type": "evidence_gap_audit",
            "task_label": task_label,
            "evidence_gaps": gaps,
            "missing_companies": gap_rows_by_terms(gaps, ("公司", "标的", "敞口")),
            "missing_metrics": gap_rows_by_terms(gaps, ("指标", "订单", "收入", "毛利", "产能", "客户")),
            "missing_risks": gap_rows_by_terms(gaps, ("风险", "反证", "不确定")),
            "suggested_sources": suggested_sources(gaps),
        }
    return {
        "schema_type": "research_brief",
        "task_label": task_label,
        "report": report,
        "company_table": company_table,
        "risk_checklist": risks,
        "evidence_gaps": gaps,
        "evidence_index": evidence_index,
    }


def ensure_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def ensure_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def card_value(card: dict[str, Any], key: str) -> str:
    return str(card.get(key) or "").strip()


def evidence_index_from_cards(cards: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for card in cards[:limit]:
        if not isinstance(card, dict):
            continue
        rows.append(
            {
                "citation_id": card_value(card, "citation_id"),
                "kind": card_value(card, "kind"),
                "title": card_value(card, "title"),
                "company": card_value(card, "company"),
                "topic": card_value(card, "topic"),
                "claim_type": card_value(card, "claim_type"),
                "source": card_value(card, "source"),
                "page": card_value(card, "page"),
            }
        )
    return rows


def evidence_items(cards: list[dict[str, Any]], claim_types: set[str], *, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        claim_type = card_value(card, "claim_type")
        if claim_type not in claim_types:
            continue
        rows.append(
            {
                "citation_id": card_value(card, "citation_id"),
                "scope": card_value(card, "company") or card_value(card, "topic") or "主题",
                "claim_type": claim_type,
                "evidence": card_value(card, "evidence") or card_value(card, "text"),
                "source": card_value(card, "source"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def compare_differences(rows: list[Any]) -> list[dict[str, str]]:
    differences: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        differences.append(
            {
                "company": str(row.get("company") or ""),
                "business_position": str(row.get("business_evidence") or ""),
                "leading_indicators": str(row.get("leading_indicators") or ""),
                "citations": str(row.get("citations") or ""),
            }
        )
    return differences


def compare_risk_differences(rows: list[Any], risks: list[Any]) -> list[dict[str, str]]:
    risk_rows: list[dict[str, str]] = []
    for row in rows:
        if isinstance(row, dict) and row.get("risks"):
            risk_rows.append(
                {
                    "company": str(row.get("company") or ""),
                    "risks": str(row.get("risks") or ""),
                    "citations": str(row.get("citations") or ""),
                }
            )
    for row in risks:
        if isinstance(row, dict):
            risk_rows.append(
                {
                    "company": str(row.get("scope") or ""),
                    "risks": str(row.get("risk") or ""),
                    "citations": str(row.get("citation_id") or ""),
                }
            )
    return risk_rows[:12]


def follow_up_indicators(risks: list[Any], gaps: list[Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for risk in risks:
        if isinstance(risk, dict):
            rows.append(
                {
                    "scope": str(risk.get("scope") or "主题"),
                    "indicator": str(risk.get("follow_up") or "持续跟踪风险披露、订单、客户需求和价格变化。"),
                    "citation_id": str(risk.get("citation_id") or ""),
                }
            )
    for gap in gaps:
        if isinstance(gap, dict) and len(rows) < 12:
            rows.append(
                {
                    "scope": "证据缺口",
                    "indicator": str(gap.get("suggested_source") or gap.get("gap") or ""),
                    "citation_id": "",
                }
            )
    return rows[:12]


def gap_rows_by_terms(gaps: list[Any], terms: tuple[str, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for gap in gaps:
        if not isinstance(gap, dict):
            continue
        text = str(gap.get("gap") or "")
        if any(term in text for term in terms):
            rows.append(
                {
                    "gap": text,
                    "priority": str(gap.get("priority") or ""),
                    "suggested_source": str(gap.get("suggested_source") or ""),
                }
            )
    return rows


def suggested_sources(gaps: list[Any]) -> list[str]:
    sources: list[str] = []
    for gap in gaps:
        if not isinstance(gap, dict):
            continue
        source = str(gap.get("suggested_source") or "").strip()
        if source and source not in sources:
            sources.append(source)
    return sources[:10]


def tool_calls_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for step in steps:
        for call in step.get("tool_calls") or []:
            if isinstance(call, dict):
                calls.append(call)
    return calls
