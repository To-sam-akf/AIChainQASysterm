from aika.agent_runner import AgentRetrievalState
from aika.agents.coverage import EvidenceCoverageChecker
from aika.agents.executor import ToolBudget, ToolExecutor
from aika.agents.planner import TaskPlanner
from aika.frontend_data import LocalKnowledgeGraph
from aika.qa_engine import QAEngine
from aika.question_planner import heuristic_plan_question
from aika.research_claims import ResearchHit


def research_hit(company: str, claim_type: str, text: str) -> ResearchHit:
    return ResearchHit(
        kind="claim",
        title=f"{company} {claim_type}",
        text=text,
        topic="光模块",
        company=company,
        claim_type=claim_type,
        source="测试报告",
        page="1",
        score=10.0,
    )


def test_task_planner_builds_dynamic_coverage_requirements() -> None:
    plan = heuristic_plan_question("中际旭创和新易盛在光模块业务上的差异、风险和跟踪指标是什么？")
    task_plan = TaskPlanner(max_retrieval_rounds=3).plan(plan.question, plan)

    assert task_plan.task_type == "risk_analysis"
    assert task_plan.budgets.max_retrieval_rounds == 3
    assert {"company_coverage", "risk_evidence", "metric_evidence"} <= set(task_plan.required_coverage)
    assert any(subtask.type == "metric_evidence" for subtask in task_plan.subtasks)


def test_coverage_checker_detects_missing_company_risk_and_metric_evidence() -> None:
    plan = heuristic_plan_question("中际旭创和新易盛在光模块业务上的差异、风险和跟踪指标是什么？")
    task_plan = TaskPlanner(max_retrieval_rounds=3).plan(plan.question, plan)
    state = AgentRetrievalState(research_hits=[research_hit("中际旭创", "company_exposure", "中际旭创拥有光模块业务。")])

    report = EvidenceCoverageChecker().check(plan, task_plan, state, retrieval_round=0)
    gaps = {gap.coverage: gap for gap in report.gaps}

    assert report.status == "warn"
    assert {"company_coverage", "risk_evidence", "metric_evidence"} <= set(report.missing)
    assert gaps["company_coverage"].companies == ["新易盛"]
    assert "风险" in gaps["risk_evidence"].query_suffix
    assert "指标" in gaps["metric_evidence"].query_suffix


def test_tool_executor_records_success_exception_and_budget_exhaustion() -> None:
    errors: list[str] = []
    executor = ToolExecutor(errors=errors, budget=ToolBudget(max_tool_calls=1))

    ok = executor.execute("search_rag", {"query": "液冷"}, lambda: [1, 2], len)
    exhausted = executor.execute("search_rag", {"query": "光模块"}, lambda: [3], len)
    failing = ToolExecutor(errors=[]).execute("query_graph", {}, lambda: (_ for _ in ()).throw(RuntimeError("boom")), len)

    assert ok.result_count == 2
    assert ok.error == ""
    assert exhausted.budget_exhausted is True
    assert exhausted.error == "Tool budget exhausted"
    assert failing.result_count == 0
    assert "boom" in failing.error


class ConditionalResearchMemory:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, question: str, plan: object, limit: int = 20) -> list[ResearchHit]:
        del plan, limit
        self.calls.append(question)
        if len(self.calls) == 1:
            return [research_hit("中际旭创", "company_exposure", "中际旭创拥有高端光模块产品。")]
        return [research_hit("新易盛", "company_exposure", "新易盛拥有高速光模块产品。")]


def test_agent_runner_supplements_missing_company_with_scoped_query() -> None:
    memory = ConditionalResearchMemory()
    engine = QAEngine(csv_graph=None, rag_index=None, research_memory=memory, llm_client=None)

    result = engine.answer_question("中际旭创和新易盛在光模块业务上的差异是什么？")
    diagnostics = result["diagnostics"]

    assert diagnostics["agent_stop_reason"] == "evidence_sufficient"
    assert diagnostics["agent_coverage"]["satisfied"] == ["company_coverage"]
    assert any("新易盛" in query and "业务 差异 指标 风险" in query for query in memory.calls[1:])


class ConditionalRiskMemory:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, question: str, plan: object, limit: int = 20) -> list[ResearchHit]:
        del question, plan, limit
        self.calls += 1
        if self.calls == 1:
            return []
        return [research_hit("英维克", "risk", "英维克液冷业务存在客户需求波动风险。")]


def test_agent_runner_supplements_missing_risk_evidence() -> None:
    memory = ConditionalRiskMemory()
    engine = QAEngine(csv_graph=None, rag_index=None, research_memory=memory, llm_client=None)

    result = engine.answer_question("英维克液冷业务主要风险是什么？")

    assert memory.calls >= 2
    assert result["diagnostics"]["agent_stop_reason"] == "evidence_sufficient"
    assert "risk_evidence" in result["diagnostics"]["agent_coverage"]["satisfied"]


def test_metric_question_refuses_when_metric_evidence_remains_missing() -> None:
    graph = LocalKnowledgeGraph(
        entities=[],
        relations=[
            {
                "head_type": "Company",
                "head_name": "浪潮信息",
                "relation": "HAS_PRODUCT",
                "tail_type": "Product",
                "tail_name": "AI服务器",
                "evidence": "浪潮信息布局AI服务器。",
                "source_title": "报告",
                "page": "1",
                "source_tier": "1",
                "section": "主营业务",
            }
        ],
    )
    engine = QAEngine(csv_graph=graph, rag_index=None, research_memory=None, llm_client=None)

    result = engine.answer_question("浪潮信息AI服务器订单指标是什么？")

    assert "无法回答该指标问题" in result["answer"]
    assert result["diagnostics"]["agent_stop_reason"] == "max_retrieval_rounds_reached"
    assert "metric_evidence" in result["diagnostics"]["agent_coverage"]["missing"]
