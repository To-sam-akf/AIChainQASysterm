from pathlib import Path

import pytest

from src.agents.models import AgentState
from src.agents.research_agent import ResearchAgent
from src.agents.store import AgentTaskNotFoundError, AgentTaskStore, InvalidAgentTaskError
from src.agents.tools import default_tool_registry


class FakeResearchEngine:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def answer_question(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict:
        del conversation_history
        self.questions.append(question)
        return {
            "question": question,
            "contextual_question": question,
            "answer": "液冷产业链投研回答 [E1]",
            "answer_type": "thematic_research",
            "evidence_cards": [
                {
                    "citation_id": "E1",
                    "evidence": "液冷证据",
                    "kind": "claim",
                    "claim_type": "company_exposure",
                    "company": "英维克",
                    "topic": "液冷",
                    "source": "测试报告",
                },
                {
                    "citation_id": "E2",
                    "evidence": "风险证据",
                    "kind": "claim",
                    "claim_type": "risk",
                    "company": "英维克",
                    "topic": "液冷",
                    "source": "测试报告",
                },
            ],
            "research_outputs": {
                "report": {
                    "title": "液冷投研简报",
                    "markdown": "## 核心判断\n液冷受益于功率密度提升。",
                    "sections": [{"title": "核心判断", "content": "液冷受益于功率密度提升。"}],
                },
                "company_compare_table": {
                    "columns": ["company", "business_evidence", "leading_indicators", "risks", "citations"],
                    "rows": [
                        {
                            "company": "英维克",
                            "chain_segment": "液冷",
                            "exposure_level": "核心敞口",
                            "business_evidence": "英维克提供液冷产品。",
                            "leading_indicators": "关注订单和客户导入。",
                            "risks": "客户需求波动。",
                            "citations": "E1、E2",
                        }
                    ],
                },
                "risk_checklist": [
                    {
                        "scope": "英维克",
                        "risk": "客户需求波动。",
                        "priority": "高",
                        "follow_up": "跟踪订单和客户导入。",
                        "citation_id": "E2",
                    }
                ],
                "evidence_gaps": [{"gap": "缺少订单验证", "priority": "高"}],
                "verification": {
                    "status": "warn",
                    "checks": {},
                    "evidence_gaps": [{"gap": "缺少订单验证", "priority": "高"}],
                    "conflict_groups": [{"conflict_group_id": "conflict_1"}],
                },
            },
            "verification": {
                "status": "warn",
                "checks": {},
                "evidence_gaps": [{"gap": "缺少订单验证", "priority": "高"}],
                "conflict_groups": [{"conflict_group_id": "conflict_1"}],
            },
            "diagnostics": {
                "agent_trace": [
                    {
                        "step": 1,
                        "phase": "plan",
                        "thought": "识别研究主题",
                        "action": "plan_question",
                        "tool_calls": [{"tool": "plan_question", "result_count": 1}],
                        "observation": "thematic_research",
                    }
                ],
                "thinking_enabled": thinking_enabled,
                "reasoning_effort": reasoning_effort or "",
            },
            "errors": [],
        }


class FailingResearchEngine:
    def answer_question(self, *args, **kwargs) -> dict:
        del args, kwargs
        raise RuntimeError("qa unavailable")


def test_agent_task_store_appends_snapshots_lists_latest_and_rejects_bad_ids(tmp_path: Path) -> None:
    store = AgentTaskStore(tmp_path)
    task = store.create_pending(task_type="research_brief", goal="液冷产业链")
    store.save(task.with_updates(status="running"))
    store.save(task.with_updates(status="completed", evidence_cards=[{"citation_id": "E1"}]))

    listed = store.list()
    loaded = store.get(task.task_id)

    assert listed[0]["task_id"] == task.task_id
    assert listed[0]["status"] == "completed"
    assert loaded["status"] == "completed"
    assert loaded["evidence_cards"][0]["citation_id"] == "E1"
    with pytest.raises(InvalidAgentTaskError):
        store.get("../bad")
    with pytest.raises(AgentTaskNotFoundError):
        store.get("missing")


def test_default_tool_registry_has_required_metadata() -> None:
    registry = default_tool_registry()
    names = registry.names()

    assert "contextualize_question" in names
    assert "search_research_claims" in names
    assert "detect_evidence_gaps" in names
    assert "build_research_outputs" in names
    for spec in registry.list():
        assert spec["name"]
        assert spec["description"]
        assert "input_schema" in spec
        assert "output_schema" in spec


def test_tool_registry_executes_registered_tools_and_reports_missing_executor() -> None:
    registry = default_tool_registry()
    registry.register_executor("search_rag", lambda payload: [{"query": payload["query"]}])

    executed = registry.execute("search_rag", {"query": "液冷"})
    missing = registry.execute("query_graph", {"cypher": "MATCH (n) RETURN n"})

    assert executed["tool"] == "search_rag"
    assert executed["result_count"] == 1
    assert executed["result"][0]["query"] == "液冷"
    assert executed["error"] == ""
    assert missing["tool"] == "query_graph"
    assert missing["result_count"] == 0
    assert "No executor registered" in missing["error"]


def test_agent_state_serializes_and_updates_task_workspace() -> None:
    state = AgentState.new(task_id="task-1", task_type="research_brief", goal="液冷产业链")
    updated = state.with_updates(
        status="running",
        current_step=1,
        evidence_pool=[{"citation_id": "E1"}],
        evidence_gaps=[{"coverage": "metric_evidence"}],
        stop_reason="max_retrieval_rounds_reached",
        budget={"max_retrieval_rounds": 3},
        verification={"status": "pass"},
    )
    restored = AgentState.from_dict(updated.to_dict())

    assert state.status == "pending"
    assert updated.status == "running"
    assert restored.task_id == "task-1"
    assert restored.evidence_pool[0]["citation_id"] == "E1"
    assert restored.evidence_gaps[0]["coverage"] == "metric_evidence"
    assert restored.stop_reason == "max_retrieval_rounds_reached"
    assert restored.budget["max_retrieval_rounds"] == 3
    assert restored.verification["status"] == "pass"


def test_research_agent_generates_completed_task_with_outputs(tmp_path: Path) -> None:
    engine = FakeResearchEngine()
    store = AgentTaskStore(tmp_path)

    task = ResearchAgent(engine, store).run(
        task_type="research_brief",
        goal="液冷产业链",
        thinking_enabled=True,
        reasoning_effort="medium",
    )

    assert task["status"] == "completed"
    assert "投研简报" in engine.questions[0]
    assert task["research_outputs"]["report"]["title"] == "液冷投研简报"
    assert task["research_outputs"]["task_outputs"]["schema_type"] == "research_brief"
    assert task["final_outputs"]["task_type"] == "research_brief"
    assert task["final_outputs"]["task_label"] == "投研简报"
    assert task["final_outputs"]["evidence_card_count"] == 2
    assert task["final_outputs"]["evidence_gap_count"] == 1
    assert task["final_outputs"]["verification_status"] == "warn"
    assert task["final_outputs"]["conflict_group_count"] == 1
    assert task["tool_calls"][0]["tool"] == "plan_question"
    assert store.get(task["task_id"])["status"] == "completed"


@pytest.mark.parametrize(
    ("task_type", "goal"),
    [
        ("research_brief", "液冷产业链"),
        ("research_brief", "光模块产业链"),
        ("research_brief", "国产算力产业链"),
        ("company_compare", "中际旭创和新易盛在光模块业务上的差异"),
        ("company_compare", "英维克和申菱环境液冷业务对比"),
        ("company_compare", "浪潮信息和工业富联AI服务器业务对比"),
        ("company_profile", "英维克液冷业务画像"),
        ("company_profile", "中际旭创光模块业务画像"),
        ("company_profile", "浪潮信息AI服务器业务画像"),
        ("risk_review", "英维克液冷业务主要风险"),
        ("risk_review", "AI服务器产业链风险审查"),
        ("risk_review", "光模块需求波动风险"),
        ("evidence_gap_audit", "液冷产业链证据缺口"),
        ("evidence_gap_audit", "国产算力指标证据缺口"),
        ("evidence_gap_audit", "光模块公司风险证据缺口"),
    ],
)
def test_research_agent_supports_multiple_task_types_with_task_outputs(
    tmp_path: Path,
    task_type: str,
    goal: str,
) -> None:
    engine = FakeResearchEngine()
    store = AgentTaskStore(tmp_path)

    task = ResearchAgent(engine, store).run(task_type=task_type, goal=goal)
    restored = store.get(task["task_id"])

    assert task["status"] == "completed"
    assert task["task_type"] == task_type
    assert task["plan"]["workflow"] == task_type
    assert task["plan"]["output_schema"]["schema_type"] == task_type
    assert task["research_outputs"]["task_outputs"]["schema_type"] == task_type
    assert task["final_outputs"]["task_type"] == task_type
    assert task["final_outputs"]["task_label"]
    assert task["final_outputs"]["task_schema_type"] == task_type
    assert restored["research_outputs"]["task_outputs"]["schema_type"] == task_type
    assert engine.questions and goal in engine.questions[0]


def test_research_agent_rejects_unsupported_task_type(tmp_path: Path) -> None:
    with pytest.raises(InvalidAgentTaskError):
        ResearchAgent(FakeResearchEngine(), AgentTaskStore(tmp_path)).run(task_type="unsupported", goal="液冷")


def test_research_agent_saves_failed_task_when_engine_errors(tmp_path: Path) -> None:
    store = AgentTaskStore(tmp_path)

    task = ResearchAgent(FailingResearchEngine(), store).run(task_type="research_brief", goal="液冷产业链")

    assert task["status"] == "failed"
    assert "qa unavailable" in task["errors"][0]
    assert task["steps"][-1]["status"] == "failed"
    assert store.get(task["task_id"])["status"] == "failed"
