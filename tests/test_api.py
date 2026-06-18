from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from aika.api import (
    app,
    get_agent_task_store,
    get_conversation_store,
    get_eval_run_store,
    get_feedback_store,
    get_knowledge_graph,
    get_qa_engine,
)
from aika.agents.store import AgentTaskStore
from aika.conversation_store import ConversationStore
from aika.eval.feedback import FeedbackStore
from aika.eval.store import EvalRunStore
from aika.frontend_data import LocalKnowledgeGraph
from aika.research_claims import ResearchMemory


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeEngine:
    def __init__(self) -> None:
        self.calls = []
        self.status = SimpleNamespace(
            graph_backend="csv",
            neo4j_enabled=False,
            rag_enabled=False,
            embedding_enabled=False,
            llm_enabled=False,
            csv_graph_enabled=True,
            graph_data_dir="",
            graph_error="",
            rag_error="",
            embedding_error="",
            llm_error="",
        )
        self.enable_agent = True
        self.agent_max_steps = 4

    def answer_question(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict:
        self.calls.append(
            {
                "question": question,
                "history": conversation_history or [],
                "thinking_enabled": thinking_enabled,
                "reasoning_effort": reasoning_effort,
            }
        )
        return {
            "question": question,
            "contextual_question": question,
            "answer": f"回答：{question}",
            "reasoning_content": "",
            "answer_type": "test",
            "plan": {},
            "cypher": "",
            "cypher_params": {},
            "cypher_source": "test",
            "graph_records": [],
            "rag_hits": [],
            "evidence_cards": [],
            "evidence": [],
            "research_outputs": {
                "report": {
                    "title": "测试投研简报",
                    "markdown": "## 核心判断\n测试回答。",
                    "sections": [{"title": "核心判断", "content": "测试回答。"}],
                },
                "evidence_gaps": [{"gap": "缺少真实证据", "priority": "中"}],
                "verification": {
                    "status": "warn",
                    "checks": {},
                    "evidence_gaps": [{"gap": "缺少真实证据", "priority": "中"}],
                    "conflict_groups": [{"conflict_group_id": "conflict_1"}],
                },
            },
            "verification": {
                "status": "warn",
                "checks": {},
                "evidence_gaps": [{"gap": "缺少真实证据", "priority": "中"}],
                "conflict_groups": [{"conflict_group_id": "conflict_1"}],
            },
            "subgraph": [],
            "diagnostics": {"agent_trace": []},
            "errors": [],
        }


def make_test_client(tmp_path: Path, engine: FakeEngine) -> httpx.AsyncClient:
    graph = LocalKnowledgeGraph(
        entities=[
            {"type": "Company", "name": "浪潮信息", "normalized_name": "浪潮信息"},
            {"type": "Report", "name": "报告", "normalized_name": "report"},
        ],
        relations=[
            {
                "head_type": "Company",
                "head_name": "浪潮信息",
                "relation": "HAS_PRODUCT",
                "tail_type": "Product",
                "tail_name": "AI服务器",
                "evidence": "浪潮信息布局AI服务器。",
            }
        ],
    )
    async def override_store() -> ConversationStore:
        return ConversationStore(tmp_path)

    async def override_agent_store() -> AgentTaskStore:
        return AgentTaskStore(tmp_path / "agent_tasks")

    async def override_eval_run_store() -> EvalRunStore:
        return EvalRunStore(tmp_path / "eval_runs")

    async def override_feedback_store() -> FeedbackStore:
        return FeedbackStore(tmp_path / "feedback")

    async def override_engine() -> FakeEngine:
        return engine

    async def override_graph() -> LocalKnowledgeGraph:
        return graph

    app.dependency_overrides[get_conversation_store] = override_store
    app.dependency_overrides[get_agent_task_store] = override_agent_store
    app.dependency_overrides[get_eval_run_store] = override_eval_run_store
    app.dependency_overrides[get_feedback_store] = override_feedback_store
    app.dependency_overrides[get_qa_engine] = override_engine
    app.dependency_overrides[get_knowledge_graph] = override_graph
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.anyio
async def test_api_conversation_lifecycle_and_multiturn_history(tmp_path: Path) -> None:
    engine = FakeEngine()
    client = make_test_client(tmp_path, engine)
    try:
        created = await client.post("/api/conversations", json={"title": ""})
        assert created.status_code == 201
        conversation_id = created.json()["id"]

        first = await client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"question": "中际旭创和新易盛在光模块业务上的差异是什么？", "thinking_enabled": False},
        )
        assert first.status_code == 200
        assert first.json()["conversation"]["turns"][0]["answer"].startswith("回答：")
        assert first.json()["turn"]["result"]["verification"]["status"] == "warn"

        second = await client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"question": "继续说它们的主要风险", "thinking_enabled": True, "reasoning_effort": "medium"},
        )
        assert second.status_code == 200
        assert engine.calls[1]["history"] == [
            {"role": "user", "content": "中际旭创和新易盛在光模块业务上的差异是什么？"},
            {"role": "assistant", "content": "回答：中际旭创和新易盛在光模块业务上的差异是什么？"},
        ]
        assert engine.calls[1]["thinking_enabled"] is True
        assert engine.calls[1]["reasoning_effort"] == "medium"

        fetched = await client.get(f"/api/conversations/{conversation_id}")
        assert fetched.status_code == 200
        assert len(fetched.json()["turns"]) == 2

        renamed = await client.patch(f"/api/conversations/{conversation_id}", json={"title": "光模块比较"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "光模块比较"

        listed = await client.get("/api/conversations")
        assert listed.status_code == 200
        assert listed.json()["conversations"][0]["turn_count"] == 2

        exported = await client.get(f"/api/conversations/{conversation_id}/export?format=md")
        assert exported.status_code == 200
        assert "继续说它们的主要风险" in exported.text

        deleted = await client.delete(f"/api/conversations/{conversation_id}")
        assert deleted.status_code == 204
    finally:
        await client.aclose()
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_api_rejects_empty_question_and_missing_conversation(tmp_path: Path) -> None:
    engine = FakeEngine()
    client = make_test_client(tmp_path, engine)
    try:
        empty = await client.post("/api/conversations/missing/messages", json={"question": "   "})
        missing = await client.get("/api/conversations/missing")

        assert empty.status_code == 400
        assert missing.status_code == 404
    finally:
        await client.aclose()
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_api_status_and_graph_endpoints(tmp_path: Path) -> None:
    engine = FakeEngine()
    client = make_test_client(tmp_path, engine)
    try:
        status_response = await client.get("/api/status")
        summary_response = await client.get("/api/graph/summary")
        subgraph_response = await client.get("/api/graph/subgraph")

        assert status_response.status_code == 200
        assert status_response.json()["stats"]["companies"] == 1
        assert status_response.json()["embedding_enabled"] is False
        assert status_response.json()["errors"]["embedding"] == ""
        assert status_response.json()["settings"]["agent_enabled"] is True
        assert status_response.json()["settings"]["agent_max_steps"] == 4
        assert summary_response.json()["relation_options"]["拥有产品"] == "HAS_PRODUCT"
        assert "<svg" in subgraph_response.json()["svg"]
    finally:
        await client.aclose()
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_api_eval_runs_and_feedback_endpoints(tmp_path: Path) -> None:
    engine = FakeEngine()
    EvalRunStore(tmp_path / "eval_runs").save(
        {
            "run_id": "eval_api_test",
            "created_at": "2026-01-01T00:00:00",
            "dataset": {"name": "qa_benchmark_v1.jsonl", "hash": "abc123"},
            "summary": {
                "cases": 1,
                "passed": 1,
                "failed": 0,
                "overall_score": 0.9,
                "metrics": {"claim_recall@k": 1.0, "evidence_precision@k": 0.8},
            },
            "category_scores": [],
            "failed_examples": [],
            "results": [{"case_id": "case_1"}],
        }
    )
    client = make_test_client(tmp_path, engine)
    try:
        listed_runs = await client.get("/api/eval/runs")
        fetched_run = await client.get("/api/eval/runs/eval_api_test")
        feedback = await client.post(
            "/api/feedback",
            json={
                "conversation_id": "conv_1",
                "turn_index": 0,
                "question": "液冷有哪些证据？",
                "answer_hash": "abcd",
                "helpful": True,
                "evidence_supported": True,
                "missing_answer": False,
                "human_score": 5,
                "note": "证据清楚",
                "citation_ids": ["E1"],
            },
        )
        listed_feedback = await client.get("/api/feedback")
        invalid_feedback = await client.post(
            "/api/feedback",
            json={"question": "液冷有哪些证据？", "citation_ids": []},
        )

        assert listed_runs.status_code == 200
        assert listed_runs.json()["runs"][0]["run_id"] == "eval_api_test"
        assert fetched_run.status_code == 200
        assert fetched_run.json()["results"][0]["case_id"] == "case_1"
        assert feedback.status_code == 201
        assert feedback.json()["feedback"]["human_score"] == 5
        assert listed_feedback.json()["feedback"][0]["citation_ids"] == ["E1"]
        assert invalid_feedback.status_code == 400
    finally:
        await client.aclose()
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_api_agent_task_lifecycle_and_export(tmp_path: Path) -> None:
    engine = FakeEngine()
    client = make_test_client(tmp_path, engine)
    try:
        task_types = ["research_brief", "company_compare", "company_profile", "risk_review", "evidence_gap_audit"]
        tasks = []
        for task_type in task_types:
            created = await client.post(
                "/api/agent/tasks",
                json={"task_type": task_type, "goal": f"{task_type} 液冷产业链", "thinking_enabled": False},
            )
            assert created.status_code == 201
            task = created.json()["task"]
            tasks.append(task)
            assert task["status"] == "completed"
            assert task["task_type"] == task_type
            assert task["final_outputs"]["task_type"] == task_type
            assert task["research_outputs"]["task_outputs"]["schema_type"] == task_type
            assert task["final_outputs"]["report_title"] == "测试投研简报"
            assert task["final_outputs"]["evidence_gap_count"] == 1
            assert task["final_outputs"]["verification_status"] == "warn"
            assert task["final_outputs"]["conflict_group_count"] == 1
        task = tasks[0]
        assert "投研简报" in engine.calls[0]["question"]

        listed = await client.get("/api/agent/tasks")
        assert listed.status_code == 200
        assert listed.json()["tasks"][0]["task_id"] in {item["task_id"] for item in tasks}

        fetched = await client.get(f"/api/agent/tasks/{task['task_id']}")
        assert fetched.status_code == 200
        assert fetched.json()["task_id"] == task["task_id"]

        exported_md = await client.get(f"/api/agent/tasks/{task['task_id']}/export?format=md")
        exported_json = await client.get(f"/api/agent/tasks/{task['task_id']}/export?format=json")
        assert exported_md.status_code == 200
        assert "测试投研简报" in exported_md.text
        assert exported_json.status_code == 200
        assert exported_json.json()["task_id"] == task["task_id"]
    finally:
        await client.aclose()
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_api_reviews_claim_through_research_store(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.research_memory = ResearchMemory(
        claims=[
            {
                "claim_id": "c1",
                "claim_type": "company_exposure",
                "topic": "液冷",
                "claim_text": "英维克 对 液冷 的公司敞口为 direct。",
                "companies": ["英维克"],
                "evidence_span": "英维克提供液冷产品。",
                "confidence": "0.80",
                "exposure_level": "direct",
                "review_status": "auto",
            }
        ],
        dossiers=[],
    )
    client = make_test_client(tmp_path / "store", engine)
    try:
        response = await client.post(
            "/api/research/claims/c1/review",
            json={
                "claim_text": "英维克 对 液冷 的公司敞口为 core：端到端液冷产品覆盖。",
                "exposure_level": "core",
                "review_status": "revised",
                "reviewer_note": "人工确认其为核心液冷标的。",
            },
        )

        assert response.status_code == 200
        assert response.json()["claim"]["exposure_level"] == "core"
        assert response.json()["review"]["reviewer"] == "frontend"
        assert engine.research_memory.get_claim("c1")["review_status"] == "revised"
    finally:
        await client.aclose()
        app.dependency_overrides.clear()
