from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api import app, get_conversation_store, get_knowledge_graph, get_qa_engine
from src.conversation_store import ConversationStore
from src.frontend_data import LocalKnowledgeGraph


class FakeEngine:
    def __init__(self) -> None:
        self.calls = []
        self.status = SimpleNamespace(
            graph_backend="csv",
            neo4j_enabled=False,
            rag_enabled=False,
            llm_enabled=False,
            csv_graph_enabled=True,
            graph_data_dir="",
            graph_error="",
            rag_error="",
            llm_error="",
        )

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
            "subgraph": [],
            "diagnostics": {},
            "errors": [],
        }


def make_test_client(tmp_path: Path, engine: FakeEngine) -> TestClient:
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

    async def override_engine() -> FakeEngine:
        return engine

    async def override_graph() -> LocalKnowledgeGraph:
        return graph

    app.dependency_overrides[get_conversation_store] = override_store
    app.dependency_overrides[get_qa_engine] = override_engine
    app.dependency_overrides[get_knowledge_graph] = override_graph
    return TestClient(app)


def test_api_conversation_lifecycle_and_multiturn_history(tmp_path: Path) -> None:
    engine = FakeEngine()
    client = make_test_client(tmp_path, engine)
    try:
        created = client.post("/api/conversations", json={"title": ""})
        assert created.status_code == 201
        conversation_id = created.json()["id"]

        first = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"question": "中际旭创和新易盛在光模块业务上的差异是什么？", "thinking_enabled": False},
        )
        assert first.status_code == 200
        assert first.json()["conversation"]["turns"][0]["answer"].startswith("回答：")

        second = client.post(
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

        fetched = client.get(f"/api/conversations/{conversation_id}")
        assert fetched.status_code == 200
        assert len(fetched.json()["turns"]) == 2

        renamed = client.patch(f"/api/conversations/{conversation_id}", json={"title": "光模块比较"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "光模块比较"

        listed = client.get("/api/conversations")
        assert listed.status_code == 200
        assert listed.json()["conversations"][0]["turn_count"] == 2

        exported = client.get(f"/api/conversations/{conversation_id}/export?format=md")
        assert exported.status_code == 200
        assert "继续说它们的主要风险" in exported.text

        deleted = client.delete(f"/api/conversations/{conversation_id}")
        assert deleted.status_code == 204
    finally:
        app.dependency_overrides.clear()


def test_api_rejects_empty_question_and_missing_conversation(tmp_path: Path) -> None:
    engine = FakeEngine()
    client = make_test_client(tmp_path, engine)
    try:
        empty = client.post("/api/conversations/missing/messages", json={"question": "   "})
        missing = client.get("/api/conversations/missing")

        assert empty.status_code == 400
        assert missing.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_api_status_and_graph_endpoints(tmp_path: Path) -> None:
    engine = FakeEngine()
    client = make_test_client(tmp_path, engine)
    try:
        status_response = client.get("/api/status")
        summary_response = client.get("/api/graph/summary")
        subgraph_response = client.get("/api/graph/subgraph")

        assert status_response.status_code == 200
        assert status_response.json()["stats"]["companies"] == 1
        assert summary_response.json()["relation_options"]["拥有产品"] == "HAS_PRODUCT"
        assert "<svg" in subgraph_response.json()["svg"]
    finally:
        app.dependency_overrides.clear()
