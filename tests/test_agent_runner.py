from aika.frontend_data import LocalKnowledgeGraph
from aika.llm_client import ChatTextResult
from aika.qa_engine import NO_EVIDENCE_ANSWER, QAEngine
from aika.rag_index import RagHit
from aika.semantic_index import SemanticHit


class FakeSemanticIndex:
    def search(self, question: str, *, top_k: int = 8):
        del question, top_k
        return [
            SemanticHit(
                doc_id="claim:s1",
                kind="claim",
                title="英维克 液冷 risk",
                text="英维克液冷业务存在客户需求波动和交付节奏不确定性风险。",
                score=0.91,
                source="语义测试报告",
                page="7",
                topic="液冷",
                company="英维克",
                claim_type="risk",
                exposure_level="core",
                ref_id="s1",
            )
        ]


class FailingSemanticIndex:
    def search(self, question: str, *, top_k: int = 8):
        del question, top_k
        raise RuntimeError("embedding service unavailable")


class UnsupportedNumberLLMClient:
    def chat_text(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2, **kwargs: object) -> str:
        del system_prompt, user_prompt, temperature, kwargs
        return "结论：浪潮信息 2026 年 AI服务器收入达到 10亿元 [E1]。"


class RecordingAgentRagIndex:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, question: str, *, top_k: int = 6, filters: dict[str, str] | None = None) -> list[RagHit]:
        del top_k, filters
        self.queries.append(question)
        return [
            RagHit(
                chunk_id="agent-hyde-rag",
                report_id="agent-hyde-report",
                source_title="测试报告",
                source_tier="1",
                source_type="annual_report",
                page="1",
                section="主营业务",
                content_type="text",
                table_id="",
                company="浪潮信息",
                text="浪潮信息布局AI服务器。",
                snippet="浪潮信息布局AI服务器。",
                score=10.0,
            )
        ]


class RecordingGraphClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def run_read_query(self, cypher: str, params: dict | None = None, *, limit: int = 50) -> list[dict]:
        del limit
        rendered = f"{cypher} {params or {}}"
        self.queries.append(rendered)
        assert "HYDE_MARKER" not in rendered
        return [
            {
                "company": "浪潮信息",
                "company_labels": ["Company"],
                "relation": "HAS_PRODUCT",
                "target": "AI服务器",
                "target_labels": ["Product"],
                "evidence": "浪潮信息布局AI服务器。",
                "source": "测试报告",
                "page": "1",
                "section": "主营业务",
                "source_tier": "1",
            }
        ]


class AgentHydeLLMClient:
    def __init__(self) -> None:
        self.hyde_calls = 0

    def chat_text(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2, **kwargs: object) -> str:
        del user_prompt, temperature, kwargs
        if "HYDE" in system_prompt:
            self.hyde_calls += 1
            return "HYDE_MARKER AI服务器 算力基础设施 产业链"
        return "核心判断：浪潮信息布局AI服务器 [E1]。"

    def chat_messages(self, *, messages: list[dict[str, str]], temperature: float = 0.2, **kwargs: object) -> ChatTextResult:
        del messages, temperature, kwargs
        return ChatTextResult(content="核心判断：浪潮信息布局AI服务器 [E1]。")

    def chat_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.0, **kwargs: object) -> dict:
        del temperature, kwargs
        import json

        payload = json.loads(user_prompt)
        if "中心调度 Agent" in system_prompt:
            return {
                "assignments": [
                    {"agent_type": "graph", "query": payload["question"], "filters": {}, "coverage_goals": payload["required_coverage"]},
                    {"agent_type": "rag", "query": payload["question"], "filters": {}, "coverage_goals": payload["required_coverage"]},
                ]
            }
        if "查询 Agent" in system_prompt:
            return {"query": payload["question"], "filters": {}}
        if "证据审核 Agent" in system_prompt:
            return {
                "accepted_ids": [item["candidate_id"] for item in payload["candidates"]],
                "rejected_ids": [],
                "missing_coverage": [],
                "conflicts": [],
            }
        if "证据归纳 Agent" in system_prompt:
            candidate_ids = [item["candidate_id"] for item in payload["candidates"][:1]]
            return {"cards": [{"candidate_ids": candidate_ids, "title": "AI服务器证据", "reason": "直接支撑问题"}]}
        return {}


def test_agent_runner_is_enabled_by_default_and_records_four_phase_trace() -> None:
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
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=None)

    result = engine.answer_question("哪些上市公司涉及AI服务器？")
    diagnostics = result["diagnostics"]
    phases = [step["phase"] for step in diagnostics["agent_trace"]]

    assert diagnostics["agent_enabled"] is True
    assert diagnostics["agent_runner"] == "langgraph"
    assert diagnostics["langgraph_enabled"] is True
    assert "search_rag" in diagnostics["langchain_tools"]
    assert diagnostics["agent_steps"] <= 4
    assert phases == ["plan", "retrieve", "supplement", "verify_answer"]
    assert result["evidence_cards"]


def test_agent_can_use_legacy_runner_when_configured() -> None:
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
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=None, agent_runner="legacy")

    result = engine.answer_question("哪些上市公司涉及AI服务器？")

    assert result["diagnostics"]["agent_enabled"] is True
    assert result["diagnostics"]["agent_runner"] == "legacy"
    assert result["diagnostics"]["langgraph_enabled"] is False
    assert result["evidence_cards"]


def test_agent_can_be_disabled_to_use_legacy_workflow() -> None:
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
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=None, enable_agent=False)

    result = engine.answer_question("哪些上市公司涉及AI服务器？")

    assert result["diagnostics"]["agent_enabled"] is False
    assert result["diagnostics"]["agent_runner"] == "workflow"
    assert result["diagnostics"]["langgraph_enabled"] is False
    assert result["diagnostics"]["agent_trace"] == []
    assert result["evidence_cards"]


def test_legacy_agent_uses_hyde_for_text_retrieval_only() -> None:
    rag = RecordingAgentRagIndex()
    graph = RecordingGraphClient()
    llm = AgentHydeLLMClient()
    engine = QAEngine(graph_client=graph, rag_index=rag, llm_client=llm, agent_runner="legacy")

    result = engine.answer_question("浪潮信息有哪些AI服务器产品？")

    assert llm.hyde_calls == 1
    assert rag.queries and all("HYDE_MARKER" in query for query in rag.queries)
    assert graph.queries
    assert result["diagnostics"]["hyde"]["generated"] is True
    assert result["diagnostics"]["agent_runner"] == "legacy"


def test_langgraph_agent_reuses_single_hyde_query_for_text_agents() -> None:
    rag = RecordingAgentRagIndex()
    graph = RecordingGraphClient()
    llm = AgentHydeLLMClient()
    engine = QAEngine(graph_client=graph, rag_index=rag, llm_client=llm)

    result = engine.answer_question("浪潮信息有哪些AI服务器产品？")

    assert llm.hyde_calls == 1
    assert rag.queries and all("HYDE_MARKER" in query for query in rag.queries)
    assert graph.queries
    assert result["diagnostics"]["hyde"]["generated"] is True
    assert result["diagnostics"]["agent_runner"] == "langgraph"


def test_agent_supplements_risk_question_when_risk_evidence_is_missing() -> None:
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
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=None)

    result = engine.answer_question("浪潮信息主要风险是什么？")
    supplement_step = result["diagnostics"]["agent_trace"][2]

    assert result["answer_type"] == "risk_analysis"
    assert supplement_step["phase"] == "supplement"
    assert supplement_step["action"] == "supplemental_retrieve"


def test_agent_verification_fails_when_no_evidence_is_available() -> None:
    engine = QAEngine(csv_graph=None, rag_index=None, research_memory=None, llm_client=None)

    result = engine.answer_question("不存在的技术有哪些公司涉及？")

    assert NO_EVIDENCE_ANSWER not in result["answer"]
    assert "证据缺口" in result["answer"]
    assert result["verification"]["status"] == "fail"
    assert result["diagnostics"]["agent_verification"]["status"] == "fail"
    assert result["diagnostics"]["agent_verification"]["checks"]["evidence_count"] == 0


def test_agent_replaces_unsupported_numeric_answer_with_evidence_limited_answer() -> None:
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
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=UnsupportedNumberLLMClient())

    result = engine.answer_question("哪些上市公司涉及AI服务器？")

    assert "收入达到 10亿元" not in result["answer"]
    assert "证据缺口" in result["answer"]
    assert result["verification"]["checks"]["numeric_support"]["unsupported"] == []


def test_agent_uses_semantic_hits_as_evidence_cards() -> None:
    engine = QAEngine(
        csv_graph=None,
        rag_index=None,
        research_memory=None,
        llm_client=None,
        semantic_index=FakeSemanticIndex(),
    )

    result = engine.answer_question("英维克液冷业务主要风险是什么？")

    assert result["diagnostics"]["embedding_enabled"] is True
    assert result["diagnostics"]["embedding_hits"] == 1
    assert result["evidence_cards"][0]["semantic_ref_id"] == "s1"
    assert result["evidence_cards"][0]["kind"] == "claim"


def test_agent_degrades_when_semantic_search_fails() -> None:
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
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=None, semantic_index=FailingSemanticIndex())

    result = engine.answer_question("哪些上市公司涉及AI服务器？")

    assert result["evidence_cards"]
    assert result["diagnostics"]["embedding_hits"] == 0
    assert any("search_semantic failed" in error for error in result["errors"])
