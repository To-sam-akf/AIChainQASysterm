from src.frontend_data import LocalKnowledgeGraph
from src.qa_engine import NO_EVIDENCE_ANSWER, QAEngine
from src.semantic_index import SemanticHit


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
    assert diagnostics["agent_steps"] <= 4
    assert phases == ["plan", "retrieve", "supplement", "verify_answer"]
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
    assert result["diagnostics"]["agent_trace"] == []
    assert result["evidence_cards"]


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
