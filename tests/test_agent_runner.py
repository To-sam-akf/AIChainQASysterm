from src.frontend_data import LocalKnowledgeGraph
from src.qa_engine import NO_EVIDENCE_ANSWER, QAEngine


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

    assert result["answer"] == NO_EVIDENCE_ANSWER
    assert result["diagnostics"]["agent_verification"]["status"] == "fail"
    assert result["diagnostics"]["agent_verification"]["checks"]["evidence_count"] == 0
