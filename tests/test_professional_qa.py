import csv
from pathlib import Path

from src.curated_graph import build_curated_graph
from src.frontend_data import LocalKnowledgeGraph
from src.llm_client import ChatStreamChunk, ChatTextResult
from src.qa_engine import QAEngine
from src.question_planner import heuristic_plan_question


ENTITY_FIELDS = [
    "entity_id",
    "type",
    "name",
    "normalized_name",
    "properties",
    "source_report_ids",
    "review_status",
    "is_core_company",
]

RELATION_FIELDS = [
    "relation_id",
    "head_type",
    "head_name",
    "relation",
    "tail_type",
    "tail_name",
    "evidence",
    "source_report_id",
    "source_title",
    "page",
    "section",
    "source_tier",
    "confidence",
    "review_status",
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_curated_graph_filters_noncore_companies_definition_noise_and_low_value_metrics(tmp_path: Path) -> None:
    entities = tmp_path / "entities.csv"
    relations = tmp_path / "relations.csv"
    output = tmp_path / "curated"
    write_csv(
        entities,
        ENTITY_FIELDS,
        [
            {"entity_id": "c1", "type": "Company", "name": "浪潮信息", "normalized_name": "浪潮信息", "properties": "{}", "source_report_ids": "[]", "review_status": "auto", "is_core_company": "true"},
            {"entity_id": "c2", "type": "Company", "name": "Amazon", "normalized_name": "amazon", "properties": "{}", "source_report_ids": "[]", "review_status": "auto", "is_core_company": "false"},
            {"entity_id": "t1", "type": "Technology", "name": "AI服务器", "normalized_name": "ai服务器", "properties": "{}", "source_report_ids": "[]", "review_status": "auto", "is_core_company": ""},
            {"entity_id": "m1", "type": "Metric", "name": "交易性金融资产期初余额", "normalized_name": "交易性金融资产期初余额", "properties": "{}", "source_report_ids": "[]", "review_status": "auto", "is_core_company": ""},
            {"entity_id": "r1", "type": "Report", "name": "报告", "normalized_name": "report_1", "properties": "{}", "source_report_ids": "[]", "review_status": "auto", "is_core_company": ""},
        ],
    )
    write_csv(
        relations,
        RELATION_FIELDS,
        [
            {"relation_id": "r1", "head_type": "Company", "head_name": "浪潮信息", "relation": "USES_TECHNOLOGY", "tail_type": "Technology", "tail_name": "AI服务器", "evidence": "公司布局AI服务器和算力基础设施。", "source_report_id": "report_1", "source_title": "报告", "page": "20", "section": "主营业务", "source_tier": "1", "confidence": "0.9", "review_status": "auto"},
            {"relation_id": "r2", "head_type": "Company", "head_name": "Amazon", "relation": "USES_TECHNOLOGY", "tail_type": "Technology", "tail_name": "AI服务器", "evidence": "Amazon采购AI服务器。", "source_report_id": "report_1", "source_title": "报告", "page": "20", "section": "主营业务", "source_tier": "2", "confidence": "0.9", "review_status": "auto"},
            {"relation_id": "r3", "head_type": "Company", "head_name": "浪潮信息", "relation": "USES_TECHNOLOGY", "tail_type": "Technology", "tail_name": "液冷", "evidence": "液冷 指 一种散热技术。", "source_report_id": "report_1", "source_title": "报告", "page": "5", "section": "释义", "source_tier": "1", "confidence": "0.9", "review_status": "auto"},
            {"relation_id": "r4", "head_type": "Company", "head_name": "浪潮信息", "relation": "HAS_METRIC", "tail_type": "Metric", "tail_name": "交易性金融资产期初余额", "evidence": "交易性金融资产期初余额 1000万元。", "source_report_id": "report_1", "source_title": "报告", "page": "90", "section": "财务报告", "source_tier": "1", "confidence": "1.0", "review_status": "auto"},
        ],
    )

    _, curated_relations = build_curated_graph(entities_csv=entities, relations_csv=relations, output_dir=output)

    assert [row["relation_id"] for row in curated_relations] == ["r1"]


def test_question_planner_identifies_professional_intents() -> None:
    compare = heuristic_plan_question("中际旭创和新易盛在光模块业务上的差异是什么？")
    bottleneck = heuristic_plan_question("AI算力产业链当前最大的瓶颈是什么？")
    risks = heuristic_plan_question("英维克液冷业务进展和主要风险是什么？")

    assert compare.answer_type == "company_compare"
    assert compare.companies == ["中际旭创", "新易盛"]
    assert "光模块" in compare.topics
    assert bottleneck.answer_type == "industry_bottleneck"
    assert risks.answer_type == "risk_analysis"
    assert risks.companies == ["英维克"]


def test_csv_backend_filters_noncore_company_answers() -> None:
    graph = LocalKnowledgeGraph(
        entities=[],
        relations=[
            {"head_type": "Company", "head_name": "浪潮信息", "relation": "HAS_PRODUCT", "tail_type": "Product", "tail_name": "AI服务器", "evidence": "浪潮信息布局AI服务器。", "source_title": "报告", "page": "1", "source_tier": "1", "section": "主营业务"},
            {"head_type": "Company", "head_name": "Amazon", "relation": "HAS_PRODUCT", "tail_type": "Product", "tail_name": "AI服务器", "evidence": "Amazon涉及AI服务器。", "source_title": "报告", "page": "1", "source_tier": "2", "section": "主营业务"},
        ],
    )
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=None)

    result = engine.answer_question("哪些上市公司涉及AI服务器？")

    assert "浪潮信息" in result["answer"]
    assert "Amazon" not in result["answer"]
    assert result["answer_type"] == "topic_to_company"
    assert result["evidence_cards"]


def test_company_compare_fallback_covers_both_companies() -> None:
    graph = LocalKnowledgeGraph(
        entities=[],
        relations=[
            {"head_type": "Company", "head_name": "中际旭创", "relation": "HAS_PRODUCT", "tail_type": "Product", "tail_name": "800G光模块", "evidence": "中际旭创800G等高端产品取得订单和市场份额。", "source_title": "研报", "page": "39", "source_tier": "2", "section": "光模块"},
            {"head_type": "Company", "head_name": "新易盛", "relation": "HAS_PRODUCT", "tail_type": "Product", "tail_name": "高速光模块", "evidence": "新易盛从高速率光模块研制等方面进行研究开发。", "source_title": "年报", "page": "48", "source_tier": "1", "section": "光模块"},
        ],
    )
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=None)

    result = engine.answer_question("中际旭创和新易盛在光模块业务上的差异是什么？")

    assert "中际旭创" in result["answer"]
    assert "新易盛" in result["answer"]
    assert result["answer_type"] == "company_compare"


def test_followup_question_uses_conversation_history() -> None:
    graph = LocalKnowledgeGraph(
        entities=[],
        relations=[
            {"head_type": "Company", "head_name": "中际旭创", "relation": "DISCLOSES_RISK", "tail_type": "Risk", "tail_name": "客户需求波动", "evidence": "中际旭创披露客户需求和产品价格波动风险。", "source_title": "年报", "page": "88", "source_tier": "1", "section": "风险因素"},
            {"head_type": "Company", "head_name": "新易盛", "relation": "DISCLOSES_RISK", "tail_type": "Risk", "tail_name": "海外市场波动", "evidence": "新易盛披露海外市场、汇率及客户需求变化风险。", "source_title": "年报", "page": "92", "source_tier": "1", "section": "风险因素"},
        ],
    )
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=None)

    result = engine.answer_question(
        "继续说它们的主要风险",
        conversation_history=[
            {"role": "user", "content": "中际旭创和新易盛在光模块业务上的差异是什么？"},
            {"role": "assistant", "content": "两家公司都涉及光模块业务。"},
        ],
    )

    assert result["diagnostics"]["contextualized"] is True
    assert "中际旭创" in result["contextual_question"]
    assert "新易盛" in result["contextual_question"]
    assert result["answer_type"] == "risk_analysis"
    assert "客户需求波动" in result["answer"]
    assert "海外市场波动" in result["answer"]


class RecordingLLMClient:
    def chat_messages(self, *, messages: list[dict[str, str]], temperature: float = 0.2, **kwargs: object) -> ChatTextResult:
        del temperature, kwargs
        system_prompt = messages[0]["content"]
        if "追问改写器" in system_prompt:
            return ChatTextResult(content="中际旭创和新易盛 继续说它们的主要风险")
        return ChatTextResult(content="基于证据回答。")

    def chat_text(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2, **kwargs: object) -> str:
        del system_prompt, user_prompt, temperature, kwargs
        return "基于证据回答。"


class StreamingLLMClient(RecordingLLMClient):
    def stream_chat_messages(self, *, messages: list[dict[str, str]], temperature: float = 0.2, **kwargs: object):
        del messages, temperature, kwargs
        yield ChatStreamChunk(content="结论：")
        yield ChatStreamChunk(content="浪潮信息涉及 AI 服务器。")


def test_fast_path_limits_llm_calls_on_first_turn() -> None:
    graph = LocalKnowledgeGraph(
        entities=[],
        relations=[
            {"head_type": "Company", "head_name": "浪潮信息", "relation": "HAS_PRODUCT", "tail_type": "Product", "tail_name": "AI服务器", "evidence": "浪潮信息布局AI服务器。", "source_title": "报告", "page": "1", "source_tier": "1", "section": "主营业务"},
        ],
    )
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=RecordingLLMClient())

    result = engine.answer_question("哪些上市公司涉及AI服务器？")

    assert result["diagnostics"]["llm_calls"]["total"] <= 1
    assert result["diagnostics"]["planner_source"] == "heuristic"
    assert result["cypher_source"] == "question_plan_csv"
    assert "timings_ms" in result["diagnostics"]


def test_independent_multiturn_question_skips_contextualizer_llm() -> None:
    graph = LocalKnowledgeGraph(
        entities=[],
        relations=[
            {"head_type": "Company", "head_name": "浪潮信息", "relation": "HAS_PRODUCT", "tail_type": "Product", "tail_name": "AI服务器", "evidence": "浪潮信息布局AI服务器。", "source_title": "报告", "page": "1", "source_tier": "1", "section": "主营业务"},
        ],
    )
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=RecordingLLMClient())

    result = engine.answer_question(
        "哪些上市公司涉及AI服务器？",
        conversation_history=[
            {"role": "user", "content": "中际旭创和新易盛在光模块业务上的差异是什么？"},
            {"role": "assistant", "content": "两家公司都涉及光模块业务。"},
        ],
    )

    assert result["diagnostics"]["contextualized"] is False
    assert result["diagnostics"]["llm_calls"]["total"] == 1


def test_followup_question_uses_contextualizer_llm_when_needed() -> None:
    graph = LocalKnowledgeGraph(
        entities=[],
        relations=[
            {"head_type": "Company", "head_name": "中际旭创", "relation": "DISCLOSES_RISK", "tail_type": "Risk", "tail_name": "客户需求波动", "evidence": "中际旭创披露客户需求和产品价格波动风险。", "source_title": "年报", "page": "88", "source_tier": "1", "section": "风险因素"},
            {"head_type": "Company", "head_name": "新易盛", "relation": "DISCLOSES_RISK", "tail_type": "Risk", "tail_name": "海外市场波动", "evidence": "新易盛披露海外市场、汇率及客户需求变化风险。", "source_title": "年报", "page": "92", "source_tier": "1", "section": "风险因素"},
        ],
    )
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=RecordingLLMClient())

    result = engine.answer_question(
        "继续说它们的主要风险",
        conversation_history=[
            {"role": "user", "content": "中际旭创和新易盛在光模块业务上的差异是什么？"},
            {"role": "assistant", "content": "两家公司都涉及光模块业务。"},
        ],
    )

    assert result["diagnostics"]["contextualized"] is True
    assert result["diagnostics"]["llm_calls"]["total"] == 2
    assert result["answer_type"] == "risk_analysis"


def test_answer_question_stream_emits_progress_deltas_and_final_result() -> None:
    graph = LocalKnowledgeGraph(
        entities=[],
        relations=[
            {"head_type": "Company", "head_name": "浪潮信息", "relation": "HAS_PRODUCT", "tail_type": "Product", "tail_name": "AI服务器", "evidence": "浪潮信息布局AI服务器。", "source_title": "报告", "page": "1", "source_tier": "1", "section": "主营业务"},
        ],
    )
    engine = QAEngine(csv_graph=graph, rag_index=None, llm_client=StreamingLLMClient())

    events = list(engine.answer_question_stream("哪些上市公司涉及AI服务器？", thinking_enabled=True, reasoning_effort="low"))

    assert any(event.get("type") == "progress" for event in events)
    assert [event.get("content") for event in events if event.get("type") == "answer_delta"] == [
        "结论：",
        "浪潮信息涉及 AI 服务器。",
    ]
    final = events[-1]
    assert final["type"] == "final"
    assert final["result"]["answer"] == "结论：浪潮信息涉及 AI 服务器。"
    assert final["result"]["diagnostics"]["llm_calls"]["stream_chat_messages"] == 1
