import json
from pathlib import Path

import pytest

from aika.cypher_guard import CypherSafetyError, ensure_limit, validate_read_only_cypher
from aika.qa_engine import NO_EVIDENCE_ANSWER, QAEngine
from aika.rag_index import LocalRagIndex, RagHit, build_rag_index


def write_chunk(path: Path, **overrides: str) -> None:
    row = {
        "chunk_id": "chunk_1",
        "report_id": "annual_000977_2025",
        "kind": "annual",
        "company": "浪潮信息",
        "source_title": "浪潮信息2025年年度报告",
        "source_url": "https://example.com/report.pdf",
        "page": "10",
        "section": "管理层讨论与分析",
        "text": "浪潮信息持续布局AI服务器和算力基础设施，服务人工智能训练与推理场景。",
    }
    row.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_build_rag_index_and_search_hits_relevant_chunk(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "rag"
    write_chunk(chunks_dir / "sample.jsonl")

    metadata = build_rag_index(chunks_dir, index_dir)
    index = LocalRagIndex.load(index_dir)
    hits = index.search("哪些公司涉及AI服务器？", top_k=3)

    assert metadata.chunk_count == 1
    assert hits
    assert hits[0].company == "浪潮信息"
    assert "AI服务器" in hits[0].snippet
    cached_hits = index.search("哪些公司涉及AI服务器？", top_k=3)
    assert cached_hits == hits


def test_rag_index_hits_industry_whitepaper_terms(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "rag"
    write_chunk(
        chunks_dir / "industry.jsonl",
        chunk_id="chunk_industry_1",
        report_id="industry_caict_green_compute_2025",
        kind="industry",
        company="",
        source_title="绿色算力发展研究报告（2025年）",
        source_tier="1",
        source_type="authority_whitepaper",
        section="智能算力与液冷",
        text="智能算力基础设施正在推动AI服务器、液冷和光模块等产业链环节协同发展。",
    )
    build_rag_index(chunks_dir, index_dir)
    index = LocalRagIndex.load(index_dir)

    hits = index.search("智能算力 液冷 光模块", top_k=3)

    assert hits
    assert hits[0].source_type == "authority_whitepaper"
    assert hits[0].source_tier == "1"
    assert "液冷" in hits[0].snippet


def test_rag_index_preserves_table_metadata_and_searches_metric_values(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "rag"
    table_text = (
        "| 指标 | 2024 | 2025 |\n|---|---|---|\n| 毛利率 | 20% | 25% |\n"
        f"| 说明 | {'长' * 1900} | |"
    )
    write_chunk(
        chunks_dir / "table.jsonl",
        chunk_id="chunk_table_1",
        content_type="table",
        table_id="table_1",
        section="主要财务指标",
        text=table_text,
    )

    build_rag_index(chunks_dir, index_dir)
    index = LocalRagIndex.load(index_dir)
    hits = index.search("2025年毛利率25%", top_k=3)

    assert hits
    assert hits[0].content_type == "table"
    assert hits[0].table_id == "table_1"
    assert "25%" in hits[0].snippet
    assert index.documents[0].text == table_text


def test_cypher_guard_allows_read_query_and_rejects_writes() -> None:
    cypher = "MATCH (c:Company)-[r]->(x) RETURN c.name AS company, r.evidence AS evidence"

    assert ensure_limit(cypher).endswith("LIMIT 50")
    assert validate_read_only_cypher(cypher) == cypher

    with pytest.raises(CypherSafetyError):
        validate_read_only_cypher("MATCH (n) DETACH DELETE n RETURN n")
    with pytest.raises(CypherSafetyError):
        validate_read_only_cypher("CALL dbms.components()")
    with pytest.raises(CypherSafetyError):
        validate_read_only_cypher("MATCH (n) RETURN n; MATCH (m) RETURN m")


class FakeGraphClient:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def run_read_query(self, cypher: str, params: dict | None = None, *, limit: int = 50) -> list[dict]:
        assert "MATCH" in cypher
        assert limit == 50
        return self.rows


class FakeLLMClient:
    def chat_text(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        if "HYDE" in system_prompt:
            return "浪潮信息 AI服务器 算力基础设施"
        assert "Neo4j" in user_prompt or "证据" in user_prompt
        return "浪潮信息涉及AI服务器，证据来自浪潮信息2025年年度报告第10页。"


class RecordingRagIndex:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, question: str, *, top_k: int = 6, filters: dict[str, str] | None = None) -> list[RagHit]:
        del top_k, filters
        self.queries.append(question)
        return [
            RagHit(
                chunk_id="chunk_hyde",
                report_id="report_hyde",
                source_title="测试报告",
                source_tier="1",
                source_type="research_report",
                page="3",
                section="正文",
                content_type="text",
                table_id="",
                company="英维克",
                text="英维克液冷业务涉及 CDU、冷板和数据中心温控。",
                snippet="英维克液冷业务涉及 CDU、冷板和数据中心温控。",
                score=10.0,
            )
        ]


class HydeLLMClient:
    def __init__(self, *, hyde_text: str = "HYDE_ONLY_TERM 液冷假想答案包含 CDU 冷板 温控", fail_hyde: bool = False) -> None:
        self.hyde_text = hyde_text
        self.fail_hyde = fail_hyde
        self.hyde_calls = 0

    def chat_text(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2, **kwargs: object) -> str:
        del user_prompt, temperature, kwargs
        if "HYDE" in system_prompt:
            self.hyde_calls += 1
            if self.fail_hyde:
                raise RuntimeError("hyde unavailable")
            return self.hyde_text
        return "核心判断：英维克涉及液冷温控 [E1]。"


def test_workflow_hyde_uses_question_and_hypothetical_answer_for_rag() -> None:
    rag = RecordingRagIndex()
    client = HydeLLMClient()
    engine = QAEngine(rag_index=rag, llm_client=client, enable_agent=False)

    result = engine.answer_question("英维克液冷业务有哪些环节？")

    assert client.hyde_calls == 1
    assert rag.queries
    assert "英维克液冷业务有哪些环节" in rag.queries[0]
    assert "HYDE_ONLY_TERM" in rag.queries[0]
    assert "CDU" in rag.queries[0]
    assert result["diagnostics"]["hyde"]["generated"] is True
    assert "hyde" in result["diagnostics"]["timings_ms"]


def test_workflow_hyde_fallbacks_keep_original_query() -> None:
    cases = [
        {"llm_client": None, "enable_hyde": True, "source": "fallback_no_llm", "has_error": False},
        {"llm_client": HydeLLMClient(), "enable_hyde": False, "source": "disabled", "has_error": False},
        {"llm_client": HydeLLMClient(fail_hyde=True), "enable_hyde": True, "source": "fallback_error", "has_error": True},
        {"llm_client": HydeLLMClient(hyde_text=""), "enable_hyde": True, "source": "fallback_empty", "has_error": False},
    ]
    for case in cases:
        rag = RecordingRagIndex()
        engine = QAEngine(
            rag_index=rag,
            llm_client=case["llm_client"],
            enable_agent=False,
            enable_hyde=bool(case["enable_hyde"]),
        )

        result = engine.answer_question("英维克液冷业务有哪些环节？")

        assert len(rag.queries) == 1
        assert rag.queries[0].startswith("英维克液冷业务有哪些环节？")
        assert "HYDE_ONLY_TERM" not in rag.queries[0]
        assert result["diagnostics"]["hyde"]["source"] == case["source"]
        assert result["diagnostics"]["hyde"]["generated"] is False
        if case["has_error"]:
            assert any("HYDE generation failed" in error for error in result["errors"])


def test_qa_engine_combines_graph_and_rag_evidence(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "rag"
    write_chunk(chunks_dir / "sample.jsonl")
    build_rag_index(chunks_dir, index_dir)
    graph_rows = [
        {
            "company": "浪潮信息",
            "company_labels": ["Company"],
            "relation": "USES_TECHNOLOGY",
            "target": "AI服务器",
            "target_labels": ["Technology"],
            "evidence": "浪潮信息持续布局AI服务器。",
            "source": "浪潮信息2025年年度报告",
            "page": "10",
        }
    ]
    engine = QAEngine(
        llm_client=FakeLLMClient(),
        graph_client=FakeGraphClient(graph_rows),
        rag_index=LocalRagIndex.load(index_dir),
        enable_llm_cypher=False,
    )

    result = engine.answer_question("浪潮信息涉及哪些技术？")

    assert "AI服务器" in result["answer"]
    assert result["graph_records"] == graph_rows
    assert result["rag_hits"]
    assert result["subgraph"][0]["source"] == "浪潮信息"


def test_qa_engine_returns_evidence_gap_answer_without_evidence() -> None:
    engine = QAEngine(
        llm_client=FakeLLMClient(),
        graph_client=FakeGraphClient([]),
        rag_index=None,
        enable_llm_cypher=False,
    )

    result = engine.answer_question("不存在的技术有哪些公司涉及？")

    assert NO_EVIDENCE_ANSWER not in result["answer"]
    assert "证据缺口" in result["answer"]
    assert result["evidence"] == []
    assert result["evidence_cards"] == []
    assert result["verification"]["status"] == "fail"
