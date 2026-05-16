import json
from pathlib import Path

import pytest

from src.cypher_guard import CypherSafetyError, ensure_limit, validate_read_only_cypher
from src.qa_engine import NO_EVIDENCE_ANSWER, QAEngine
from src.rag_index import LocalRagIndex, build_rag_index


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
        assert "Neo4j" in user_prompt
        return "浪潮信息涉及AI服务器，证据来自浪潮信息2025年年度报告第10页。"


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


def test_qa_engine_returns_fixed_answer_without_evidence() -> None:
    engine = QAEngine(
        llm_client=FakeLLMClient(),
        graph_client=FakeGraphClient([]),
        rag_index=None,
        enable_llm_cypher=False,
    )

    result = engine.answer_question("不存在的技术有哪些公司涉及？")

    assert result["answer"] == NO_EVIDENCE_ANSWER
    assert result["evidence"] == []
