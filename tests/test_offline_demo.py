from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src import cli
from src import qa_engine
from src.qa_engine import QAEngine


def write_minimal_graph(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "entities.csv").write_text(
        "\n".join(
            [
                "type,name,normalized_name,is_core_company",
                "Company,浪潮信息,浪潮信息,true",
                "Product,AI服务器,ai服务器,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "relations.csv").write_text(
        "\n".join(
            [
                "relation_id,head_type,head_name,relation,tail_type,tail_name,evidence,source_title,page,section,source_tier",
                "r1,Company,浪潮信息,HAS_PRODUCT,Product,AI服务器,浪潮信息布局AI服务器。,测试报告,1,主营业务,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_from_env_skips_postgres_when_disabled(monkeypatch, tmp_path: Path) -> None:
    graph_dir = tmp_path / "graph"
    write_minimal_graph(graph_dir)

    def fail_from_env(cls):  # noqa: ANN001
        raise AssertionError("PostgreSQL retrieval store should not be initialized")

    monkeypatch.setattr(qa_engine.PostgresRetrievalStore, "from_env", classmethod(fail_from_env))
    monkeypatch.setenv("QA_DISABLE_POSTGRES", "true")
    monkeypatch.setenv("QA_GRAPH_BACKEND", "csv")
    monkeypatch.setenv("KG_DATA_DIR", str(graph_dir))
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    monkeypatch.setenv("LLM_API_KEY", "")

    engine = QAEngine.from_env()
    try:
        assert engine.retrieval_store is None
        assert engine.rag_index is None
        assert engine.research_memory is None
        assert engine.semantic_index is None
        assert engine.status.graph_backend == "csv"
        assert engine.status.rag_enabled is False
        assert engine.status.research_enabled is False
        assert engine.status.embedding_enabled is False
    finally:
        engine.close()


def test_build_engine_sets_postgres_disable_for_offline(monkeypatch) -> None:
    captured = {}

    class FakeEngine:
        def __init__(self) -> None:
            self.csv_graph = object()
            self.graph_client = object()
            self.llm_client = object()
            self.semantic_index = object()
            self.status = SimpleNamespace(
                llm_enabled=True,
                embedding_enabled=True,
                neo4j_enabled=True,
                graph_backend="neo4j",
            )

    def fake_from_env() -> FakeEngine:
        captured["QA_DISABLE_POSTGRES"] = cli.os.environ.get("QA_DISABLE_POSTGRES")
        captured["QA_GRAPH_BACKEND"] = cli.os.environ.get("QA_GRAPH_BACKEND")
        captured["EMBEDDING_MODEL"] = cli.os.environ.get("EMBEDDING_MODEL")
        return FakeEngine()

    monkeypatch.setattr(cli.QAEngine, "from_env", staticmethod(fake_from_env))

    engine = cli.build_engine(
        SimpleNamespace(
            offline=True,
            use_llm=False,
            use_embedding=False,
        )
    )

    assert captured["QA_DISABLE_POSTGRES"] == "true"
    assert captured["QA_GRAPH_BACKEND"] == "csv"
    assert captured["EMBEDDING_MODEL"] == ""
    assert engine.status.graph_backend == "csv"
    assert engine.status.llm_enabled is False
    assert engine.status.embedding_enabled is False
    assert engine.status.neo4j_enabled is False
