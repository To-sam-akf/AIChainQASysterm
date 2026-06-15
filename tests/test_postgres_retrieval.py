from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from src.postgres_retrieval import (
    EMBEDDING_DIMENSIONS,
    PostgresRagIndex,
    PostgresResearchMemory,
    PostgresRetrievalStore,
    PostgresSemanticIndex,
    migrate_database,
    stable_content_hash,
)
from src.rag_index import RagDocument
from src.semantic_index import SemanticIndexMetadata


def test_stable_content_hash_ignores_mapping_order() -> None:
    assert stable_content_hash({"topic": "液冷", "companies": ["英维克"]}) == stable_content_hash(
        {"companies": ["英维克"], "topic": "液冷"}
    )


def test_postgres_rag_row_mapping_preserves_source_fields() -> None:
    hit = PostgresRagIndex._hit_from_row(
        {
            "chunk_id": "chunk_1",
            "report_id": "report_1",
            "source_title": "液冷技术报告",
            "source_tier": "1",
            "source_type": "technical_roadmap",
            "page": "8",
            "section": "冷板式液冷",
            "content_type": "table",
            "table_id": "table_1",
            "company": "英维克",
            "text": "英维克提供冷板式液冷产品。",
            "bm25_score": 2.0,
        },
        "英维克液冷技术",
        ["英维克", "液冷"],
    )

    assert hit.chunk_id == "chunk_1"
    assert hit.content_type == "table"
    assert hit.table_id == "table_1"
    assert hit.score > 2.0


def test_postgres_semantic_index_rejects_wrong_query_dimension() -> None:
    metadata = SemanticIndexMetadata(
        index_version="test",
        built_at="",
        document_count=0,
        vector_count=0,
        dimension=EMBEDDING_DIMENSIONS,
        embedding_model="test",
        rag_dir="postgresql",
        research_dir="postgresql",
        index_dir="postgresql",
    )
    store = SimpleNamespace(semantic_metadata=lambda: metadata)
    client = SimpleNamespace(embed_texts=lambda texts: [[0.1, 0.2]])
    index = PostgresSemanticIndex(store, embedding_client=client)

    with pytest.raises(ValueError, match="expected 2048"):
        index.search("液冷")


@pytest.mark.integration
def test_postgres_retrieval_end_to_end() -> None:
    database_url = os.getenv("TEST_DATABASE_URL", "").strip()
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    migrate_database(database_url)
    assert migrate_database(database_url) == []
    store = PostgresRetrievalStore(database_url, min_size=1, max_size=2)
    try:
        store.ensure_ready()
        with store.pool.connection() as connection:
            connection.execute(
                """
                TRUNCATE claim_reviews, retrieval_builds, segment_dossiers,
                         research_claims, rag_chunks RESTART IDENTITY CASCADE
                """
            )
            connection.commit()

        documents = [
            _rag_document("chunk_liquid", "英维克提供冷板式液冷产品。", company="英维克"),
            _rag_document("chunk_optical", "中际旭创提供高速光模块。", company="中际旭创"),
        ]
        first_sync = store.sync_rag_documents(documents)
        assert first_sync["record_count"] == 2
        assert PostgresRagIndex(store).search("英维克 液冷", top_k=1)[0].chunk_id == "chunk_liquid"

        second_sync = store.sync_rag_documents(documents[:1])
        assert second_sync["deleted"] == 1
        assert store.chunk_ids() == {"chunk_liquid"}

        claim = {
            "claim_id": "claim_liquid",
            "claim_type": "company_exposure",
            "topic": "液冷",
            "claim_text": "英维克对液冷具有直接敞口。",
            "companies": ["英维克"],
            "evidence_span": "英维克提供冷板式液冷产品。",
            "confidence": "0.90",
            "exposure_level": "direct",
            "review_status": "auto",
        }
        research_result = store.sync_research([claim])
        assert research_result["claims"] == 1
        assert research_result["dossiers"] == 1

        pending = store.pending_embeddings()
        vectors = [_unit_vector(index) for index, _ in enumerate(pending)]
        store.update_embeddings(pending, vectors, model="integration-test")
        semantic = PostgresSemanticIndex(
            store,
            embedding_client=SimpleNamespace(embed_texts=lambda texts: [_unit_vector(0)]),
        )
        assert semantic.search("液冷", top_k=3, filters={"kinds": ["rag"]})[0].ref_id == "chunk_liquid"
        assert semantic.search(
            "英维克",
            top_k=3,
            filters={"kinds": ["claim"], "companies": ["英维克"]},
        )[0].ref_id == "claim_liquid"

        memory = PostgresResearchMemory(store)
        updated = memory.review_claim(
            "claim_liquid",
            {"review_status": "revised", "exposure_level": "core"},
        )
        assert updated["exposure_level"] == "core"
        assert semantic.search("英维克", top_k=3, filters={"kinds": ["claim"]}) == []
        with store.pool.connection() as connection:
            review_count = connection.execute(
                "SELECT count(*) AS count FROM claim_reviews WHERE claim_id = %s",
                ("claim_liquid",),
            ).fetchone()["count"]
        assert review_count == 1
    finally:
        store.close()


def _rag_document(chunk_id: str, text: str, *, company: str) -> RagDocument:
    return RagDocument(
        chunk_id=chunk_id,
        report_id="report_1",
        kind="annual_report",
        company=company,
        source_title="测试报告",
        source_url="",
        source_tier="1",
        source_type="annual_report",
        page="1",
        section="主营业务",
        content_type="text",
        table_id="",
        text=text,
        token_counts={"液冷": 1},
        token_count=1,
    )


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index % EMBEDDING_DIMENSIONS] = 1.0
    return vector
