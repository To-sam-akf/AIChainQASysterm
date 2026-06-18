import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aika.embedding_client import OpenAICompatibleEmbeddingClient, embedding_configured
from aika.extraction_schema import load_jsonl, write_jsonl
from aika.semantic_index import (
    SEMANTIC_DOCUMENTS_FILE,
    SEMANTIC_METADATA_FILE,
    SEMANTIC_VECTORS_FILE,
    SemanticIndex,
    build_semantic_documents,
    build_semantic_index,
)


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_embedding_client_batches_openai_compatible_requests(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        data = [
            {"index": index, "embedding": [float(index + 1), 0.0, 0.0]}
            for index, _ in enumerate(json["input"])
        ]
        return FakeResponse({"data": data})

    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-test")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "")
    monkeypatch.setattr("aika.embedding_client.requests.post", fake_post)

    client = OpenAICompatibleEmbeddingClient(batch_size=2)
    vectors = client.embed_texts(["a", "b", "c"])

    assert [call["url"] for call in calls] == [
        "https://embedding.example/v1/embeddings",
        "https://embedding.example/v1/embeddings",
    ]
    assert calls[0]["headers"]["Authorization"] == "Bearer embedding-key"
    assert calls[0]["json"] == {"model": "text-embedding-test", "input": ["a", "b"]}
    assert vectors == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_embedding_client_sends_optional_dimensions(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        del url, headers, timeout
        calls.append(json)
        return FakeResponse({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-test")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setattr("aika.embedding_client.requests.post", fake_post)

    client = OpenAICompatibleEmbeddingClient()
    client.embed_texts(["a"])

    assert calls[0]["dimensions"] == 1024


def test_embedding_model_is_required(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")

    assert embedding_configured() is False
    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        OpenAICompatibleEmbeddingClient()


def test_embedding_client_reports_request_errors(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers, json, timeout):
        del url, headers, json, timeout
        calls.append(1)
        raise RuntimeError("network down")

    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-test")
    monkeypatch.setattr("aika.embedding_client.requests.post", fake_post)

    client = OpenAICompatibleEmbeddingClient(max_retries=1)
    with pytest.raises(RuntimeError, match="Embedding request failed"):
        client.embed_texts(["a"])
    assert len(calls) == 2


class FakeEmbeddingClient:
    model = "fake-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        if "液冷" in text:
            return [3.0, 0.0, 0.0]
        if "光模块" in text:
            return [0.0, 3.0, 0.0]
        return [0.0, 0.0, 3.0]


def write_claims(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "claim_id",
        "claim_type",
        "topic",
        "claim_text",
        "companies",
        "mechanism",
        "direction",
        "horizon",
        "metric",
        "value",
        "unit",
        "source_report_id",
        "source_title",
        "page",
        "section",
        "source_tier",
        "evidence_span",
        "confidence",
        "as_of_date",
        "exposure_level",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "claim_id": "c1",
                "claim_type": "risk",
                "topic": "液冷",
                "claim_text": "液冷需求存在交付节奏和客户资本开支波动风险。",
                "companies": json.dumps(["英维克"], ensure_ascii=False),
                "source_title": "测试报告",
                "page": "3",
                "section": "风险",
                "source_tier": "1",
                "confidence": "0.8",
                "exposure_level": "core",
            }
        )


def test_semantic_index_builds_normalized_vectors_and_searches(tmp_path: Path) -> None:
    rag_dir = tmp_path / "rag"
    research_dir = tmp_path / "curated"
    index_dir = tmp_path / "semantic"
    write_jsonl(
        rag_dir / "documents.jsonl",
        [
            {
                "chunk_id": "r1",
                "report_id": "report",
                "company": "英维克",
                "source_title": "液冷报告",
                "source_tier": "1",
                "source_type": "broker_report",
                "page": "1",
                "section": "液冷",
                "text": "液冷用于降低智算中心散热压力。",
            },
            {
                "chunk_id": "r2",
                "report_id": "report",
                "company": "中际旭创",
                "source_title": "光模块报告",
                "source_tier": "1",
                "source_type": "broker_report",
                "page": "2",
                "section": "光模块",
                "text": "光模块需求来自AI集群网络升级。",
            },
        ],
    )
    write_claims(research_dir / "claims.csv")
    write_jsonl(research_dir / "segment_dossiers.jsonl", [{"topic": "液冷", "summary": "液冷产业链涉及CDU和冷板。"}])

    documents = build_semantic_documents(rag_dir=rag_dir, research_dir=research_dir)
    metadata = build_semantic_index(
        documents=documents,
        embedding_client=FakeEmbeddingClient(),
        output_dir=index_dir,
        rag_dir=rag_dir,
        research_dir=research_dir,
    )
    vectors = load_jsonl(index_dir / SEMANTIC_VECTORS_FILE)
    loaded = SemanticIndex.load(index_dir, embedding_client=FakeEmbeddingClient())
    hits = loaded.search("液冷风险是什么", top_k=2)

    assert metadata.document_count == 4
    assert metadata.vector_count == 4
    assert (index_dir / SEMANTIC_DOCUMENTS_FILE).exists()
    assert (index_dir / SEMANTIC_METADATA_FILE).exists()
    assert all(math.isclose(math.sqrt(sum(value * value for value in row["vector"])), 1.0) for row in vectors)
    assert hits[0].topic == "液冷" or "液冷" in hits[0].text


def test_embedding_index_requires_database_url() -> None:
    env = os.environ.copy()
    env["DATABASE_URL"] = ""
    env["EMBEDDING_DIMENSIONS"] = "2048"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_embedding_index.py",
            "--dry-run",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "DATABASE_URL is required" in result.stderr
