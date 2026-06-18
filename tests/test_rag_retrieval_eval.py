import json
import math
from pathlib import Path

import pytest

from aika.cli import parse_positive_ints, parse_rag_retrievers
from aika.eval.rag_dataset import (
    RagRetrievalCase,
    RagRetrievalDatasetError,
    load_rag_retrieval_cases,
)
from aika.eval.rag_metrics import RetrievedChunk, score_retrieval
from aika.eval.rag_runner import (
    RagRetrievalEvalError,
    reciprocal_rank_fusion,
    retrieve_semantic,
    run_rag_retrieval_benchmark,
    validate_semantic_corpus,
)
from aika.eval.store import EvalRunStore
from aika.extraction_schema import write_jsonl
from aika.question_planner import heuristic_plan_question
from aika.rag_index import LocalRagIndex, document_from_chunk
from aika.semantic_index import SemanticDocument, SemanticIndex, SemanticIndexMetadata


def case_payload(**overrides):
    payload = {
        "case_id": "rag_001",
        "split": "test",
        "category": "technical",
        "question": "液冷如何降低散热压力？",
        "filters": {},
        "evidence_units": [
            {
                "unit_id": "liquid_cooling",
                "required": True,
                "description": "液冷散热机理",
                "alternatives": [{"chunk_id": "chunk_a", "grade": 2}],
            }
        ],
        "judged_irrelevant_chunk_ids": [],
        "hard_negatives": [],
        "notes": "",
    }
    payload.update(overrides)
    return payload


def write_cases(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_rag_dataset_rejects_duplicate_cases_and_unknown_chunks(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    row = case_payload()
    write_cases(path, [row, row])

    with pytest.raises(RagRetrievalDatasetError, match="duplicate case_id"):
        load_rag_retrieval_cases(path)

    write_cases(path, [row])
    with pytest.raises(RagRetrievalDatasetError, match="unknown chunk_id"):
        load_rag_retrieval_cases(path, known_chunk_ids={"another_chunk"})


def test_rag_dataset_limit_still_validates_the_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    second = case_payload(
        case_id="rag_002",
        evidence_units=[
            {
                "unit_id": "stale",
                "required": True,
                "description": "",
                "alternatives": [{"chunk_id": "missing_chunk", "grade": 2}],
            }
        ],
    )
    write_cases(path, [case_payload(), second])

    with pytest.raises(RagRetrievalDatasetError, match="unknown chunk_id"):
        load_rag_retrieval_cases(path, known_chunk_ids={"chunk_a"}, limit=1)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            case_payload(
                evidence_units=[
                    {
                        "unit_id": "background",
                        "required": False,
                        "description": "",
                        "alternatives": [{"chunk_id": "chunk_a", "grade": 1}],
                    }
                ]
            ),
            "at least one evidence unit must be required",
        ),
        (
            case_payload(
                evidence_units=[
                    {
                        "unit_id": "u1",
                        "required": True,
                        "description": "",
                        "alternatives": [{"chunk_id": "chunk_a", "grade": 2}],
                    },
                    {
                        "unit_id": "u2",
                        "required": False,
                        "description": "",
                        "alternatives": [{"chunk_id": "chunk_a", "grade": 1}],
                    },
                ]
            ),
            "conflicting grades",
        ),
    ],
)
def test_rag_dataset_rejects_invalid_evidence_units(payload: dict, message: str) -> None:
    with pytest.raises(RagRetrievalDatasetError, match=message):
        RagRetrievalCase.from_dict(payload, line_no=1)


def test_chunk_metrics_cover_equivalent_unit_once_and_track_unjudged() -> None:
    case = RagRetrievalCase.from_dict(
        case_payload(
            evidence_units=[
                {
                    "unit_id": "u1",
                    "required": True,
                    "description": "",
                    "alternatives": [
                        {"chunk_id": "a1", "grade": 2},
                        {"chunk_id": "a2", "grade": 2},
                    ],
                },
                {
                    "unit_id": "u2",
                    "required": True,
                    "description": "",
                    "alternatives": [{"chunk_id": "b", "grade": 2}],
                },
                {
                    "unit_id": "background",
                    "required": False,
                    "description": "",
                    "alternatives": [{"chunk_id": "c", "grade": 1}],
                },
            ],
            judged_irrelevant_chunk_ids=["negative"],
        ),
        line_no=1,
    )
    hits = [
        RetrievedChunk("a1", 5.0),
        RetrievedChunk("a2", 4.0),
        RetrievedChunk("c", 3.0),
        RetrievedChunk("negative", 2.0),
        RetrievedChunk("unknown", 1.0),
    ]

    scored = score_retrieval(case, hits, ks=(5,))
    metrics = scored["metrics"]

    assert metrics["recall@5"] == 0.5
    assert metrics["precision@5"] == 0.6
    assert metrics["hit_rate@5"] == 1.0
    assert metrics["mrr@5"] == 1.0
    assert metrics["duplicate_rate@5"] == 0.2
    assert metrics["unjudged_rate@5"] == 0.2
    assert 0.0 < metrics["ndcg@5"] < 1.0
    assert scored["details"]["5"]["covered_required_units"] == ["u1"]
    assert scored["details"]["5"]["missed_required_units"] == ["u2"]


def test_precision_and_rates_use_requested_k_when_fewer_hits_are_returned() -> None:
    case = RagRetrievalCase.from_dict(case_payload(), line_no=1)

    scored = score_retrieval(
        case,
        [RetrievedChunk("chunk_a", 1.0), RetrievedChunk("unknown", 0.5)],
        ks=(6,),
    )

    assert scored["metrics"]["precision@6"] == round(1 / 6, 4)
    assert scored["metrics"]["unjudged_rate@6"] == round(1 / 6, 4)
    assert scored["details"]["6"]["returned_chunks"] == 2


class FakeEmbeddingClient:
    model = "fake"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def semantic_index(documents: list[SemanticDocument]) -> SemanticIndex:
    metadata = SemanticIndexMetadata(
        index_version="test",
        built_at="2026-01-01T00:00:00+00:00",
        document_count=len(documents),
        vector_count=len(documents),
        dimension=2,
        embedding_model="fake",
        rag_dir="",
        research_dir="",
        index_dir="",
    )
    return SemanticIndex(
        documents=documents,
        vectors=[[1.0, 0.0] for _ in documents],
        metadata=metadata,
        embedding_client=FakeEmbeddingClient(),
    )


def test_semantic_retrieval_only_returns_rag_chunks() -> None:
    index = semantic_index(
        [
            SemanticDocument(doc_id="claim:c1", kind="claim", title="Claim", text="液冷", ref_id="c1"),
            SemanticDocument(
                doc_id="rag:chunk_a",
                kind="rag",
                title="报告",
                text="液冷降低散热压力",
                source="报告",
                ref_id="chunk_a",
            ),
        ]
    )
    case = RagRetrievalCase.from_dict(case_payload(), line_no=1)
    hits = retrieve_semantic(
        index,
        case,
        heuristic_plan_question(case.question),
        top_k=5,
    )

    assert [hit.chunk_id for hit in hits] == ["chunk_a"]


def test_rrf_is_deterministic_and_deduplicates_chunk_ids() -> None:
    bm25 = [
        RetrievedChunk("a", 10.0, source_title="A"),
        RetrievedChunk("b", 9.0, source_title="B"),
    ]
    semantic = [
        RetrievedChunk("b", 0.9, source_title="B"),
        RetrievedChunk("c", 0.8, source_title="C"),
    ]

    first = reciprocal_rank_fusion(bm25, semantic, top_k=3)
    second = reciprocal_rank_fusion(bm25, semantic, top_k=3)

    assert [hit.chunk_id for hit in first] == ["b", "a", "c"]
    assert first == second
    assert first[0].component_ranks == {"bm25": 2, "semantic": 1}
    assert math.isclose(first[0].score, 1 / 62 + 1 / 61, rel_tol=1e-6)


def test_semantic_corpus_validation_detects_stale_index() -> None:
    index = semantic_index(
        [SemanticDocument(doc_id="rag:old", kind="rag", title="旧报告", text="旧", ref_id="old")]
    )

    with pytest.raises(RagRetrievalEvalError, match="does not match"):
        validate_semantic_corpus(index, {"new"})


def test_rag_runner_writes_report_and_unjudged_review_queue(tmp_path: Path) -> None:
    documents = [
        document_from_chunk(
            {
                "chunk_id": "chunk_a",
                "report_id": "r1",
                "company": "",
                "source_title": "液冷报告",
                "source_tier": "1",
                "source_type": "authority_whitepaper",
                "page": "1",
                "section": "液冷",
                "text": "液冷通过冷板降低高功率设备散热压力。",
            }
        ),
        document_from_chunk(
            {
                "chunk_id": "chunk_b",
                "report_id": "r2",
                "company": "",
                "source_title": "液冷背景",
                "source_tier": "2",
                "source_type": "technical_paper",
                "page": "2",
                "section": "液冷",
                "text": "液冷技术仍需关注冷却液和运维问题。",
            }
        ),
    ]
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    index = LocalRagIndex([document for document in documents if document], index_dir=rag_dir)
    benchmark = tmp_path / "benchmark.jsonl"
    write_cases(benchmark, [case_payload()])
    store = EvalRunStore(tmp_path / "runs")

    report = run_rag_retrieval_benchmark(
        index,
        benchmark_path=benchmark,
        retrievers=("bm25",),
        ks=(1, 2),
        candidate_k=2,
        store=store,
        review_dir=tmp_path / "reviews",
    )

    assert report["summary"]["cases"] == 1
    assert report["summary"]["by_retriever"]["bm25"]["metrics"]["recall@1"] == 1.0
    assert report["review_queue"]["unjudged"] >= 1
    assert Path(report["review_queue"]["path"]).exists()
    assert store.get(report["run_id"])["kind"] == "rag_retrieval"


def test_rag_cli_argument_parsers() -> None:
    assert parse_positive_ints("6,1,3,6") == (1, 3, 6)
    assert parse_rag_retrievers("auto", use_embedding=False) == ("bm25",)
    assert parse_rag_retrievers("auto", use_embedding=True) == ("bm25", "semantic", "rrf")
    assert parse_rag_retrievers("semantic,rrf", use_embedding=True) == ("semantic", "rrf")

    with pytest.raises(ValueError):
        parse_positive_ints("0")
    with pytest.raises(ValueError):
        parse_rag_retrievers("unknown", use_embedding=False)
