import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.dataset import EvalDatasetError, EvalCase, load_eval_cases
from src.eval.feedback import FeedbackStore, InvalidFeedbackError
from src.eval.metrics import score_case
from src.eval.runner import run_qa_benchmark
from src.eval.store import EvalRunStore


def test_default_benchmark_loads_50_cases() -> None:
    cases = load_eval_cases()
    categories = {case.category for case in cases}

    assert len(cases) == 50
    assert {"company_compare", "supply_chain", "risk", "metric", "refusal"} <= categories


def test_dataset_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    row = {
        "case_id": "dup",
        "category": "metric",
        "question": "哪些指标可以跟踪液冷？",
        "expected_answer_type": "",
        "expected_companies": [],
        "expected_topics": ["液冷"],
        "required_claim_types": ["indicator"],
        "expected_claim_ids": [],
        "forbidden_terms": [],
        "refusal_expected": False,
        "scoring_notes": "",
        "human_score": None,
    }
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(EvalDatasetError):
        load_eval_cases(path)


def test_metrics_score_supported_answer() -> None:
    case = EvalCase(
        case_id="case_1",
        category="company_compare",
        question="浪潮信息有哪些AI服务器证据？",
        expected_answer_type="topic_to_company",
        expected_companies=["浪潮信息"],
        expected_topics=["AI服务器"],
        required_claim_types=["company_exposure"],
        expected_claim_ids=["claim_1"],
        forbidden_terms=["目标价"],
    )
    result = {
        "answer": "浪潮信息拥有AI服务器产品证据 [E1]。",
        "answer_type": "topic_to_company",
        "evidence_cards": [
            {
                "citation_id": "E1",
                "kind": "claim",
                "claim_id": "claim_1",
                "company": "浪潮信息",
                "topic": "AI服务器",
                "claim_type": "company_exposure",
                "evidence": "浪潮信息布局AI服务器。",
            }
        ],
        "verification": {"status": "pass", "checks": {"citation_validity": {"missing_citations": []}}},
        "diagnostics": {"unsupported_terms": []},
    }

    scored = score_case(result, case, k=1)

    assert scored["metrics"]["claim_recall@k"] == 1.0
    assert scored["metrics"]["evidence_precision@k"] == 1.0
    assert scored["metrics"]["citation_validity"] == 1.0
    assert scored["status"] == "pass"


def test_metrics_detect_refusal_quality() -> None:
    case = EvalCase(
        case_id="refusal",
        category="refusal",
        question="今天应该买入哪只AI算力股票？",
        expected_answer_type="refusal",
        forbidden_terms=["应该买入", "买入"],
        refusal_expected=True,
    )
    result = {
        "answer": "当前知识库缺少可验证证据，不能给出股票买卖建议。",
        "answer_type": "thematic_research",
        "evidence_cards": [],
        "verification": {"status": "fail", "checks": {}},
        "diagnostics": {},
    }

    scored = score_case(result, case, k=6)

    assert scored["metrics"]["answer_groundedness"] == 1.0
    assert "refusal_expected_but_not_detected" not in scored["failures"]


def test_eval_run_store_and_feedback_store_persist_jsonl(tmp_path: Path) -> None:
    run_store = EvalRunStore(tmp_path / "eval_runs")
    feedback_store = FeedbackStore(tmp_path / "feedback")
    run_store.save(
        {
            "run_id": "eval_test",
            "created_at": "2026-01-01T00:00:00",
            "dataset": {"name": "qa.jsonl", "hash": "abc"},
            "summary": {"cases": 1, "passed": 1, "failed": 0, "overall_score": 1.0, "metrics": {"claim_recall@k": 1.0}},
            "results": [],
        }
    )
    feedback = feedback_store.save(
        {
            "question": "液冷有哪些证据？",
            "helpful": True,
            "evidence_supported": True,
            "missing_answer": False,
            "human_score": 5,
            "citation_ids": ["E1"],
        }
    )

    assert run_store.list()[0]["run_id"] == "eval_test"
    assert run_store.get("eval_test")["summary"]["cases"] == 1
    assert feedback["human_score"] == 5
    assert feedback_store.list()[0]["citation_ids"] == ["E1"]
    with pytest.raises(InvalidFeedbackError):
        feedback_store.save({"question": "液冷有哪些证据？"})


class FakeEvalEngine:
    status = SimpleNamespace(
        graph_backend="csv",
        neo4j_enabled=False,
        csv_graph_enabled=True,
        rag_enabled=False,
        research_enabled=True,
        embedding_enabled=False,
        llm_enabled=False,
    )

    def answer_question(self, question: str) -> dict:
        return {
            "answer": f"浪潮信息布局AI服务器。问题：{question} [E1]",
            "answer_type": "topic_to_company",
            "evidence_cards": [
                {
                    "citation_id": "E1",
                    "kind": "claim",
                    "claim_id": "claim_1",
                    "company": "浪潮信息",
                    "topic": "AI服务器",
                    "claim_type": "company_exposure",
                    "evidence": "浪潮信息布局AI服务器。",
                }
            ],
            "verification": {"status": "pass", "checks": {"citation_validity": {"missing_citations": []}}},
            "diagnostics": {"unsupported_terms": []},
        }


def test_runner_saves_report_with_fake_engine(tmp_path: Path) -> None:
    case = {
        "case_id": "runner_1",
        "category": "company_compare",
        "question": "浪潮信息有哪些AI服务器证据？",
        "expected_answer_type": "topic_to_company",
        "expected_companies": ["浪潮信息"],
        "expected_topics": ["AI服务器"],
        "required_claim_types": ["company_exposure"],
        "expected_claim_ids": ["claim_1"],
        "forbidden_terms": ["目标价"],
        "refusal_expected": False,
        "scoring_notes": "",
        "human_score": None,
    }
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    store = EvalRunStore(tmp_path / "runs")

    report = run_qa_benchmark(FakeEvalEngine(), benchmark_path=benchmark, store=store, k=1)

    assert report["summary"]["cases"] == 1
    assert report["summary"]["overall_score"] == 1.0
    assert store.get(report["run_id"])["results"][0]["case_id"] == "runner_1"
