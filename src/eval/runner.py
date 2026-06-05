"""QA benchmark runner and report aggregation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.eval.dataset import DEFAULT_QA_BENCHMARK, EvalCase, load_eval_cases
from src.eval.metrics import score_case
from src.eval.store import EvalRunStore


def run_qa_benchmark(
    engine: Any,
    *,
    benchmark_path: Path | str = DEFAULT_QA_BENCHMARK,
    limit: int = 0,
    k: int = 6,
    store: EvalRunStore | None = None,
    save: bool = True,
) -> dict[str, Any]:
    path = Path(benchmark_path)
    cases = load_eval_cases(path, limit=limit)
    run_id = new_run_id()
    results = []
    for case in cases:
        started_at = datetime.now().isoformat(timespec="seconds")
        try:
            result = engine.answer_question(case.question)
            scored = score_case(result, case, k=k)
            results.append(case_report(case, result, scored, started_at=started_at, k=k))
        except Exception as exc:
            results.append(error_case_report(case, exc, started_at=started_at, k=k))
    report = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": path.name,
            "path": str(path),
            "hash": file_sha256(path),
            "version": path.stem,
            "cases": len(cases),
        },
        "environment": {
            "git_commit": git_commit(),
            "k": max(1, int(k or 6)),
            "limit": int(limit or 0),
            "engine_status": engine_status(engine),
        },
        "summary": summarize_results(results),
        "category_scores": category_scores(results),
        "failed_examples": failed_examples(results),
        "results": results,
    }
    if save and store is not None:
        store.save(report)
    return report


def case_report(case: EvalCase, result: dict[str, Any], scored: dict[str, Any], *, started_at: str, k: int) -> dict[str, Any]:
    cards = list(result.get("evidence_cards") or result.get("evidence") or [])
    return {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "started_at": started_at,
        "expected_answer_type": case.expected_answer_type,
        "answer_type": str(result.get("answer_type") or ""),
        "refusal_expected": case.refusal_expected,
        "metrics": scored["metrics"],
        "score": scored["score"],
        "status": scored["status"],
        "failures": scored["failures"],
        "evidence_gaps": scored["evidence_gaps"],
        "answer": str(result.get("answer") or ""),
        "answer_preview": short_text(str(result.get("answer") or ""), 260),
        "evidence_cards": compact_cards(cards[: max(1, int(k or 6))]),
        "evidence_card_count": len(cards),
        "verification": result.get("verification") if isinstance(result.get("verification"), dict) else {},
        "diagnostics": compact_diagnostics(result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}),
        "expected": {
            "companies": case.expected_companies,
            "topics": case.expected_topics,
            "claim_types": case.required_claim_types,
            "claim_ids": case.expected_claim_ids,
            "forbidden_terms": case.forbidden_terms,
            "scoring_notes": case.scoring_notes,
        },
    }


def error_case_report(case: EvalCase, exc: Exception, *, started_at: str, k: int) -> dict[str, Any]:
    del k
    return {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "started_at": started_at,
        "expected_answer_type": case.expected_answer_type,
        "answer_type": "",
        "refusal_expected": case.refusal_expected,
        "metrics": {
            "claim_recall@k": 0.0,
            "evidence_precision@k": 0.0,
            "citation_validity": 0.0,
            "answer_groundedness": 0.0,
            "unsupported_claim_rate": 1.0,
            "human_score": case.human_score,
        },
        "score": 0.0,
        "status": "fail",
        "failures": [f"exception:{type(exc).__name__}"],
        "evidence_gaps": [{"gap": str(exc), "priority": "高", "suggested_source": "检查评测运行时异常。"}],
        "answer": "",
        "answer_preview": "",
        "evidence_cards": [],
        "evidence_card_count": 0,
        "verification": {},
        "diagnostics": {},
        "expected": {
            "companies": case.expected_companies,
            "topics": case.expected_topics,
            "claim_types": case.required_claim_types,
            "claim_ids": case.expected_claim_ids,
            "forbidden_terms": case.forbidden_terms,
            "scoring_notes": case.scoring_notes,
        },
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = metric_means(results)
    passed = sum(1 for item in results if item.get("status") == "pass")
    failed = sum(1 for item in results if item.get("status") == "fail")
    return {
        "cases": len(results),
        "passed": passed,
        "warned": sum(1 for item in results if item.get("status") == "warn"),
        "failed": failed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "overall_score": round(sum(float(item.get("score") or 0.0) for item in results) / len(results), 4) if results else 0.0,
        "metrics": metrics,
    }


def category_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[str(item.get("category") or "unknown")].append(item)
    rows = []
    for category, items in sorted(grouped.items()):
        rows.append(
            {
                "category": category,
                "cases": len(items),
                "overall_score": round(sum(float(item.get("score") or 0.0) for item in items) / len(items), 4),
                "pass_rate": round(sum(1 for item in items if item.get("status") == "pass") / len(items), 4),
                "metrics": metric_means(items),
            }
        )
    return rows


def failed_examples(results: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    rows = [item for item in results if item.get("status") != "pass"]
    rows.sort(key=lambda item: (float(item.get("score") or 0.0), str(item.get("case_id") or "")))
    return [
        {
            "case_id": item.get("case_id"),
            "category": item.get("category"),
            "question": item.get("question"),
            "score": item.get("score"),
            "failures": item.get("failures", []),
            "evidence_gaps": item.get("evidence_gaps", [])[:3],
            "answer_preview": item.get("answer_preview", ""),
        }
        for item in rows[:limit]
    ]


def metric_means(results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "claim_recall@k",
        "evidence_precision@k",
        "citation_validity",
        "answer_groundedness",
        "unsupported_claim_rate",
    ]
    metrics: dict[str, Any] = {}
    for key in keys:
        metrics[key] = round(sum(float(item.get("metrics", {}).get(key) or 0.0) for item in results) / len(results), 4) if results else 0.0
    human_scores = [float(item.get("metrics", {}).get("human_score")) for item in results if item.get("metrics", {}).get("human_score") is not None]
    metrics["human_score"] = round(sum(human_scores) / len(human_scores), 4) if human_scores else None
    return metrics


def compact_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "citation_id",
        "kind",
        "claim_id",
        "title",
        "evidence",
        "source",
        "page",
        "company",
        "topic",
        "claim_type",
        "exposure_level",
        "score",
    )
    output = []
    for card in cards:
        row = {key: card.get(key, "") for key in fields}
        row["evidence"] = short_text(str(row.get("evidence") or ""), 360)
        output.append(row)
    return output


def compact_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "graph_backend",
        "graph_records",
        "rag_hits",
        "research_hits",
        "embedding_hits",
        "evidence_cards",
        "planner_source",
        "agent_runner",
        "langgraph_enabled",
        "agent_steps",
        "unsupported_terms",
        "timings_ms",
        "llm_calls",
    )
    return {key: diagnostics.get(key) for key in keep if key in diagnostics}


def engine_status(engine: Any) -> dict[str, Any]:
    status = getattr(engine, "status", None)
    if status is None:
        return {}
    return {
        "graph_backend": getattr(status, "graph_backend", ""),
        "neo4j_enabled": bool(getattr(status, "neo4j_enabled", False)),
        "csv_graph_enabled": bool(getattr(status, "csv_graph_enabled", False)),
        "rag_enabled": bool(getattr(status, "rag_enabled", False)),
        "research_enabled": bool(getattr(status, "research_enabled", False)),
        "embedding_enabled": bool(getattr(status, "embedding_enabled", False)),
        "llm_enabled": bool(getattr(status, "llm_enabled", False)),
    }


def new_run_id() -> str:
    return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def short_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)] + "..."
