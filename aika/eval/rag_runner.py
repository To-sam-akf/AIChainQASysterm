"""Pure chunk retrieval benchmark for BM25, semantic search, and RRF fusion."""

from __future__ import annotations

import hashlib
import subprocess
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from aika.eval.rag_dataset import (
    DEFAULT_RAG_RETRIEVAL_BENCHMARK,
    RagRetrievalCase,
    load_rag_retrieval_cases,
)
from aika.eval.rag_metrics import RetrievedChunk, hit_judgment, normalize_ks, score_retrieval
from aika.eval.store import EvalRunStore
from aika.extraction_schema import write_jsonl
from aika.question_planner import QuestionPlan, heuristic_plan_question
from aika.rag_index import RagHit
from aika.semantic_index import SemanticHit


DEFAULT_RETRIEVERS = ("bm25",)
SUPPORTED_RETRIEVERS = {"bm25", "semantic", "rrf"}
RRF_CONSTANT = 60


class RagRetrievalEvalError(ValueError):
    """Raised when retrieval evaluation cannot be configured safely."""


def run_rag_retrieval_benchmark(
    rag_index: Any,
    *,
    benchmark_path: Path | str = DEFAULT_RAG_RETRIEVAL_BENCHMARK,
    semantic_index: Any | None = None,
    retrievers: tuple[str, ...] = DEFAULT_RETRIEVERS,
    ks: tuple[int, ...] = (1, 3, 6, 12),
    candidate_k: int = 30,
    limit: int = 0,
    store: EvalRunStore | None = None,
    save: bool = True,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    requested = normalize_retrievers(retrievers)
    normalized_ks = normalize_ks(ks)
    candidate_k = max(int(candidate_k), max(normalized_ks))
    known_chunk_ids = set(rag_index.chunk_ids())
    cases = load_rag_retrieval_cases(
        benchmark_path,
        known_chunk_ids=known_chunk_ids,
        limit=limit,
    )
    if ("semantic" in requested or "rrf" in requested) and semantic_index is None:
        raise RagRetrievalEvalError("semantic and rrf evaluation require a configured SemanticIndex")
    if semantic_index is not None:
        validate_semantic_corpus(semantic_index, known_chunk_ids)

    run_id = new_run_id()
    results: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for case in cases:
        plan = heuristic_plan_question(case.question)
        component_hits: dict[str, list[RetrievedChunk]] = {}
        if "bm25" in requested or "rrf" in requested:
            component_hits["bm25"] = retrieve_bm25(rag_index, case, plan, top_k=candidate_k)
        if "semantic" in requested or "rrf" in requested:
            assert semantic_index is not None
            component_hits["semantic"] = retrieve_semantic(semantic_index, case, plan, top_k=candidate_k)
        if "rrf" in requested:
            component_hits["rrf"] = reciprocal_rank_fusion(
                component_hits["bm25"],
                component_hits["semantic"],
                top_k=candidate_k,
            )
        for retriever in requested:
            hits = component_hits[retriever]
            scored = score_retrieval(case, hits, ks=normalized_ks)
            result = case_report(case, plan, retriever, hits, scored, ks=normalized_ks)
            results.append(result)
            review_rows.extend(unjudged_review_rows(case, retriever, hits, top_k=candidate_k))

    benchmark_path = Path(benchmark_path)
    review_rows = dedupe_review_rows(review_rows)
    report = {
        "run_id": run_id,
        "kind": "rag_retrieval",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": benchmark_path.name,
            "path": str(benchmark_path),
            "hash": file_sha256(benchmark_path),
            "version": benchmark_path.stem,
            "cases": len(cases),
            "annotation_statuses": annotation_status_counts(cases),
        },
        "environment": {
            "git_commit": git_commit(),
            "ks": list(normalized_ks),
            "candidate_k": candidate_k,
            "retrievers": list(requested),
            "corpus_chunks": len(known_chunk_ids),
            "corpus_hash": corpus_hash(rag_index),
            "rag_index": rag_index_metadata(rag_index),
            "semantic_index": semantic_index_metadata(semantic_index),
            "rrf_constant": RRF_CONSTANT,
        },
        "summary": summarize_results(results, requested, normalized_ks, case_count=len(cases)),
        "category_scores": category_scores(results, requested, normalized_ks),
        "review_queue": {
            "unjudged": len(review_rows),
            "path": "",
        },
        "results": results,
    }
    if save and review_dir is not None and review_rows:
        review_path = review_dir / f"{run_id}_unjudged.jsonl"
        write_jsonl(review_path, review_rows)
        report["review_queue"]["path"] = str(review_path)
    if save and store is not None:
        store.save(report)
    return report


def retrieve_bm25(
    index: Any,
    case: RagRetrievalCase,
    plan: QuestionPlan,
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    query = " ".join([case.question, *plan.expanded_topics]).strip()
    filters = retrieval_filters(case, plan)
    return [retrieved_from_rag_hit(hit, rank=rank) for rank, hit in enumerate(
        index.search(query, top_k=top_k, filters=filters),
        start=1,
    )]


def retrieve_semantic(
    index: Any,
    case: RagRetrievalCase,
    plan: QuestionPlan,
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    query = " ".join(
        part
        for part in [case.question, plan.answer_type, *plan.companies, *plan.expanded_topics]
        if part
    )
    filters: dict[str, list[str]] = {"kinds": ["rag"]}
    company = retrieval_filters(case, plan).get("company")
    if company:
        filters["companies"] = [company]
    hits = index.search(query, top_k=top_k, filters=filters)
    output: list[RetrievedChunk] = []
    for rank, hit in enumerate(hits, start=1):
        if hit.kind != "rag":
            continue
        chunk_id = semantic_chunk_id(hit)
        if not chunk_id:
            continue
        output.append(retrieved_from_semantic_hit(hit, chunk_id=chunk_id, rank=rank))
    return output


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedChunk],
    top_k: int,
    constant: int = RRF_CONSTANT,
) -> list[RetrievedChunk]:
    if constant < 0:
        raise ValueError("RRF constant must be non-negative")
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    metadata: dict[str, RetrievedChunk] = {}
    component_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    names = ("bm25", "semantic")
    for list_index, hits in enumerate(ranked_lists):
        component = names[list_index] if list_index < len(names) else f"component_{list_index + 1}"
        for rank, hit in enumerate(hits, start=1):
            if not hit.chunk_id:
                continue
            scores[hit.chunk_id] += 1.0 / (constant + rank)
            best_rank[hit.chunk_id] = min(best_rank.get(hit.chunk_id, rank), rank)
            component_ranks[hit.chunk_id][component] = rank
            metadata.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], best_rank[chunk_id], chunk_id))
    output = []
    for chunk_id in ordered[: max(0, top_k)]:
        source = metadata[chunk_id]
        output.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                score=round(scores[chunk_id], 8),
                source_title=source.source_title,
                page=source.page,
                section=source.section,
                company=source.company,
                snippet=source.snippet,
                component_ranks=dict(sorted(component_ranks[chunk_id].items())),
            )
        )
    return output


def retrieval_filters(case: RagRetrievalCase, plan: QuestionPlan) -> dict[str, str]:
    filters = dict(case.filters)
    if not filters.get("company") and len(plan.companies) == 1 and plan.answer_type in {
        "risk_analysis",
        "company_profile",
    }:
        filters["company"] = plan.companies[0]
    return filters


def semantic_chunk_id(hit: SemanticHit) -> str:
    ref_id = str(hit.ref_id or "")
    if ref_id:
        return ref_id
    doc_id = str(hit.doc_id or "")
    return doc_id.removeprefix("rag:") if doc_id.startswith("rag:") else ""


def retrieved_from_rag_hit(hit: RagHit, *, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=hit.chunk_id,
        score=float(hit.score),
        source_title=hit.source_title,
        page=str(hit.page),
        section=hit.section,
        company=hit.company,
        snippet=hit.snippet,
        component_ranks={"bm25": rank},
    )


def retrieved_from_semantic_hit(hit: SemanticHit, *, chunk_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=float(hit.score),
        source_title=hit.source,
        page=str(hit.page),
        section=hit.section,
        company=hit.company,
        snippet=hit.text,
        component_ranks={"semantic": rank},
    )


def case_report(
    case: RagRetrievalCase,
    plan: QuestionPlan,
    retriever: str,
    hits: list[RetrievedChunk],
    scored: dict[str, Any],
    *,
    ks: tuple[int, ...],
) -> dict[str, Any]:
    max_k = max(ks)
    primary_k = 6 if 6 in ks else max_k
    detail = scored["details"][str(primary_k)]
    return {
        "case_id": case.case_id,
        "split": case.split,
        "category": case.category,
        "question": case.question,
        "retriever": retriever,
        "annotation_status": case.annotation_status,
        "query_plan": {
            "answer_type": plan.answer_type,
            "companies": plan.companies,
            "expanded_topics": plan.expanded_topics,
            "filters": retrieval_filters(case, plan),
        },
        "metrics": scored["metrics"],
        "missed_required_units": detail["missed_required_units"],
        "hits": [
            {
                **hit.to_dict(),
                **hit_judgment(case, hit.chunk_id),
                "rank": rank,
                "snippet": short_text(hit.snippet, 360),
            }
            for rank, hit in enumerate(hits[:max_k], start=1)
        ],
    }


def summarize_results(
    results: list[dict[str, Any]],
    retrievers: tuple[str, ...],
    ks: tuple[int, ...],
    *,
    case_count: int,
) -> dict[str, Any]:
    primary_k = 6 if 6 in ks else max(ks)
    by_retriever = {}
    for retriever in retrievers:
        rows = [row for row in results if row["retriever"] == retriever]
        means = metric_means(rows, ks)
        by_retriever[retriever] = {
            "cases": len(rows),
            "metrics": means,
            "primary": {
                key: means.get(f"{key}@{primary_k}", 0.0)
                for key in ("recall", "precision", "hit_rate", "mrr", "ndcg", "duplicate_rate", "unjudged_rate")
            },
        }
    return {
        "cases": case_count,
        "retrievers": list(retrievers),
        "primary_k": primary_k,
        "by_retriever": by_retriever,
    }


def category_scores(
    results: list[dict[str, Any]],
    retrievers: tuple[str, ...],
    ks: tuple[int, ...],
) -> list[dict[str, Any]]:
    output = []
    categories = sorted({str(row.get("category") or "unknown") for row in results})
    for retriever in retrievers:
        for category in categories:
            rows = [
                row
                for row in results
                if row["retriever"] == retriever and row["category"] == category
            ]
            if rows:
                output.append(
                    {
                        "retriever": retriever,
                        "category": category,
                        "cases": len(rows),
                        "metrics": metric_means(rows, ks),
                    }
                )
    return output


def metric_means(rows: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, float]:
    names = ("recall", "precision", "hit_rate", "mrr", "ndcg", "duplicate_rate", "unjudged_rate")
    keys = [f"{name}@{k}" for k in ks for name in names]
    if not rows:
        return {key: 0.0 for key in keys}
    return {
        key: round(sum(float(row.get("metrics", {}).get(key) or 0.0) for row in rows) / len(rows), 4)
        for key in keys
    }


def unjudged_review_rows(
    case: RagRetrievalCase,
    retriever: str,
    hits: list[RetrievedChunk],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    output = []
    for rank, hit in enumerate(hits[:top_k], start=1):
        judgment = hit_judgment(case, hit.chunk_id)
        if judgment["judgment"] != "unjudged":
            continue
        output.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "question": case.question,
                "retriever": retriever,
                "rank": rank,
                "chunk_id": hit.chunk_id,
                "source_title": hit.source_title,
                "page": hit.page,
                "section": hit.section,
                "company": hit.company,
                "snippet": short_text(hit.snippet, 600),
                "suggested_grade": None,
                "suggested_unit_id": "",
            }
        )
    return output


def dedupe_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for row in rows:
        key = (row["case_id"], row["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def validate_semantic_corpus(index: Any, known_chunk_ids: set[str]) -> None:
    semantic_chunk_ids = set(index.rag_chunk_ids())
    missing = known_chunk_ids - semantic_chunk_ids
    stale = semantic_chunk_ids - known_chunk_ids
    if missing or stale:
        raise RagRetrievalEvalError(
            "semantic index does not match the RAG corpus: "
            f"missing={len(missing)}, stale={len(stale)}; rebuild the embedding index"
        )


def normalize_retrievers(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    output = []
    for value in values:
        name = str(value).strip().casefold()
        if name not in SUPPORTED_RETRIEVERS:
            raise RagRetrievalEvalError(f"unsupported retriever: {name}")
        if name not in output:
            output.append(name)
    if not output:
        raise RagRetrievalEvalError("at least one retriever is required")
    return tuple(output)


def annotation_status_counts(cases: list[RagRetrievalCase]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[case.annotation_status] += 1
    return dict(sorted(counts.items()))


def rag_index_metadata(index: Any) -> dict[str, Any]:
    return dict(index.metadata_dict())


def semantic_index_metadata(index: Any | None) -> dict[str, Any]:
    return dict(index.metadata_dict()) if index is not None else {}


def corpus_hash(index: Any) -> str:
    return str(index.corpus_hash())


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


def new_run_id() -> str:
    return f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def short_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)] + "..."
