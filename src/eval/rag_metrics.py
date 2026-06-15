"""Deterministic chunk-level retrieval metrics for RAG evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from src.eval.rag_dataset import RagRetrievalCase


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    source_title: str = ""
    page: str = ""
    section: str = ""
    company: str = ""
    snippet: str = ""
    component_ranks: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_retrieval(
    case: RagRetrievalCase,
    hits: list[RetrievedChunk],
    *,
    ks: tuple[int, ...] = (1, 3, 6, 12),
) -> dict[str, Any]:
    normalized_ks = normalize_ks(ks)
    metrics: dict[str, float] = {}
    details: dict[str, Any] = {}
    for k in normalized_ks:
        scored = score_at_k(case, hits, k=k)
        metrics.update({f"{name}@{k}": round(value, 4) for name, value in scored["metrics"].items()})
        details[str(k)] = scored["details"]
    return {"metrics": metrics, "details": details}


def score_at_k(case: RagRetrievalCase, hits: list[RetrievedChunk], *, k: int) -> dict[str, Any]:
    requested_k = max(1, int(k))
    top_hits = hits[:requested_k]
    grade_by_chunk, units_by_chunk = judgment_maps(case)
    required_units = {unit.unit_id for unit in case.evidence_units if unit.required}
    covered_required: set[str] = set()
    covered_any: set[str] = set()
    relevant_count = 0
    duplicate_count = 0
    unjudged_count = 0
    first_direct_rank = 0
    gains: list[int] = []
    judged_negative = set(case.judged_irrelevant_chunk_ids) | set(case.hard_negatives)

    for rank, hit in enumerate(top_hits, start=1):
        grade = grade_by_chunk.get(hit.chunk_id)
        units = units_by_chunk.get(hit.chunk_id, set())
        if grade is None:
            if hit.chunk_id not in judged_negative:
                unjudged_count += 1
            gains.append(0)
            continue
        relevant_count += 1
        gains.append((2**grade) - 1)
        direct_units = {
            unit_id
            for unit_id in units
            if unit_id in required_units and grade >= 2
        }
        covered_required.update(direct_units)
        if grade >= 2 and not first_direct_rank:
            first_direct_rank = rank
        if units and units <= covered_any:
            duplicate_count += 1
        covered_any.update(units)

    denominator = requested_k
    recall = len(covered_required) / len(required_units) if required_units else 0.0
    precision = relevant_count / denominator if denominator else 0.0
    hit_rate = 1.0 if first_direct_rank else 0.0
    mrr = 1.0 / first_direct_rank if first_direct_rank else 0.0
    dcg = discounted_gain(gains)
    ideal_gains = sorted(((2**grade) - 1 for grade in grade_by_chunk.values()), reverse=True)[:requested_k]
    idcg = discounted_gain(ideal_gains)
    ndcg = dcg / idcg if idcg else 0.0
    duplicate_rate = duplicate_count / denominator if denominator else 0.0
    unjudged_rate = unjudged_count / denominator if denominator else 0.0
    return {
        "metrics": {
            "recall": recall,
            "precision": precision,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "ndcg": ndcg,
            "duplicate_rate": duplicate_rate,
            "unjudged_rate": unjudged_rate,
        },
        "details": {
            "covered_required_units": sorted(covered_required),
            "missed_required_units": sorted(required_units - covered_required),
            "relevant_chunks": relevant_count,
            "returned_chunks": len(top_hits),
            "duplicate_chunks": duplicate_count,
            "unjudged_chunks": unjudged_count,
            "first_direct_rank": first_direct_rank,
        },
    }


def judgment_maps(case: RagRetrievalCase) -> tuple[dict[str, int], dict[str, set[str]]]:
    grade_by_chunk: dict[str, int] = {}
    units_by_chunk: dict[str, set[str]] = {}
    for unit in case.evidence_units:
        for alternative in unit.alternatives:
            grade_by_chunk[alternative.chunk_id] = max(
                grade_by_chunk.get(alternative.chunk_id, 0),
                alternative.grade,
            )
            units_by_chunk.setdefault(alternative.chunk_id, set()).add(unit.unit_id)
    return grade_by_chunk, units_by_chunk


def hit_judgment(case: RagRetrievalCase, chunk_id: str) -> dict[str, Any]:
    grade_by_chunk, units_by_chunk = judgment_maps(case)
    if chunk_id in grade_by_chunk:
        return {
            "judgment": "relevant",
            "grade": grade_by_chunk[chunk_id],
            "unit_ids": sorted(units_by_chunk.get(chunk_id, set())),
        }
    if chunk_id in set(case.hard_negatives):
        return {"judgment": "hard_negative", "grade": 0, "unit_ids": []}
    if chunk_id in set(case.judged_irrelevant_chunk_ids):
        return {"judgment": "irrelevant", "grade": 0, "unit_ids": []}
    return {"judgment": "unjudged", "grade": None, "unit_ids": []}


def discounted_gain(gains: list[int]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def normalize_ks(ks: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    normalized = sorted({int(k) for k in ks if int(k) > 0})
    if not normalized:
        raise ValueError("At least one positive K value is required")
    return tuple(normalized)
