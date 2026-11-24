"""Independent per-answer metrics with explicit pending and N/A denominators."""

from __future__ import annotations

from typing import Any

from aika.eval.judge_schema import Judgment
from aika.eval.benchmark_v2 import normalize_quote


def ratio(statuses: list[str], *, positive: set[str], pending: set[str] | None = None, applicable: bool = True) -> dict[str, Any]:
    pending = pending or {"pending"}
    unresolved = sum(status in pending for status in statuses)
    denominator = len(statuses) - unresolved
    numerator = sum(status in positive for status in statuses)
    return {
        "numerator": numerator if applicable else None, "denominator": denominator if applicable else None,
        "pending": unresolved if applicable else 0, "total": len(statuses) if applicable else 0,
        "value": numerator / denominator if applicable and denominator else None,
        "applicable": applicable and bool(statuses), "provisional": applicable and bool(unresolved),
        "review_coverage": denominator / len(statuses) if applicable and statuses else None,
    }


def score_metrics(case, row, judgment: Judgment, coverage: dict[str, Any]) -> dict[str, Any]:
    failed = row["status"] != "success"
    points = {point.point_id: point.status for point in judgment.points}
    point_statuses = [points[point.point_id] if not failed else "missing" for point in case.answer_points if point.critical]
    distinct = {}
    for claim in judgment.claims:
        text = normalize_quote(claim.statement)
        if text in distinct and distinct[text] != claim.status:
            distinct[text] = "pending"
        else:
            distinct[text] = claim.status
    claims = list(distinct.values())
    # Every unresolved segment adds an explicit unresolved claim inventory placeholder.
    claims.extend("pending" for segment in judgment.segments if segment.kind == "pending")
    citations = [citation.status for citation in judgment.citations]
    for segment in judgment.segments:
        if segment.kind == "pending":
            definition = next(s for s in coverage["segments"] if s["segment_id"] == segment.segment_id)
            citations.extend("pending" for _ in definition["citation_ids"])
    evidence = {}
    for stage in ("retrieved", "selected"):
        statuses = [unit["status"] if not failed else "missing" for unit in coverage[stage]]
        evidence[stage] = ratio(statuses, positive={"matched"}, applicable=row["variant"] != "A")
    refuses = case.answer_behavior.mode in ("partial", "refuse")
    answerable = case.answer_behavior.mode != "refuse"
    def boolean_metric(value, applicable):
        return ratio(["pending" if value is None else "yes" if value else "no"], positive={"yes"}, applicable=applicable)
    return {
        "answer_point_coverage": ratio(point_statuses, positive={"correct"}),
        "evidence_coverage": evidence,
        "fact_support": ratio(claims, positive={"supported"}, applicable=not failed),
        "claim_inventory_complete": not failed and all(segment.kind != "pending" for segment in judgment.segments),
        "pending_inventory_segments": sum(segment.kind == "pending" for segment in judgment.segments) if not failed else 0,
        "unsupported_claims": ratio(claims, positive={"contradicted", "unsupported"}, applicable=not failed),
        "contradicted_count": sum(status == "contradicted" for status in claims),
        "unsupported_count": sum(status == "unsupported" for status in claims),
        "citation_correctness": ratio(citations, positive={"correct"}, applicable=not failed),
        "missing_citations": bool(not citations and judgment.claims) if not failed else None,
        "unsupported_answer": boolean_metric(judgment.task.unsupported_answer, refuses and not failed),
        "over_refusal": boolean_metric(judgment.task.over_refusal, answerable and not failed),
        "task_usability": "unusable" if failed else judgment.task.usability,
        "serious_error_count": len(judgment.task.serious_errors),
        "serious_error_case": True if judgment.task.serious_errors else None if judgment.task.serious_errors_pending else False,
    }


def is_pending(metrics: dict[str, Any]) -> bool:
    def visit(value):
        if isinstance(value, dict):
            return value.get("provisional", False) or any(visit(item) for item in value.values())
        return False
    return visit(metrics) or metrics["task_usability"] == "pending" or metrics["serious_error_case"] is None
