"""Human calibration, attributed review import and reproducible review queues."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from aika.eval.batch_runner import append_jsonl, read_journal
from aika.eval.judge_schema import HumanReview


def calibration_sample(rows: list[dict[str, Any]], seed: int, answer_hashes: list[str] | None = None) -> dict[str, Any]:
    if answer_hashes is not None:
        mapping = {row['answer_sha256']: row for row in rows}
        if len(answer_hashes) != 8 or len(set(answer_hashes)) != 8 or not set(answer_hashes) <= mapping.keys():
            raise ValueError("Previous calibration must select eight unchanged answers")
        selected = [mapping[value] for value in answer_hashes]
        if (Counter(r['category'] for r in selected) != Counter(dict.fromkeys(
                ('simple_lookup', 'complex_analysis', 'numeric_conflict', 'insufficient_evidence'), 2))
                or Counter(r['variant'] for r in selected) != Counter(dict.fromkeys(('A', 'B', 'E', 'F'), 2))
                or not any(a['variant'] == 'E' and b['variant'] == 'F' and a['case_id'] == b['case_id']
                           for a in selected for b in selected)):
            raise ValueError("Previous calibration does not satisfy category/variant/E-F requirements")
        return {"seed": seed, "status": "awaiting_human_review", "required": 8,
                "answer_ids": [r['answer_id'] for r in selected]}
    rng = random.Random(seed)
    categories = ("simple_lookup", "complex_analysis", "numeric_conflict", "insufficient_evidence")
    groups = {category: sorted([row for row in rows if row["category"] == category], key=lambda r: r["answer_id"]) for category in categories}
    for group in groups.values():
        rng.shuffle(group)
    anchors = [(a, b) for group in groups.values() for a in group for b in group
               if a["variant"] == "E" and b["variant"] == "F" and a["case_id"] == b["case_id"]]
    def search(index, chosen, counts, anchor):
        if index == len(categories):
            return chosen if all(counts[v] == 2 for v in ("A", "B", "E", "F")) else None
        category = categories[index]
        pairs = [anchor] if category == anchor[0]["category"] else itertools.combinations(groups[category], 2)
        for pair in pairs:
            updated = counts + Counter(row["variant"] for row in pair)
            if max(updated.values()) <= 2:
                found = search(index + 1, chosen + list(pair), updated, anchor)
                if found is not None:
                    return found
        return None
    for anchor in anchors:
        selected = search(0, [], Counter(), anchor)
        if selected:
            return {"seed": seed, "status": "awaiting_human_review", "required": 8, "answer_ids": [r["answer_id"] for r in selected]}
    return {"seed": seed, "status": "insufficient_coverage", "required": 8,
            "available_categories": {category: len(group) for category, group in groups.items()}, "answer_ids": []}


def assessment_signature(assessment: dict[str, Any]) -> dict[str, Any]:
    judgment = assessment["judgment"]
    return {"points": {p["point_id"]: p["status"] for p in judgment["points"]},
            "claims": {c["statement"]: c["status"] for c in judgment["claims"]},
            "task": judgment["task"]["usability"],
            "citations": {(c['claim_id'] + ':' + c['citation_id']): c['status'] for c in judgment.get('citations', [])},
            "serious_errors": judgment['task'].get('serious_errors', []),
            "serious_errors_pending": judgment['task'].get('serious_errors_pending', False),
            "minor_revision_claim_ids": sorted(judgment['task'].get('minor_revision_claim_ids', [])),
            "unsupported_answer": judgment["task"]["unsupported_answer"], "over_refusal": judgment["task"]["over_refusal"]}


def review_queue(rows: list[dict[str, Any]], calibration: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    reasons = defaultdict(set)
    for answer_id in calibration["answer_ids"]:
        reasons[answer_id].add("calibration")
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
        if row["metrics"]["serious_error_case"] is True:
            reasons[row["answer_id"]].add("serious_error")
        if row["pending"] or row["status"] == "judge_error":
            reasons[row["answer_id"]].add("pending")
    for group in by_case.values():
        # Compare task and common gold points; different answer lengths are not themselves disagreements.
        signatures = {(tuple(sorted(assessment_signature(row)["points"].items())),
                       row["metrics"]["task_usability"], row["metrics"]["serious_error_case"]) for row in group}
        if len(signatures) > 1:
            for row in group:
                reasons[row["answer_id"]].add("version_disagreement")
    rng = random.Random(seed)
    rest = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["answer_id"]):
        if row["answer_id"] not in reasons:
            rest[row["category"]].append(row)
    for group in rest.values():
        for row in rng.sample(group, max(1, math.ceil(len(group) * 0.2))):
            reasons[row["answer_id"]].add("stratified_random_20_percent")
    return [{"answer_id": answer_id, "reasons": sorted(values)} for answer_id, values in sorted(reasons.items())]


def import_reviews(path: Path, destination: Path, assessments: dict[str, dict[str, Any]]) -> None:
    reviews = [HumanReview.model_validate(row) for row in read_journal(path)]
    if not reviews:
        raise ValueError("No human reviews to import")
    seen = set()
    for review in reviews:
        if review.answer_id in seen:
            raise ValueError("Duplicate answer in review import")
        seen.add(review.answer_id)
        assessment = assessments.get(review.answer_id)
        if assessment is None or review.answer_sha256 != assessment["answer_sha256"] or review.judgment_sha256 != assessment["judgment_sha256"]:
            raise ValueError("Human review targets an unknown or changed answer/judgment")
    # Validate the full import before appending any audit entries.
    for review in reviews:
        append_jsonl(destination, review.model_dump(mode="json"))


def calibration_status(sample: dict[str, Any], model_rows: list[dict[str, Any]], reviewed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    original = {row["answer_id"]: row for row in model_rows}
    reviewed = {row["answer_id"]: row for row in reviewed_rows if row.get("human_review")}
    differences, missing, pending = [], [], []
    for answer_id in sample["answer_ids"]:
        if answer_id not in reviewed:
            missing.append(answer_id)
        else:
            if reviewed[answer_id]["pending"]:
                pending.append(answer_id)
            if assessment_signature(original[answer_id]) != assessment_signature(reviewed[answer_id]):
                differences.append({"answer_id": answer_id, "model": assessment_signature(original[answer_id]),
                                    "human": assessment_signature(reviewed[answer_id])})
    ready = len(sample["answer_ids"]) == 8 and not (missing or pending or differences)
    return {**sample, "status": "calibrated" if ready else sample["status"] if not sample["answer_ids"] else "needs_review_or_rubric_revision",
            "missing_human_reviews": missing, "pending_human_reviews": pending, "differences": differences,
            "ready_for_product_conclusions": ready}
