"""Benchmark dataset loading and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_QA_BENCHMARK = ROOT_DIR / "data" / "eval" / "qa_benchmark_v1.jsonl"
REQUIRED_FIELDS = {
    "case_id",
    "category",
    "question",
    "expected_answer_type",
    "expected_companies",
    "expected_topics",
    "required_claim_types",
    "expected_claim_ids",
    "forbidden_terms",
    "refusal_expected",
    "scoring_notes",
    "human_score",
}


class EvalDatasetError(ValueError):
    """Raised when a benchmark dataset is malformed."""


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str
    question: str
    expected_answer_type: str = ""
    expected_companies: list[str] = field(default_factory=list)
    expected_topics: list[str] = field(default_factory=list)
    required_claim_types: list[str] = field(default_factory=list)
    expected_claim_ids: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)
    refusal_expected: bool = False
    scoring_notes: str = ""
    human_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, line_no: int = 0) -> "EvalCase":
        missing = sorted(REQUIRED_FIELDS - set(payload))
        if missing:
            raise EvalDatasetError(f"line {line_no}: missing fields: {', '.join(missing)}")
        case_id = clean_str(payload.get("case_id"))
        category = clean_str(payload.get("category"))
        question = clean_str(payload.get("question"))
        if not case_id:
            raise EvalDatasetError(f"line {line_no}: case_id cannot be empty")
        if not category:
            raise EvalDatasetError(f"line {line_no}: category cannot be empty")
        if not question:
            raise EvalDatasetError(f"line {line_no}: question cannot be empty")
        human_score = payload.get("human_score")
        if human_score in ("", None):
            normalized_human_score = None
        else:
            try:
                normalized_human_score = float(human_score)
            except (TypeError, ValueError) as exc:
                raise EvalDatasetError(f"line {line_no}: human_score must be numeric or null") from exc
            if normalized_human_score < 0 or normalized_human_score > 5:
                raise EvalDatasetError(f"line {line_no}: human_score must be in [0, 5]")
        refusal_expected = bool(payload.get("refusal_expected", False))
        forbidden_terms = clean_list(payload.get("forbidden_terms"))
        if refusal_expected and not forbidden_terms:
            raise EvalDatasetError(f"line {line_no}: refusal cases must define forbidden_terms")
        return cls(
            case_id=case_id,
            category=category,
            question=question,
            expected_answer_type=clean_str(payload.get("expected_answer_type")),
            expected_companies=clean_list(payload.get("expected_companies")),
            expected_topics=clean_list(payload.get("expected_topics")),
            required_claim_types=clean_list(payload.get("required_claim_types")),
            expected_claim_ids=clean_list(payload.get("expected_claim_ids")),
            forbidden_terms=forbidden_terms,
            refusal_expected=refusal_expected,
            scoring_notes=clean_str(payload.get("scoring_notes")),
            human_score=normalized_human_score,
        )


def load_eval_cases(path: Path | str = DEFAULT_QA_BENCHMARK, *, limit: int = 0) -> list[EvalCase]:
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        raise EvalDatasetError(f"benchmark file does not exist: {benchmark_path}")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for line_no, line in enumerate(benchmark_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalDatasetError(f"line {line_no}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise EvalDatasetError(f"line {line_no}: case payload must be an object")
        case = EvalCase.from_dict(payload, line_no=line_no)
        if case.case_id in seen:
            raise EvalDatasetError(f"line {line_no}: duplicate case_id: {case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
        if limit and len(cases) >= limit:
            break
    if not cases:
        raise EvalDatasetError(f"benchmark file has no cases: {benchmark_path}")
    return cases


def clean_str(value: Any) -> str:
    return str(value or "").strip()


def clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise EvalDatasetError("list field must be an array or comma-separated string")
