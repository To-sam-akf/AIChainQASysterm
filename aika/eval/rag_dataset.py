"""Dataset schema and validation for chunk-level RAG retrieval evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RAG_RETRIEVAL_BENCHMARK = ROOT_DIR / "data" / "eval" / "rag_retrieval_v1.jsonl"
REQUIRED_FIELDS = {
    "case_id",
    "split",
    "category",
    "question",
    "filters",
    "evidence_units",
    "judged_irrelevant_chunk_ids",
    "hard_negatives",
    "notes",
}


class RagRetrievalDatasetError(ValueError):
    """Raised when a retrieval benchmark is malformed or stale."""


@dataclass(frozen=True)
class EvidenceAlternative:
    chunk_id: str
    grade: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, location: str) -> "EvidenceAlternative":
        chunk_id = clean_str(payload.get("chunk_id"))
        if not chunk_id:
            raise RagRetrievalDatasetError(f"{location}: chunk_id cannot be empty")
        try:
            grade = int(payload.get("grade"))
        except (TypeError, ValueError) as exc:
            raise RagRetrievalDatasetError(f"{location}: grade must be 1 or 2") from exc
        if grade not in {1, 2}:
            raise RagRetrievalDatasetError(f"{location}: grade must be 1 or 2")
        return cls(chunk_id=chunk_id, grade=grade)


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    required: bool
    description: str
    alternatives: list[EvidenceAlternative] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, location: str) -> "EvidenceUnit":
        unit_id = clean_str(payload.get("unit_id"))
        if not unit_id:
            raise RagRetrievalDatasetError(f"{location}: unit_id cannot be empty")
        alternatives_payload = payload.get("alternatives")
        if not isinstance(alternatives_payload, list) or not alternatives_payload:
            raise RagRetrievalDatasetError(f"{location}: alternatives cannot be empty")
        alternatives = [
            EvidenceAlternative.from_dict(item, location=f"{location}.alternatives[{index}]")
            for index, item in enumerate(alternatives_payload)
            if isinstance(item, dict)
        ]
        if len(alternatives) != len(alternatives_payload):
            raise RagRetrievalDatasetError(f"{location}: every alternative must be an object")
        chunk_ids = [alternative.chunk_id for alternative in alternatives]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RagRetrievalDatasetError(f"{location}: duplicate alternative chunk_id")
        required = bool(payload.get("required", False))
        if required and not any(alternative.grade == 2 for alternative in alternatives):
            raise RagRetrievalDatasetError(f"{location}: required units need at least one grade=2 alternative")
        return cls(
            unit_id=unit_id,
            required=required,
            description=clean_str(payload.get("description")),
            alternatives=alternatives,
        )


@dataclass(frozen=True)
class RagRetrievalCase:
    case_id: str
    split: str
    category: str
    question: str
    filters: dict[str, str] = field(default_factory=dict)
    evidence_units: list[EvidenceUnit] = field(default_factory=list)
    judged_irrelevant_chunk_ids: list[str] = field(default_factory=list)
    hard_negatives: list[str] = field(default_factory=list)
    notes: str = ""
    annotation_status: str = "draft"
    reviewers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, line_no: int = 0) -> "RagRetrievalCase":
        missing = sorted(REQUIRED_FIELDS - set(payload))
        if missing:
            raise RagRetrievalDatasetError(f"line {line_no}: missing fields: {', '.join(missing)}")
        case_id = clean_str(payload.get("case_id"))
        split = clean_str(payload.get("split"))
        category = clean_str(payload.get("category"))
        question = clean_str(payload.get("question"))
        if not case_id:
            raise RagRetrievalDatasetError(f"line {line_no}: case_id cannot be empty")
        if not split:
            raise RagRetrievalDatasetError(f"line {line_no}: split cannot be empty")
        if not category:
            raise RagRetrievalDatasetError(f"line {line_no}: category cannot be empty")
        if not question:
            raise RagRetrievalDatasetError(f"line {line_no}: question cannot be empty")

        filters_payload = payload.get("filters")
        if not isinstance(filters_payload, dict):
            raise RagRetrievalDatasetError(f"line {line_no}: filters must be an object")
        filters = {
            clean_str(key): clean_str(value)
            for key, value in filters_payload.items()
            if clean_str(key) and clean_str(value)
        }

        units_payload = payload.get("evidence_units")
        if not isinstance(units_payload, list) or not units_payload:
            raise RagRetrievalDatasetError(f"line {line_no}: evidence_units cannot be empty")
        units = [
            EvidenceUnit.from_dict(item, location=f"line {line_no}.evidence_units[{index}]")
            for index, item in enumerate(units_payload)
            if isinstance(item, dict)
        ]
        if len(units) != len(units_payload):
            raise RagRetrievalDatasetError(f"line {line_no}: every evidence unit must be an object")
        unit_ids = [unit.unit_id for unit in units]
        if len(unit_ids) != len(set(unit_ids)):
            raise RagRetrievalDatasetError(f"line {line_no}: duplicate unit_id")
        if not any(unit.required for unit in units):
            raise RagRetrievalDatasetError(f"line {line_no}: at least one evidence unit must be required")

        irrelevant = clean_string_list(payload.get("judged_irrelevant_chunk_ids"), location=f"line {line_no}")
        hard_negatives = clean_string_list(payload.get("hard_negatives"), location=f"line {line_no}")
        positive_grades: dict[str, int] = {}
        for unit in units:
            for alternative in unit.alternatives:
                previous = positive_grades.get(alternative.chunk_id)
                if previous is not None and previous != alternative.grade:
                    raise RagRetrievalDatasetError(
                        f"line {line_no}: conflicting grades for chunk_id {alternative.chunk_id}"
                    )
                positive_grades[alternative.chunk_id] = alternative.grade
        negative_ids = set(irrelevant) | set(hard_negatives)
        overlap = sorted(set(positive_grades) & negative_ids)
        if overlap:
            raise RagRetrievalDatasetError(
                f"line {line_no}: positive and negative judgments overlap: {', '.join(overlap)}"
            )

        return cls(
            case_id=case_id,
            split=split,
            category=category,
            question=question,
            filters=filters,
            evidence_units=units,
            judged_irrelevant_chunk_ids=irrelevant,
            hard_negatives=hard_negatives,
            notes=clean_str(payload.get("notes")),
            annotation_status=clean_str(payload.get("annotation_status")) or "draft",
            reviewers=clean_string_list(payload.get("reviewers", []), location=f"line {line_no}"),
        )


def load_rag_retrieval_cases(
    path: Path | str = DEFAULT_RAG_RETRIEVAL_BENCHMARK,
    *,
    known_chunk_ids: set[str] | None = None,
    limit: int = 0,
) -> list[RagRetrievalCase]:
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        raise RagRetrievalDatasetError(f"benchmark file does not exist: {benchmark_path}")
    cases: list[RagRetrievalCase] = []
    seen: set[str] = set()
    for line_no, line in enumerate(benchmark_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RagRetrievalDatasetError(f"line {line_no}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RagRetrievalDatasetError(f"line {line_no}: case payload must be an object")
        case = RagRetrievalCase.from_dict(payload, line_no=line_no)
        if case.case_id in seen:
            raise RagRetrievalDatasetError(f"line {line_no}: duplicate case_id: {case.case_id}")
        seen.add(case.case_id)
        if known_chunk_ids is not None:
            unknown = sorted(referenced_chunk_ids(case) - known_chunk_ids)
            if unknown:
                preview = ", ".join(unknown[:8])
                suffix = "..." if len(unknown) > 8 else ""
                raise RagRetrievalDatasetError(
                    f"line {line_no}: unknown chunk_id values: {preview}{suffix}"
                )
        cases.append(case)
    if not cases:
        raise RagRetrievalDatasetError(f"benchmark file has no cases: {benchmark_path}")
    return cases[:limit] if limit and limit > 0 else cases


def referenced_chunk_ids(case: RagRetrievalCase) -> set[str]:
    chunk_ids = {
        alternative.chunk_id
        for unit in case.evidence_units
        for alternative in unit.alternatives
    }
    chunk_ids.update(case.judged_irrelevant_chunk_ids)
    chunk_ids.update(case.hard_negatives)
    return chunk_ids


def clean_str(value: Any) -> str:
    return str(value or "").strip()


def clean_string_list(value: Any, *, location: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RagRetrievalDatasetError(f"{location}: expected an array of chunk_id strings")
    output = [clean_str(item) for item in value if clean_str(item)]
    if len(output) != len(set(output)):
        raise RagRetrievalDatasetError(f"{location}: duplicate chunk_id in judgment list")
    return output
