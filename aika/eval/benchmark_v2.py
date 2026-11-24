"""Source-backed v2 benchmarks, separate from the legacy QA/RAG datasets.

Loading validates annotations, not answers. Source checks require the private
local corpus and PyMuPDF; ordinary schema checks do not open PDFs or use an LLM.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, ValidationError, model_validator


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_V2 = ROOT_DIR / "data/eval/benchmark_v2_dev.jsonl"
CATEGORIES = {
    "simple_lookup": "简单资料查询",
    "complex_analysis": "公司比较或产业链分析",
    "numeric_conflict": "数字、年份与口径",
    "insufficient_evidence": "资料不足或拒答",
}
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Digest = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class BenchmarkV2Error(ValueError):
    """Invalid annotations, stale sources or missing human review."""


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileSnapshot(Annotation):
    path: Text
    sha256: Digest

    @model_validator(mode="after")
    def relative_path(self) -> "FileSnapshot":
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("snapshot paths must be relative to the repository")
        return self


class CorpusSnapshot(Annotation):
    rag_corpus: FileSnapshot
    reports_manifest: FileSnapshot


def publication_end(value: str) -> date:
    """Conservatively interpret a month/year as its last possible date."""
    if re.fullmatch(r"\d{4}", value):
        return date(int(value), 12, 31)
    if re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = map(int, value.split("-"))
        return date(year, month, calendar.monthrange(year, month)[1])
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("published_at must be YYYY, YYYY-MM or YYYY-MM-DD")
    return date.fromisoformat(value)


class SourceReference(Annotation):
    source_id: Text
    report_id: Text
    title: Text
    source_url: Text
    published_at: Text
    pdf: FileSnapshot
    pdf_page: int = Field(strict=True, ge=1)
    section: Text
    context: Text
    quotes: list[Text] = Field(min_length=1)
    chunks: dict[Text, Digest] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_publication(self) -> "SourceReference":
        publication_end(self.published_at)
        return self


class NumericValue(Annotation):
    subject: Text
    metric: Text
    period: Text
    value: Text
    unit: Text

    @model_validator(mode="after")
    def decimal_value(self) -> "NumericValue":
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", self.value):
            raise ValueError("numeric value must be an unformatted decimal string")
        return self


class StandardFact(Annotation):
    fact_id: Text
    statement: Text
    source_ids: list[Text] = Field(min_length=1)
    numeric_values: list[NumericValue] = Field(default_factory=list)


class AnswerPoint(Annotation):
    point_id: Text
    text: Text
    critical: StrictBool
    kind: Literal["fact", "synthesis", "boundary"]
    fact_ids: list[Text]


class EvidenceUnit(Annotation):
    unit_id: Text
    description: Text
    required: StrictBool
    fact_ids: list[Text] = Field(min_length=1)
    primary_sources: list[Text] = Field(min_length=1)
    # Each group is a complete substitute for primary_sources, not an extra hit.
    alternative_sources: list[list[Text]]


class AnswerBehavior(Annotation):
    mode: Literal["complete", "partial", "refuse"]
    answerable: list[Text]
    unanswerable: list[Text]
    reason: Text

    @model_validator(mode="after")
    def consistent_mode(self) -> "AnswerBehavior":
        if self.mode == "complete" and (not self.answerable or self.unanswerable):
            raise ValueError("complete answers need answerable parts and no unanswerable parts")
        if self.mode == "partial" and (not self.answerable or not self.unanswerable):
            raise ValueError("partial answers need both answerable and unanswerable parts")
        if self.mode == "refuse" and (self.answerable or not self.unanswerable):
            raise ValueError("refuse answers need unanswerable parts only")
        return self


class HumanReview(Annotation):
    ai_status: Literal["pdf_checked"]
    ai_checked_on: date
    human_status: Literal["pending", "approved", "changes_requested"]
    human_reviewer: Text | None
    human_reviewed_at: datetime | None
    reviewed_content_sha256: Digest | None
    notes: str

    @model_validator(mode="after")
    def review_record(self) -> "HumanReview":
        if self.human_reviewed_at is not None and self.human_reviewed_at.tzinfo is None:
            raise ValueError("human review timestamp must include a timezone")
        fields = (self.human_reviewer, self.human_reviewed_at, self.reviewed_content_sha256)
        if self.human_status == "pending" and any(item is not None for item in fields):
            raise ValueError("pending review must not contain an approval record")
        if self.human_status != "pending" and any(item is None for item in fields):
            raise ValueError("human decisions require reviewer, timestamp and content hash")
        return self


class Provenance(Annotation):
    dataset: Text
    case_id: Text


class BenchmarkV2Case(Annotation):
    schema_version: Literal["benchmark_v2"]
    dataset_version: Text
    case_id: Text
    split: Literal["dev", "test"]
    pilot: StrictBool
    category: Literal["simple_lookup", "complex_analysis", "numeric_conflict", "insufficient_evidence"]
    question: Text
    as_of_date: date
    leakage_group: Text
    provenance: list[Provenance]
    corpus_snapshot: CorpusSnapshot
    sources: list[SourceReference] = Field(min_length=1)
    facts: list[StandardFact] = Field(min_length=1)
    answer_points: list[AnswerPoint] = Field(min_length=1)
    evidence_units: list[EvidenceUnit] = Field(min_length=1)
    forbidden_inferences: list[Text] = Field(min_length=1)
    answer_behavior: AnswerBehavior
    scoring_notes: list[Text] = Field(min_length=1)
    review: HumanReview

    def public_input(self) -> dict[str, str]:
        """Only public task text is safe to pass to the answering system."""
        return {"question": self.question}

    def content_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"review"})
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def cross_references(self) -> "BenchmarkV2Case":
        def ids(items: list, field: str) -> set[str]:
            values = [getattr(item, field) for item in items]
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {field}")
            return set(values)

        def refs(values: list[str], known: set[str], label: str) -> None:
            if len(set(values)) != len(values) or set(values) - known:
                raise ValueError(f"duplicate or unknown {label}: {values}")

        source_ids = ids(self.sources, "source_id")
        fact_ids = ids(self.facts, "fact_id")
        ids(self.answer_points, "point_id")
        ids(self.evidence_units, "unit_id")
        if self.pilot and self.split != "dev":
            raise ValueError("pilot cases must remain in dev")
        if self.as_of_date.isoformat() not in self.question:
            raise ValueError("question must expose as_of_date to the answering system")
        for source in self.sources:
            if publication_end(source.published_at) > self.as_of_date:
                raise ValueError(f"source {source.source_id} exceeds as_of_date")
        for fact in self.facts:
            refs(fact.source_ids, source_ids, "fact source")
        for point in self.answer_points:
            refs(point.fact_ids, fact_ids, "answer point fact")
            if point.kind != "boundary" and not point.fact_ids:
                raise ValueError("factual/synthesis answer points need standard facts")
        if not any(point.critical for point in self.answer_points):
            raise ValueError("at least one critical answer point is required")
        if self.answer_behavior.mode != "refuse" and not any(unit.required for unit in self.evidence_units):
            raise ValueError("answerable cases need at least one required evidence unit")
        facts_by_id = {fact.fact_id: fact for fact in self.facts}
        for unit in self.evidence_units:
            refs(unit.fact_ids, fact_ids, "evidence fact")
            for group in [unit.primary_sources, *unit.alternative_sources]:
                if not group:
                    raise ValueError("empty evidence alternative")
                refs(group, source_ids, "evidence source")
            needed = {sid for fid in unit.fact_ids for sid in facts_by_id[fid].source_ids}
            if not needed <= set(unit.primary_sources):
                raise ValueError("primary evidence does not cover its standard fact sources")
        if self.answer_behavior.mode != "complete" and not any(
            point.kind == "boundary" for point in self.answer_points
        ):
            raise ValueError("partial/refuse cases need an explicit boundary answer point")
        if self.review.human_status != "pending" and (
            self.review.reviewed_content_sha256 != self.content_sha256()
        ):
            raise ValueError("human review content hash is stale")
        return self


def normalize_quote(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def validate_collection(cases: list[BenchmarkV2Case]) -> None:
    seen: set[str] = set()
    questions: set[str] = set()
    groups: dict[str, str] = {}
    # Identical facts can leak even when the manually chosen group differs.
    source_facts: dict[tuple, str] = {}
    for case in cases:
        if case.case_id in seen:
            raise BenchmarkV2Error(f"duplicate case_id: {case.case_id}")
        seen.add(case.case_id)
        question = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", case.question).casefold())
        if question in questions:
            raise BenchmarkV2Error(f"duplicate question: {case.case_id}")
        questions.add(question)
        if groups.setdefault(case.leakage_group, case.split) != case.split:
            raise BenchmarkV2Error(f"leakage_group crosses dev/test: {case.leakage_group}")
        sources = {source.source_id: source for source in case.sources}
        for fact in case.facts:
            key = (normalize_quote(fact.statement), tuple(sorted(
                sources[sid].report_id for sid in fact.source_ids
            )))
            if source_facts.setdefault(key, case.split) != case.split:
                raise BenchmarkV2Error(f"same standard fact crosses dev/test: {case.case_id}")


def load_benchmark_v2(
    path: str | Path = DEFAULT_BENCHMARK_V2, *, require_human_review: bool = False,
) -> list[BenchmarkV2Case]:
    cases = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkV2Error(f"cannot read benchmark: {path}: {exc}") from exc
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            cases.append(BenchmarkV2Case.model_validate_json(line))
        except ValidationError as exc:
            raise BenchmarkV2Error(f"{path}: line {line_no}: {exc}") from exc
    if not cases:
        raise BenchmarkV2Error(f"benchmark has no cases: {path}")
    validate_collection(cases)
    if require_human_review:
        require_approved(cases)
    return cases


def require_approved(cases: list[BenchmarkV2Case]) -> None:
    pending = [case.case_id for case in cases if case.review.human_status != "approved"]
    if pending:
        raise BenchmarkV2Error(f"human review required: {', '.join(pending)}")
    for case in cases:
        review = case.review
        if (review.human_reviewer is None or review.human_reviewed_at is None
                or review.reviewed_content_sha256 != case.content_sha256()):
            raise BenchmarkV2Error(f"human review record missing or stale: {case.case_id}")


def validate_pilot(cases: list[BenchmarkV2Case]) -> None:
    expected = {category: 5 for category in CATEGORIES}
    if len(cases) != 20 or dict(Counter(case.category for case in cases)) != expected:
        raise BenchmarkV2Error("pilot must contain exactly 20 cases, 5 per category")
    if not all(case.pilot and case.split == "dev" for case in cases):
        raise BenchmarkV2Error("all pilot cases must be flagged pilot=true and split=dev")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sources(cases: list[BenchmarkV2Case], *, root: Path = ROOT_DIR) -> dict[str, int]:
    """Verify frozen files, source metadata, chunk locators and PDF excerpts.

    This is a consistency check, not a human judgement that a fact follows from
    an excerpt or that an absence-of-disclosure claim is established.
    """
    try:
        import fitz
    except ImportError as exc:
        raise BenchmarkV2Error("source checks require PyMuPDF (install the full/dev extra)") from exc
    root = root.resolve()
    checked_files: dict[Path, str] = {}
    corpora: dict[Path, dict] = {}
    manifests: dict[Path, dict] = {}
    pages: dict[tuple[Path, int], str] = {}
    excerpts = 0

    def checked(snapshot: FileSnapshot) -> Path:
        path = (root / snapshot.path).resolve()
        if not path.is_relative_to(root):
            raise BenchmarkV2Error(f"source path escapes root: {snapshot.path}")
        if path not in checked_files:
            checked_files[path] = file_sha256(path)
        if checked_files[path] != snapshot.sha256:
            raise BenchmarkV2Error(f"source hash changed: {snapshot.path}")
        return path

    try:
        for case in cases:
            rag_path = checked(case.corpus_snapshot.rag_corpus)
            manifest_path = checked(case.corpus_snapshot.reports_manifest)
            if rag_path not in corpora:
                rows = [json.loads(line) for line in rag_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                corpora[rag_path] = {row["chunk_id"]: row for row in rows}
                if len(corpora[rag_path]) != len(rows):
                    raise BenchmarkV2Error("duplicate chunk_id in source corpus")
            if manifest_path not in manifests:
                with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                manifests[manifest_path] = {row["report_id"]: row for row in rows}
                if len(manifests[manifest_path]) != len(rows):
                    raise BenchmarkV2Error("duplicate report_id in source manifest")
            for source in case.sources:
                report = manifests[manifest_path].get(source.report_id)
                if report is None:
                    raise BenchmarkV2Error(f"unknown report: {source.report_id}")
                for field, expected in {
                    "published_at": source.published_at, "local_path": source.pdf.path,
                    "sha256": source.pdf.sha256, "source_url": source.source_url, "title": source.title,
                }.items():
                    if report.get(field) != expected:
                        raise BenchmarkV2Error(f"manifest mismatch: {source.report_id}.{field}")
                pdf_path = checked(source.pdf)
                for cid, digest in source.chunks.items():
                    chunk = corpora[rag_path].get(cid)
                    if chunk is None:
                        raise BenchmarkV2Error(f"unknown chunk_id: {cid}")
                    if chunk.get("report_id") != source.report_id or str(chunk.get("page")) != str(source.pdf_page):
                        raise BenchmarkV2Error(f"chunk location mismatch: {cid}")
                    if hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest() != digest:
                        raise BenchmarkV2Error(f"chunk text hash changed: {cid}")
                key = (pdf_path, source.pdf_page)
                if key not in pages:
                    with fitz.open(pdf_path) as pdf:
                        if source.pdf_page > len(pdf):
                            raise BenchmarkV2Error(f"PDF page out of range: {source.source_id}")
                        pages[key] = normalize_quote(pdf[source.pdf_page - 1].get_text())
                for quote in source.quotes:
                    if normalize_quote(quote) not in pages[key]:
                        raise BenchmarkV2Error(f"quote not found in PDF: {case.case_id}/{source.source_id}")
                    excerpts += 1
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        if isinstance(exc, BenchmarkV2Error):
            raise
        raise BenchmarkV2Error(f"cannot verify local sources: {exc}") from exc
    return {"cases": len(cases), "files": len(checked_files), "pdf_pages": len(pages), "excerpts": excerpts}
