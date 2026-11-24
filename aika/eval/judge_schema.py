"""Versioned independent judgments; unresolved items never become implicit passes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Support(Record):
    source_id: str
    quote: str = Field(min_length=1)


class PointJudgment(Record):
    point_id: str
    status: Literal["correct", "missing", "incorrect", "pending"]
    answer_quote: str
    reason: str = Field(min_length=1)
    support: list[Support]


class ClaimJudgment(Record):
    claim_id: str
    segment_id: str
    statement: str = Field(min_length=1)
    status: Literal["supported", "contradicted", "unsupported", "pending"]
    reason: str = Field(min_length=1)
    support: list[Support]
    citation_ids: list[str]


class SegmentJudgment(Record):
    segment_id: str
    kind: Literal["factual", "nonfactual", "pending"]
    reason: str = Field(min_length=1)


class CitationJudgment(Record):
    claim_id: str
    citation_id: str
    status: Literal["correct", "incorrect", "pending"]
    reason: str = Field(min_length=1)
    support: list[Support]


class SeriousError(Record):
    kind: Literal["wrong_subject", "numeric_or_period", "fabricated_source", "unsupported_conclusion", "post_cutoff"]
    answer_quote: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    support: list[Support]


class TaskJudgment(Record):
    usability: Literal["usable", "minor_revision", "major_revision", "unusable", "pending"]
    reason: str = Field(min_length=1)
    unsupported_answer: StrictBool | None
    over_refusal: StrictBool | None
    serious_errors: list[SeriousError]
    serious_errors_pending: StrictBool
    minor_revision_claim_ids: list[str] = Field(default_factory=list)


class Judgment(Record):
    points: list[PointJudgment]
    segments: list[SegmentJudgment]
    claims: list[ClaimJudgment]
    citations: list[CitationJudgment]
    task: TaskJudgment


class CoverageLink(Record):
    source_id: str
    original_quote: str = Field(min_length=1)
    material_id: str
    material_quote: str = Field(min_length=1)


class CoverageDecision(Record):
    unit_id: str
    status: Literal["matched", "missing", "pending"]
    reason: str = Field(min_length=1)
    support: list[CoverageLink]


class HumanReview(Record):
    answer_id: str
    answer_sha256: str
    judgment_sha256: str
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime
    decision: Literal["approved", "corrected"]
    notes: str = Field(min_length=1)
    minutes: float | None = Field(default=None, ge=0)
    judgment: Judgment | None
    evidence_coverage: dict[Literal["retrieved", "selected"], list[CoverageDecision]] | None = None

    @model_validator(mode="after")
    def attributed(self):
        if self.reviewed_at.tzinfo is None or not self.reviewer.strip():
            raise ValueError("Human review requires a reviewer and timezone-aware timestamp")
        if self.decision == "corrected" and self.judgment is None:
            raise ValueError("Corrected reviews must contain the complete replacement judgment")
        if self.decision == "approved" and self.judgment is not None:
            raise ValueError("Approved reviews reuse the recorded judgment; use corrected to change it")
        return self
