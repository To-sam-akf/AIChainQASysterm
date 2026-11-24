"""Apply explicitly attributed partial reviews without claiming full review.

The caller supplies scoring functions from the source experiment's frozen code.
Partial reviews are separate from ``HumanReview`` and cannot establish that a
whole answer has been reviewed.
"""
from __future__ import annotations

import copy
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from aika.eval.judge_schema import CitationJudgment, ClaimJudgment, CoverageDecision, Judgment


class ScopedReview(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answer_id: str = Field(min_length=1)
    answer_sha256: str = Field(min_length=1)
    judgment_sha256: str = Field(min_length=1)
    expected_state_sha256: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime
    notes: str = Field(min_length=1)
    coverage: dict[Literal['retrieved', 'selected'], list[CoverageDecision]] = Field(default_factory=dict)
    claims: list[ClaimJudgment] = Field(default_factory=list)
    citations: list[CitationJudgment] = Field(default_factory=list)
    over_refusal: StrictBool | None = None

    @model_validator(mode='after')
    def validate_scope(self):
        if not self.reviewer.strip() or self.reviewed_at.utcoffset() is None:
            raise ValueError('A scoped review requires an attributed reviewer and timezone-aware date')
        if not any(self.coverage.values()) and not self.claims and not self.citations and self.over_refusal is None:
            raise ValueError('A scoped review must explicitly address at least one item')
        for values, key in ((self.claims, lambda x: x.claim_id),
                            (self.citations, lambda x: (x.claim_id, x.citation_id))):
            if len({key(x) for x in values}) != len(values):
                raise ValueError('A scoped review cannot contain duplicate judgment targets')
        for decisions in self.coverage.values():
            if len({d.unit_id for d in decisions}) != len(decisions):
                raise ValueError('A scoped review cannot contain duplicate coverage units')
        return self


def review_state_digest(row, digest):
    """Bind a review to the exact effective state, including earlier partial edits."""
    return digest({'judgment': row['judgment'], 'coverage': row['coverage']})


def apply_scoped_review(row, review, *, case, generation, payload, digest,
                        apply_coverage_review, validate_judgment, score_metrics, is_pending):
    review = ScopedReview.model_validate(review)
    if any(getattr(review, key) != row[key] for key in ('answer_id', 'answer_sha256', 'judgment_sha256')):
        raise ValueError('Scoped review targets a different answer or original judgment')
    if review.expected_state_sha256 != review_state_digest(row, digest):
        raise ValueError('Scoped review targets a stale effective state')
    result = copy.deepcopy(row)
    if review.coverage:
        # The legacy validator requires all required units in a stage. Supply
        # pending placeholders solely for validation, then apply only the
        # explicitly reviewed units; existing units are never overwritten.
        required = [u.unit_id for u in case.evidence_units if u.required]
        from types import SimpleNamespace
        expanded = {}
        for stage, decisions in review.coverage.items():
            selected = {d.unit_id: d for d in decisions}
            if not set(selected) <= set(required):
                raise ValueError('Unknown coverage unit')
            expanded[stage] = [selected.get(uid) or CoverageDecision(unit_id=uid, status='pending',
                                reason='未在此次局部审核范围内', support=[]) for uid in required]
        checked = apply_coverage_review(SimpleNamespace(evidence_coverage=expanded), result['coverage'],
                                        result['coverage_materials'], case)
        for stage, decisions in review.coverage.items():
            ids = {d.unit_id for d in decisions}
            replacements = {d['unit_id']: d for d in checked[stage] if d['unit_id'] in ids}
            result['coverage'][stage] = [replacements.get(d['unit_id'], d) for d in result['coverage'][stage]]

    judgment = copy.deepcopy(result['judgment'])
    for kind, updates, fields in (
            ('claims', review.claims, ('claim_id', 'segment_id', 'statement', 'citation_ids')),
            ('citations', review.citations, ('claim_id', 'citation_id'))):
        for update in updates:
            replacement = update.model_dump()
            matching = [i for i, old in enumerate(judgment[kind])
                        if all(old[f] == replacement[f] for f in fields)]
            if len(matching) != 1:
                raise ValueError('Scoped judgment target is missing or changes its identity')
            judgment[kind][matching[0]] = replacement
    if review.over_refusal is not None:
        judgment['task']['over_refusal'] = review.over_refusal
    if review.claims or review.citations or review.over_refusal is not None:
        checked = validate_judgment(Judgment.model_validate(copy.deepcopy(judgment)), payload).model_dump()
        if checked != judgment:
            raise ValueError('Judgment validation changed an item outside the proposed scope or rejected its evidence')
        result['judgment'] = checked

    result['metrics'] = score_metrics(case, generation, Judgment.model_validate(result['judgment']), result['coverage'])
    result['pending'] = is_pending(result['metrics'])
    result.setdefault('scoped_human_reviews', []).append(review.model_dump(mode='json'))
    # In particular, do not set, remove, or replace row['human_review'].
    assert result.get('human_review') == row.get('human_review')
    return result
