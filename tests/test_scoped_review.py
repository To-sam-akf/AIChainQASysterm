import copy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from aika.eval.experiment_artifacts import digest
from aika.eval.scoped_review import ScopedReview, apply_scoped_review, review_state_digest


def row_and_review():
    row = {'answer_id': 'a', 'answer_sha256': 'answer', 'judgment_sha256': 'judge',
           'judgment': {'points': [], 'segments': [], 'claims': [], 'citations': [],
                        'task': {'usability': 'pending', 'reason': 'unresolved', 'unsupported_answer': None,
                                 'over_refusal': None, 'serious_errors': [], 'serious_errors_pending': False,
                                 'minor_revision_claim_ids': []}},
           'coverage': {'retrieved': [{'unit_id': 'one', 'status': 'pending'}, {'unit_id': 'two', 'status': 'missing'}],
                        'selected': [], 'segments': []},
           'coverage_materials': {'retrieved': [], 'selected': []},
           'human_review': {'reviewer': 'previous reviewer'}}
    review = {'answer_id': 'a', 'answer_sha256': 'answer', 'judgment_sha256': 'judge',
              'expected_state_sha256': review_state_digest(row, digest), 'reviewer': 'test reviewer',
              'reviewed_at': datetime.now(timezone.utc).isoformat(), 'notes': 'test only',
              'coverage': {'retrieved': [{'unit_id': 'one', 'status': 'matched', 'reason': 'test source', 'support': []}]}}
    return row, review


def apply(row, review, *, coverage_validator=None, judgment_validator=None):
    def validate_coverage(review, coverage, materials, case):
        assert [d.unit_id for d in review.evidence_coverage['retrieved']] == ['one', 'two']
        assert review.evidence_coverage['retrieved'][1].status == 'pending'
        result = copy.deepcopy(coverage)
        result['retrieved'] = [d.model_dump() for d in review.evidence_coverage['retrieved']]
        return result
    return apply_scoped_review(
        row, review, case=SimpleNamespace(evidence_units=[SimpleNamespace(unit_id='one', required=True),
                                                           SimpleNamespace(unit_id='two', required=True)]),
        generation={}, payload={}, digest=digest, apply_coverage_review=coverage_validator or validate_coverage,
        validate_judgment=judgment_validator or (lambda j, payload: j),
        score_metrics=lambda *args: {'task_usability': 'pending'}, is_pending=lambda metrics: True)


def test_partial_coverage_preserves_unreviewed_item_and_existing_human_review():
    row, review = row_and_review()
    result = apply(row, review)
    assert result['coverage']['retrieved'][0]['status'] == 'matched'
    assert result['coverage']['retrieved'][1] == row['coverage']['retrieved'][1]
    assert result['human_review'] == row['human_review']
    assert result['scoped_human_reviews'][0]['reviewer'] == 'test reviewer'
    assert row['coverage']['retrieved'][0]['status'] == 'pending'


def test_stale_or_unattributed_review_is_rejected():
    row, review = row_and_review()
    row['judgment']['task']['reason'] = 'changed after review'
    with pytest.raises(ValueError, match='stale'):
        apply(row, review)
    review['reviewer'] = ' '
    with pytest.raises(ValueError, match='reviewer'):
        ScopedReview.model_validate(review)


def test_unreviewed_state_or_other_fields_cannot_be_silently_changed():
    row, review = row_and_review()
    review['coverage'] = {}
    review['over_refusal'] = False
    def alters_other_fields(judgment, payload):
        judgment.task.usability = 'usable'
        return judgment
    with pytest.raises(ValueError, match='outside the proposed scope'):
        apply(row, review, judgment_validator=alters_other_fields)


def test_invalid_coverage_evidence_is_not_accepted():
    row, review = row_and_review()
    def reject(*args):
        raise ValueError('quote not present in original')
    with pytest.raises(ValueError, match='quote not present'):
        apply(row, review, coverage_validator=reject)
