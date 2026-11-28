from copy import deepcopy
from types import SimpleNamespace

import pytest
from aika.cli import qa_quality_exit
from aika.eval.runner import summarize_results


def report(statuses):
    metrics = dict.fromkeys(('claim_recall@k', 'evidence_precision@k', 'citation_validity',
                             'answer_groundedness', 'unsupported_claim_rate'), 0.8)
    metrics['human_score'] = None
    rows = [{'case_id': str(i), 'question': f'question {i}', 'category': 'test',
             'score': 0.8, 'metrics': metrics.copy(), 'status': status}
            for i, status in enumerate(statuses)]
    cases = [SimpleNamespace(case_id=row['case_id'], question=row['question'], category=row['category']) for row in rows]
    return {'run_id': 'run', 'created_at': 'today',
            'dataset': {'cases': len(rows), 'name': 'qa', 'path': 'qa', 'hash': 'abc', 'version': 'v1'},
            'environment': {'git_commit': '', 'k': 1, 'limit': 0, 'engine_status': {}},
            'summary': summarize_results(rows), 'results': rows}, cases


@pytest.mark.parametrize('statuses,diagnostic,regression,acceptance', [
    (['pass'], 0, 0, 0), (['warn'], 0, 0, 1), (['fail'], 0, 1, 1),
    ([], 1, 1, 1), ([None], 1, 1, 1)])
def test_quality_policies(statuses, diagnostic, regression, acceptance):
    value, cases = report(statuses)
    for mode, expected in [('diagnostic', diagnostic), ('regression', regression), ('acceptance', acceptance)]:
        assert qa_quality_exit(value, mode, saved=True, expected_cases=cases) == expected


def test_missing_report_or_inconsistent_results_fail():
    value, cases = report(['pass'])
    assert qa_quality_exit({}, 'acceptance', saved=True, expected_cases=cases) == 1
    assert qa_quality_exit(value, 'acceptance', saved=False, expected_cases=cases) == 1
    original_repro = {'dataset': {'cases': 1}, 'summary': {'cases': 1, 'passed': 1, 'warned': 0, 'failed': 0, 'overall_score': 0.8}, 'results': [{'status': 'pass'}]}
    for mode in ('diagnostic', 'regression', 'acceptance'):
        assert qa_quality_exit(original_repro, mode, saved=True, expected_cases=cases) == 1


@pytest.mark.parametrize('change', ['duplicate', 'replacement', 'missing_score', 'missing_metric', 'bad_summary', 'missing_metadata', 'wrong_question'])
def test_incomplete_or_inconsistent_report_rejected(change):
    value, cases = report(['pass', 'pass'])
    value = deepcopy(value)
    if change == 'duplicate':
        value['results'][1] = deepcopy(value['results'][0])
    elif change == 'replacement':
        value['results'][0]['case_id'] = 'other'
    elif change == 'missing_score':
        del value['results'][0]['score']
    elif change == 'missing_metric':
        del value['results'][0]['metrics']['citation_validity']
    elif change == 'bad_summary':
        value['summary']['overall_score'] = 1.0
    elif change == 'missing_metadata':
        del value['dataset']['hash']
    else:
        value['results'][0]['question'] = 'different'
    assert qa_quality_exit(value, 'acceptance', saved=True, expected_cases=cases) == 1
