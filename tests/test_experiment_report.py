import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('report_experiments', Path(__file__).resolve().parents[1] / 'scripts/report_experiments.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_report_excludes_pending_pairs_and_preserves_unknown_cost(tmp_path):
    generation = tmp_path / 'generation'; generation.mkdir()
    score = tmp_path / 'score'; score.mkdir()
    def ratio(value, pending=0, applicable=True):
        return {'value': value, 'pending': pending, 'provisional': bool(pending), 'applicable': applicable}
    def row(variant, task, pending):
        metrics = {'task_usability': task, 'answer_point_coverage':ratio(1.0), 'unsupported_claims':ratio(None, applicable=False),
                   'citation_correctness':ratio(None, applicable=False), 'fact_support':ratio(None, applicable=False),
                   'evidence_coverage':{'retrieved':ratio(None, applicable=False), 'selected':ratio(None, applicable=False)},
                   'serious_error_case':False}
        return {'case_id':'case1','variant':variant,'category':'simple_lookup','answer_id':variant,'pending':pending,'metrics':metrics,
                'judgment':{'task':{'usability':task,'reason':'reason','serious_errors':[]},'claims':[]}}
    results=[{'case_id':'case1','variant':v,'status':'success','timings_ms':{'total':1000}} for v in 'ABEF']
    (generation/'manifest.json').write_text(json.dumps({'run_id':'gen'}))
    (generation/'results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in results))
    (score/'manifest.json').write_text(json.dumps({'run_id':'score','generation_dir':str(generation),'planned_assessments':4,
       'dataset_sha256':'data','code_hash':'code','status':'needs_review','ready_for_product_conclusions':False,'billing':{'amount':None}}))
    scores=[row('A','unusable',False),row('B','usable',False),row('E','usable',True),row('F','usable',False)]
    (score/'scores.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in scores))
    result=report.render(score,tmp_path/'out')
    assert result['paired_task_changes']['A->B']=={'improved':1}
    assert result['paired_task_changes']['E->F']=={'pending':1}
    assert result['billing']['amount'] is None
    assert result['variants']['A']['evidence_coverage']['retrieved']['mean'] is None
    assert result['variants']['E']['pending_answers']==1
    assert result['ready_for_product_conclusions'] is False


def test_replay_report_keeps_historical_judging_cost(tmp_path):
    generation = tmp_path / 'generation'; generation.mkdir()
    score = tmp_path / 'score'; score.mkdir()
    ratio = {'value': None, 'pending': 0, 'provisional': False, 'applicable': False}
    metrics = {'task_usability': 'unusable', 'answer_point_coverage':ratio, 'unsupported_claims':ratio,
               'citation_correctness':ratio, 'fact_support':ratio,
               'evidence_coverage':{'retrieved':ratio, 'selected':ratio}, 'serious_error_case':False}
    row = {'case_id':'case1','variant':'A','category':'simple_lookup','answer_id':'a','pending':False,'metrics':metrics,
           'judgment':{'task':{'usability':'unusable','reason':'reason','serious_errors':[]},'claims':[]}}
    (generation/'manifest.json').write_text(json.dumps({'run_id':'gen'}))
    (generation/'results.jsonl').write_text(json.dumps({'case_id':'case1','variant':'A','status':'success','timings_ms':{'total':1000}})+'\n')
    historical = {'path':'earlier/manifest.json','billing':{'unique_attempts':78,'amount':None}}
    (score/'manifest.json').write_text(json.dumps({'run_id':'replay','generation_dir':str(generation),'planned_assessments':1,
       'dataset_sha256':'data','code_hash':'code','status':'needs_review','ready_for_product_conclusions':False,
       'billing':{'judging':{'unique_attempts':0}},'scoring_mode':'rule_revalidation_of_saved_model_responses',
       'new_network_requests':0,'historical_judging_billing_reference':historical}))
    (score/'scores.jsonl').write_text(json.dumps(row)+'\n')
    result=report.render(score,tmp_path/'out')
    assert result['historical_judging_billing_reference']==historical
    assert '原评审共78次调用' in (tmp_path/'out'/'report.md').read_text()
    assert '费用未知' in (tmp_path/'out'/'report.md').read_text()
