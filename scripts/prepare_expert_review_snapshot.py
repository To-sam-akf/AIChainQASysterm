"""Apply safely representable expert decisions to the confirmed pilot snapshot."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path('/tmp/aiqasys_frozen_222235_4e268bb1')
sys.path.insert(0, str(FROZEN))
from aika.eval.answer_judge import validate_judgment
from aika.eval.benchmark_v2 import load_benchmark_v2
from aika.eval.faithfulness import is_pending, score_metrics
from aika.eval.judge_schema import Judgment
from aika.eval.score_runner import judging_code_hash

BASE = ROOT / 'data/eval_runs/closeouts/continuation_20261124'
SCORE = BASE / 'confirmed_review_1_10/scores.jsonl'
CSV = BASE / 'reviewer_packet_30/填写表.csv'
RUN = ROOT / 'data/eval_runs/experiments/experiment_20261123_215354_ca07ed1a'
ORIGINAL = ROOT / 'data/eval_runs/scores/score_20261123_222235_4e268bb1'
OUT = BASE / 'expert_review_snapshot_20261124'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main():
    manifest = json.loads((ORIGINAL/'manifest.json').read_text())
    assert judging_code_hash() == manifest['code_hash']
    assert sha(ORIGINAL/'source.zip') == manifest['code']['source_sha256']
    assert sha(SCORE) == '6e9681959a183dbdcf39886994de523fe72ab281f6ffdf71fe1b78ada5527236'
    assert sha(CSV) == '844f61e11c8322f281880ca9d45a4fcb51ff4172c562d1efeedc5fdfa70ddba0'
    rows = read(SCORE)
    by_id = {r['answer_id']: r for r in rows}
    gen = {(r['case_id'], r['variant']): r for r in read(RUN/'results.jsonl')}
    cases = {c.case_id: c for c in load_benchmark_v2(RUN/'benchmark.jsonl', require_human_review=True)}
    packets = {r['answer_id']: r['input'] for r in read(ORIGINAL/'review_packet.jsonl')}
    with CSV.open(encoding='utf-8-sig', newline='') as f:
        reviews = list(csv.DictReader(f))
    assert len(reviews) == 113 and len({(x['答案ID'], x['项目类型'], x['项目ID']) for x in reviews}) == 113
    actions = []
    for review in reviews:
        row = by_id[review['答案ID']]
        assert row['case_id'] == review['题号'] and row['variant'] == review['版本']
        kind, item_id, decision = review['项目类型'], review['项目ID'], review['结论']
        action = {'answer_id': row['answer_id'], 'case_id': row['case_id'], 'variant': row['variant'],
                  'type': kind, 'item_id': item_id, 'expert_decision': decision,
                  'reason': review['理由及原文页码'], 'reviewer_as_recorded': review['审核人'],
                  'review_date': review['审核日期'], 'applied': False, 'disposition': ''}
        if decision == '无法判断':
            action['disposition'] = '专家保留待核'
        elif kind == '事实陈述' and decision in ('支持', '支持，限输入范围'):
            previous = copy.deepcopy(row['judgment'])
            item = next(x for x in row['judgment']['claims'] if x['claim_id'] == item_id)
            assert item['status'] == 'pending'
            item['status'] = 'supported'
            item['reason'] = review['理由及原文页码']
            try:
                checked = validate_judgment(Judgment.model_validate(copy.deepcopy(row['judgment'])), packets[row['answer_id']]).model_dump()
                if checked != row['judgment']:
                    raise ValueError('冻结评分校验未接受该字段或改变其他字段')
                action.update(applied=True, disposition='按事实陈述支持处理')
            except ValueError as exc:
                row['judgment'] = previous
                action['disposition'] = f'待证据映射：{exc}'
        elif kind == '整体可用性' and decision == '需要大改并重新核实':
            assert row['judgment']['task']['usability'] == 'pending'
            row['judgment']['task']['usability'] = 'major_revision'
            row['judgment']['task']['reason'] = review['理由及原文页码']
            action.update(applied=True, disposition='任务判为需大改')
        elif kind == '严重错误候选' and decision == '未发现严重错误，但存在重要漏答':
            assert row['judgment']['task']['serious_errors_pending']
            assert not row['judgment']['task']['serious_errors']
            row['judgment']['task']['serious_errors_pending'] = False
            action.update(applied=True, disposition='本份候选按专家复核未发现严重错误；保留重要漏答及其他待核字段')
        else:
            action['disposition'] = '意见已保存；该类别或措辞尚未完成逐字段证据映射，不自动更改评分'
        actions.append(action)
    for row in rows:
        key = (row['case_id'], row['variant'])
        row['metrics'] = score_metrics(cases[key[0]], gen[key], Judgment.model_validate(row['judgment']), row['coverage'])
        row['pending'] = is_pending(row['metrics'])
    assert len(rows) == 80 and len(actions) == 113
    OUT.mkdir(exist_ok=False)
    (OUT/'scores.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    (OUT/'expert_decisions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in actions))
    summary = {'status': 'partial_expert_review_snapshot', 'scoring_code_hash': manifest['code_hash'],
               'source_scores_sha256': sha(SCORE), 'expert_csv_sha256': sha(CSV),
               'user_clarification': '审核人为专家，使用AI辅助；表中保留原署名，姓名及是否独立于开发未记录。',
               'answers': len(rows), 'expert_items': len(actions),
               'expert_unable_to_judge': sum(x['expert_decision']=='无法判断' for x in actions),
               'applied': sum(x['applied'] for x in actions),
               'not_applied_by_type': dict(Counter(x['type'] for x in actions if not x['applied'] and x['expert_decision']!='无法判断')),
               'pending_answers': sum(r['pending'] for r in rows),
               'serious_candidates_pending': sum(r['judgment']['task']['serious_errors_pending'] for r in rows),
               'full_answer_review_not_inferred': True, 'ready_for_product_conclusions': False,
               'scores_sha256': sha(OUT/'scores.jsonl')}
    (OUT/'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
