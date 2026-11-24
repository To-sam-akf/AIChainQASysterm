"""Map expert decisions on pilot_014 to frozen evidence without editing inputs."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/eval_runs/closeouts/continuation_20261124/expert_review_snapshot_20261124'
RUN=ROOT/'data/eval_runs/experiments/experiment_20261123_215354_ca07ed1a'
sys.path.insert(0,'/tmp/aiqasys_frozen_222235_4e268bb1')
from aika.eval.answer_judge import apply_coverage_review
from aika.eval.benchmark_v2 import load_benchmark_v2
from aika.eval.judge_schema import CoverageDecision, CoverageLink, Judgment
from aika.eval.faithfulness import score_metrics, is_pending

def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 rows=read(BASE/'scores.jsonl'); actions=read(BASE/'expert_decisions.jsonl')
 cases={c.case_id:c for c in load_benchmark_v2(RUN/'benchmark.jsonl',require_human_review=True)}
 gen={(x['case_id'],x['variant']):x for x in read(RUN/'results.jsonl')}; changed=[]
 for row in rows:
  if row['case_id']!='pilot_014' or row['variant'] not in ('B','E','F'): continue
  case=cases[row['case_id']]; source=next(s for s in case.sources if s.source_id=='cambricon_rd')
  for stage in ('retrieved','selected'):
   material=next(m for m in row['coverage_materials'][stage] if m['material_id']=='material_34d8fc1e338f9460d540')
   for s in ('1,169,100,962.15','17.99','91.30','73.31'):
    assert s in material['text'] and s in '\n'.join(source.quotes)
   link=CoverageLink(source_id='cambricon_rd',original_quote='1,169,100,962.15',material_id=material['material_id'],material_quote='1,169,100,962.15')
   decisions=[CoverageDecision(unit_id='e_f1',status='matched',reason='专家确认研发投入及比例；链接为同表金额，另核对比例和变动数字。',support=[link]),
              CoverageDecision(unit_id='e_f2',status='pending',reason='不在该项检索阶段的本次覆盖映射内',support=[])]
   if stage=='selected':
    assert all('不包含股份支付费用' not in m['text'] for m in row['coverage_materials'][stage])
    decisions[1]=CoverageDecision(unit_id='e_f2',status='missing',reason='专家核查选用材料缺少表下注释；参考年报不能补充至生成输入。',support=[])
   validated=apply_coverage_review(SimpleNamespace(evidence_coverage={stage:decisions}),row['coverage'],row['coverage_materials'],case)
   ids={'e_f1'} | ({'e_f2'} if stage=='selected' else set())
   previous={x['unit_id']:x for x in row['coverage'][stage]}
   for item in validated[stage]:
    if item['unit_id'] in ids:
     row['coverage'][stage][next(i for i,x in enumerate(row['coverage'][stage]) if x['unit_id']==item['unit_id'])]=item
     changed.append({'case_id':row['case_id'],'variant':row['variant'],'stage':stage,'unit_id':item['unit_id'],'before':previous[item['unit_id']],'after':item})
   for a in actions:
    if a['case_id']==row['case_id'] and a['variant']==row['variant'] and a['type']=='材料覆盖' and a['item_id'] in {f'{stage}:{uid}' for uid in ids}:
     a['applied']=True;a['disposition']='依据原始材料和专家意见完成冻结评分器的证据链接校验'
  row['metrics']=score_metrics(case,gen[(row['case_id'],row['variant'])],Judgment.model_validate(row['judgment']),row['coverage']);row['pending']=is_pending(row['metrics'])
 assert len(changed)==9
 out=BASE/'coverage_mapped';out.mkdir(exist_ok=False)
 (out/'scores.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 (out/'expert_decisions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in actions))
 (out/'coverage_changes.json').write_text(json.dumps(changed,ensure_ascii=False,indent=2)+'\n')
 manifest={'source_sha256':sha(BASE/'scores.jsonl'),'output_sha256':sha(out/'scores.jsonl'),'additional_mapped_items':len(changed),'applied_total':sum(x['applied'] for x in actions),'pending_answers':sum(x['pending'] for x in rows),'ready_for_product_conclusions':False}
 (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(manifest,ensure_ascii=False))

if __name__=='__main__':main()
