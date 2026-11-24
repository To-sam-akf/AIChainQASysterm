"""Produce a reproducible provisional comparison report from an existing scoring run."""
from __future__ import annotations
import argparse,json,statistics,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aika.eval.batch_runner import read_journal,latest_results

def render(score:Path, output:Path):
 m=json.loads((score/'manifest.json').read_text()); rows=read_journal(score/'scores.jsonl')
 if len(rows)!=m['planned_assessments'] or len({(x['case_id'],x['variant']) for x in rows})!=len(rows):raise ValueError('Incomplete or duplicate scores')
 generation=Path(m['generation_dir']); gm=json.loads((generation/'manifest.json').read_text())
 generated=latest_results(read_journal(generation/'results.jsonl'))
 if set(generated)!={(r['case_id'],r['variant']) for r in rows}:raise ValueError('Generation and score cases differ')
 def metric(group,name):
  vals=[r['metrics'][name] for r in group]; known=[v['value'] for v in vals if v['applicable'] and v['value'] is not None]
  return {'mean':statistics.mean(known) if known else None,'applicable_cases':len(known),'pending_items':sum(v['pending'] for v in vals),'provisional_cases':sum(v['provisional'] for v in vals)}
 def view(group):
  t=Counter(r['metrics']['task_usability'] for r in group); times=[generated[(r['case_id'],r['variant'])].get('timings_ms',{}).get('total') for r in group];times=[x for x in times if isinstance(x,(int,float))]
  def coverage(stage):
   values=[r['metrics']['evidence_coverage'][stage] for r in group]; known=[x['value'] for x in values if x['applicable'] and x['value'] is not None]
   return {'mean':statistics.mean(known) if known else None,'applicable_cases':len(known),'pending_items':sum(x['pending'] for x in values)}
  return {'answers':len(group),'generation_failures':sum(generated[(r['case_id'],r['variant'])]['status']!='success' for r in group),'pending_answers':sum(r['pending'] for r in group),
   'answer_point_coverage':metric(group,'answer_point_coverage'),'unsupported_claims':metric(group,'unsupported_claims'),'citation_correctness':metric(group,'citation_correctness'),
   'evidence_coverage':{stage:coverage(stage) for stage in ('retrieved','selected')},
   'fact_support':metric(group,'fact_support'),'serious_error_cases':sum(r['metrics']['serious_error_case'] is True for r in group),'serious_error_pending':sum(r['metrics']['serious_error_case'] is None for r in group),
   'task_usability':dict(t),'usable_fraction':(t['usable']+t['minor_revision'])/sum(t[k] for k in rank if k in t) if sum(t[k] for k in rank if k in t) else None,'mean_time_ms':statistics.mean(times) if times else None}
 rank={'unusable':0,'major_revision':1,'minor_revision':2,'usable':2}
 variants=sorted({r['variant'] for r in rows}); cats=sorted({r['category'] for r in rows}); groups={v:view([r for r in rows if r['variant']==v]) for v in variants}
 bycat={cat:{v:view([r for r in rows if r['category']==cat and r['variant']==v]) for v in variants} for cat in cats}
 bykey={(r['case_id'],r['variant']):r for r in rows};cases=sorted({r['case_id'] for r in rows});pairs={}
 for a,b in [('A','B'),('B','F'),('E','F')]:
  counts=Counter()
  for case in cases:
   x,y=bykey.get((case,a)),bykey.get((case,b))
   if not x or not y:counts['not_comparable']+=1;continue
   u,v=x['metrics']['task_usability'],y['metrics']['task_usability']
   if x['pending'] or y['pending'] or u not in rank or v not in rank:counts['pending']+=1;continue
   counts['improved' if rank[v]>rank[u] else 'declined' if rank[v]<rank[u] else 'unchanged']+=1
  pairs[a+'->'+b]=dict(counts)
 errors=[]
 for r in rows:
  t=r['judgment']['task'];g=generated[(r['case_id'],r['variant'])]
  if t['usability'] not in ('usable','minor_revision') or r['pending'] or g['status']!='success':
   errors.append({'case_id':r['case_id'],'variant':r['variant'],'answer_id':r['answer_id'],'task':t['usability'],'pending':r['pending'],'task_reason':t['reason'],
    'generation_status':g['status'],'serious_errors':t['serious_errors'],'unsupported_claims':[{'statement':c['statement'],'status':c['status'],'reason':c['reason']} for c in r['judgment']['claims'] if c['status'] in ('contradicted','unsupported')],
    'cause':'待逐题归因','fix':'待复核','retest':'未记录'})
 result={'score_id':m['run_id'],'generation_id':gm['run_id'],'dataset_sha256':m['dataset_sha256'],'score_code_hash':m['code_hash'],
  'ready_for_product_conclusions':m.get('ready_for_product_conclusions',False),'status':m['status'],'variants':groups,'categories':bycat,'paired_task_changes':pairs,'errors':errors,'billing':m.get('billing'),
  'scoring_mode':m.get('scoring_mode','model_judging'),'new_network_requests':m.get('new_network_requests'),
  'historical_judging_billing_reference':m.get('historical_judging_billing_reference'),
  'notes':['待复核不作正确或错误；比例均为逐题适用已判定值的均值，另列待复核数。','同题任一侧未决即待比较；失败保留。','E/F共享生成，不把重复的总耗时之差当作独立请求。','错误原因和修改措施需依据具体证据人工归因，未判定不自动编造。']}
 output.mkdir(parents=True,exist_ok=True);(output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 def f(x):return '不适用' if x is None else f'{x:.1%}'
 lines=['# 实验对比报告','',f"评分 {m['run_id']}；生成 {gm['run_id']}。",'',('**结果未完成校准，仅供复核，不作为产品结论。**' if not result['ready_for_product_conclusions'] else '校准状态见评分清单。'),'', '|版本|答案|要点覆盖（适用题）|无依据陈述（适用题）|严重错误题/待判|任务档位|未决答案|平均耗时ms|','|---|---:|---|---|---|---|---:|---:|']
 for v,g in groups.items():
  a,b=g['answer_point_coverage'],g['unsupported_claims'];lines.append(f"|{v}|{g['answers']}|{f(a['mean'])}（{a['applicable_cases']}，未决{a['pending_items']}）|{f(b['mean'])}（{b['applicable_cases']}，未决{b['pending_items']}）|{g['serious_error_cases']}/{g['serious_error_pending']}|{g['task_usability']}|{g['pending_answers']}|{g['mean_time_ms']:.0f}|" if g['mean_time_ms'] is not None else f'|{v}|{g["answers"]}|不适用|不适用|不适用|{g["task_usability"]}|{g["pending_answers"]}|未知|')
 billing_note='费用：详见 comparison.json 的用量和账单字段；缺费率时费用未知。'
 if result['historical_judging_billing_reference']:
  historical=result['historical_judging_billing_reference']['billing']
  billing_note=(f"本评分为保存回复的规则回放，本轮新增请求{result['new_network_requests']}；"
                f"原评审共{historical['unique_attempts']}次调用，费用{'未知' if historical.get('amount') is None else historical['amount']}。"
                '历史账单引用和用量见comparison.json，不能将回放零调用误读为原评审无成本。')
 lines+=['',billing_note,'','## 同题任务比较','']+[f'- {k}: {v}' for k,v in pairs.items()]+['','## 按类别及错误详情','','见 [comparison.json](comparison.json)，错误原因、措施和重测未有依据时保留待填。','','比例按逐题已判定值求均值；未决数另列。E/F共享生成阶段。']
 (output/'report.md').write_text('\n'.join(lines)+'\n');return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--score-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();result=render(args.score_dir,args.output_dir);print(json.dumps({'output_dir':str(args.output_dir),'answers':sum(x['answers'] for x in result['variants'].values()),'ready':result['ready_for_product_conclusions']},ensure_ascii=False))
if __name__=='__main__':main()
