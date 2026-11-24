"""Rebuild the exploratory README figures from the preserved pilot records.

Requires matplotlib. Run from the repository root with .venv/bin/python.
"""
from __future__ import annotations
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/eval_runs/closeouts/continuation_20261124/expert_review_snapshot_20261124/coverage_mapped'
RUN=ROOT/'data/eval_runs/experiments/experiment_20261123_215354_ca07ed1a'
OUT=ROOT/'docs/imgs'
DOC=ROOT/'docs/eval-pilot'
FONT=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc').get_name()
plt.rcParams.update({'font.family':FONT,'axes.unicode_minus':False,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'font.size':11,'savefig.dpi':160})
COLORS={'A':'#667085','B':'#2563a5','E':'#9a6b20','F':'#168065'}

def read(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(values): return statistics.fmean(values) if values else None

def main():
 scores=read(BASE/'scores.jsonl'); results=read(RUN/'results.jsonl'); manifest=json.loads((RUN/'manifest.json').read_text()); audit=json.loads((BASE/'manifest.json').read_text())
 assert manifest['valid_for_comparison'] is True and audit['output_sha256']==sha(BASE/'scores.jsonl')
 assert len(scores)==len(results)==80 and all(x['status']=='success' for x in results)
 byrun={(r['case_id'],r['variant']):r for r in results}; assert len(byrun)==80
 vals={}; records=[]
 for variant in 'ABEF':
  group=[r for r in scores if r['variant']==variant]; assert len(group)==20
  point=[r['metrics']['answer_point_coverage']['value'] for r in group if r['metrics']['answer_point_coverage']['applicable'] and r['metrics']['answer_point_coverage']['value'] is not None]
  cites=[r['metrics']['citation_correctness']['value'] for r in group if r['metrics']['citation_correctness']['applicable'] and r['metrics']['citation_correctness']['value'] is not None]
  task=Counter(r['metrics']['task_usability'] for r in group); judged=20-task['pending']; usable=(task['usable']+task['minor_revision'])/judged if judged else None
  times=[byrun[(r['case_id'],variant)]['timings_ms']['total']/1000 for r in group]
  if variant=='F': assert all(byrun[(r['case_id'],'F')]['timings_ms']['total']>=byrun[(r['case_id'],'F')]['timings_ms']['shared_generation'] for r in group)
  missing=sum(not r['judgment']['citations'] for r in group)
  pend_cites=sum(r['metrics']['citation_correctness']['pending'] for r in group)
  serious=sum(r['metrics']['serious_error_case'] is True for r in group);serious_pending=sum(r['metrics']['serious_error_case'] is None for r in group)
  vals[variant]={'point':mean(point),'point_n':len(point),'citation':mean(cites),'citation_n':len(cites),'citation_pending':pend_cites,'citation_absent':missing,'usable':usable,'usable_n':judged,'usable_pending':task['pending'],'median':statistics.median(times),'min':min(times),'max':max(times),'serious':serious,'serious_pending':serious_pending,'failures':0}
  for key,n,pending,unit in [('point',len(point),sum(r['metrics']['answer_point_coverage']['pending'] for r in group),'ratio'),('citation',len(cites),pend_cites,'ratio'),('usable',judged,task['pending'],'ratio'),('median',20,0,'seconds'),('min',20,0,'seconds'),('max',20,0,'seconds')]:
   records.append({'实验ID':manifest['run_id'],'题集SHA256':manifest['dataset']['sha256'],'版本':variant,'指标':key,'值':vals[variant][key] if vals[variant][key] is not None else '', '适用题数':n,'失败数':0,'待核数':pending,'单位':unit,'计算脚本':'scripts/plot_pilot_evaluation.py'})
 scores_by={(r['case_id'],r['variant']):r for r in scores}; paired=Counter(); pairs=[]; rank={'unusable':0,'major_revision':1,'minor_revision':2,'usable':2}
 for cid in sorted({r['case_id'] for r in scores}):
  er=scores_by[(cid,'E')];fr=scores_by[(cid,'F')];e=er['metrics']['task_usability'];f=fr['metrics']['task_usability']
  comparison='待比较' if er['pending'] or fr['pending'] or e=='pending' or f=='pending' else '改善' if rank[f]>rank[e] else '退步' if rank[f]<rank[e] else '持平'
  paired[comparison]+=1;pairs.append({'题号':cid,'E可用性':e,'F可用性':f,'E答案含待核':er['pending'],'F答案含待核':fr['pending'],'比较':comparison})
 assert sum(paired.values())==20
 OUT.mkdir(parents=True,exist_ok=True); DOC.mkdir(parents=True,exist_ok=True)
 with (DOC/'pilot-summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 with (DOC/'pilot-pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
 fig,axes=plt.subplots(1,3,figsize=(12,4.8)); names=['A','B','F']; metrics=[('point','关键要点覆盖'),('citation','引用正确率'),('usable','任务可用率')]
 for ax,(key,title) in zip(axes,metrics):
  for i,v in enumerate(names):
   value=vals[v][key]
   if value is None: ax.text(i,4,'不适用',ha='center',fontsize=9,color='#475467'); continue
   ax.bar(i,value*100,color=COLORS[v],width=.58)
   n=vals[v][key+'_n'];ax.text(i,value*100+2,f'{value*100:.1f}%\n(n={n})',ha='center',va='bottom',fontsize=9)
  ax.set_title(title,fontweight='bold');ax.set_xticks(range(3),names);ax.set_ylim(0,120);ax.set_yticks([0,25,50,75,100]);ax.set_axisbelow(True);ax.yaxis.grid(color='#e8ecf1');ax.tick_params(axis='y',labelsize=8)
 fig.suptitle('回答质量｜20 道开发题 · 探索性结果',fontweight='bold',fontsize=16,y=.98)
 fig.text(.5,.035,'按已判定题目计算；轻微修改计入可用。引用缺失 A/B/F：20/1/0 题；任务待核：0/4/3 题。',ha='center',fontsize=9)
 fig.text(.5,.008,'F 有 1 个引用待核；严重错误候选仍有 B 版 1 份待核。其他答案未全部完成人工复核。',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.09,1,.92));fig.savefig(OUT/'eval-quality.png',bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4.5));labels=['改善','持平','退步','待比较'];counts=[paired[x] for x in labels];bars=ax.barh(labels,counts,color=['#168065','#728096','#c37b34','#d5dae3']);ax.invert_yaxis();ax.set_xlim(0,21);ax.set_xticks([0,5,10,15,20]);ax.xaxis.grid(color='#e8ecf1');ax.set_axisbelow(True)
 for b,n in zip(bars,counts):ax.text(n+.25,b.get_y()+b.get_height()/2,str(n),va='center',fontweight='bold')
 ax.set_title('答案检查与干预｜E → F 同题任务可用性',loc='left',fontweight='bold',fontsize=15);ax.set_xlabel('题数（20 题）');fig.text(.12,.015,'E/F 使用同一生成候选；任一答案有待核项即列入“待比较”，不归因于图谱或单个组件。',fontsize=9);fig.tight_layout(rect=(0,.07,1,1));fig.savefig(OUT/'eval-ablation.png',bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5));
 for v in 'ABEF':
  d=vals[v]; ax.errorbar(d['median'],d['point']*100,xerr=[[d['median']-d['min']],[d['max']-d['median']]],fmt='o',color=COLORS[v],capsize=3,markersize=9,label=v);ax.annotate(f" {v}  {d['median']:.1f}秒 / {d['point']*100:.1f}%",(d['median'],d['point']*100),xytext=(4,7 if v!='F' else -16),textcoords='offset points',fontsize=9,color=COLORS[v])
 ax.set_ylim(0,115);ax.set_xlim(left=0);ax.set_xlabel('端到端耗时中位数（秒）；横线为范围');ax.set_ylabel('关键要点覆盖（%）');ax.grid(color='#e8ecf1');ax.set_axisbelow(True);ax.set_title('质量与耗时｜20 道开发题 · 探索性',loc='left',fontweight='bold',fontsize=15);fig.text(.12,.015,'每方案 20 题，失败 0；F 含共享生成及检查。费用未知；单轮耗时不代表生产水平。',fontsize=9);fig.tight_layout(rect=(0,.07,1,1));fig.savefig(OUT/'eval-tradeoff.png',bbox_inches='tight');plt.close(fig)
 summary={'source_scores_sha256':sha(BASE/'scores.jsonl'),'run_id':manifest['run_id'],'dataset_sha256':manifest['dataset']['sha256'],'variants':vals,'ef_task_comparison':dict(paired),'expert_decisions_applied':audit['applied_total'],'expert_items_unresolved':60,'expert_items_not_mapped_with_opinion':34,'status':'exploratory_pending_review'}
 (DOC/'pilot-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
