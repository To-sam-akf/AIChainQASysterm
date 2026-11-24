from dataclasses import replace

from aika.agent_runner import AgentRetrievalState
from aika.agents.coverage import EvidenceCoverageChecker
from aika.agents.planner import TaskPlanner
from aika.professional_qa import EvidenceCard
from aika.question_planner import QuestionPlan
from aika.rd_evidence import requirements, matrix, queries, coverage_cards
from aika.rag_index import RagHit


TOTAL = '''| 项目 | 2025年 | 2024年 |
|---|---|---|
| 研发投入金额（万元） | 120.00 | 90.00 |
| 研发投入资本化的金额（万元） | 20.00 | 10.00 |
| 研发投入占营业收入比例 | 2.35% | 3.15% |'''
EXPENSE = '''单位：万元
| 项目 | 2025年 | 2024年 |
|---|---|---|
| 研发费用 | 100.00 | 80.00 |'''
POLICY = '本公司研究阶段支出计入当期损益，开发阶段支出满足条件才资本化。'
OVERVIEW = '2025年研发费用约0.01亿元。'


def plan():
    return QuestionPlan(question='截至2026-05-15的资料，浪潮信息2025年的研发费用、研发投入总额、资本化研发投入、投入占营收比例是多少，能否相加？',
                        answer_type='company_profile', companies=['浪潮信息'], needs_metrics=True)


def hit(text, name='raw', company='浪潮信息', source='2025年年度报告'):
    return RagHit(chunk_id=name, report_id='report', company=company, source_title=source,
                  source_type='company_annual_report', source_tier='1', page='42', section='',
                  content_type='table', table_id='', text=text, snippet=text[:30], score=1)


def test_precision_requirement_is_derived_from_question_not_cutoff_date():
    p = plan()
    assert len(requirements(p)) == 5 and {r[1] for r in requirements(p)} == {'2025'}
    assert requirements(replace(p, question='浪潮信息研发投入有多少？')) == []
    assert requirements(replace(p, question='浪潮信息2025年报研发费用与研发投入能否相加？'))


def test_overview_and_total_do_not_replace_current_year_exact_expense_row():
    p = plan(); items = [hit(TOTAL), hit(OVERVIEW), hit(POLICY)]
    missing = [c['facet'] for c in matrix(p, items) if not c['matched']]
    assert missing == ['expense']
    report = EvidenceCoverageChecker().check(p, TaskPlanner().plan(p.question, p),
                                           AgentRetrievalState(rag_hits=items), retrieval_round=0)
    assert 'metric_evidence' in report.satisfied and 'rd_evidence' in report.missing
    assert all(c['matched'] for c in matrix(p, [*items, hit(EXPENSE)]))


def test_wrong_year_company_parent_or_unitless_table_does_not_satisfy_precision():
    p = plan()
    for bad in [hit(EXPENSE, source='2024年年度报告'), hit(EXPENSE, company='其他公司'),
                hit('母公司利润表\n'+EXPENSE), hit(EXPENSE.replace('单位：万元\n',''))]:
        assert not next(c['matched'] for c in matrix(p,[bad]) if c['facet']=='expense')


def test_missing_query_and_selection_need_no_answer_values_or_page_locators():
    p = plan(); q = queries(p, [hit(TOTAL), hit(POLICY)])
    assert len(q)==1 and q[0][0]=='浪潮信息' and q[0][2]=='expense'
    assert '万元' in q[0][1] and '2025' in q[0][1]
    assert all(s not in q[0][1] for s in ['chunk_', '第42页', '100.00', '381,198'])
    cards=[EvidenceCard('', 'rag','概述', OVERVIEW, company='浪潮信息',source='2025年年度报告',score=999)]
    cards += [EvidenceCard('', 'rag','原表',t,company='浪潮信息',source='2025年年度报告') for t in (TOTAL,EXPENSE,POLICY)]
    selected=coverage_cards(cards,p,3)
    assert len(selected)==3 and all(c['matched'] for c in matrix(p, selected))
    assert all(c in cards for c in selected)


def test_shared_agent_targets_missing_expense_and_keeps_it_after_summary():
    from aika.qa_engine import QAEngine
    from tests.test_centralized_multi_agent import StructuredMultiAgentLLM

    class Index:
        def __init__(self): self.calls=[]
        def search(self, question, *, top_k, filters):
            self.calls.append((question, filters))
            if '费用表' in question and '万元' in question:
                return [hit(EXPENSE,'exact-fee')]
            return [hit(TOTAL,'total'),hit(POLICY,'policy'),hit(OVERVIEW,'overview')]

    class SummaryOnlyOverview(StructuredMultiAgentLLM):
        def chat_json(self, *, system_prompt, user_prompt, **kwargs):
            if '证据归纳 Agent' in system_prompt:
                import json
                payload=json.loads(user_prompt)
                overview=next(c for c in payload['candidates'] if c['evidence']==OVERVIEW)
                return {'cards':[{'candidate_ids':[overview['candidate_id']], 'title':'概述', 'reason':'summary'}]}
            return super().chat_json(system_prompt=system_prompt,user_prompt=user_prompt,**kwargs)

    index=Index();engine=QAEngine(rag_index=index,llm_client=SummaryOnlyOverview())
    result=engine.answer_question(plan().question)
    assert any('费用表' in q and f.get('company')=='浪潮信息' for q,f in index.calls)
    assert all(c['matched'] for c in matrix(plan(), result['evidence_cards']))
    assert result['diagnostics']['multi_agent']['summary']['rd_coverage_selection']
    assert any(c['args'].get('source')=='public_question_rd_requirements'
               for t in result['diagnostics']['agent_trace'] for c in t['tool_calls'])
