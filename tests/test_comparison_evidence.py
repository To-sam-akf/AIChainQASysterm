from dataclasses import replace

from aika.agents.coverage import EvidenceCoverageChecker
from aika.agents.planner import TaskPlanner
from aika.agent_runner import AgentRetrievalState
from aika.comparison_evidence import matrix, requirements, queries, coverage_cards
from aika.professional_qa import EvidenceCard
from aika.question_planner import QuestionPlan
from aika.rag_index import RagHit


def plan():
    return QuestionPlan(question='根据中际旭创和新易盛2025年报比较业务、产品速率和技术布局。',
                        answer_type='company_compare', needs_comparison=True,
                        companies=['中际旭创', '新易盛'], topics=['光模块'])


def hit(company, text, source='2025年年度报告', source_type='annual_report'):
    return RagHit(chunk_id=company+text[:6], report_id='report', company=company,
                  source_title=source, source_type=source_type, source_tier='1', page='11',
                  section='主营业务', content_type='text', table_id='', text=text, snippet=text[:10], score=10)


BUSINESS='公司主营业务为光模块，为云数据中心客户提供400G、800G和1.6T产品。'
TECH='主要产品 产品系列：1.6T OSFP产品采用EML和硅光平台设计方案。'


def test_company_names_income_and_industry_introduction_do_not_complete_comparison():
    p=plan(); hits=[hit(c,'营业收入100亿元，同比增长60%；行业普遍认为LPO适用于短距离场景，800G技术继续发展。') for c in p.companies]
    report=EvidenceCoverageChecker().check(p,TaskPlanner().plan(p.question,p),AgentRetrievalState(rag_hits=hits),retrieval_round=0)
    assert 'company_coverage' in report.satisfied and 'comparison_evidence' in report.missing
    assert not any(c['matched'] for c in matrix(p,hits))


def test_annual_company_facets_are_independent_and_require_current_primary_source():
    p=plan(); hits=[hit(p.companies[0],BUSINESS),hit(p.companies[0],TECH),hit(p.companies[1],BUSINESS)]
    assert [c['facet'] for c in matrix(p,hits) if not c['matched']]==['technology']
    for bad in [hit(p.companies[1],TECH,'2024年年度报告'),hit(p.companies[1],TECH,'2025年报点评','broker_report')]:
        assert any(not c['matched'] for c in matrix(p,[*hits,bad]))
    hits.append(hit(p.companies[1],TECH))
    assert all(c['matched'] for c in matrix(p,hits))


def test_generic_research_marketing_does_not_count_as_technical_layout():
    cells=matrix(plan(),[hit('中际旭创',BUSINESS+'公司注重技术研发，持续推动产品发展。')])
    assert not next(c['matched'] for c in cells if c['company']=='中际旭创' and c['facet']=='technology')


def test_company_business_table_supports_service_without_counting_industry_table():
    from aika.comparison_evidence import supports

    row = '| 宝之云 | 聚焦数据中心托管服务、算力服务、算力运营，为企业提供基础设施解决方案。 |'
    header = '| 业务分类 | 经营模式及行业情况说明 |\n| --- | --- |\n'
    requirement = ('宝信软件', '2025', 'business')
    company_table = '### 一、报告期内公司从事的业务情况\n\n' + header + row
    assert supports(hit('宝信软件', company_table), requirement)
    assert supports(hit('宝信软件', company_table), ('宝信软件', '2025', 'products'))
    assert not supports(hit('宝信软件', '### 行业发展情况\n' + header + row), requirement)
    assert not supports(hit('宝信软件', company_table, '2024年度报告'), requirement)
    assert not supports(hit('数据港', company_table), requirement)


def test_missing_queries_are_derived_from_question_and_scoped_by_company():
    p=plan(); result=queries(p,[hit(p.companies[0],BUSINESS),hit(p.companies[0],TECH)])
    assert len(result)==3 and all(c=='新易盛' and '2025' in q for c,q in result)
    assert all('chunk_' not in q and '第11页' not in q for _,q in result)
    assert requirements(replace(p,question='比较两家公司业务'))==[]


def test_comparison_fallback_does_not_add_unrequested_optical_terms():
    p = QuestionPlan(question='根据宝信软件和数据港2025年年报说明数据中心业务和服务。',
                     answer_type='company_compare', needs_comparison=True,
                     companies=['宝信软件', '数据港'], topics=['数据中心'])
    task = TaskPlanner().plan(p.question, p)
    suffix = next(s.query_suffix for s in task.subtasks if s.type == 'comparison_evidence')
    assert '产品速率' not in suffix and '技术布局' not in suffix
    assert '服务' in suffix
    assert {'business'} == {facet for _, _, facet in requirements(p)}
    assert any('速率' in query for _, query in queries(plan(), []))


def test_card_selection_keeps_annual_facets_before_high_scoring_research():
    p=plan(); cards=[EvidenceCard(citation_id='',kind='rag',title='收入',company=c,source='2025年年度报告',evidence='收入100亿元',score=999) for c in p.companies]
    cards += [EvidenceCard(citation_id='',kind='rag',title='原文',company=c,source='2025年年度报告',evidence=t,score=1) for c in p.companies for t in (BUSINESS,TECH)]
    selected=coverage_cards(cards,p,4)
    assert len(selected)==4 and all(c['matched'] for c in matrix(p,selected))
    assert all(c in cards for c in selected)


def test_agent_scopes_retrieval_and_keeps_both_companies_in_final_evidence():
    from aika.qa_engine import QAEngine
    from tests.test_centralized_multi_agent import StructuredMultiAgentLLM

    class Index:
        def __init__(self): self.calls=[]
        def search(self, query, *, top_k, filters):
            self.calls.append((query,filters))
            company=filters.get('company')
            return [hit(company,BUSINESS),hit(company,TECH)] if company else []

    index=Index()
    result=QAEngine(rag_index=index,llm_client=StructuredMultiAgentLLM()).answer_question(plan().question)
    assert {'中际旭创','新易盛'} <= {f.get('company') for _,f in index.calls}
    assert all(c['matched'] for c in matrix(plan(),result['evidence_cards']))
    assert result['diagnostics']['agent_coverage']['missing']==[]
    assert result['diagnostics']['multi_agent']['summary']['comparison_coverage_selection']
    assert any(c['args'].get('source')=='public_question_requirements'
               for t in result['diagnostics']['agent_trace'] for c in t['tool_calls'])
