import pytest
from dataclasses import replace

from aika.agents.verification import build_evidence_limited_answer, verify_answer_support
from aika.professional_qa import EvidenceCard
from aika.question_planner import QuestionPlan
from aika.semantic_evidence import rate_ceiling_issues


def cards(second='乙公司已推出硅光与LPO、CPO产品。', first=None):
    return [EvidenceCard(citation_id='E1', kind='rag', title='年报', source='2025年年度报告', company='甲公司',
                         evidence=first or '甲公司已推出硅光与LPO、CPO产品，包括400G、800G、1.6T、6.4T NPO和12.8T XPO模块。'),
            EvidenceCard(citation_id='E2', kind='rag', title='年报', source='2025年年度报告', company='乙公司', evidence=second)]


def test_shared_bands_do_not_establish_a_shared_maximum():
    text = '两家均披露400G、800G、1.6T产品，最高披露速率均到1.6T[E1][E2]。'
    issues = rate_ceiling_issues(text, cards(), question='比较2025年年报')
    assert issues and issues[0]['support'][0]['company'] == '甲公司'
    assert rate_ceiling_issues('共同覆盖400G、800G、1.6T。', cards()) == []


@pytest.mark.parametrize('text', [
    '共同覆盖最高速率为1.6T[E1][E2]。',
    '单通道最高披露速率为1.6T[E1]。',
    '乙公司最高披露速率为1.6T[E2]。',
    '截至2025年末在售产品的最高披露速率为1.6T[E1]。',
])
def test_qualified_or_other_company_rates_do_not_borrow_aggregate_products(text):
    assert rate_ceiling_issues(text, cards(), question='2025年产品') == []


def test_old_forecasts_and_industry_principles_are_not_company_product_counterexamples():
    evidence = cards(first='甲公司预计明年推出12.8T XPO模块。')
    assert rate_ceiling_issues('甲公司最高披露速率为1.6T。', evidence, question='2025年') == []
    evidence[0] = replace(evidence[0], evidence='行业原理：12.8T是聚合带宽，并非公司实际产品。')
    assert rate_ceiling_issues('甲公司最高披露速率为1.6T。', evidence, question='2025年') == []
    evidence[0] = replace(evidence[0], evidence='甲公司已推出12.8T产品。', source='2024年年度报告')
    assert rate_ceiling_issues('甲公司最高披露速率为1.6T。', evidence, question='2025年') == []


def test_f_removes_bad_ceiling_and_shared_cpo_without_destroying_other_comparison():
    evidence = cards('乙公司已推出硅光与LPO产品。')
    plan = QuestionPlan(question='比较两家公司2025年产品', answer_type='company_comparison', companies=['甲公司', '乙公司'])
    text = ('两家均披露硅光及LPO产品[E1][E2]。\n'
            '最高披露速率均到1.6T[E1][E2]。\n'
            '两家均有CPO产品布局[E1][E2]。')
    initial = verify_answer_support(text, plan, evidence, evidence, question=plan.question)
    assert initial['checks']['semantic_support']['status'] == 'fail'
    fixed = build_evidence_limited_answer(plan, evidence, initial, answer=text)
    assert '最高披露速率均到1.6T' not in fixed and '两家均有CPO' not in fixed
    assert '两家均披露硅光及LPO产品' in fixed
    assert verify_answer_support(fixed, plan, evidence, evidence, question=plan.question)['status'] != 'fail'


@pytest.mark.parametrize('text', [
    '- 业务性质：均以光模块研发、生产、销售为主营，服务云数据中心、AI算力集群、电信网络等客户[E1][E3]。',
    '- 均以光模块研发、生产、销售为主营，均服务云数据中心、电信网络及AI算力相关场景[E1][E3]。',
])
def test_implicit_shared_customer_list_is_narrowed_to_each_cited_company(text):
    evidence=[EvidenceCard('E1','rag','业务','公司产品服务于人工智能算力集群、云数据中心、电信网络。',company='新易盛',source='2025年年度报告'),
              EvidenceCard('E3','rag','业务','公司产品服务于云计算数据中心、数据通信、电信传输和固网接入。',company='中际旭创',source='2025年年度报告')]
    plan=QuestionPlan(question='比较两家公司2025年业务与场景',answer_type='company_compare',companies=['新易盛','中际旭创'])
    checked=verify_answer_support(text,plan,evidence,evidence,question=plan.question)
    assert checked['checks']['semantic_support']['status']=='fail'
    revised=build_evidence_limited_answer(plan,evidence,checked,answer=text)
    assert '服务云数据中心及电信网络[E1][E3]' in revised
    assert '新易盛另明确提及AI算力集群[E1]' in revised
    assert '均以光模块研发、生产、销售为主营' in revised
    assert verify_answer_support(revised,plan,evidence,evidence,question=plan.question)['status']!='fail'


def test_bare_plural_with_one_company_is_not_automatically_two_companies():
    from aika.semantic_evidence import semantic_support_check
    evidence=[EvidenceCard('E1','rag','业务','公司产品服务于云数据中心。',company='新易盛',source='2025年年度报告')]
    assert semantic_support_check('均服务云数据中心[E1]。',evidence)['issues']==[]
