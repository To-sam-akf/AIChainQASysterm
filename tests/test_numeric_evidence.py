import pytest

from aika.agents.verification import build_evidence_limited_answer, numeric_support_check, support_text, verify_answer_support
from aika.numeric_evidence import accounting_consistency_check, arithmetic, audit_financial_numbers, conversion_checks, quantities
from aika.professional_qa import EvidenceCard
from aika.question_planner import QuestionPlan


TABLE = '''单位：万元
| 项目 | 2025 年 | 2024 年 |
|---|---|---|
| 研发费用 | 100.00 | 80.00 |
| 研发投入金额（万元） | 120.00 | 90.00 |
| 研发投入资本化的金额（万元） | 20.00 | 10.00 |
| 研发投入占营业收入比例 | 2.35% | 3.15% |
'''


def card(text=TABLE, citation='E1', company='浪潮信息'):
    return EvidenceCard(citation_id=citation, kind='rag', title='年报原文', evidence=text,
                        source='2025年年度报告', page='15', company=company)


def check(answer, cards=None):
    cards = cards if cards is not None else [card()]
    return numeric_support_check(answer, support_text(cards), cards=cards, company='浪潮信息')


def test_count_translation_is_pending_not_a_reason_to_delete():
    cards = [EvidenceCard(citation_id='E1', kind='rag', title='UALink',
                          evidence='The UPLI allows up to 1024 Accelerators or endpoints.',
                          source='UALink Specification')]
    result = numeric_support_check('最多支持1024个加速器[E1]。', support_text(cards), cards=cards)
    assert '1024个' not in result['unsupported']
    assert any('1024' in value for value in result['pending'])
    assert result['status'] == 'warn'
    missing = numeric_support_check('最多支持1025个加速器[E1]。', support_text(cards), cards=cards)
    assert missing['status'] == 'fail'


def test_publication_date_boundary_is_not_an_unsupported_company_year():
    cards = [card('2025年研发费用100万元。')]
    answer = '所给资料未标注公开日期，无法确认其是否不晚于2026-05-15。'
    result = numeric_support_check(answer, support_text(cards), cards=cards)
    assert '2026' not in result['unsupported']
    assert '公开日期限定待核实' in result['pending']
    assert '入库截止日资格待核实' not in result['pending']


def test_table_unit_and_period_are_bound_to_each_value():
    data = quantities(TABLE)
    assert any(q.number == 100 and q.unit == '万元' and q.year == '2025' and q.label == 'rd_expense' for q in data)
    assert any(q.number == 80 and q.year == '2024' for q in data)
    result = check('浪潮信息2025年研发费用100.00万元，研发投入120万元，资本化研发投入20万元[E1]。')
    assert result['unsupported'] == [] and result['status'] == 'pass'


def test_sparse_pdf_columns_split_unit_and_notes_do_not_change_year_mapping():
    text = '''单位：万元
| 项目 | | 2025 年 | | 2024 年 | | 同比增减 |
|---|---|---|---|---|---|---|
| 财务费用 | 5.00 | | 8.00 | | -3.00% | 主要系本期利息支出增加所致 |
| 研发投入资本化的金额（万 | 20.00 | | 10.00 | | 100.00% | |
| 元） | | | | | | |
'''
    data = quantities(text, source_year='2025')
    assert any(q.number == 20 and q.unit == '万元' and q.year == '2025' for q in data)
    assert any(q.number == 10 and q.year == '2024' for q in data)
    assert not any(q.number == 100 and q.year == '2025' for q in data)


def test_sparse_header_starting_with_year_does_not_shift_columns():
    text = '''| | | 2025 年 | | 2024 年 | | 2023 年 |
|---|---|---|---|---|---|---|
| 研发投入占营业收入比例 | 2.83% | | 4.66% | | 4.32% | |
'''
    values = quantities(text, source_year='2025', citation='E1')
    assert [(str(q.number), q.year) for q in values] == [('2.83', '2025'), ('4.66', '2024'), ('4.32', '2023')]


def test_current_year_table_labels_capitalized_share_not_total_amount():
    text = '''| 项目 | 本年度 | 上年度 |
|---|---|---|
| 研发投入资本化的比重（%） | 9.26 | 17.04 |
'''
    values = quantities(text, source_year='2025', citation='E1')
    assert [(str(q.number), q.year, q.label) for q in values] == [
        ('9.26', '2025', 'rd_cap_ratio'), ('17.04', '2024', 'rd_cap_ratio')]


def test_previous_period_in_prose_uses_previous_year_without_reversing_columns():
    table = '''| 项目 | 本年度 | 上年度 |
|---|---|---|
| 研发投入占营业收入比例（%） | 17.99 | 91.30 |
'''
    evidence = [card(table)]
    good = check('浪潮信息2025年报。上年度占比为**91.30%**，本年度为17.99%[E1]。', evidence)
    assert good['unsupported'] == []
    bad = check('浪潮信息2025年报。上年度占比为**17.99%**，本年度为91.30%[E1]。', evidence)
    assert '17.99%' in bad['unsupported']


def test_report_period_revenue_ratio_retains_rd_numerator():
    evidence = [card('报告期内公司研发投入为116,910.10万元，占报告期内营业收入的比例为17.99%。')]
    result = check('浪潮信息2025年研发投入116,910.10万元、占营业收入17.99%[E1]。', evidence)
    assert '17.99%' not in result['unsupported']


def test_sparse_current_year_header_does_not_shift_periods():
    text = '''| | 本年度 | 上年度 | 变化幅度（%） |
|---|---|---|---|
| 研发投入总额占营业收入比例（%） | 31.78 | 37.61 | 减少5.83个百分点 |
| 研发投入资本化的比重（%） | 9.26 | 17.04 | 减少7.78个百分点 |
'''
    values = quantities(text, source_year='2025', citation='E1')
    assert any(str(q.number) == '31.78' and q.year == '2025' for q in values)
    assert any(str(q.number) == '9.26' and q.year == '2025' and q.label == 'rd_cap_ratio' for q in values)


def test_approximate_calculation_without_operands_is_pending():
    result = check('资本化比重 = 23万元 ÷ 123万元 ≈ 18.699%，与披露的比例一致。')
    assert '18.699%' not in result['unsupported']
    assert '18.699%' in result['pending']


def test_grouped_year_headers_do_not_assign_values_to_wrong_year():
    text = '''单位：元
| | | | 2025 年 | | | | | 2024 年 | | | | 同比增减 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | 金额 | | | 占营业收入 | | 金额 | | | 占营业收入 | | |
| 机房温控节能产品 | | 3,448,477,492.62 | | 56.83% | | | 2,440,951,426.94 | | 53.19% | | | 41.28% |
'''
    values = quantities(text, source_year='2025', citation='E1')
    current = next(q for q in values if str(q.number) == '3448477492.62')
    assert current.year == ''
    result = numeric_support_check('2025年机房温控节能产品收入3,448,477,492.62元[E1]。', text,
                                   cards=[card(text)])
    assert not any('3,448,477,492.62' in value for value in result['unsupported'])


def test_no_table_unit_or_year_is_pending_rather_than_assumed():
    result = check('2025年研发费用100万元[E1]。', [card('| 研发费用 | 100.00 | 80.00 |')])
    assert result['status'] == 'warn' and result['pending']


@pytest.mark.parametrize('answer', [
    '2025年研发费用100元[E1]。',  # Wrong unit.
    '2024年研发费用100万元[E1]。',  # Wrong column.
    '2025年营业收入100万元[E1]。',  # Number belongs to another metric.
    '英维克2025年研发费用100万元[E1]。',  # Wrong subject.
    '2025年研发费用100万元[E2]。',  # Other card cannot support this citation.
])
def test_wrong_unit_period_metric_company_or_citation_does_not_pass(answer):
    result = check(answer, [card(), card('2025年营业收入300万元。', citation='E2')])
    assert result['status'] == 'fail' and result['unsupported']


def test_table_answer_without_repeated_units_is_checked():
    result = check('''2025年，单位：万元
| 项目 | 金额 |
|---|---|
| 研发费用 | 999.00 |
''')
    assert '999.00万元' in result['unsupported']


def test_table_cell_with_parenthesized_unit_conversion_is_checked():
    result=check('''2025年
| 指标 | 数值 | 来源 |
|---|---|---|
| 研发费用 | 100万元（约0.01亿元） | [E1] |
''')
    assert result['unsupported']==[] and result['status']=='pass'
    assert {q['text'] for q in result['audit']['quantities']}=={'100万元','0.01亿元'}
    bad=check('''2025年
| 研发费用 | 100万元（0.1亿元） | [E1] |
''')
    assert '0.1亿元' in bad['unsupported']


def test_decimal_unit_conversion_and_explicit_rounding():
    evidence = [card('2025年营业收入为2,853,087,157.96元。')]
    assert check('2025年营业收入约285,308.72万元[E1]。', evidence)['status'] == 'pass'
    assert check('2025年营业收入为28.5308715796亿元[E1]。', evidence)['status'] == 'pass'
    assert check('2025年营业收入约28.54亿元[E1]。', evidence)['status'] != 'pass'
    assert check('2025年营业收入285,308.72万元[E1]。', evidence)['status'] == 'warn'


@pytest.mark.parametrize('expression', [
    '100万元 + 20万元 = 120万元',
    '120万元 - 100万元 = 20万元',
    '120万元 = 100万元 + 20万元',
    '120.00 ≈ 100.00 + 20.00，差额0.00万元',
    '100万元 + 0.002亿元 = 120万元',
])
def test_explicit_arithmetic_uses_source_bound_operands(expression):
    result = check('2025年，' + expression + '[E1]。')
    assert result['unsupported'] == []
    assert result['audit']['calculations'][0]['status'] == 'supported'


def test_bad_arithmetic_fails_even_if_result_appears_elsewhere_in_source():
    result = check('2025年，100万元 + 20万元 = 100万元[E1]。')
    assert result['status'] == 'fail'
    assert result['audit']['calculations'][0]['status'] == 'unsupported'


def test_ratio_arithmetic_and_currency_dimensions_are_verified():
    assert check('2025年，20万元 / 100万元 = 20%[E1]。')['status'] == 'pass'
    assert check('2025年，20万元 / 100万元 = 30%[E1]。')['status'] == 'fail'
    assert check('2025年，100万元 × 20万元 = 120万元[E1]。')['status'] == 'fail'
    assert check('2025年，100万元 + 20万元 / 120万元 = 1[E1]。')['status'] != 'pass'


def test_source_free_formula_or_cross_period_math_cannot_self_validate():
    assert check('2025年计算：777万元 + 222万元 = 999万元[E1]。')['status'] != 'pass'
    result = arithmetic('100万元 + 10万元 = 110万元', quantities(TABLE))
    assert result[0]['status'] == 'pending'


def test_percentage_points_are_distinct_from_currency_units():
    result = check('2025年研发投入占营业收入比例2.35%[E1]。')
    assert result['status'] == 'pass'
    assert check('2025年研发投入2.35万元[E1]。')['status'] == 'fail'


def test_pending_calculation_warns_without_shrinking_the_whole_answer():
    plan = QuestionPlan(question='浪潮信息研发投入是多少？', answer_type='company_profile', companies=['浪潮信息'])
    cards = [card()]
    result = verify_answer_support('2025年研发投入120万元[E1]。计算后的差额0.00万元，公式未列出。',
                                   plan, cards, cards, question=plan.question)
    assert result['checks']['numeric_support']['pending']
    assert result['status'] == 'warn'


@pytest.mark.parametrize('shorthand', ['占营收2.35%', '占营业收入2.35%', '占营收比例2.35%'])
def test_revenue_share_shorthand_inherits_the_explicit_numerator(shorthand):
    answer = '2025年研发投入金额120万元、' + shorthand + '[E1]。'
    assert check(answer)['status'] == 'pass'
    # An expense ratio must not borrow the total-R&D ratio just because it matches.
    assert check('2025年研发费用100万元、' + shorthand + '[E1]。')['status'] == 'fail'


def test_capitalized_amount_shorthand_after_revenue_ratio_keeps_its_own_metric():
    answer = '2025年研发投入120万元、占营收2.35%、资本化20万元[E1]。'
    result = check(answer)
    assert '20万元' not in result['unsupported']


def test_ambiguous_capitalized_share_shorthand_remains_pending():
    answer = '2025年研发投入120万元、资本化20万元、比例1.69%[E1]。'
    data = TABLE + '| 资本化研发投入占研发投入比例 | 1.69% | 1.88% |\n'
    result = check(answer, [card(data)])
    assert '1.69%' not in result['unsupported']
    assert '1.69%' in result['pending']


def test_later_comparison_year_does_not_change_preceding_current_value():
    values = quantities('2025年研发投入：\n研发投入占营业收入比例2.35%（2024年为3.15%）[E1]。')
    assert [(q.text, q.year) for q in values] == [('2.35%', '2025'), ('3.15%', '2024')]


def test_source_value_does_not_inherit_previous_unrelated_line_year():
    text = '2023年推出上一代产品。\n新款冷板支撑100%全液冷需求。'
    values = quantities(text, citation='E1', source_year='2025')
    assert next(q for q in values if q.text == '100%').year == ''
    cards = [card(text)]
    result = numeric_support_check('2025年冷板支持100%全液冷需求[E1]。', support_text(cards), cards=cards)
    assert '100%' not in result['unsupported']
    assert '100%' in result['pending']


def test_bare_revenue_share_without_a_numerator_remains_pending():
    assert check('2025年占营收2.35%[E1]。')['status'] == 'warn'
    assert check('2024年研发投入金额90万元、占营收2.35%[E1]。')['status'] == 'fail'


def test_percentage_point_calculation_with_ambiguous_sources_is_pending():
    cards = [card('| 光通信收发模块 | 42.61% | 34.65% |')]
    good = numeric_support_check('毛利率变化：42.61% − 34.65% = **上升7.96个百分点**[E1]。', support_text(cards), cards=cards)
    assert not any('7.96' in value for value in good['unsupported'])
    assert any('7.96' in value for value in good['pending'])
    bad = numeric_support_check('毛利率变化：42.61% − 34.65% = **上升7.97个百分点**[E1]。', support_text(cards), cards=cards)
    assert any('7.97' in value for value in bad['unsupported'])
    assert check('2025年研发投入占营业收入比例2.35个百分点[E1]。')['status'] == 'fail'


@pytest.mark.parametrize('sentence', [
    '相加会重复计算资本化金额。',
    '二者不应相加，因为资本化部分已包含在研发投入总额中，相加会重复计算。',
])
def test_correct_numbers_do_not_hide_a_wrong_accounting_explanation(sentence):
    text = '研发投入总额由费用化与资本化组成，研发费用是费用化子集。' + sentence
    result = accounting_consistency_check(text)
    assert result['status'] == 'fail'
    assert result['issues'][0]['statement'] == sentence


@pytest.mark.parametrize('sentence', [
    '相加会重复计算费用化部分。',
    '相加不是重复计算资本化金额。',
    '研发投入总额加资本化研发投入，相加会重复计算资本化金额。',
])
def test_accounting_check_respects_negation_and_the_actual_addition(sentence):
    assert accounting_consistency_check('研发费用与研发投入总额不同。' + sentence)['status'] == 'pass'


def test_local_accounting_rejection_keeps_all_numeric_rows_and_other_explanations():
    plan = QuestionPlan(question='浪潮信息研发费用和研发投入能否相加？', answer_type='company_overview', companies=['浪潮信息'])
    answer = ('2025年浪潮信息：研发费用与研发投入总额口径不同。\n'
              '| 项目 | 金额 |\n|---|---|\n| 研发费用 | 100万元 [E1] |\n'
              '| 研发投入总额 | 120万元 [E1] |\n| 资本化金额 | 20万元 [E1] |\n'
              '研发投入金额120万元、占营收2.35%[E1]。\n'
              '研发费用是费用化子集。相加会重复计算资本化金额。应按包含关系解释。')
    cards = [card()]
    initial = verify_answer_support(answer, plan, cards, cards, question=plan.question)
    assert initial['status'] == 'fail'
    assert initial['checks']['numeric_support']['unsupported'] == []
    fixed = build_evidence_limited_answer(plan, cards, initial, answer=answer)
    assert '相加会重复计算资本化金额' not in fixed
    assert '| 资本化金额 | 20万元 [E1] |' in fixed
    assert '占营收2.35%' in fixed and '应按包含关系解释' in fixed
    assert verify_answer_support(fixed, plan, cards, cards, question=plan.question)['status'] != 'fail'


def test_local_bad_amount_does_not_remove_supported_rows_or_pending_calculations():
    plan = QuestionPlan(question='浪潮信息研发投入？', answer_type='company_overview', companies=['浪潮信息'])
    cards = [card()]
    answer = '2025年研发费用100万元[E1]。\n2025年研发投入999万元[E1]。\n计算差额0.00万元，公式待核对。'
    initial = verify_answer_support(answer, plan, cards, cards, question=plan.question)
    fixed = build_evidence_limited_answer(plan, cards, initial, answer=answer)
    assert '999万元' not in fixed and '研发费用100万元' in fixed
    assert '差额0.00万元' in fixed  # A pending calculation is never a deletion trigger.


def test_evidence_only_fallback_preserves_the_complete_financial_table():
    plan = QuestionPlan(question='浪潮信息研发投入？', answer_type='company_overview', companies=['浪潮信息'])
    cards = [card()]
    initial = verify_answer_support('2025年研发投入999万元[E1]。', plan, cards, cards, question=plan.question)
    fixed = build_evidence_limited_answer(plan, cards, initial)
    assert TABLE.strip() in fixed and '不能替代完整作答' in fixed
    assert '999万元' not in fixed
    assert verify_answer_support(fixed, plan, cards, cards, question=plan.question)['status'] != 'fail'


@pytest.mark.parametrize('objects', ['"研发费用"与"资本化研发投入"', '资本化研发投入＋研发费用'])
def test_addition_checks_the_actual_operands(objects):
    answer = '研发投入总额=研发费用+资本化研发投入。把' + objects + '相加会重复计算。'
    result = accounting_consistency_check(answer)
    assert result['status'] == 'fail' and result['issues'][0]['kind'] == 'rd_addition_operands'


@pytest.mark.parametrize('claim', [
    '研发费用与资本化研发投入相加不会重复计算。',
    '不能认为研发费用与资本化研发投入相加会重复计算。',
    '研发费用加研发投入总额会重复计算费用化部分。',
    '研发费用与资本化研发投入相加会重复计算，这个说法不成立。',
])
def test_addition_distinguishes_valid_composition_and_quoted_errors(claim):
    assert accounting_consistency_check('研发投入总额含费用化与资本化。' + claim)['status'] == 'pass'


@pytest.mark.parametrize('claim', [
    '两者不能相加：把研发费用与研发投入总额相加，会把费用化部分重复计算；若要得到总投入，应使用研发投入总额，或费用化研发支出加资本化研发投入，而不是总额再加其中子项。',
    '研发费用与资本化研发投入可以相加，但研发费用与研发投入总额不能相加，会重复计算费用化部分。',
    '把研发费用与研发投入总额相加会重复计算费用化部分，费用化研发支出加资本化研发投入才是总额。',
])
def test_addition_does_not_borrow_a_verdict_from_other_operands(claim):
    assert accounting_consistency_check('研发费用与研发投入总额不同。' + claim)['status'] == 'pass'


@pytest.mark.parametrize('claim', [
    '研发费用与资本化研发投入不能相加。',
    '不能把研发费用与资本化研发投入相加。',
    '研发费用与资本化研发投入相加，会重复计算。',
])
def test_addition_keeps_explicit_wrong_operand_rejections(claim):
    assert accounting_consistency_check('研发费用与研发投入总额不同。' + claim)['status'] == 'fail'


def test_two_individually_sourced_numbers_are_not_automatically_equivalent():
    cards = [card(), card('2025年母公司研发费用800,000元。', citation='E2')]
    result = check('2025年研发费用100.00万元（即800,000元）[E1][E2]。', cards)
    assert all(q['status'] == 'supported' for q in result['audit']['quantities'])
    assert result['audit']['conversions'][0]['status'] == 'unsupported'
    assert result['status'] == 'fail'
    assert conversion_checks('381,198.60万元≈2,853,087,157.96元，尾差为四舍五入。')[0]['status'] == 'unsupported'


def test_unit_equivalence_rounding_and_reconciliation_are_separate():
    assert conversion_checks('100万元即1,000,000元。')[0]['status'] == 'supported'
    assert conversion_checks('381,198.60万元（约38.12亿元）。')[0]['status'] == 'supported'
    assert conversion_checks('381,198.60万元即38.12亿元。')[0]['status'] == 'pending'
    assert conversion_checks('100万元+20万元=120万元。') == []
    assert conversion_checks('100万元与800,000元不同。') == []


def test_capitalisation_share_and_unverified_cutoff_are_not_false_hard_failures():
    answer = ('2025年资本化占研发投入比例20.00%[E1]。\n'
              '资料未提供2025年资本化研发投入占营业收入比例，仅提供其占研发投入比例20.00%[E1]。\n'
              '上述资料仅指截至2026-05-15已入库资料。')
    table = TABLE.replace('2.35% | 3.15%', '2.35% | 3.15%') + '| 资本化研发投入占研发投入 | 20.00% | 11.11% |\n'
    result = check(answer, [card(table)])
    assert result['unsupported'] == []
    assert '入库截止日资格待核实' in result['pending']
    assert check('2025年资本化研发投入占营业收入比例2.35%[E1]。')['status'] == 'fail'


def test_wrong_addition_is_removed_but_correct_capitalisation_evidence_survives():
    plan = QuestionPlan(question='浪潮信息研发费用与研发投入总额能否相加？', answer_type='company_overview', companies=['浪潮信息'])
    answer = ('2025年研发费用100万元、研发投入总额120万元、资本化研发投入20万元[E1]。\n'
              '研发投入总额=研发费用+资本化研发投入。\n'
              '把"研发费用"与"资本化研发投入"相加会重复计算[E1]。')
    initial = verify_answer_support(answer, plan, [card()], [card()], question=plan.question)
    assert initial['checks']['accounting_consistency']['status'] == 'fail'
    fixed = build_evidence_limited_answer(plan, [card()], initial, answer=answer)
    assert '相加会重复计算' not in fixed
    assert '资本化研发投入20万元' in fixed and '研发投入总额=研发费用+资本化研发投入' in fixed
