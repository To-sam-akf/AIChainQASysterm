"""Public-question-derived R&D evidence coverage, never benchmark locators/values."""
from __future__ import annotations

import re

from aika.numeric_evidence import quantities


FACETS = {'expense': '研发费用 费用表 单位 万元', 'total': '公司研发投入情况 研发投入金额',
          'capital': '研发投入资本化的金额', 'ratio': '研发投入占营业收入比例',
          'policy': '研发支出 会计政策 费用化 资本化'}


def value(item, key):
    return str((item.get(key) if isinstance(item, dict) else getattr(item, key, '')) or '')


def requirements(plan):
    text = plan.question
    if not ('研发费用' in text and re.search('研发投入|研发支出', text)):
        return []
    years = sorted(set(re.findall(r'((?:19|20)\d{2})年(?:度)?(?:的)?研发', text)
                       + re.findall(r'((?:19|20)\d{2})\s*年?\s*(?:年报|年度报告)', text)))
    if not years or not plan.companies:
        return []
    facets = ['expense', 'total']
    if '资本化' in text:
        facets.append('capital')
    if re.search('占.*(?:营收|营业收入)|投入占', text):
        facets.append('ratio')
    if re.search('为何|为什么|相加|口径|解释|不同', text):
        facets.append('policy')
    return [(company, year, facet) for company in plan.companies for year in years for facet in facets]


def supports(item, requirement):
    company, year, facet = requirement
    text = value(item, 'text') or value(item, 'evidence')
    source = value(item, 'source_title') or value(item, 'source')
    if (value(item, 'kind') not in ('', 'rag', 'annual') or value(item, 'company') != company
            or year not in source or not re.search('年报|年度报告', source) or re.search('研报|点评', source)
            or value(item, 'source_type') not in ('', 'annual_report', 'company_annual_report')):
        return False
    if re.search('母公司(?:利润表|损益表)', value(item, 'section') + text):
        return False
    compact = re.sub(r'\s+', '', text)
    if facet == 'policy':
        return all(term in compact for term in ('研究阶段', '开发阶段', '计入当期损益', '资本化'))
    label = {'expense': 'rd_expense', 'total': 'rd_total', 'capital': 'rd_cap', 'ratio': 'rd_ratio'}[facet]
    for q in quantities(text, company=company, source_year=year, citation='source'):
        if q.label != label or q.year != year:
            continue
        if facet == 'expense':
            # The narrative in billions is rounded; it cannot replace the
            # precise current-year expense row for a reconciliation question.
            if '|' in text and q.unit in ('元', '万元') and not q.approximate:
                return True
        elif q.unit in (('%', '％') if facet == 'ratio' else ('元', '万元', '亿元')):
            return True
    return False


def matrix(plan, items):
    return [{'company': c, 'year': y, 'facet': f, 'matched': any(supports(item, (c, y, f)) for item in items)}
            for c, y, f in requirements(plan)]


def missing_cells(plan, items):
    return [c for c in matrix(plan, items) if not c['matched']]


def queries(plan, items):
    return [(c['company'], f"{c['company']} {c['year']}年年度报告 {FACETS[c['facet']]}", c['facet'])
            for c in missing_cells(plan, items)]


def coverage_cards(cards, plan, limit):
    remaining = set(requirements(plan)); selected = []; pool = list(cards)
    while remaining and pool and len(selected) < limit:
        best = max(pool, key=lambda c: sum(supports(c, req) for req in remaining))
        covered = {req for req in remaining if supports(best, req)}
        if not covered:
            break
        selected.append(best); pool.remove(best); remaining -= covered
    return (selected + [c for c in cards if c not in selected])[:limit]
