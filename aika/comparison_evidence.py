"""Question-derived annual-report comparison requirements, never benchmark answer keys."""
from __future__ import annotations

import re
from typing import Any


FACETS = {'business': '主营业务 产品服务', 'rates': '主要产品 产品系列 高速产品 速率',
          'technology': '主要产品 技术平台 研发 布局 设计方案', 'products': '主要产品 产品覆盖'}


def value(item: Any, name: str) -> str:
    return str((item.get(name) if isinstance(item, dict) else getattr(item, name, '')) or '')


def business_table_rows(text: str) -> list[str]:
    """Find business rows only in a company business-classification table."""
    lines = text.splitlines()
    rows = []
    in_table = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith('|'):
            in_table = False
            continue
        cells = [cell.strip() for cell in stripped.strip('|').split('|')]
        if len(cells) < 2:
            in_table = False
            continue
        if re.search(r'业务分类|业务类别', cells[0]) and re.search(r'经营模式|业务情况|业务说明', cells[1]):
            context = ''.join(lines[max(0, index - 5):index])
            in_table = bool(re.search(r'公司(?:从事的|主要|主营|的)?业务', re.sub(r'\s+', '', context)))
            continue
        if in_table and not re.fullmatch(r'[-: ]+', cells[0]) and cells[0]:
            rows.append(cells[1])
    return rows


def requirements(plan):
    if not (plan.needs_comparison or plan.answer_type == 'company_compare'):
        return []
    years = sorted(set(re.findall(r'((?:19|20)\d{2})\s*年?\s*(?:年报|年度报告)', plan.question)))
    if not years or len(plan.companies) < 2:
        return []
    facets = ['business']
    if re.search('速率|速度|带宽', plan.question):
        facets.append('rates')
    elif re.search('产品|覆盖', plan.question):
        facets.append('products')
    if re.search('技术|布局|方案', plan.question):
        facets.append('technology')
    return [(company, year, facet) for company in plan.companies for year in years for facet in facets]


def supports(item, requirement):
    company, year, facet = requirement
    text = value(item, 'text') or value(item, 'evidence')
    source = value(item, 'source_title') or value(item, 'source')
    kind = value(item, 'kind')
    if kind not in ('', 'annual', 'rag') or value(item, 'company') != company:
        return False
    if (not re.search(r'年报|年度报告', source) or year not in source
            or re.search('研报|点评', source)
            or value(item, 'source_type') not in ('', 'annual_report', 'company_annual_report')):
        return False
    compact = re.sub(r'\s+', '', text)
    table_rows = business_table_rows(text)
    owned = re.search(r'公司(?:主营|主要|业务|已|拥有|为)|主营业务|业务主要涵盖|主要产品|产品系列|为.{0,20}客户提供|产品.{0,20}(?:采用|继承|设计方案)', compact)
    table_business = any(re.search(r'(?:提供|聚焦|从事|开展).{0,80}(?:产品|服务|业务|运营|解决方案)', re.sub(r'\s+', '', row)) for row in table_rows)
    if not owned and not table_business:
        return False
    if facet == 'business':
        return bool(re.search(r'主营业务|业务主要涵盖|公司主要|主要产品|为.{0,20}客户提供', compact)) or table_business
    if facet == 'rates':
        return bool(re.search(r'\d+(?:\.\d+)?[GT](?:bps)?', compact, re.I))
    if facet == 'technology':
        # Generic statements such as “公司注重技术研发” are not an actual technical layout.
        return bool(re.search(r'(?:推出|构建|采用|继承|设计方案|技术平台|研发项目|技术路线|产品布局)', compact)
                    and re.search('技术|平台|方案|研发|布局', compact))
    return bool(re.search('产品|服务', compact))


def matrix(plan, items):
    return [{'company': company, 'year': year, 'facet': facet,
             'matched': any(supports(item, (company, year, facet)) for item in items)}
            for company, year, facet in requirements(plan)]


def missing_cells(plan, items):
    return [cell for cell in matrix(plan, items) if not cell['matched']]


def coverage_cards(cards, plan, limit):
    """Keep annual evidence for each requested cell before filling remaining slots."""
    remaining = set(requirements(plan)); selected = []
    pool = list(cards)
    while remaining and pool and len(selected) < limit:
        best = max(pool, key=lambda card: sum(supports(card, req) for req in remaining))
        covered = {req for req in remaining if supports(best, req)}
        if not covered:
            break
        selected.append(best); pool.remove(best); remaining -= covered
    return (selected + [card for card in cards if card not in selected])[:limit]


def queries(plan, items):
    # Requirements, filters and queries derive exclusively from the public question.
    topics = ' '.join(plan.topics)
    return [(cell['company'], f"{cell['company']} {cell['year']}年年度报告 {topics} {FACETS[cell['facet']]}")
            for cell in missing_cells(plan, items)]
