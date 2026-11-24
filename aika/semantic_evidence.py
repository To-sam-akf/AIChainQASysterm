"""Limited source-bound conjunction and rate-ceiling checks.

No benchmark facts, expected answer, product score or model calls are consulted.
The independent judge supplies frozen originals; F supplies its actual evidence.
"""
from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal

from aika.numeric_evidence import YEAR, VALUE, field
from aika.question_planner import extract_companies


def product_context(text, start, end):
    left = max(0, start-150, max(text.rfind(mark, 0, start) for mark in '。；') + 1)
    right = min([end+150, *[pos for mark in '。；' if (pos := text.find(mark, end)) >= 0]])
    window = re.sub(r'\s+', '', text[left:right])
    if re.search(r'(?:并非|不是|不等于|不代表).{0,8}(?:公司|本公司).{0,8}(?:产品|布局)', window):
        return False
    action = re.search(r'(?:公司|本公司).{0,100}(?:构建|采用|推出|发布|研发|提供|拥有|生产|销售|基于)'
                       r'|(?:展示|推出|发布).{0,70}(?:模块|产品)', window)
    table = '|' in text and re.search(r'产品系列|产品特性', text)
    return bool(action or table)


def collective_scope_gaps(statement, citation_ids, citation_sources, originals) -> list[str]:
    """A conjunction cannot borrow one company's technology from another.

    This checks the *cited* originals only. Missing support is not a finding
    that the company lacks a technology. Unrecognised wording stays with the
    semantic judge; this guard does not claim to recognise every paraphrase.
    """
    explicit = bool(re.search(r'(?:两家(?:公司)?|双方|两者).{0,12}均', statement))
    implicit = bool(re.search(r'(?:^|[:：])\s*(?:[-*+]\s*)?均(?:以|为|面向|服务|覆盖)', statement))
    if not (explicit or implicit):
        return []
    aspects = {}
    if (re.search(r'布局|平台|产品|技术体系', statement)
            and not re.search(r'(?:不等同|不代表|不足以|不能).{0,20}(?:产品|布局)', statement)):
        for name, pattern in {
            'CPO': r'(?<![A-Za-z])CPO(?![A-Za-z])|光电共封装|共封装光学',
            'LPO': r'(?<![A-Za-z])LPO(?![A-Za-z])', 'LRO': r'(?<![A-Za-z])LRO(?![A-Za-z])',
            'NPO': r'(?<![A-Za-z])NPO(?![A-Za-z])', 'XPO': r'(?<![A-Za-z])XPO(?![A-Za-z])',
            'SiPh': r'(?<![A-Za-z])SiPh(?![A-Za-z])|硅光',
        }.items():
            if re.search(pattern, statement, re.I):
                aspects[name] = (pattern, True)
    if re.search(r'面向|服务|覆盖', statement):
        if '数据中心' in statement:
            aspects['数据中心场景'] = (r'(?:云计算)?数据中心', False)
        if (re.search(r'AI算力|人工智能算力', statement)
                and not re.search(r'(?:另|仅|单独).{0,12}(?:提及|披露|面向|服务).{0,10}AI', statement)):
            aspects['AI算力场景'] = (r'AI\s*(?:算力|Cluster)|人工智能算力', False)
    if not aspects:
        return []
    by_company = defaultdict(list)
    for citation_id in citation_ids:
        for source_id in citation_sources.get(citation_id, set()):
            source = originals.get(source_id, {})
            if source.get('company'):
                by_company[source['company']].append(source['text'])
    if implicit and not explicit and len(by_company) < 2:
        return []  # A bare "均" need not quantify multiple companies.
    if len(by_company) < 2:
        return ['共同点未定位到至少两家公司的引用原文']
    gaps = []
    for company, texts in sorted(by_company.items()):
        for name, (pattern, needs_product) in aspects.items():
            supported = False
            for text in texts:
                for mention in re.finditer(pattern, text, re.I):
                    if not needs_product or product_context(text, mention.start(), mention.end()):
                        supported = True
                        break
                if supported:
                    break
            if not supported:
                gaps.append(company + '：' + name)
    return gaps


def narrow_shared_ai(statement, cards):
    """Narrow a proven common customer list while retaining supported facts.

    Only applies to the unambiguous cloud/telecom intersection and an AI cluster
    explicitly served by some cited companies. It never infers absence elsewhere.
    """
    refs = re.findall(r'\[(E\d+)\]', statement)
    selected = [c for c in cards if field(c, 'citation_id') in refs]
    originals = {str(i): {'company': field(c, 'company'), 'text': field(c, 'evidence') or field(c, 'text')}
                 for i, c in enumerate(selected)}
    mapping = defaultdict(set)
    for i, c in enumerate(selected):
        mapping[field(c, 'citation_id')].add(str(i))
    gaps = collective_scope_gaps(statement, refs, mapping, originals)
    if not gaps or any(not g.endswith('：AI算力场景') for g in gaps):
        return None
    by_company = defaultdict(list)
    for c in selected:
        if field(c, 'company'):
            by_company[field(c, 'company')].append(c)
    ai = []
    for company, items in sorted(by_company.items()):
        texts = [field(c, 'evidence') or field(c, 'text') for c in items]
        if not (any(re.search(r'云(?:计算)?数据中心', t) for t in texts)
                and any(re.search('电信网络|电信传输', t) for t in texts)):
            return None
        supported = []
        for c, text in zip(items, texts):
            if re.search(r'(?:服务于|服务|面向|应用于)[^。；\n]{0,35}(?:人工智能算力集群|AI\s*(?:算力集群|Cluster))', text, re.I):
                supported.append(field(c, 'citation_id'))
        if supported:
            ai.append(company + '另明确提及AI算力集群' + ''.join(f'[{r}]' for r in dict.fromkeys(supported)))
    if not ai or len(ai) >= len(by_company):
        return None
    start = re.search(r'(?:服务|面向|覆盖)[^。；]*?(?:人工智能|AI\s*)算力(?:集群|相关场景)', statement)
    if not start:
        return None
    common_refs = ''.join(f'[{r}]' for r in dict.fromkeys(refs))
    return (statement[:start.start()] + '服务云数据中心及电信网络' + common_refs + '。'
            + '；'.join(ai) + '。')


RATE = re.compile(rf'(?<![A-Za-z\d.])(?P<value>{VALUE})\s*(?P<unit>T|G)(?![A-Za-z\d])', re.I)
CEILING = re.compile(rf'(?:最高(?:披露)?(?:产品)?速率|最高披露速率|速率上限)'
                     rf'\s*(?:均|都)?\s*(?:为|到|达|是|不超过|止于|至)\s*'
                     rf'(?P<value>{VALUE})\s*(?P<unit>T|G)(?![A-Za-z\d])', re.I)


def rate_ceiling_issues(answer, cards, *, question=''):
    """Find counterexamples to an asserted ceiling, never infer a true maximum.

    Common covered bands, per-lane rates and aggregate modules have different
    meanings. Qualified per-lane/architecture claims remain for semantic review.
    """
    issues = []
    years = set(YEAR.findall(question))
    answer_years = set(YEAR.findall(answer))
    if not years and len(answer_years) == 1:
        years = answer_years
    companies = {field(card, 'company') for card in cards if field(card, 'company')}
    for line_index, line in enumerate(answer.splitlines()):
        if re.search(r'单通道|每通道|单波|单端口|可插拔|在售|量产|年末|期末|per.lane|共同覆盖.{0,8}最高', line, re.I):
            continue
        for ceiling in CEILING.finditer(line):
            limit = Decimal(ceiling['value'].replace(',', '')) * (1000 if ceiling['unit'].upper() == 'T' else 1)
            named = (set(extract_companies(line)) & companies) | {c for c in companies if c in line}
            applicable = named or companies
            counters = []
            for card in cards:
                company = field(card, 'company')
                if not company or company not in applicable:
                    continue
                title = field(card, 'source') or field(card, 'title')
                source_years = set(YEAR.findall(title))
                if years and source_years != years:
                    continue
                if re.search(r'研报|研究报告', title) and not re.search('年报|年度报告', title):
                    continue
                text = field(card, 'evidence') or field(card, 'text')
                for rate in RATE.finditer(text):
                    value = Decimal(rate['value'].replace(',', '')) * (1000 if rate['unit'].upper() == 'T' else 1)
                    prefix = re.sub(r'\s+', '', text[max(0, rate.start()-60):rate.start()])
                    if re.search(r'(?:预计|有望|计划|目标|将于)[^。；]*$', prefix):
                        continue
                    if value > limit and product_context(text, rate.start(), rate.end()):
                        counters.append({'company': company, 'rate': rate.group(),
                                         'citation_id': field(card, 'citation_id'),
                                         'source_id': field(card, 'id'),
                                         'quote': text[max(0, rate.start()-50):min(len(text), rate.end()+60)]})
            if counters:
                issues.append({'kind': 'rate_ceiling', 'statement': ceiling.group(), 'line': line_index,
                               'reason': '共同覆盖档位不等于最高披露上限；同年度公司产品原文存在更高速率，不能据此作跨架构技术排名',
                               'support': counters})
    return issues


def semantic_support_check(answer, cards, *, question=''):
    originals, citations = {}, defaultdict(set)
    for index, card in enumerate(cards):
        source_id = field(card, 'id') or f'input:{index}'
        originals[source_id] = {'company': field(card, 'company'),
                                'text': field(card, 'evidence') or field(card, 'text')}
        citations[field(card, 'citation_id')].add(source_id)
    issues = rate_ceiling_issues(answer, cards, question=question)
    for segment in re.finditer(r'[^。！？；\n]+[。！？；]?', answer):
        refs = re.findall(r'\[(E\d+)\]', segment.group())
        gaps = collective_scope_gaps(segment.group(), refs, citations, originals)
        if gaps:
            issues.append({'kind': 'collective_scope', 'statement': segment.group().strip(),
                           'line': answer.count('\n', 0, segment.start()),
                           'reason': '所给引用未充分支持双方共同概括：' + '、'.join(gaps)
                                     + '；不等同于公司实际没有该能力'})
    return {'status': 'fail' if issues else 'pass', 'issues': issues}

