"""Independent rule findings against saved answers and frozen source excerpts.

No product verification or product self-rating enters this module. Shared pure
rules are given independently resolved originals, never the product's verdict.
"""
from __future__ import annotations

import re
from decimal import Decimal

from aika.eval.judge_schema import ClaimJudgment, CitationJudgment, SeriousError, Support
from aika.numeric_evidence import SCALE, QUANTITY, accounting_consistency_check, conversion_checks, metric_label
from aika.semantic_evidence import semantic_support_check


def findings(payload):
    cards = []
    sources = {s['id']: s for s in payload['sources']}
    for citation in payload['citations']:
        for source_id in citation['original_source_ids']:
            if source_id in sources:
                cards.append({**sources[source_id], 'citation_id': citation['citation_id'],
                              'source': sources[source_id].get('title', '')})
    answer = payload['answer']
    comparison = []
    if (re.search(r'比较', payload['question']) and re.search('产品|光模块|技术', payload['question'])
            and not re.search('片段主题|摘录主题|材料主题', payload['question'])):
        for line in answer.splitlines():
            if (re.search('本次所给资料|所给片段', line)
                    and re.search(r'披露侧重(?:经营数据|财务|产销)', line)
                    and re.search(r'披露侧重(?:技术|研发|产品)', line)):
                comparison.append({'kind': 'sampling_difference', 'statement': line.strip(), 'point_only': True,
                    'reason': '片段取样的财务/研发主题差异不等于完成同口径公司披露比较；该范围说明可为真，但不足以命中比较要点'})
    return [*accounting_consistency_check(answer)['issues'], *comparison,
            *[{**c, 'statement': c['expression']} for c in conversion_checks(answer) if c['status'] == 'unsupported'],
            *semantic_support_check(answer, cards, question=payload['question'])['issues']]


def numeric_proofs(payload, numeric):
    result = []
    for fact, value in numeric:
        plain = value['value'].replace(',', '')
        number = Decimal(plain)
        spellings = [plain, format(number, ',f')]
        for source_id in fact['source_ids']:
            sid = 'gold:' + source_id
            source = next((s for s in payload['sources'] if s['id'] == sid), None)
            if not source:
                continue
            for spelling in spellings:
                offset = source['text'].find(spelling)
                if offset < 0:
                    continue
                text = source['text']
                result.append(Support(source_id=sid, quote=text[max(0, offset-45):offset+len(spelling)+25]))
                break
    return result


def rd_identity(payload):
    groups = {}
    for fact in payload.get('facts', []):
        for v in fact.get('numeric_values', []):
            if v['unit'] not in ('元', '万元', '亿元'):
                continue
            key = (v['subject'], v['period'])
            groups.setdefault(key, {})[metric_label(v['metric'])] = (fact, v)
    # Multiple subjects/periods must not silently borrow the first identity.
    if len(groups) != 1:
        return []
    for group in groups.values():
        if not {'rd_expense', 'rd_total', 'rd_cap'} <= group.keys():
            continue
        values = {k: Decimal(v['value']) * SCALE[v['unit']] for k, (_, v) in group.items()}
        if values['rd_expense'] + values['rd_cap'] == values['rd_total']:
            return numeric_proofs(payload, [group[k] for k in ('rd_expense', 'rd_cap', 'rd_total')])
    return []


def affects_point(issue, definition, point, payload):
    text = definition['text']
    if issue['kind'].startswith('rd_'):
        return bool(re.search(r'相加|加总投入|勾稽|费用再加', text))
    if issue['kind'] == 'sampling_difference':
        return definition['kind'] == 'synthesis' and bool(re.search('侧重点差异|披露.*差异', text))
    if issue['kind'] == 'rate_ceiling':
        return bool(re.search(r'高速产品|产品覆盖|速率', text))
    if issue['kind'] == 'money_conversion' and definition['kind'] == 'fact':
        bases = {Decimal(m['value'].replace(',', '')) * SCALE[m['unit']]
                 for m in QUANTITY.finditer(issue['statement']) if m['unit'] in ('元', '万元', '亿元')}
        return any(f['fact_id'] in definition['fact_ids']
                   and any(v['unit'] in ('元', '万元', '亿元') and Decimal(v['value']) * SCALE[v['unit']] in bases
                           for v in f['numeric_values']) for f in payload.get('facts', []))
    return False


def apply_findings(judgment, payload):
    """Do not let a model's supported/usable label erase a known contradiction."""
    for index, issue in enumerate(findings(payload)):
        if issue.get('point_only'):
            for point in judgment.points:
                definition = next(d for d in payload['answer_points'] if d['point_id'] == point.point_id)
                if definition['critical'] and affects_point(issue, definition, point, payload):
                    if point.status == 'correct':
                        point.status = 'missing'
                        point.reason += '；独立规则核对：' + issue['reason']
                    if judgment.task.usability in ('usable', 'minor_revision'):
                        judgment.task.usability = 'major_revision'
            continue  # Do not label an honest description of the input a false fact.
        matching = [s for s in payload['segments'] if issue['statement'] in s['text']
                    or (len(s['text']) > 12 and s['text'] in issue['statement'])]
        for segment in matching:
            statement = issue['statement'] if issue['statement'] in segment['text'] else segment['text']
            items = [c for c in judgment.claims if c.segment_id == segment['segment_id']
                     and (statement in c.statement or c.statement in statement)]
            inventory = next(s for s in judgment.segments if s.segment_id == segment['segment_id'])
            inventory.kind = 'factual'
            inventory.reason += '；规则核对发现实质陈述，必须纳入事实清单'
            if not items:
                claim_id = f'rule_{index}_{segment["segment_id"]}'
                while any(c.claim_id == claim_id for c in judgment.claims):
                    claim_id += '_'
                item = ClaimJudgment(claim_id=claim_id, segment_id=segment['segment_id'], statement=statement,
                    status='pending', reason=issue['reason'], support=[], citation_ids=segment['citation_ids'])
                judgment.claims.append(item)
                items = [item]
                for citation_id in item.citation_ids:
                    judgment.citations.append(CitationJudgment(claim_id=claim_id, citation_id=citation_id,
                        status='incorrect', reason='该引用不能支持规则核对发现的整个实质结论', support=[]))
            proofs = []
            if issue['kind'] == 'rate_ceiling':
                proofs = [Support(source_id=s['source_id'], quote=s['quote']) for s in issue['support'] if s['source_id']]
            elif issue['kind'].startswith('rd_'):
                proofs = rd_identity(payload)
            elif issue['kind'] == 'money_conversion':
                bases = {Decimal(m['value'].replace(',', '')) * SCALE[m['unit']] for m in QUANTITY.finditer(statement)
                         if m['unit'] in ('元', '万元', '亿元')}
                numeric = [(f, v) for f in payload.get('facts', []) for v in f['numeric_values']
                           if v['unit'] in ('元', '万元', '亿元') and Decimal(v['value']) * SCALE[v['unit']] in bases]
                proofs = numeric_proofs(payload, numeric)
            for item in items:
                if issue['kind'] == 'collective_scope':
                    if item.status == 'supported':
                        item.status = 'pending'
                else:
                    item.status = 'contradicted' if proofs else 'unsupported'
                item.reason += '；独立规则核对：' + issue['reason']
                if issue['kind'] != 'collective_scope':
                    item.support = proofs
                for c in judgment.citations:
                    if c.claim_id == item.claim_id and c.status == 'correct':
                        c.status = 'incorrect'
                        c.reason += '；该引用不能支持错误的等价关系、运算对象或整句上限/共同点'
            critical = False
            for point in judgment.points:
                definition = next(d for d in payload['answer_points'] if d['point_id'] == point.point_id)
                if definition['critical'] and affects_point(issue, definition, point, payload):
                    critical = True
                    if point.status == 'correct':
                        point.status = 'incorrect'
                        point.reason += '；独立规则核对：' + issue['reason']
            if critical:
                if judgment.task.usability in ('usable', 'minor_revision', 'pending'):
                    judgment.task.usability = 'major_revision'
                judgment.task.reason += '；关键要点存在规则已确认的问题：' + issue['reason']
                can_confirm = (issue['kind'] in ('rate_ceiling', 'money_conversion')
                               or payload.get('generation_materials', {}).get('has_materials') is True)
                def same_error(error):
                    if error.answer_quote in statement or statement in error.answer_quote:
                        return True
                    if issue['kind'] == 'money_conversion':
                        def values(text):
                            return sorted(Decimal(m['value'].replace(',', '')) * SCALE[m['unit']]
                                          for m in QUANTITY.finditer(text) if m['unit'] in ('元', '万元', '亿元'))
                        return values(error.answer_quote) == values(statement)
                    return error.reason == issue['reason'] and error.support == proofs
                if proofs and can_confirm and not any(same_error(e) for e in judgment.task.serious_errors):
                    judgment.task.serious_errors.append(SeriousError(
                        kind='numeric_or_period' if issue['kind'] in ('rate_ceiling', 'money_conversion') else 'unsupported_conclusion',
                        answer_quote=statement, reason=issue['reason'], support=proofs))
    return judgment
