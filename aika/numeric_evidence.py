"""Source-bound financial quantities, unit conversions and explicit arithmetic.

This checks numeric provenance, not the truth of the surrounding interpretation.
Unknown derivations stay pending; no source-free expression is accepted as proof.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from aika.question_planner import extract_companies

SCALE = {'元': Decimal(1), '万元': Decimal(10000), '亿元': Decimal(100000000),
         '%': Decimal('.01'), '％': Decimal('.01'), '个百分点': Decimal('.01'), '?': Decimal(1), 'ratio': Decimal(1)}
VALUE = r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?'
UNIT = r'个百分点|亿元|万元|元|%|％'
QUANTITY = re.compile(rf'(?<![\d.])(?P<value>{VALUE})\s*(?P<unit>{UNIT})')
CELL = re.compile(rf'(?P<value>{VALUE})\s*(?P<unit>{UNIT})?')
YEAR = re.compile(r'((?:19|20)\d{2})\s*年')
LABELS = (
    ('资本化研发投入占营业收入比例', 'rd_cap_revenue_ratio'), ('资本化研发投入占营收比例', 'rd_cap_revenue_ratio'),
    ('资本化研发投入占营业收入', 'rd_cap_revenue_ratio'), ('资本化研发投入占营收', 'rd_cap_revenue_ratio'),
    ('研发投入占营业收入比例', 'rd_ratio'), ('研发投入占营收比例', 'rd_ratio'),
    ('研发投入资本化的比重', 'rd_cap_ratio'), ('研发投入资本化比重', 'rd_cap_ratio'),
    ('资本化比重', 'rd_cap_ratio'), ('资本化比例', 'rd_cap_ratio'),
    ('资本化研发投入占研发投入', 'rd_cap_ratio'), ('研发投入资本化的金额', 'rd_cap'),
    ('资本化占研发投入比例', 'rd_cap_ratio'), ('资本化占研发投入', 'rd_cap_ratio'),
    ('占研发投入比例', 'rd_share_unspecified'),
    ('资本化率', 'rd_cap_ratio'), ('占营业收入比例', 'revenue_ratio_unspecified'),
    ('占营收比例', 'revenue_ratio_unspecified'), ('占营业收入', 'revenue_ratio_unspecified'),
    ('占营收', 'revenue_ratio_unspecified'), ('占比', 'ratio_unspecified'),
    ('资本化研发投入', 'rd_cap'), ('资本化金额', 'rd_cap'),
    ('研发投入总额', 'rd_total'), ('研发投入金额', 'rd_total'), ('研发投入', 'rd_total'),
    ('研发费用', 'rd_expense'), ('营业收入', 'revenue'), ('营收', 'revenue'), ('收入', 'revenue'),
    ('毛利率', 'gross_margin'), ('净利润', 'profit'), ('订单金额', 'orders'),
)
TOKEN = rf'(?:{VALUE})\s*(?:{UNIT})?'
EXPR = rf'{TOKEN}(?:\s*[+＋\-−*×/÷]\s*{TOKEN}){{1,3}}'
EQUATION = re.compile(rf'(?P<left>{EXPR})\s*(?:=|＝|≈)\s*(?P<right>{TOKEN})(?![\d.,])|'
                        rf'(?P<result>{TOKEN})\s*(?:=|＝|≈)\s*(?P<expr>{EXPR})')


def field(item, name):
    return str(item.get(name, '') if isinstance(item, dict) else getattr(item, name, '') or '')


def metric_label(text):
    compact = re.sub(r'\s+', '', text)
    compact = compact.replace('占报告期内营业收入的比例', '占营业收入比例')
    matches = [(compact.rfind(term), -len(term), label) for term, label in LABELS if term in compact]
    if not matches:
        return ''
    # Prefer the longest expression at a position over its shorter prefix.
    start = max(pos for pos, _, _ in matches)
    covering = [(pos, neg_len, label) for pos, neg_len, label in matches
                if pos == start or pos - neg_len >= start]
    pos, _, label = min(covering, key=lambda m: m[1])
    if label == 'rd_share_unspecified':
        numerator = metric_label(compact[:pos])
        return {'rd_cap': 'rd_cap_ratio', 'rd_cap_revenue_ratio': 'rd_cap_ratio',
                'rd_expense': 'rd_expense_share'}.get(numerator, label)
    if label == 'revenue_ratio_unspecified':
        # A shorthand inherits its numerator, never the denominator's label.
        # E.g. "研发投入金额…、占营收…" is not a revenue amount or a
        # capitalisation ratio. Bare "占营收" remains ambiguous.
        numerator = metric_label(compact[:pos])
        return {'rd_total': 'rd_ratio', 'rd_expense': 'rd_expense_revenue_ratio',
                'rd_cap': 'rd_cap_revenue_ratio', 'orders': 'orders_revenue_ratio'}.get(numerator, label)
    return label


@dataclass(frozen=True)
class Quantity:
    number: Decimal
    unit: str
    text: str
    label: str
    year: str
    company: str
    citation: str
    line: int
    approximate: bool = False

    @property
    def base(self):
        return self.number * SCALE[self.unit]

    @property
    def dimension(self):
        return 'unknown' if self.unit == '?' else 'money' if self.unit in ('元', '万元', '亿元') else 'rate'


def quantities(text, *, citation='', company='', source_year=''):
    """Read explicit units and table-wide/row units without joining unrelated cells."""
    default_unit = ''
    current_year = source_year
    current_company = company
    headers = []
    result = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        compact = re.sub(r'\s+', '', line)
        previous_year = current_year
        companies = extract_companies(line)
        if len(companies) == 1:
            current_company = companies[0]
        years = YEAR.findall(line)
        if len(set(years)) == 1:
            current_year = years[0]
        unit_heading = re.search(rf'单位[:：]\s*({UNIT})', line)
        if unit_heading:
            default_unit = unit_heading[1]
        if '|' in line:
            cells = line.strip().strip('|').split('|')
            row_label = next((c.strip() for c in cells if c.strip()), '')
            if any(YEAR.search(cell) or re.fullmatch(r'(?:本期|上期|本年度|上年度)(?:发生额|金额|数)?', cell.strip()) for cell in cells):
                nonempty = [c.strip() for c in cells if c.strip()]
                # In sparse PDF tables the first non-empty cell can itself be
                # a year, not a label. Keep it when every header is a year.
                if len(nonempty) >= 2 and all(YEAR.fullmatch(c) for c in nonempty):
                    headers = nonempty
                elif (nonempty and re.fullmatch(r'(?:本期|本年度)(?:发生额|金额|数)?', nonempty[0])
                      and any(re.fullmatch(r'(?:上期|上年度)(?:发生额|金额|数)?', c) for c in nonempty[1:])):
                    headers = nonempty
                elif len(nonempty) >= 2 and YEAR.fullmatch(nonempty[0]) and sum(bool(YEAR.search(c)) for c in nonempty) >= 2:
                    # Multi-level headers may repeat amount/ratio columns under
                    # each year. Without a verified column span, avoid shifting
                    # the first year's value into the second year.
                    headers = []
                else:
                    headers = [c.strip() for c in cells[cells.index(next(c for c in cells if c.strip())) + 1:] if c.strip()]
            # PDF table extraction can put the last half of a row label on the next line.
            if index + 1 < len(lines) and '|' in lines[index + 1]:
                continuation = next((c.strip() for c in lines[index + 1].strip().strip('|').split('|') if c.strip()), '')
                if re.search(r'[（(]万?$', row_label) and continuation.startswith('元'):
                    row_label += continuation
            label_unit = re.search(rf'[（(]\s*({UNIT})\s*[）)]', row_label)
            if '项目' in row_label and label_unit:
                default_unit = label_unit[1]
            numeric_column = 0
            for col, cell in enumerate(cells):
                match = CELL.fullmatch(cell.strip())
                matches = [match] if match else list(QUANTITY.finditer(cell))
                if not matches:
                    continue
                header = headers[numeric_column] if numeric_column < len(headers) else ''
                numeric_column += 1
                year_match = YEAR.search(header)
                year = year_match[1] if year_match else current_year if ('本期' in header or '本年度' in header) else str(int(source_year)-1) if ('上期' in header or '上年度' in header) and source_year else ''
                if not year and not source_year and not citation:
                    year = current_year  # Answer table inherits its own explicitly stated year.
                label = metric_label(row_label)
                if '变动' in header or '同比' in header:
                    label += '_change'
                for match in matches:
                    unit = match['unit'] or (label_unit[1] if label_unit else default_unit) or '?'
                    approximate = bool(re.search('约|折合|换算|四舍五入', cell[:match.start()][-15:]))
                    result.append(Quantity(Decimal(match['value'].replace(',', '')), unit,
                        match['value'] + unit, label, year, current_company, citation, index, approximate))
        else:
            headers = []
            for match in QUANTITY.finditer(line):
                prefix = line[:match.start()]
                local_years = YEAR.findall(prefix)
                local_companies = extract_companies(prefix)
                year = local_years[-1] if local_years else '' if citation else previous_year if years else current_year
                # Only bind an immediately preceding, explicit prior-period
                # label; other comparisons remain subject to normal checking.
                if year and re.search(r'(?:上年度|上期)(?:占比|比例|为|是|[:：\s*])*$', prefix):
                    year = str(int(year) - 1)
                # An explicit difference/converted amount is not a repeated assertion of the preceding metric.
                label = '' if re.search(r'相差|差额|差值|折合|约合', prefix[-12:]) else metric_label(prefix)
                if match['unit'] in ('元', '万元', '亿元') and re.search(r'资本化\s*[:：]?\s*$', prefix):
                    label = 'rd_cap'
                if match['unit'] in ('%', '％') and re.search(r'资本化[^。；\n]{0,60}[、，,]比例\s*[:：]?\s*$', prefix):
                    label = 'rd_share_unspecified'
                result.append(Quantity(Decimal(match['value'].replace(',', '')), match['unit'], match.group().strip(),
                    label, year,
                    local_companies[-1] if local_companies else current_company, citation, index,
                    bool(re.search(r'约|折合|换算|四舍五入', prefix[-15:]))))
    return result


def compatible(answer, source):
    return (answer.dimension == source.dimension
            and (answer.unit == '个百分点') == (source.unit == '个百分点')
            and not (answer.company and source.company and answer.company != source.company)
            and not (answer.year and source.year and answer.year != source.year)
            and not (answer.label not in ('', 'ratio_unspecified', 'revenue_ratio_unspecified', 'rd_share_unspecified')
                     and source.label and answer.label != source.label)
            and not (answer.label == 'revenue_ratio_unspecified' and source.label
                     and source.label != 'rd_ratio' and not source.label.endswith('_revenue_ratio'))
            and not (answer.label == 'rd_share_unspecified' and source.label
                     and source.label not in ('rd_cap_ratio', 'rd_expense_share')))


def arithmetic(line, observations):
    """Validate explicit arithmetic with source-grounded operands, using Decimal."""
    outcomes = []
    for match in EQUATION.finditer(line):
        expr = match['left'] or match['expr']
        target = match['right'] or match['result']
        parsed, ops, cursor = [], [], 0
        while cursor < len(expr):
            operand = re.compile(TOKEN).match(expr, cursor)
            if operand is None:
                break
            parsed.append(CELL.fullmatch(operand.group().strip()))
            cursor = operand.end()
            separator = re.match(r'\s*([+＋\-−*×/÷])\s*', expr[cursor:])
            if separator is None:
                break
            ops.append(separator[1])
            cursor += separator.end()
        output = CELL.fullmatch(target.strip())
        if not parsed or not all(parsed) or output is None:
            continue
        units = {m['unit'] for m in [*parsed, output] if m['unit']}
        if not units:
            inferred = [{q.unit for q in observations if q.unit != '?' and q.number == Decimal(m['value'].replace(',', ''))}
                        for m in parsed]
            common = set.intersection(*inferred) if inferred else set()
            if len(common) != 1:
                outcomes.append({'expression': match.group(), 'status': 'pending', 'reason': '计算的单位无法唯一确认'})
                continue
            units = common
        default = next(iter(units)) if len(units) == 1 else None
        values, supports = [], []
        for m in parsed:
            unit = m['unit'] or default
            if not unit:
                break
            base = Decimal(m['value'].replace(',', '')) * SCALE[unit]
            found = [q for q in observations if q.base == base and q.dimension == ('money' if unit in ('元', '万元', '亿元') else 'rate')]
            if not found:
                break
            supports.append(found)
            values.append(base)
        else:
            # All operands must have at least one common company/period context.
            contexts = [{(q.company, q.year) for q in found} for found in supports]
            common_contexts = set.intersection(*contexts)
            if not common_contexts or not any(year for _, year in common_contexts):
                outcomes.append({'expression': match.group(), 'status': 'pending', 'reason': '计算主体或年份口径未对齐'})
                continue
            if any(op in ('+', '＋', '-', '−') for op in ops) and any(op in ('*', '×', '/', '÷') for op in ops):
                outcomes.append({'expression': match.group(), 'status': 'pending', 'reason': '混合运算需进一步核对计算步骤'})
                continue
            total = values[0]
            dimension = supports[0][0].dimension
            valid_dimensions = True
            for op, value, support in zip(ops, values[1:], supports[1:]):
                other = support[0].dimension
                if op in ('+', '＋', '-', '−'):
                    valid_dimensions &= dimension == other
                    total = total + value if op in ('+', '＋') else total - value
                elif op in ('*', '×'):
                    valid_dimensions &= not (dimension == other == 'money')
                    total *= value
                    dimension = 'money' if 'money' in (dimension, other) else 'rate'
                elif value == 0:
                    valid_dimensions = False
                else:
                    valid_dimensions &= not (dimension == 'rate' and other == 'money')
                    total /= value
                    dimension = 'rate' if dimension == other else 'money'
            divided_rate = dimension == 'rate' and any(op in ('/', '÷') for op in ops)
            output_unit = output['unit'] or ('ratio' if divided_rate else default)
            if not output_unit:
                outcomes.append({'expression': match.group(), 'status': 'pending', 'reason': '计算结果的单位无法确认'})
                continue
            target_base = Decimal(output['value'].replace(',', '')) * SCALE[output_unit]
            same_dimension = dimension == ('money' if output_unit in ('元', '万元', '亿元') else 'rate')
            equal = total == target_base
            if not equal and '≈' in match.group():
                precision = Decimal(1).scaleb(Decimal(output['value'].replace(',', '')).as_tuple().exponent)
                equal = (total / SCALE[output_unit]).quantize(precision, rounding=ROUND_HALF_UP) == Decimal(output['value'].replace(',', ''))
            valid = equal and same_dimension and valid_dimensions
            outcomes.append({'expression': match.group(), 'status': 'supported' if valid else 'unsupported',
                             'reason': '原文数值及单位的运算一致' if valid else '等式计算结果或单位不一致',
                             'result_base': str(target_base), 'unit': output_unit})
            continue
        outcomes.append({'expression': match.group(), 'status': 'pending', 'reason': '计算操作数没有对应的原文数值'})
    return outcomes


def audit_financial_numbers(answer: str, cards: list[Any], *, company=''):
    observations = []
    for card in cards:
        years = YEAR.findall(field(card, 'source'))
        observations += quantities(field(card, 'evidence') or field(card, 'evidence_span'),
            citation=field(card, 'citation_id'), company=field(card, 'company'), source_year=years[0] if len(set(years)) == 1 else '')
    answers = [q for q in quantities(answer, company=company) if q.unit != '?']
    lines = answer.splitlines()
    decisions, calculations = [], []
    for index, line in enumerate(lines):
        refs = set(re.findall(r'\[(E\d+)\]', line))
        available = [q for q in observations if not refs or q.citation in refs]
        line_companies = extract_companies(line)
        scoped_company = line_companies[0] if len(line_companies) == 1 else company
        line_years = set(YEAR.findall(line))
        scoped = [q for q in available if not (scoped_company and q.company and q.company != scoped_company)
                  and not (len(line_years) == 1 and q.year and q.year not in line_years)]
        computed = arithmetic(line, scoped)
        calculations.extend(computed)
        for item in [q for q in answers if q.line == index]:
            candidates = [q for q in available if compatible(item, q)]
            exact = [q for q in candidates if q.base == item.base]
            rounded = [q for q in candidates if (q.base / SCALE[item.unit]).quantize(
                Decimal(1).scaleb(item.number.as_tuple().exponent), rounding=ROUND_HALF_UP) == item.number]
            derived = any(c['status'] == 'supported' and (Decimal(c['result_base']) == item.base
                or (item.number == 0 and re.search(r'差额|差值|误差', line))) for c in computed)
            certain = [q for q in exact if not (item.year and not q.year)]
            unknown = [q for q in available if q.unit == '?' and q.number == item.number]
            rounded_certain = [q for q in rounded if not (item.year and not q.year)]
            pp = re.search(rf'({VALUE})\s*[%％]\s*[−-]\s*({VALUE})\s*[%％]\s*[=＝]'
                           rf'\s*[*`\s]*(?:上升|下降|增加|减少)?\s*({VALUE})\s*个百分点', line)
            pp_consistent = (pp and item.unit == '个百分点'
                and Decimal(pp[3].replace(',', '')) == item.number
                and Decimal(pp[1].replace(',', '')) - Decimal(pp[2].replace(',', '')) == item.number)
            if item.label in ('revenue_ratio_unspecified', 'rd_share_unspecified') and (exact or rounded):
                status, reason = 'pending', '占营收比例的分子指标未明确，不能仅按相同数字确认'
            elif certain or (rounded_certain and item.approximate) or derived:
                status, reason = 'supported', '匹配原文数值、明确单位换算或已核验的计算'
            elif exact or unknown:
                status, reason = 'pending', '原文存在相同数值，但表格单位或年份列缺失，不能自动确认'
            elif pp_consistent:
                status, reason = 'pending', '百分点差值算术一致；操作数主体、年份和来源仍须核对'
            elif rounded:
                status, reason = 'pending', '仅在四舍五入后相符，答案未明确标注近似'
            elif re.search(r'计算|相差|差额|差值|约|折合|换算|≈', line):
                status, reason = 'pending', '派生数值缺少可核验的明确等式或对应口径'
            else:
                status, reason = 'unsupported', '未在对应引用、主体、年份和指标口径找到等价数值'
            decisions.append({'text': item.text, 'line': index, 'status': status, 'reason': reason,
                              'source_citations': sorted({q.citation for q in exact or rounded})})
    return {'quantities': decisions, 'calculations': calculations,
            'conversions': conversion_checks(answer)}


def conversion_checks(answer: str) -> list[dict[str, Any]]:
    """Check an asserted monetary equivalence, even if both numbers have sources.

    A rounded representation is not an exact reconciliation. Merely listing two
    different amounts (or explicitly saying they differ) is not an equivalence.
    """
    outcomes = []
    for index, line in enumerate(answer.splitlines()):
        amounts = [m for m in QUANTITY.finditer(line) if m['unit'] in ('元', '万元', '亿元')]
        equations = list(EQUATION.finditer(line))
        for left, right in zip(amounts, amounts[1:]):
            if any(e.start() <= left.start() and right.end() <= e.end() for e in equations):
                continue  # A+b=c is arithmetic, not the assertion b=c.
            connector = line[left.end():right.start()]
            if not re.fullmatch(r'[\s*`（()）\[\]E\d，,：:]*'
                                r'(?:即|等于|等价于|折合|约合|约为|约|≈|=|＝)'
                                r'[\s*`（()）约]*', connector):
                continue
            a, b = Decimal(left['value'].replace(',', '')), Decimal(right['value'].replace(',', ''))
            a_base, b_base = a * SCALE[left['unit']], b * SCALE[right['unit']]
            rounded = ((a_base / SCALE[right['unit']]).quantize(Decimal(1).scaleb(b.as_tuple().exponent), rounding=ROUND_HALF_UP) == b
                       or (b_base / SCALE[left['unit']]).quantize(Decimal(1).scaleb(a.as_tuple().exponent), rounding=ROUND_HALF_UP) == a)
            approximate = bool(re.search('约|≈', connector))
            status = ('supported' if a_base == b_base or (rounded and approximate)
                      else 'pending' if rounded else 'unsupported')
            outcomes.append({'expression': line[left.start():right.end()], 'line': index,
                             'status': status, 'kind': 'money_conversion',
                             'reason': '金额换算一致' if status == 'supported' else
                             '仅舍入后相符，不能作为精确勾稽' if status == 'pending' else
                             '两个金额按元/万元/亿元换算不等价，不能用口径一致或四舍五入解释',
                             'difference_yuan': str(abs(a_base - b_base))})
    return outcomes


def accounting_consistency_check(answer: str) -> dict[str, Any]:
    """Check a narrow logical error, not an enterprise accounting reconciliation.

    When the answer treats R&D expense as a subset of total R&D, adding the
    expense to that total repeats the expensed part, not the capitalised part.
    This does not establish that a company's two reported figures reconcile.
    """
    issues = []
    offset = 0
    for index, line in enumerate(answer.splitlines(keepends=True)):
        offset += len(line)
        context = answer[:offset]
        if not ('研发费用' in context and re.search('研发投入总额|研发投入金额|研发投入合计', context)):
            continue
        for sentence in re.findall(r'[^。！？]+[。！？]?', line):
            compact = re.sub(r'\s+', '', sentence)
            compact = re.sub(r'[*`"“”\'‘’]', '', compact)
            expense = r'(?:研发费用|费用化(?:研发)?(?:费用|支出|部分))'
            capital = r'资本化(?:研发)?(?:投入|金额|支出|部分)'
            pair = re.search(rf'(?:{expense})(?:与|和|及|加上?|[+＋])(?:{capital})'
                             rf'|(?:{capital})(?:与|和|及|加上?|[+＋])(?:{expense})', compact)
            # Bind the verdict to this pair of operands. Another clause can
            # correctly reject total + expense while accepting expense + cap.
            pair_prefix = compact[:pair.start()] if pair else ''
            pair_suffix = compact[pair.end():] if pair else ''
            pair_rejected = bool(pair and (
                re.search(r'(?:不能|不可|不应|不得)(?:把|将)?$', pair_prefix)
                or re.match(r'(?:不能|不可|不应|不得)相加', pair_suffix)
                or re.match(r'(?:相加|加总)?[，,]?(?:会|将)?重复(?:计算|计入)', pair_suffix)
            ))
            meta_negation = re.search(r'不会重复|不重复|并非不能相加|不是不能相加|不能(?:说|认为|认定).{0,25}重复'
                                      r'|(?:说法|表述|解释).{0,8}(?:错误|不对|不成立)', compact)
            if pair_rejected and not meta_negation:
                issues.append({'statement': sentence.strip(), 'line': index, 'kind': 'rd_addition_operands',
                               'reason': '禁止相加的运算对象写错：费用化研发费用与资本化投入可组成总投入；不能把这两个组成部分说成重复计算'})
                continue
            wrong_part = re.search(r'(?:相加|加总).{0,8}重复(?:计算|计入)(?:的(?:是)?)?资本化(?:金额|部分|支出)', compact)
            wrong_reason = re.search(r'(?:因为|由于)资本化部分已包含.{0,30}(?:相加|加总).{0,8}重复计算', compact)
            if not (wrong_part or wrong_reason):
                continue
            if meta_negation:
                continue
            if re.search(r'(?:不(?:会|是)|并非|而非|不是).{0,8}重复|总额\s*[+＋加]\s*资本化', compact):
                continue
            # An explicitly different addition can legitimately repeat capitalisation.
            if re.search(r'(?:总额|总投入).{0,8}(?:与|和|加|＋|\+).{0,4}资本化(?:研发)?(?:投入|支出|金额)', compact):
                continue
            issues.append({'statement': sentence.strip(), 'line': index,
                           'kind': 'rd_duplicate_part',
                           'reason': '总投入再加研发费用重复的是费用化部分；不能以资本化部分已包含解释该重复项'})
    return {'status': 'fail' if issues else 'pass', 'issues': issues}
