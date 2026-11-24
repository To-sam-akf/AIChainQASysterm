"""Prepare a local review packet from the existing, scoped evaluation snapshot."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/eval_runs/closeouts/continuation_20261124/confirmed_review_1_10'
RUN = ROOT / 'data/eval_runs/experiments/experiment_20261123_215354_ca07ed1a'
PACKET = ROOT / 'data/eval_runs/scores/score_20261123_222235_4e268bb1/review_packet.jsonl'
OUT = ROOT / 'data/eval_runs/closeouts/continuation_20261124/reviewer_packet_30'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_value(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def main():
    scores = [r for r in read(BASE / 'scores.jsonl') if r['pending']]
    answers = {(r['case_id'], r['variant']): r for r in read(RUN / 'results.jsonl')}
    cases = {r['case_id']: r for r in read(RUN / 'benchmark.jsonl')}
    packets = {r['answer_id']: r['input'] for r in read(PACKET)}
    assert len(scores) == 30 and len({r['answer_id'] for r in scores}) == 30
    OUT.mkdir(parents=True, exist_ok=False)
    files = OUT / 'answers'
    files.mkdir()
    review_rows, index = [], []
    for number, score in enumerate(scores, 1):
        case_id, variant = score['case_id'], score['variant']
        answer = answers[(case_id, variant)]
        case = cases[case_id]
        packet = packets[score['answer_id']]
        assert answer['answer_id'] == score['answer_id'] if 'answer_id' in answer else True
        assert packet['answer'] == answer['final_answer']
        section = []

        def add(kind, field, question, text, reason):
            review_rows.append({'编号': number, '题号': case_id, '版本': variant,
                                '答案ID': score['answer_id'], '项目类型': kind, '项目ID': field,
                                '需要判断': question, '待核内容': text, '现有待核原因': reason,
                                '结论': '', '理由及原文页码': '', '审核人': '', '审核日期': ''})
            section.append((kind, field, question, text, reason))

        j = score['judgment']
        for kind, key, id_field, text_field in [('关键要点', 'points', 'point_id', 'answer_quote'),
                                                ('事实陈述', 'claims', 'claim_id', 'statement'),
                                                ('引用', 'citations', 'citation_id', 'claim_id')]:
            for item in j[key]:
                if item.get('status') == 'pending':
                    add(kind, item.get(id_field, ''), '核对原文后填写判断；无法判断请注明。', item.get(text_field, ''), item.get('reason', ''))
        segments = {x['segment_id']: x['text'] for x in score['coverage'].get('segments', [])}
        for item in j['segments']:
            if item.get('kind') == 'pending':
                add('段落性质', item['segment_id'], '该段是否包含需要核实的事实？', segments.get(item['segment_id'], ''), item.get('reason', ''))
        for stage, label in [('retrieved', '检索到的材料'), ('selected', '最终选用的材料')]:
            for item in score['coverage'].get(stage, []):
                if item.get('status') == 'pending':
                    unit = next((u for u in case.get('evidence_units', []) if u['unit_id'] == item['unit_id']), {})
                    add('材料覆盖', f"{stage}:{item['unit_id']}", f"{label}是否包含该要点所需的直接依据？", f"需核对：{unit.get('description', '未记录')}\n保存的匹配记录：{render_value(item.get('sources', {}))}", item.get('reason', ''))
        task = j['task']
        if task.get('usability') == 'pending':
            add('整体可用性', 'usability', '整份回答可直接使用、需要大改，还是不可用？', '', task.get('reason', ''))
        if task.get('over_refusal') is None:
            add('是否错误拒答', 'over_refusal', '可回答的内容是否被错误地拒答？', '', task.get('reason', ''))
        if task.get('unsupported_answer') is None:
            add('无依据回答', 'unsupported_answer', '回答是否存在无依据的重要结论？', '', task.get('reason', ''))
        if task.get('serious_errors_pending'):
            add('严重错误候选', 'serious_errors_pending', '是否有影响任务的重要事实、口径、来源或结论错误？请写明具体位置。', '', task.get('reason', ''))
        assert section, f'{case_id} {variant} has pending status without mapped fields'

        path = files / f'{number:02d}_{case_id}_{variant}.md'
        index.append((number, case_id, variant, len(section), bool(task.get('serious_errors_pending')), path.name))
        lines = [f'# {number:02d}｜{case_id}｜版本 {variant}', '', f"答案 ID：`{score['answer_id']}`；答案 SHA-256：`{score['answer_sha256']}`。", '',
                 '## 题目', '', case['question'], '', '## 本份需要判断的项目', '',
                 '请在回填表中按“题号、版本、项目类型、项目ID”逐行填写；不确定时写“无法判断”。', '']
        for kind, field, question, text, reason in section:
            lines.extend([f'### {kind}｜{field}', '', question, '', f'待核内容：{text or "请结合整份回答。"}', '', f'此前留下的原因：{reason or "未记录。"}', ''])
        lines.extend(['## 完整回答', '', answer['final_answer'], '', '## 题目参考原文', '',
                      '以下是评测端参考资料，不代表系统作答时看到了这些原文。页码以 PDF 页为准，需核对表头、脚注和上下文。', ''])
        for src in case['sources']:
            pdf = src.get('pdf', {})
            pathstr = pdf.get('path', '')
            exists = bool(pathstr and (ROOT / pathstr).is_file())
            lines.extend([f"### {src.get('title', '')}｜第 {src.get('pdf_page', '未知')} 页", '',
                          f"来源：{src.get('source_url', '未记录')}；本地文件：`{pathstr or '未记录'}`（{'存在' if exists else '未确认可用'}）。", ''])
            for quote in src.get('quotes', []):
                lines.extend([f'> {quote.replace(chr(10), chr(10) + "> ")}', ''])
        lines.extend(['## 系统作答时实际收到的材料', '',
                      '以下为保存的生成输入中的资料部分；空白代表没有保存资料内容。不要用参考原文补写当时输入。', '',
                      '```text', str(answer.get('generation_context', {}).get('text') or '（无资料内容）'), '```', '',
                      '## 已有评测材料', '',
                      '以下是原评测包保存的引用和资料定位，供进一步核对；不等于已审定。', '',
                      '<details><summary>展开保存的引用和原文定位</summary>', '', '```json',
                      render_value({'citations': packet.get('citations', []), 'sources': packet.get('sources', []),
                                    'coverage_materials': score.get('coverage_materials')}), '```', '', '</details>', ''])
        path.write_text('\n'.join(lines), encoding='utf-8')
    with (OUT / '填写表.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    lines = ['# 30 份回答核查包', '',
             '本包供有资料核查经验的人使用，不是最终结论。文件可能含本地受限资料摘录，请只在有权访问的范围内分享。', '',
             '## 怎么做', '', '1. 打开下表对应文件，阅读题目、待核项目和完整回答。',
             '2. 核对参考原文及页码；判断系统当时是否掌握材料时，要看“实际收到的材料”，不要用后来找到的资料补写。',
             '3. 在“填写表.csv”逐行填写结论、理由或出处、审核人和日期；无法判断就写“无法判断”及原因，不必猜。',
             '4. 事实支持、引用、是否漏答、可用性和严重错误分别判断；严重错误候选不等于已经判定有严重错误。',
             '5. 保存修改后的 CSV 交回。仅填写的具体项目视作已审核，未填项目保留待核；此前确认的范围不需要重复确认。', '',
             '同题 E/F 来自同一份生成候选，但最终回答可能不同，须分别核对。此前模型建议和用户接受建议不等于本次独立人工核查。', '',
             '严重错误参考：公司、关键数字、单位、年份或口径错误；编造来源；证据不足却做重要排名或因果判断等。请结合具体任务影响说明。',
             '可用性参考：可直接使用；需要大改并重新核实；不可用。拿不准就保留待核。', '',
             '本包只列当前 30 份含待核答案，不代表其他答案已完成全部版本差异审查或随机抽查；也不替代独立测试。', '',
             f'共 {len(index)} 份回答，{len(review_rows)} 个待核项目；其中 5 份含严重错误候选。', '',
             '| 编号 | 题号 | 版本 | 待核项目数 | 严重错误候选 | 文件 |', '|---:|---|---|---:|---|---|']
    for number, case, variant, count, serious, name in index:
        lines.append(f'| {number} | {case} | {variant} | {count} | {"是" if serious else "—"} | [打开](answers/{name}) |')
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest = {'status': 'awaiting_independent_review', 'answers': len(index), 'items': len(review_rows),
                'serious_candidate_answers': sum(x[4] for x in index),
                'sources': {str(p.relative_to(ROOT)): digest(p) for p in [BASE/'scores.jsonl', RUN/'results.jsonl', RUN/'benchmark.jsonl', PACKET]},
                'results_not_imported': True}
    (OUT / 'manifest.json').write_text(render_value(manifest) + '\n', encoding='utf-8')
    print(render_value(manifest))


if __name__ == '__main__':
    main()
