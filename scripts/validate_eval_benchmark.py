#!/usr/bin/env python3
"""Validate a v2 question set and optionally render its human review sheet."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from aika.eval.benchmark_v2 import (
    CATEGORIES,
    DEFAULT_BENCHMARK_V2,
    BenchmarkV2Case,
    BenchmarkV2Error,
    file_sha256,
    load_benchmark_v2,
    require_approved,
    validate_collection,
    validate_pilot,
    validate_sources,
)


def render_review(cases: list[BenchmarkV2Case], dataset_sha256: str) -> str:
    """A generated reading aid; JSONL remains the annotation source of truth."""
    approved = sum(case.review.human_status == "approved" for case in cases)
    lines = [
        "# AIKA v2 首轮题集人工复核清单", "",
        f"题数：{len(cases)}；人工通过：{approved}；待确认或需修改：{len(cases) - approved}。",
        "本清单由 JSONL 题集生成。AI 已回查本地 PDF；这不等于人工核验通过，也不证明资料中不存在其他披露。",
        f"题集 SHA256：`{dataset_sha256}`。", "",
        "## 使用与校验", "",
        "```bash",
        "# 只检查结构与首轮配额，不调用模型或访问数据库",
        ".venv/bin/python scripts/validate_eval_benchmark.py --pilot",
        "# 回查私有本地资料：文件哈希、来源、原文块位置和 PDF 摘录",
        ".venv/bin/python scripts/validate_eval_benchmark.py --pilot --check-sources",
        "# 根据修改后的题集重新生成本清单（会覆盖清单中的手写内容）",
        ".venv/bin/python scripts/validate_eval_benchmark.py --pilot --check-sources \\",
        "  --review-output data/eval/benchmark_v2_review.md",
        "# 人工审核门槛：仍有未通过题目时，此命令返回退出码 2",
        ".venv/bin/python scripts/validate_eval_benchmark.py --require-human-review",
        "```", "",
        "需安装项目 dev/full 依赖；普通结构检查不需要 PDF，来源检查需要本地私有资料和 PyMuPDF。",
        "使用 `--benchmark` 指定其他 v2 文件，用 `--additional-benchmark` 加入其他划分并检查跨集分组/相同事实泄漏；",
        "旧 QA/RAG loader 不支持本格式，不能将本文件直接传给现有 `main.py eval`。",
        "本轮 20 题全部属于 dev，pilot=true；试运行通过后再扩为 40 道开发题与 60 道独立测试题。", "",
        "## 字段与审核方式", "",
        "`facts` 保存标准事实与数字原值；`answer_points` 关联事实并区分关键要点、事实、综合分析与边界。",
        "`evidence_units` 的 primary_sources 必须共同满足；每组 alternative_sources 是完整替代方案；空列表表示尚无核实的替代材料。",
        "required=false 表示背景或可选证据，不计入必需证据分母；纯拒答题无必需证据时，必需证据覆盖记为不适用，不因未检索背景材料扣分。",
        "比较题注明的可选路径只需满足其中一种，不能将全部可选证据或标准事实当成必须逐项复述的答案。",
        "sources 包含 PDF 页码（从 1 开始，可能不同于印刷页码）、上下文、摘录、来源和文件/原文块哈希。",
        "corpus_snapshot 固定本地资料文件，不代表运行时 PostgreSQL 索引已冻结。被测系统只接收 public_input() 的问题文本。", "",
        "逐题打开 PDF，核对主体、年份、表头、单位、脚注、推理边界与可替代证据；尤其检查资料不足题的范围是否足够明确。",
        "请把审核结果记录回 JSONL 的 review，清单中的勾选只作阅读辅助，不会自动改变状态：", "",
        "- 初始 human_status=pending，human_reviewer、human_reviewed_at、reviewed_content_sha256 都为 null。",
        "- 实际审核后写 approved 或 changes_requested，填写真实审核者、带时区的审核时间、下列当前内容哈希和意见。",
        "- 内容哈希覆盖题目及全部评分依据，排除 review；修改内容后旧审核失效，需要重新核对。",
        "- 所有题目人工通过前只用于调试；AI 检查与脚本通过均不会自动授予正式评测资格。", "",
        "sam 的首轮审核、修改前内容哈希及后续复核记录保存在 benchmark_v2_reviews.jsonl；每次通过仅对应实际审核的内容版本。",
        "人工审核记录中的时间为意见录入时间；用户未提供实际逐题审核时刻。", "",
        "## 题目索引", "",
        "| 题号 | 类别 | 预期行为 | 人工状态 | 问题 |",
        "|---|---|---|---|---|",
    ]
    for case in cases:
        lines.append(
            f"| {case.case_id} | {CATEGORIES[case.category]} | {case.answer_behavior.mode} | "
            f"{case.review.human_status} | {case.question.replace('|', '/')} |"
        )
    for case in cases:
        lines += ["", f"## {case.case_id} · {CATEGORIES[case.category]}", "", case.question, "",
                  f"截止日期：{case.as_of_date}；分组：`{case.leakage_group}`；划分：{case.split}。",
                  f"AI 核对日期：{case.review.ai_checked_on}；人工状态：{case.review.human_status}。",
                  "来源题号：" + "；".join(f"{p.dataset}/{p.case_id}" for p in case.provenance) + "。",
                  f"当前内容 SHA256：`{case.content_sha256()}`。", "",
                  "### 必答要点", ""]
        for point in case.answer_points:
            lines.append(f"- {point.point_id} [{'关键' if point.critical else '补充'}/{point.kind}] {point.text}"
                         f"（事实：{', '.join(point.fact_ids) or '边界判定'}）")
        lines += ["", "### 标准事实与数字", ""]
        for fact in case.facts:
            lines.append(f"- {fact.fact_id}：{fact.statement}（依据：{', '.join(fact.source_ids)}）")
            for value in fact.numeric_values:
                lines.append(f"  - {value.subject} / {value.period} / {value.metric}：{value.value} {value.unit}")
        lines += ["", "### 回答行为与禁止推导", "",
                  f"预期：**{case.answer_behavior.mode}**。{case.answer_behavior.reason}", "",
                  "可回答：" + "；".join(case.answer_behavior.answerable or ["无需给出请求中的实质结论"]),
                  "不可回答：" + "；".join(case.answer_behavior.unanswerable or ["无"]), ""]
        lines.extend(f"- 禁止：{text}" for text in case.forbidden_inferences)
        lines.extend(f"- 判分：{text}" for text in case.scoring_notes)
        lines += ["", "### 证据要求与替代材料", ""]
        for unit in case.evidence_units:
            alternatives = " 或 ".join(" + ".join(group) for group in unit.alternative_sources) or "无已核实替代材料"
            lines.append(f"- {unit.unit_id}（{'必需' if unit.required else '补充'}）：{unit.description}；"
                         f"事实 {', '.join(unit.fact_ids)}；主证据 {' + '.join(unit.primary_sources)}；替代：{alternatives}。")
        lines += ["", "### 原文核对", ""]
        for source in case.sources:
            lines += [f"**{source.source_id} · {source.report_id} · {source.title}**", "",
                      f"[本地 PDF 第 {source.pdf_page} 页](../../{source.pdf.path}#page={source.pdf_page})；"
                      f"[来源页面]({source.source_url})；公开日期：{source.published_at}。",
                      f"章节：{source.section}。{source.context}",
                      f"原文块：{', '.join(source.chunks)}；PDF SHA256：`{source.pdf.sha256}`。", ""]
            lines.extend(f"> {quote}" for quote in source.quotes)
            lines.append("")
        lines += ["### 人工审核栏", "",
                  f"- [{'x' if case.review.human_status == 'approved' else ' '}] 当前版本已打开原始 PDF 并核对全部事实、数字、表头和脚注。",
                  f"- [{'x' if case.review.human_status == 'approved' else ' '}] 当前版本已确认必答要点、证据替代、禁止推导及回答/拒答边界。",
                  f"- [{'x' if case.review.human_status == 'approved' else ' '}] 当前版本已将实际决定、审核者、时间、内容哈希及意见写回 JSONL。",
                  f"审核者：{case.review.human_reviewer or '待复核'}；录入时间：{case.review.human_reviewed_at or '无当前版本审核记录'}。",
                  f"当前审核意见：{case.review.notes or '待人工确认。'}", ""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_V2)
    parser.add_argument("--additional-benchmark", type=Path, action="append", default=[])
    parser.add_argument("--pilot", action="store_true", help="require 20 development cases, 5 per category")
    parser.add_argument("--check-sources", action="store_true", help="verify the private local corpus and PDFs")
    parser.add_argument("--require-human-review", action="store_true")
    parser.add_argument("--review-output", type=Path, help="write a generated review sheet for --benchmark")
    args = parser.parse_args(argv)
    try:
        primary = load_benchmark_v2(args.benchmark)
        cases = list(primary)
        for path in args.additional_benchmark:
            cases.extend(load_benchmark_v2(path))
        validate_collection(cases)
        if args.pilot:
            validate_pilot(primary)
        sources = validate_sources(cases) if args.check_sources else None
        if args.require_human_review:
            require_approved(cases)
        if args.review_output is not None:
            args.review_output.parent.mkdir(parents=True, exist_ok=True)
            args.review_output.write_text(render_review(primary, file_sha256(args.benchmark)), encoding="utf-8")
        print(json.dumps({
            "cases": len(cases), "categories": dict(Counter(case.category for case in cases)),
            "human_review": dict(Counter(case.review.human_status for case in cases)),
            "source_checks": sources if sources is not None else "not_requested",
            "review_output": str(args.review_output) if args.review_output else None,
        }, ensure_ascii=False, indent=2))
        return 0
    except (BenchmarkV2Error, OSError) as exc:
        print(f"benchmark validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
