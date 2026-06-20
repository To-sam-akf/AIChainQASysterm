"""Markdown renderer for ReportSpec."""

from __future__ import annotations

from typing import Any

from aika.report.spec import ReportSpec


def render_markdown(spec: ReportSpec | dict[str, Any]) -> str:
    active = _ensure_spec(spec)
    return "\n\n".join(f"## {section['title']}\n{section['content']}" for section in render_markdown_sections(active))


def render_markdown_sections(spec: ReportSpec | dict[str, Any]) -> list[dict[str, str]]:
    active = _ensure_spec(spec)
    return [
        {"title": "一页结论", "content": _executive_page(active)},
        {"title": "证据覆盖审计", "content": _coverage_audit(active)},
        {"title": "核心判断", "content": _core_judgment(active)},
        {"title": "证据缺口", "content": _gaps(active)},
        {"title": "图表", "content": _charts(active)},
        {"title": "公司对比", "content": _company_table(active)},
        {"title": "风险清单", "content": _risks(active)},
        {"title": "证据摘要", "content": _evidence_summary(active, limit=8)},
        {"title": "证据附录", "content": _evidence_appendix(active, limit=8)},
    ]


def _executive_page(spec: ReportSpec) -> str:
    lines = ["### 本次报告能回答什么"]
    lines.extend(f"- {item}" for item in (spec.executive_summary.can_answer or ["当前证据不足，无法形成可验证结论。"])[:4])
    lines.extend(["", "### 本次报告不能回答什么"])
    lines.extend(f"- {item}" for item in (spec.executive_summary.cannot_answer or ["当前证据不足。"])[:5])
    lines.extend(
        [
            "",
            "### 当前报告适合的使用方式",
            f"- 研究主题：{spec.topic}",
            f"- Report Type: `{spec.report_type}`（{spec.report_type_label}）",
            f"- Coverage: {spec.coverage.coverage_score:.0%}；Direct Claim Ratio: {spec.coverage.direct_claim_ratio:.0%}",
            f"- Conclusion Usability: {spec.usability}",
        ]
    )
    if spec.report_type == "evidence_coverage_audit":
        lines.append("- 覆盖审计提醒：当前证据覆盖不足，本报告只适合识别证据边界和补数方向，不应当作为完整深度结论使用。")
    return "\n".join(lines)


def _coverage_audit(spec: ReportSpec) -> str:
    coverage = spec.coverage
    lines = [
        f"- Report Type: `{spec.report_type}`（{spec.report_type_label}）",
        f"- Coverage: {coverage.coverage_score:.0%}",
        f"- Company Coverage: {coverage.company_coverage:.0%}",
        f"- Direct Claim Ratio: {coverage.direct_claim_ratio:.0%}",
        f"- Direct Claims: {coverage.direct_claims}",
        f"- Indirect Claims: {coverage.indirect_claims}",
        f"- Mentioned Claims: {coverage.mentioned_claims}",
        f"- Unsupported Claims: {coverage.unsupported_claims}",
        f"- Covered Companies: {coverage.covered_companies}/{coverage.target_companies or coverage.covered_companies}",
        f"- Source Freshness: {coverage.freshness_status}",
        f"- Conclusion Usability: {spec.usability}",
    ]
    if spec.report_type == "evidence_coverage_audit":
        lines.append("- 覆盖审计提醒：证据覆盖低或直接证据占比不足，系统已自动降级为覆盖审计报告。")
    return "\n".join(lines)


def _core_judgment(spec: ReportSpec) -> str:
    findings = spec.executive_summary.key_findings or spec.executive_summary.can_answer
    return findings[0] if findings else "当前证据不足。"


def _gaps(spec: ReportSpec) -> str:
    gaps = spec.appendix.evidence_gaps
    if not gaps:
        return "当前证据包未识别出关键缺口。"
    return "\n".join(f"- {gap.get('gap', '')} 建议：{gap.get('suggested_source', '')}" for gap in gaps[:8])


def _charts(spec: ReportSpec) -> str:
    heatmap = spec.charts.company_coverage_heatmap
    bar = spec.charts.evidence_strength_bar
    flow = spec.charts.flow_map
    timeline = spec.charts.source_freshness_timeline
    lines = ["### 公司覆盖热力图"]
    lines.append(_heatmap_table(spec) if heatmap.cells else heatmap.empty_message)
    lines.extend(["", "### 证据强度柱状图"])
    if any(value for value in bar.counts.values()):
        lines.extend(f"- {key}: {value}" for key, value in bar.counts.items())
    else:
        lines.append(bar.empty_message)
    lines.extend(["", "### Evidence-weighted supply chain map"])
    if flow.links:
        lines.extend(f"- {link.source} -> {link.target}: {link.value:g}（证据：{'、'.join(link.evidence_ids) or '未标注'}）" for link in flow.links)
        lines.append(f"- 图注：{flow.caption}")
    else:
        lines.append(flow.empty_message)
    lines.extend(["", "### Source freshness timeline"])
    if timeline.items:
        lines.extend(f"- {item.year}: {item.count}（{item.freshness}）" for item in timeline.items)
    else:
        lines.append(timeline.empty_message)
    return "\n".join(lines)


def _heatmap_table(spec: ReportSpec) -> str:
    heatmap = spec.charts.company_coverage_heatmap
    if not heatmap.rows or not heatmap.columns:
        return heatmap.empty_message
    lookup = {(cell.company, cell.segment): cell for cell in heatmap.cells}
    lines = ["| 公司 | " + " | ".join(heatmap.columns) + " |", "| --- | " + " | ".join("---" for _ in heatmap.columns) + " |"]
    for company in heatmap.rows:
        scores = [str(lookup.get((company, segment)).score if lookup.get((company, segment)) else 0) for segment in heatmap.columns]
        lines.append("| " + " | ".join([company, *scores]) + " |")
    return "\n".join(lines)


def _company_table(spec: ReportSpec) -> str:
    table = spec.appendix.company_compare_table or {}
    rows = table.get("rows") or []
    if not rows:
        return "当前证据不足以生成公司对比表。"
    lines = ["| 公司 | 环节 | 敞口 | 业务证据 | 指标 | 风险 | 证据 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows[:12]:
        lines.append(
            "| {company} | {chain_segment} | {exposure_level} | {business_evidence} | {leading_indicators} | {risks} | {citations} |".format(
                company=_cell(row.get("company")),
                chain_segment=_cell(row.get("chain_segment")),
                exposure_level=_cell(row.get("exposure_level")),
                business_evidence=_cell(row.get("business_evidence")),
                leading_indicators=_cell(row.get("leading_indicators")),
                risks=_cell(row.get("risks")),
                citations=_cell(row.get("citations")),
            )
        )
    return "\n".join(lines)


def _risks(spec: ReportSpec) -> str:
    rows = spec.appendix.risk_checklist or []
    if not rows:
        return "当前证据不足以生成风险清单。"
    return "\n".join(
        f"- {row.get('scope', '主题')}：{row.get('risk', '')}（优先级：{row.get('priority', '')}，跟踪：{row.get('follow_up', '')}）"
        + (f" [{row.get('citation_id')}]" if row.get("citation_id") else "")
        for row in rows[:8]
    )


def _evidence_summary(spec: ReportSpec, *, limit: int) -> str:
    cards = spec.appendix.evidence_cards
    if not cards:
        return "当前证据不足。"
    return "\n".join(
        f"- [{card.get('citation_id') or card.get('evidence_id') or 'E?'}] 来源：{card.get('source') or card.get('source_title') or '未标注'}；"
        f"claim 类型：{card.get('claim_type') or 'unknown'}；敞口：{card.get('exposure_level') or '未分级'}；"
        f"证据：{_short_text(card.get('evidence'), 120)}"
        for card in cards[:limit]
    )


def _evidence_appendix(spec: ReportSpec, *, limit: int) -> str:
    cards = spec.appendix.evidence_cards
    if not cards:
        return "当前证据不足。"
    blocks = []
    for card in cards[:limit]:
        citation = card.get("citation_id") or card.get("evidence_id") or "E?"
        source = card.get("source") or card.get("source_title") or "未标注"
        blocks.append(
            "\n".join(
                [
                    f"### [{citation}] {source}",
                    f"- 来源：{source}",
                    f"- Claim 类型：{card.get('claim_type') or 'unknown'}",
                    f"- 敞口级别：{card.get('exposure_level') or '未分级'}",
                    f"- 证据原文：{_short_text(card.get('evidence'), 360) or '未提供。'}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _ensure_spec(spec: ReportSpec | dict[str, Any]) -> ReportSpec:
    return spec if isinstance(spec, ReportSpec) else ReportSpec.model_validate(spec)


def _cell(value: Any) -> str:
    text = str(value or "当前证据不足").replace("|", "\\|")
    return text


def _short_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."
