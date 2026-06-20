from __future__ import annotations

from pathlib import Path

from aika.report.render_html import render_html
from aika.report.render_markdown import render_markdown
from aika.report.render_pdf import render_report_pdf
from aika.report.spec import (
    AppendixSpec,
    ChartsSpec,
    CoverageSpec,
    ExecutiveSummarySpec,
    FlowMapLinkSpec,
    FlowMapNodeSpec,
    FlowMapSpec,
    FreshnessItemSpec,
    FreshnessTimelineSpec,
    ReportSpec,
)


def sample_spec() -> ReportSpec:
    return ReportSpec(
        report_type="evidence_coverage_audit",
        topic="液冷产业链",
        title="液冷产业链证据覆盖审计报告",
        report_type_label="证据覆盖审计报告",
        usability="Limited",
        coverage=CoverageSpec(
            coverage_score=0.15,
            direct_claims=0,
            indirect_claims=1,
            mentioned_claims=1,
            unsupported_claims=3,
            covered_companies=1,
            target_companies=9,
            direct_claim_ratio=0.0,
            company_coverage=1 / 9,
            freshness_status="aging",
        ),
        executive_summary=ExecutiveSummarySpec(
            can_answer=["当前证据只能支持以下有限判断：液冷存在成本不确定性 [E1]"],
            cannot_answer=["不能回答完整公司排序。"],
            key_findings=["当前证据只能支持以下有限判断：液冷存在成本不确定性 [E1]"],
        ),
        charts=ChartsSpec(),
        appendix=AppendixSpec(evidence_cards=[]),
    )


def test_markdown_renderer_outputs_executive_page() -> None:
    markdown = render_markdown(sample_spec())

    assert "## 一页结论" in markdown
    assert "## 证据覆盖审计" in markdown
    assert "当前证据不足" in markdown


def test_html_renderer_outputs_hierarchy_and_empty_chart_message() -> None:
    html = render_html(sample_spec())

    assert "<h1>液冷产业链证据覆盖审计报告</h1>" in html
    assert "<h2>证据覆盖审计</h2>" in html
    assert "<h2>核心判断</h2>" in html
    assert "<h2>Evidence Review</h2>" in html
    assert "公司覆盖热力图" in html
    assert "证据强度柱状图" in html
    assert "Evidence-weighted supply chain map" in html
    assert "Source freshness timeline" in html
    assert "当前证据不足" in html
    assert "覆盖审计提醒" in html


def test_html_renderer_outputs_flow_caption_when_chart_is_present() -> None:
    spec = sample_spec().model_copy(
        update={
            "charts": ChartsSpec(
                flow_map=FlowMapSpec(
                    nodes=[FlowMapNodeSpec(id="AI服务器", label="AI服务器"), FlowMapNodeSpec(id="液冷", label="液冷")],
                    links=[FlowMapLinkSpec(source="AI服务器", target="液冷", value=1.2, evidence_ids=["E1"])],
                ),
                source_freshness_timeline=FreshnessTimelineSpec(items=[FreshnessItemSpec(year="2025", count=2, freshness="fresh")]),
            )
        }
    )

    html = render_html(spec)

    assert "不代表市场规模或收入占比" in html
    assert "AI服务器 → 液冷" in html


def test_pdf_renderer_writes_html_and_invokes_playwright_export(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[Path, Path]] = []

    def fake_export(html_path: Path, pdf_path: Path) -> None:
        calls.append((html_path, pdf_path))
        pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr("aika.report.render_pdf._export_pdf_with_playwright", fake_export)

    result = render_report_pdf(sample_spec(), output_dir=tmp_path)

    assert Path(result["html_path"]).exists()
    assert Path(result["pdf_path"]).read_bytes().startswith(b"%PDF")
    assert calls == [(Path(result["html_path"]), Path(result["pdf_path"]))]
