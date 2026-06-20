"""Pydantic models for structured AIKA report output."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CoverageSpec(StrictReportModel):
    coverage_score: float
    direct_claims: int
    indirect_claims: int
    mentioned_claims: int = 0
    unsupported_claims: int
    covered_companies: int
    target_companies: int
    direct_claim_ratio: float = 0.0
    company_coverage: float = 0.0
    freshness_status: str = "unknown"


class ExecutiveSummarySpec(StrictReportModel):
    can_answer: list[str]
    cannot_answer: list[str]
    key_findings: list[str]


class HeatmapCellSpec(StrictReportModel):
    company: str
    segment: str
    score: int
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class CompanyCoverageHeatmapSpec(StrictReportModel):
    title: str = "公司覆盖热力图"
    columns: list[str] = Field(default_factory=list)
    rows: list[str] = Field(default_factory=list)
    cells: list[HeatmapCellSpec] = Field(default_factory=list)
    empty_message: str = "当前证据不足"


class EvidenceStrengthBarSpec(StrictReportModel):
    title: str = "证据强度柱状图"
    counts: dict[Literal["direct", "indirect", "mentioned", "unsupported", "no_evidence"], int] = Field(
        default_factory=lambda: {
            "direct": 0,
            "indirect": 0,
            "mentioned": 0,
            "unsupported": 0,
            "no_evidence": 0,
        }
    )
    empty_message: str = "当前证据不足"


class FlowMapNodeSpec(StrictReportModel):
    id: str
    label: str


class FlowMapLinkSpec(StrictReportModel):
    source: str
    target: str
    value: float
    evidence_ids: list[str] = Field(default_factory=list)


class FlowMapSpec(StrictReportModel):
    title: str = "Evidence-weighted supply chain map"
    nodes: list[FlowMapNodeSpec] = Field(default_factory=list)
    links: list[FlowMapLinkSpec] = Field(default_factory=list)
    caption: str = "线条粗细代表 AIKA 本地证据数量/强度，不代表市场规模或收入占比。"
    empty_message: str = "当前证据不足"


class FreshnessItemSpec(StrictReportModel):
    year: str
    count: int
    freshness: str


class FreshnessTimelineSpec(StrictReportModel):
    title: str = "Source freshness timeline"
    items: list[FreshnessItemSpec] = Field(default_factory=list)
    empty_message: str = "当前证据不足"


class ChartsSpec(StrictReportModel):
    flow_map: FlowMapSpec = Field(default_factory=FlowMapSpec)
    company_coverage_heatmap: CompanyCoverageHeatmapSpec = Field(default_factory=CompanyCoverageHeatmapSpec)
    evidence_strength_bar: EvidenceStrengthBarSpec = Field(default_factory=EvidenceStrengthBarSpec)
    source_freshness_timeline: FreshnessTimelineSpec = Field(default_factory=FreshnessTimelineSpec)


class AppendixSpec(StrictReportModel):
    evidence_cards: list[dict[str, Any]]
    evidence_gaps: list[dict[str, Any]] = Field(default_factory=list)
    company_compare_table: dict[str, Any] = Field(default_factory=dict)
    risk_checklist: list[dict[str, Any]] = Field(default_factory=list)


class ReportSpec(StrictReportModel):
    report_type: str
    topic: str
    title: str
    report_type_label: str = ""
    usability: str = ""
    coverage: CoverageSpec
    executive_summary: ExecutiveSummarySpec
    charts: ChartsSpec
    appendix: AppendixSpec
