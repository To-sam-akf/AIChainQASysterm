"""Report type classification from evidence coverage metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any


REPORT_TYPE_LABELS = {
    "evidence_coverage_audit": "证据覆盖审计报告",
    "preliminary_research_brief": "初步研究简报",
    "industry_research_report": "分析报告",
    "deep_research_report": "深度研究报告",
}

REPORT_TYPE_USABILITY = {
    "evidence_coverage_audit": "Limited",
    "preliminary_research_brief": "Preliminary",
    "industry_research_report": "Usable with caveats",
    "deep_research_report": "High",
}


@dataclass(frozen=True)
class CoverageMetrics:
    coverage_score: float
    company_coverage: float
    direct_claim_ratio: float
    unsupported_claims: int = 0
    freshness_status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_report_type(coverage_score: float, company_coverage: float, direct_claim_ratio: float) -> str:
    del company_coverage
    coverage_score = _ratio(coverage_score)
    direct_claim_ratio = _ratio(direct_claim_ratio)
    if coverage_score < 0.3 or direct_claim_ratio < 0.25:
        return "evidence_coverage_audit"
    if coverage_score < 0.6:
        return "preliminary_research_brief"
    if coverage_score < 0.8:
        return "industry_research_report"
    return "deep_research_report"


def report_type_label(report_type: str) -> str:
    return REPORT_TYPE_LABELS.get(report_type, REPORT_TYPE_LABELS["evidence_coverage_audit"])


def report_usability(report_type: str) -> str:
    return REPORT_TYPE_USABILITY.get(report_type, REPORT_TYPE_USABILITY["evidence_coverage_audit"])


def report_title(subject: str, report_type: str) -> str:
    clean_subject = str(subject or "").strip() or "AI 算力产业链"
    return f"{clean_subject}{report_type_label(report_type)}"


def aggregate_freshness_status(values: list[Any]) -> str:
    statuses = [normalize_freshness_status(value) for value in values if normalize_freshness_status(value)]
    if not statuses:
        return "unknown"
    for status in ("stale", "aging", "fresh", "unknown"):
        if status in statuses:
            return status
    return "unknown"


def normalize_freshness_status(value: Any, *, today: date | None = None) -> str:
    text = str(value or "").strip().lower()
    if text in {"fresh", "aging", "stale", "unknown"}:
        return text
    parsed = _parse_date(text)
    if parsed is None:
        return "unknown"
    current = today or date.today()
    months = (current.year - parsed.year) * 12 + (current.month - parsed.month)
    if current.day < parsed.day:
        months -= 1
    if months <= 18:
        return "fresh"
    if months <= 36:
        return "aging"
    return "stale"


def _ratio(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _parse_date(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m", "%Y/%m", "%Y.%m", "%Y"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed.date()
    return None
