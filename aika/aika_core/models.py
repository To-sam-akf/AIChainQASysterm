"""Stable data models and backend interface for AIKA Core."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, tuple | set):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError:
                loaded = []
            if isinstance(loaded, list):
                return [_clean_text(item) for item in loaded if _clean_text(item)]
        return [item.strip() for item in re.split(r"[;；,，、|]", text) if item.strip()]
    return [_clean_text(value)] if _clean_text(value) else []


def _floatish(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dict_from_any(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return dict(value.to_dict())
    if is_dataclass(value):
        return asdict(value)
    return {}


@dataclass(frozen=True)
class EvidenceCard:
    citation_id: str
    kind: str
    title: str
    evidence: str
    claim_id: str = ""
    source: str = ""
    page: str = ""
    section: str = ""
    company: str = ""
    relation: str = ""
    target: str = ""
    source_tier: str = ""
    score: float = 0.0
    reason: str = ""
    topic: str = ""
    claim_type: str = ""
    exposure_level: str = ""
    confidence: str = ""
    as_of_date: str = ""
    source_url: str = ""
    published_at: str = ""
    paragraph_id: str = ""
    freshness_status: str = ""
    counter_evidence_status: str = ""
    counter_evidence_summary: str = ""
    supported_conclusion_ids: list[str] = field(default_factory=list)
    contradicted_conclusion_ids: list[str] = field(default_factory=list)
    evidence_span: str = ""
    review_status: str = ""
    reviewer_note: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: Any) -> "EvidenceCard":
        if isinstance(value, cls):
            return value
        row = _dict_from_any(value)
        if not row:
            row = {field.name: getattr(value, field.name, "") for field in fields(cls) if hasattr(value, field.name)}
        as_of_date = _clean_text(row.get("as_of_date"))
        published_at = _clean_text(row.get("published_at")) or as_of_date
        source = _clean_text(row.get("source") or row.get("source_title"))
        title = _clean_text(row.get("title") or source)
        return cls(
            citation_id=_clean_text(row.get("citation_id")),
            kind=_clean_text(row.get("kind")),
            title=title,
            evidence=_clean_text(row.get("evidence") or row.get("text")),
            claim_id=_clean_text(row.get("claim_id")),
            source=source,
            page=_clean_text(row.get("page")),
            section=_clean_text(row.get("section")),
            company=_clean_text(row.get("company")),
            relation=_clean_text(row.get("relation")),
            target=_clean_text(row.get("target")),
            source_tier=_clean_text(row.get("source_tier")),
            score=_floatish(row.get("score")),
            reason=_clean_text(row.get("reason")),
            topic=_clean_text(row.get("topic")),
            claim_type=_clean_text(row.get("claim_type")),
            exposure_level=_clean_text(row.get("exposure_level")),
            confidence=_clean_text(row.get("confidence")),
            as_of_date=as_of_date,
            source_url=_clean_text(row.get("source_url") or row.get("url")),
            published_at=published_at,
            paragraph_id=_clean_text(row.get("paragraph_id") or row.get("paragraph")),
            freshness_status=_clean_text(row.get("freshness_status")),
            counter_evidence_status=_clean_text(row.get("counter_evidence_status")),
            counter_evidence_summary=_clean_text(row.get("counter_evidence_summary")),
            supported_conclusion_ids=_as_list(row.get("supported_conclusion_ids")),
            contradicted_conclusion_ids=_as_list(row.get("contradicted_conclusion_ids")),
            evidence_span=_clean_text(row.get("evidence_span")),
            review_status=_clean_text(row.get("review_status")),
            reviewer_note=_clean_text(row.get("reviewer_note")),
            raw=row,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConclusionCard:
    conclusion_id: str
    conclusion_text: str
    conclusion_type: str = "fact"
    confidence: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    counter_evidence_ids: list[str] = field(default_factory=list)
    evidence_status: str = "insufficient"
    counter_evidence_status: str = "unknown"
    counter_evidence_summary: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceLink:
    conclusion_id: str
    evidence_id: str
    support_type: str = "supports"
    rationale: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    claim_type: str
    topic: str
    claim_text: str
    companies: list[str] = field(default_factory=list)
    mechanism: str = ""
    direction: str = ""
    horizon: str = ""
    metric: str = ""
    value: str = ""
    unit: str = ""
    source_report_id: str = ""
    source_title: str = ""
    page: str = ""
    section: str = ""
    source_tier: str = ""
    evidence_span: str = ""
    confidence: str = ""
    as_of_date: str = ""
    exposure_level: str = ""
    review_status: str = ""
    reviewer_note: str = ""
    quality_flags: str = ""
    conflict_group_id: str = ""
    score: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any], *, score: float = 0.0) -> "ClaimRecord":
        return cls(
            claim_id=_clean_text(row.get("claim_id")),
            claim_type=_clean_text(row.get("claim_type")),
            topic=_clean_text(row.get("topic")),
            claim_text=_clean_text(row.get("claim_text") or row.get("text")),
            companies=_as_list(row.get("companies") or row.get("company")),
            mechanism=_clean_text(row.get("mechanism")),
            direction=_clean_text(row.get("direction")),
            horizon=_clean_text(row.get("horizon")),
            metric=_clean_text(row.get("metric")),
            value=_clean_text(row.get("value")),
            unit=_clean_text(row.get("unit")),
            source_report_id=_clean_text(row.get("source_report_id")),
            source_title=_clean_text(row.get("source_title") or row.get("source")),
            page=_clean_text(row.get("page")),
            section=_clean_text(row.get("section")),
            source_tier=_clean_text(row.get("source_tier")),
            evidence_span=_clean_text(row.get("evidence_span")),
            confidence=_clean_text(row.get("confidence")),
            as_of_date=_clean_text(row.get("as_of_date")),
            exposure_level=_clean_text(row.get("exposure_level")),
            review_status=_clean_text(row.get("review_status")),
            reviewer_note=_clean_text(row.get("reviewer_note")),
            quality_flags=_clean_text(row.get("quality_flags")),
            conflict_group_id=_clean_text(row.get("conflict_group_id")),
            score=_floatish(row.get("score")) or score,
            raw=dict(row),
        )

    @classmethod
    def from_research_hit(cls, hit: Any) -> "ClaimRecord":
        row = _dict_from_any(hit)
        if not row:
            row = {
                "claim_id": getattr(hit, "claim_id", ""),
                "claim_type": getattr(hit, "claim_type", ""),
                "topic": getattr(hit, "topic", ""),
                "claim_text": getattr(hit, "text", ""),
                "companies": [getattr(hit, "company", "")] if getattr(hit, "company", "") else [],
                "source_title": getattr(hit, "source", ""),
                "page": getattr(hit, "page", ""),
                "section": getattr(hit, "section", ""),
                "source_tier": getattr(hit, "source_tier", ""),
                "evidence_span": getattr(hit, "evidence_span", ""),
                "confidence": getattr(hit, "confidence", ""),
                "as_of_date": getattr(hit, "as_of_date", ""),
                "exposure_level": getattr(hit, "exposure_level", ""),
                "review_status": getattr(hit, "review_status", ""),
                "reviewer_note": getattr(hit, "reviewer_note", ""),
                "score": getattr(hit, "score", 0.0),
            }
        return cls.from_row(row, score=_floatish(row.get("score")))

    def to_evidence_card(self, *, citation_id: str = "") -> EvidenceCard:
        return EvidenceCard(
            citation_id=citation_id,
            kind="claim",
            title=f"{self.topic} {self.claim_type}".strip(),
            evidence=self.claim_text,
            claim_id=self.claim_id,
            source=self.source_title,
            page=self.page,
            section=self.section,
            company=self.companies[0] if self.companies else "",
            source_tier=self.source_tier,
            score=self.score,
            reason="curated claim",
            topic=self.topic,
            claim_type=self.claim_type,
            exposure_level=self.exposure_level,
            confidence=self.confidence,
            as_of_date=self.as_of_date,
            published_at=self.as_of_date,
            evidence_span=self.evidence_span,
            review_status=self.review_status,
            reviewer_note=self.reviewer_note,
            raw=self.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    name: str
    type: str
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    label: str = ""
    source_type: str = ""
    target_type: str = ""
    evidence: str = ""
    source_title: str = ""
    page: str = ""
    section: str = ""
    source_tier: str = ""
    report_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceGap:
    gap: str
    priority: str = "中"
    suggested_source: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Any) -> "EvidenceGap":
        if isinstance(row, cls):
            return row
        data = _dict_from_any(row)
        return cls(
            gap=_clean_text(data.get("gap") or data.get("reason")),
            priority=_clean_text(data.get("priority")) or "中",
            suggested_source=_clean_text(data.get("suggested_source")),
            raw=data,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompanyProfile:
    company: str
    topic: str = ""
    summary: str = ""
    business_position: str = ""
    technology_products: list[str] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)
    evidence_cards: list[EvidenceCard] = field(default_factory=list)
    graph_edges: list[GraphEdge] = field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = field(default_factory=list)
    research_outputs: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompanyComparison:
    companies: list[str]
    topic: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    evidence_cards: list[EvidenceCard] = field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = field(default_factory=list)
    research_outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchBrief:
    title: str
    markdown: str
    sections: list[dict[str, Any]] = field(default_factory=list)
    evidence_cards: list[EvidenceCard] = field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    research_outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchBackend(ABC):
    """Backend contract shared by CLI, MCP, web, and future storage engines."""

    @abstractmethod
    def search_evidence(self, query: str, *, top_k: int = 8, **filters: Any) -> list[EvidenceCard]:
        raise NotImplementedError

    @abstractmethod
    def search_claims(self, query: str, *, top_k: int = 8, **filters: Any) -> list[ClaimRecord]:
        raise NotImplementedError

    @abstractmethod
    def query_graph(
        self,
        *,
        company: str = "",
        technology: str = "",
        relation_type: str = "",
        limit: int = 80,
    ) -> list[GraphEdge]:
        raise NotImplementedError

    @abstractmethod
    def get_company_profile(self, company: str, *, topic: str = "") -> CompanyProfile:
        raise NotImplementedError
