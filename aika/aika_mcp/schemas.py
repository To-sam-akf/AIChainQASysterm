"""Stable request schemas for AIKA MCP tools."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aika.aika_core.backends.sqlite_backend import DEFAULT_PROFILE


BackendMode = Literal["auto", "sqlite", "csv"]
TaskType = Literal["research_brief"]
Depth = Literal["light", "standard", "deep"]


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,，;；|、]", value) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [_clean_text(item) for item in value if _clean_text(item)]
    return [_clean_text(value)] if _clean_text(value) else []


def _non_empty(value: Any, *, field_name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


class AikaMcpRequest(BaseModel):
    """Common backend-selection fields shared by all MCP tools."""

    model_config = ConfigDict(extra="forbid")

    backend: BackendMode = Field(default="auto", description="Backend mode: auto, sqlite, or csv.")
    home: str | None = Field(default=None, description="Optional AIKA home directory for SQLite indexes.")
    profile: str = Field(default=DEFAULT_PROFILE, description="AIKA knowledge profile name.")

    @field_validator("home", mode="before")
    @classmethod
    def normalize_home(cls, value: Any) -> str | None:
        text = _clean_text(value)
        return text or None

    @field_validator("profile", mode="before")
    @classmethod
    def normalize_profile(cls, value: Any) -> str:
        return _clean_text(value) or DEFAULT_PROFILE


class FilteredSearchRequest(AikaMcpRequest):
    query: str = Field(..., description="Search query.")
    top_k: int = Field(default=8, ge=0, le=50, description="Maximum number of records to return.")
    companies: list[str] = Field(default_factory=list, description="Optional company filters.")
    topics: list[str] = Field(default_factory=list, description="Optional topic filters.")
    claim_types: list[str] = Field(default_factory=list, description="Optional claim type filters.")

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: Any) -> str:
        return _non_empty(value, field_name="query")

    @field_validator("companies", "topics", "claim_types", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _string_list(value)


class SearchEvidenceRequest(FilteredSearchRequest):
    """Search local evidence spans and return citation-ready cards."""


class SearchClaimsRequest(FilteredSearchRequest):
    """Search local curated claims and return structured records."""


class CompanyProfileRequest(AikaMcpRequest):
    company: str = Field(..., description="Company name.")
    topic: str = Field(default="", description="Optional topic focus.")

    @field_validator("company", mode="before")
    @classmethod
    def normalize_company(cls, value: Any) -> str:
        return _non_empty(value, field_name="company")

    @field_validator("topic", mode="before")
    @classmethod
    def normalize_topic(cls, value: Any) -> str:
        return _clean_text(value)


class CompareCompaniesRequest(AikaMcpRequest):
    companies: list[str] = Field(..., description="Two or more company names.")
    topic: str = Field(default="", description="Optional topic focus.")

    @field_validator("companies", mode="before")
    @classmethod
    def normalize_companies(cls, value: Any) -> list[str]:
        return _string_list(value)

    @field_validator("topic", mode="before")
    @classmethod
    def normalize_topic(cls, value: Any) -> str:
        return _clean_text(value)

    @model_validator(mode="after")
    def require_two_companies(self) -> "CompareCompaniesRequest":
        if len(self.companies) < 2:
            raise ValueError("companies must contain at least two company names")
        return self


class QueryIndustryGraphRequest(AikaMcpRequest):
    company: str = Field(default="", description="Optional company filter.")
    technology: str = Field(default="", description="Optional technology or topic filter.")
    relation_type: str = Field(default="", description="Optional relation type filter.")
    limit: int = Field(default=80, ge=0, le=200, description="Maximum graph edges to return.")

    @field_validator("company", "technology", "relation_type", mode="before")
    @classmethod
    def normalize_filters(cls, value: Any) -> str:
        return _clean_text(value)


class BuildResearchBriefRequest(AikaMcpRequest):
    query: str = Field(default="", description="Brief query or research question.")
    topic: str = Field(default="", description="Optional topic focus.")

    @field_validator("query", "topic", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        return _clean_text(value)

    @model_validator(mode="after")
    def require_subject(self) -> "BuildResearchBriefRequest":
        if not self.query and not self.topic:
            raise ValueError("query or topic is required")
        return self


class AuditEvidenceGapsRequest(AikaMcpRequest):
    query: str = Field(default="", description="Question or subject to audit.")
    companies: list[str] = Field(default_factory=list, description="Optional company filters.")
    topic: str = Field(default="", description="Optional topic focus.")

    @field_validator("query", "topic", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        return _clean_text(value)

    @field_validator("companies", mode="before")
    @classmethod
    def normalize_companies(cls, value: Any) -> list[str]:
        return _string_list(value)


class RunResearchTaskRequest(AikaMcpRequest):
    task_type: TaskType = Field(default="research_brief", description="Research task type.")
    topic: str = Field(..., description="Research topic.")
    companies: list[str] = Field(default_factory=list, description="Optional company focus list.")
    depth: Depth = Field(default="standard", description="Retrieval depth.")
    require_citations: bool = Field(default=True, description="Whether citation coverage is required.")

    @field_validator("topic", mode="before")
    @classmethod
    def normalize_topic(cls, value: Any) -> str:
        return _non_empty(value, field_name="topic")

    @field_validator("companies", mode="before")
    @classmethod
    def normalize_companies(cls, value: Any) -> list[str]:
        return _string_list(value)


class RenderReportPdfRequest(AikaMcpRequest):
    report_spec: dict[str, Any] = Field(default_factory=dict, description="Optional structured ReportSpec to render.")
    query: str = Field(default="", description="Research question used when report_spec is not provided.")
    topic: str = Field(default="", description="Research topic used when report_spec is not provided.")
    output_dir: str = Field(default="", description="Optional directory for generated HTML and PDF files.")

    @field_validator("query", "topic", "output_dir", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        return _clean_text(value)

    @model_validator(mode="after")
    def require_report_or_subject(self) -> "RenderReportPdfRequest":
        if not self.report_spec and not self.query and not self.topic:
            raise ValueError("report_spec, query, or topic is required")
        return self


REQUEST_MODELS: dict[str, type[AikaMcpRequest]] = {
    "search_evidence": SearchEvidenceRequest,
    "search_claims": SearchClaimsRequest,
    "get_company_profile": CompanyProfileRequest,
    "compare_companies": CompareCompaniesRequest,
    "query_industry_graph": QueryIndustryGraphRequest,
    "build_research_brief": BuildResearchBriefRequest,
    "audit_evidence_gaps": AuditEvidenceGapsRequest,
    "run_research_task": RunResearchTaskRequest,
    "render_report_pdf": RenderReportPdfRequest,
}


def request_schema_catalog() -> dict[str, dict[str, Any]]:
    return {name: model.model_json_schema() for name, model in REQUEST_MODELS.items()}
