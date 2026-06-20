from __future__ import annotations

import pytest
from pydantic import ValidationError

from aika.report.spec import AppendixSpec, ChartsSpec, CoverageSpec, ExecutiveSummarySpec, ReportSpec


def test_report_spec_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        ReportSpec.model_validate({"topic": "液冷产业链"})


def test_appendix_evidence_cards_are_required_and_serializable() -> None:
    with pytest.raises(ValidationError):
        AppendixSpec.model_validate({})

    spec = ReportSpec(
        report_type="evidence_coverage_audit",
        topic="液冷产业链",
        title="液冷产业链证据覆盖审计报告",
        coverage=CoverageSpec(
            coverage_score=0.15,
            direct_claims=1,
            indirect_claims=1,
            unsupported_claims=2,
            covered_companies=1,
            target_companies=3,
        ),
        executive_summary=ExecutiveSummarySpec(can_answer=[], cannot_answer=[], key_findings=[]),
        charts=ChartsSpec(),
        appendix=AppendixSpec(evidence_cards=[{"citation_id": "E1", "evidence": "液冷证据"}]),
    )

    payload = spec.model_dump()
    assert payload["appendix"]["evidence_cards"][0]["citation_id"] == "E1"
    assert payload["coverage"]["coverage_score"] == 0.15
