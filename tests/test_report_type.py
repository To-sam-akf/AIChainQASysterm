from aika.report_type import classify_report_type


def test_classify_report_type_uses_coverage_and_direct_evidence_thresholds() -> None:
    assert classify_report_type(0.15, 1.0, 0.2) == "evidence_coverage_audit"
    assert classify_report_type(0.45, 1.0, 0.25) == "preliminary_research_brief"
    assert classify_report_type(0.75, 1.0, 0.25) == "industry_research_report"
    assert classify_report_type(0.85, 1.0, 0.25) == "deep_research_report"


def test_direct_evidence_shortfall_forces_audit_even_when_coverage_is_high() -> None:
    assert classify_report_type(0.85, 1.0, 0.24) == "evidence_coverage_audit"
