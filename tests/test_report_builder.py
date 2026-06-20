from __future__ import annotations

from types import SimpleNamespace

from aika.report.builder import build_report_spec


def test_builder_downgrades_low_coverage_to_audit() -> None:
    spec = build_report_spec(
        question="液冷产业链有哪些上市公司？",
        plan=SimpleNamespace(topics=["液冷"], companies=["英维克", "申菱环境"]),
        evidence_cards=[
            {
                "citation_id": "E1",
                "company": "英维克",
                "topic": "液冷",
                "claim_type": "risk",
                "exposure_level": "mentioned",
                "evidence": "液冷部署存在成本和交付节奏不确定性。",
            }
        ],
        graph_records=[],
        gaps=[{"gap": "申菱环境缺少直接证据", "priority": "高"}],
        verification={"checks": {"unsupported_terms": ["完整产业链排序"]}},
    )

    assert spec.report_type == "evidence_coverage_audit"
    assert spec.coverage.covered_companies == 1
    assert spec.coverage.target_companies == 2
    assert spec.coverage.unsupported_claims == 2


def test_builder_counts_claim_strength_and_companies_without_markdown() -> None:
    spec = build_report_spec(
        question="液冷产业链",
        plan=SimpleNamespace(topics=["液冷"], companies=["英维克", "申菱环境", "高澜股份"]),
        evidence_cards=[
            {"citation_id": "E1", "company": "英维克", "topic": "液冷", "claim_type": "company_exposure", "exposure_level": "direct", "evidence": "直接证据"},
            {"citation_id": "E2", "company": "申菱环境", "topic": "液冷", "claim_type": "mechanism", "exposure_level": "indirect", "evidence": "间接证据"},
            {"citation_id": "E3", "company": "英维克", "topic": "液冷", "claim_type": "risk", "exposure_level": "mentioned", "evidence": "提及证据"},
        ],
        graph_records=[],
        gaps=[{"gap": "缺少指标证据", "priority": "高"}],
        verification={"checks": {}},
    )

    assert spec.coverage.direct_claims == 1
    assert spec.coverage.indirect_claims == 1
    assert spec.coverage.mentioned_claims == 1
    assert spec.coverage.covered_companies == 2
    assert spec.coverage.target_companies == 3
    assert "markdown" not in spec.model_dump()
