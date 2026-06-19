from __future__ import annotations

from datetime import date

from aika.aika_core.evidence import build_evidence_ux_bundle, freshness_status
from aika.aika_core.models import ClaimRecord, EvidenceCard


def test_freshness_status_classifies_dates() -> None:
    today = date(2026, 6, 19)

    assert freshness_status("", today=today) == "unknown"
    assert freshness_status("2020", today=today) == "stale"
    assert freshness_status("2025", today=today) in {"fresh", "aging"}


def test_build_evidence_ux_bundle_links_claim_conclusions_to_cards() -> None:
    claim = ClaimRecord(
        claim_id="claim_1",
        claim_type="mechanism",
        topic="液冷",
        claim_text="液冷需求受到高功率密度服务器散热约束推动。",
        source_title="测试报告",
        page="5",
        evidence_span="高功率密度服务器需要更强散热能力。",
        confidence="0.82",
        as_of_date="2025",
    )
    card = claim.to_evidence_card(citation_id="E1")

    bundle = build_evidence_ux_bundle("液冷产业链", [card], claims=[claim])

    assert bundle["conclusions"]
    assert bundle["evidence_cards"]
    assert bundle["evidence_links"]
    conclusion = bundle["conclusions"][0]
    evidence_ids = {card["evidence_id"] for card in bundle["evidence_cards"]}
    assert conclusion["evidence_status"] == "supported"
    assert conclusion["evidence_ids"]
    assert set(conclusion["evidence_ids"]).issubset(evidence_ids)
    assert all(link["evidence_id"] in evidence_ids for link in bundle["evidence_links"])


def test_build_evidence_ux_bundle_marks_counter_evidence() -> None:
    card = EvidenceCard(
        citation_id="E2",
        kind="claim",
        title="液冷 risk",
        evidence="液冷部署仍存在成本和交付节奏不确定性。",
        source="测试报告",
        page="9",
        topic="液冷",
        claim_type="risk",
        confidence="0.7",
        published_at="2024",
    )

    bundle = build_evidence_ux_bundle("液冷产业链", [card])

    assert bundle["counter_evidence"]
    assert bundle["evidence_cards"][0]["counter_evidence_status"] == "possible"
