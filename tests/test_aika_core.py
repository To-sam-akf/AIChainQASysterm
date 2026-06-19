import json

from aika.aika_core import CSVResearchBackend, ClaimRecord, ConclusionCard, EvidenceCard, EvidenceLink, GraphEdge
from aika.aika_core.claims import load_claims
from aika.aika_core.data_paths import DEFAULT_CLAIMS_CSV


def test_load_claims_from_curated_csv() -> None:
    claims = load_claims(DEFAULT_CLAIMS_CSV)

    assert claims
    assert all(isinstance(claim, ClaimRecord) for claim in claims[:5])
    assert claims[0].claim_id
    assert claims[0].claim_text


def test_csv_backend_search_claims_returns_structured_traceable_results() -> None:
    backend = CSVResearchBackend()

    results = backend.search_claims("液冷", top_k=5)

    assert 0 < len(results) <= 5
    for record in results:
        assert isinstance(record, ClaimRecord)
        assert record.claim_id
        assert record.claim_text
        assert record.source_title or record.source_report_id or record.evidence_span


def test_csv_backend_search_evidence_returns_cards() -> None:
    backend = CSVResearchBackend()

    cards = backend.search_evidence("液冷", top_k=4)

    assert 0 < len(cards) <= 4
    assert all(isinstance(card, EvidenceCard) for card in cards)
    assert all(card.citation_id for card in cards)
    assert all(card.evidence for card in cards)


def test_csv_backend_query_graph_by_company_returns_relation_edges() -> None:
    backend = CSVResearchBackend()

    edges = backend.query_graph(company="中际旭创", limit=20)

    assert edges
    assert all(isinstance(edge, GraphEdge) for edge in edges)
    assert any(edge.source == "中际旭创" for edge in edges)
    assert any(edge.relation for edge in edges)


def test_csv_backend_company_profile_contains_evidence_and_gap_fields() -> None:
    backend = CSVResearchBackend()

    profile = backend.get_company_profile("中际旭创")
    payload = profile.to_dict()

    assert profile.company == "中际旭创"
    assert "evidence_cards" in payload
    assert "risks" in payload
    assert "evidence_gaps" in payload
    assert profile.evidence_cards or profile.risks or profile.evidence_gaps


def test_evidence_card_from_any_populates_evidence_ux_fields() -> None:
    card = EvidenceCard.from_any(
        {
            "citation_id": "E1",
            "kind": "claim",
            "source_title": "测试报告",
            "text": "证据文本",
            "as_of_date": "2025",
            "supported_conclusion_ids": "C1",
        }
    )

    assert card.source == "测试报告"
    assert card.title == "测试报告"
    assert card.evidence == "证据文本"
    assert card.published_at == "2025"
    assert card.supported_conclusion_ids == ["C1"]


def test_claim_to_evidence_card_maps_as_of_date_to_published_at() -> None:
    claim = ClaimRecord(
        claim_id="claim_1",
        claim_type="mechanism",
        topic="液冷",
        claim_text="液冷需求受到散热约束推动。",
        as_of_date="2025",
    )

    card = claim.to_evidence_card(citation_id="E1")

    assert card.published_at == "2025"


def test_conclusion_and_evidence_link_are_json_serializable() -> None:
    conclusion = ConclusionCard(conclusion_id="C1", conclusion_text="结论", evidence_ids=["E1"])
    link = EvidenceLink(conclusion_id="C1", evidence_id="E1")

    encoded = json.dumps({"conclusion": conclusion.to_dict(), "link": link.to_dict()}, ensure_ascii=False)

    assert "C1" in encoded
    assert "E1" in encoded
