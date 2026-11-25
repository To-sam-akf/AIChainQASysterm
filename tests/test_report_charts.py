from __future__ import annotations

from datetime import date

from aika.report.charts.evidence_bar import build_evidence_strength_bar
from aika.report.charts.flow_map import build_flow_map
from aika.report.charts.freshness import build_source_freshness_timeline
from aika.report.charts.heatmap import build_company_coverage_heatmap


def card(**overrides: object) -> dict[str, object]:
    base = {
        "citation_id": "E1",
        "company": "英维克",
        "topic": "液冷",
        "claim_type": "company_exposure",
        "exposure_level": "direct",
        "source": "报告A",
        "published_at": "2025",
        "evidence": "液冷证据",
    }
    base.update(overrides)
    return base


def test_company_coverage_heatmap_scores_evidence_strength() -> None:
    heatmap = build_company_coverage_heatmap(
        [
            card(citation_id="E1", company="直接公司", exposure_level="direct", source="报告A"),
            card(citation_id="E2", company="提及公司", exposure_level="mentioned", claim_type="risk"),
            card(citation_id="E3", company="加强公司", exposure_level="direct", source="报告A", published_at="2025", freshness_status="fresh"),
            card(citation_id="E4", company="加强公司", exposure_level="indirect", claim_type="indicator", source="报告B", published_at="2025"),
        ],
        target_companies=["直接公司", "提及公司", "加强公司", "未覆盖公司"],
    )

    scores = {(cell.company, cell.segment): cell.score for cell in heatmap.cells}
    assert scores[("未覆盖公司", "液冷")] == 0
    assert scores[("直接公司", "液冷")] > scores[("提及公司", "液冷")]
    assert scores[("加强公司", "液冷")] == 5


def test_evidence_strength_bar_counts_levels() -> None:
    bar = build_evidence_strength_bar(
        [
            card(citation_id="E1", exposure_level="direct"),
            card(citation_id="E2", exposure_level="indirect", claim_type="mechanism"),
            card(citation_id="E3", exposure_level="mentioned", claim_type="risk"),
        ],
        unsupported_claims=2,
        no_evidence=4,
    )

    assert bar.counts == {"direct": 1, "indirect": 1, "mentioned": 1, "unsupported": 2, "no_evidence": 4}


def test_company_exposure_keeps_explicit_indirect_and_mentioned_levels() -> None:
    cards = [card(company="间接公司", exposure_level="indirect"), card(company="提及公司", exposure_level="mentioned")]
    counts = build_evidence_strength_bar(cards).counts
    assert (counts["direct"], counts["indirect"], counts["mentioned"]) == (0, 1, 1)
    heatmap = build_company_coverage_heatmap(cards)
    scores = {(cell.company, cell.segment): cell.score for cell in heatmap.cells}
    assert scores[("间接公司", "液冷")] == 2
    assert scores[("提及公司", "液冷")] == 1


def test_flow_map_prefers_graph_entity_to_document_source() -> None:
    flow = build_flow_map([{"source": "2025年年度报告", "head_name": "浪潮信息", "company": "浪潮信息", "target": "液冷服务器", "relation": "HAS_PRODUCT"}])
    assert flow.links[0].source == "浪潮信息"


def test_flow_map_links_include_evidence_ids_and_caption() -> None:
    flow = build_flow_map(
        [
            {"source": "AI芯片/高功率封装", "target": "AI服务器/超节点", "relation": "ENABLES", "evidence_id": "G1"},
            {"source": "AI芯片/高功率封装", "target": "AI服务器/超节点", "relation": "HAS_PRODUCT", "evidence_id": "G2"},
        ]
    )

    assert flow.links
    link = flow.links[0]
    assert link.source == "AI芯片/高功率封装"
    assert link.target == "AI服务器/超节点"
    assert link.value > 1
    assert link.evidence_ids == ["G1", "G2"]
    assert "不代表市场规模或收入占比" in flow.caption


def test_source_freshness_timeline_aggregates_years() -> None:
    timeline = build_source_freshness_timeline(
        [
            card(citation_id="E1", published_at="2022-01-01"),
            card(citation_id="E2", published_at="2024-01-01"),
            card(citation_id="E3", published_at="2025-12-01"),
            card(citation_id="E4", published_at="2025-03-01"),
        ],
        today=date(2026, 6, 20),
    )

    rows = {item.year: item for item in timeline.items}
    assert rows["2022"].count == 1
    assert rows["2022"].freshness == "stale"
    assert rows["2024"].freshness == "aging"
    assert rows["2025"].count == 2
    assert rows["2025"].freshness == "fresh"
