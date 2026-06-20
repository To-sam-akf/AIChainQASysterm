"""Company coverage heatmap chart data."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from aika.domain_lexicon import company_segment
from aika.report.charts.common import dedupe, evidence_id, freshness_status, source_key, text_attr
from aika.report.spec import CompanyCoverageHeatmapSpec, HeatmapCellSpec


EXPOSURE_RANK = {"mentioned": 1, "indirect": 2, "direct": 3, "core": 3}


def build_company_coverage_heatmap(
    evidence_cards: Iterable[Any],
    *,
    target_companies: Iterable[str] | None = None,
    graph_records: Iterable[Any] | None = None,
) -> CompanyCoverageHeatmapSpec:
    cards = list(evidence_cards or [])
    records = list(graph_records or [])
    companies = dedupe([*(target_companies or []), *[_card_company(card) for card in cards], *[_record_company(record) for record in records]])
    segments = dedupe([_segment_for_card(card) for card in cards] + [_segment_for_record(record) for record in records])
    if not companies:
        companies = dedupe(_card_company(card) for card in cards)
    if not segments:
        segments = dedupe(_segment_for_card(card) for card in cards)

    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for card in cards:
        company = _card_company(card)
        segment = _segment_for_card(card)
        if company and segment:
            grouped[(company, segment)].append(card)

    cells: list[HeatmapCellSpec] = []
    for company in companies:
        for segment in segments:
            score, reason = _score_cards(grouped.get((company, segment), []))
            ids = dedupe(evidence_id(card) for card in grouped.get((company, segment), []))
            cells.append(HeatmapCellSpec(company=company, segment=segment, score=score, evidence_ids=ids, reason=reason))
    return CompanyCoverageHeatmapSpec(columns=segments, rows=companies, cells=cells)


def _card_company(card: Any) -> str:
    return text_attr(card, "company")


def _record_company(record: Any) -> str:
    return text_attr(record, "company") or text_attr(record, "source")


def _segment_for_card(card: Any) -> str:
    return text_attr(card, "chain_segment") or company_segment(_card_company(card)) or text_attr(card, "topic")


def _segment_for_record(record: Any) -> str:
    return text_attr(record, "chain_segment") or text_attr(record, "target") or text_attr(record, "topic")


def _score_cards(cards: list[Any]) -> tuple[int, str]:
    if not cards:
        return 0, "无证据"
    best = max(_base_score(card) for card in cards)
    source_count = len(dedupe(source_key(card) for card in cards))
    has_direct = any(_base_score(card) >= 3 for card in cards)
    has_fresh = any((text_attr(card, "freshness_status") or freshness_status(text_attr(card, "published_at") or text_attr(card, "as_of_date"))) == "fresh" for card in cards)
    has_metric = any(text_attr(card, "claim_type") == "indicator" or text_attr(card, "metric") or text_attr(card, "value") for card in cards)
    if has_direct and source_count >= 2 and has_fresh and has_metric:
        return 5, "direct + 多来源 + 新鲜 + 有指标"
    if has_direct and source_count >= 2:
        return 4, "direct + 多来源"
    if best >= 3:
        return 3, "direct"
    if best == 2:
        return 2, "indirect"
    return 1, "mentioned"


def _base_score(card: Any) -> int:
    exposure = text_attr(card, "exposure_level").casefold()
    claim_type = text_attr(card, "claim_type").casefold()
    if exposure in {"core", "direct"} or claim_type == "company_exposure":
        return 3
    if exposure == "indirect" or claim_type in {"mechanism", "indicator", "supply_chain", "trend", "policy"}:
        return 2
    return EXPOSURE_RANK.get(exposure, 1)
