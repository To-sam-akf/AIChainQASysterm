"""Source freshness timeline chart data."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from typing import Any, Iterable

from aika.report.charts.common import freshness_status, parse_year, text_attr
from aika.report.spec import FreshnessItemSpec, FreshnessTimelineSpec


FRESHNESS_ORDER = {"stale": 0, "aging": 1, "fresh": 2, "unknown": 3}


def build_source_freshness_timeline(
    evidence_cards: Iterable[Any],
    *,
    today: date | None = None,
) -> FreshnessTimelineSpec:
    statuses_by_year: dict[str, Counter[str]] = defaultdict(Counter)
    for card in evidence_cards or []:
        date_text = text_attr(card, "published_at") or text_attr(card, "as_of_date")
        year = parse_year(date_text)
        if not year:
            continue
        status = text_attr(card, "freshness_status") or freshness_status(date_text, today=today)
        statuses_by_year[year][status] += 1
    items = [
        FreshnessItemSpec(year=year, count=sum(counter.values()), freshness=_dominant_status(counter))
        for year, counter in sorted(statuses_by_year.items())
    ]
    return FreshnessTimelineSpec(items=items)


def _dominant_status(counter: Counter[str]) -> str:
    if not counter:
        return "unknown"
    return sorted(counter.items(), key=lambda item: (-item[1], FRESHNESS_ORDER.get(item[0], 99), item[0]))[0][0]
