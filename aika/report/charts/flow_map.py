"""Evidence-weighted supply chain map data."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from aika.report.charts.common import dedupe, evidence_id, text_attr
from aika.report.spec import FlowMapLinkSpec, FlowMapNodeSpec, FlowMapSpec


RELATION_WEIGHTS = {
    "HAS_PRODUCT": 1.2,
    "BELONGS_TO_CHAIN": 1.2,
    "PART_OF_CHAIN": 1.2,
    "SUPPLIES": 1.2,
    "USES_TECHNOLOGY": 1.1,
    "ENABLES": 1.1,
    "DEPENDS_ON": 1.1,
}


def build_flow_map(
    graph_records: Iterable[Any],
    *,
    evidence_cards: Iterable[Any] | None = None,
) -> FlowMapSpec:
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"value": 0.0, "evidence_ids": []})
    for index, record in enumerate(graph_records or [], start=1):
        source = text_attr(record, "source") or text_attr(record, "head_name") or text_attr(record, "company")
        target = text_attr(record, "target") or text_attr(record, "tail_name") or text_attr(record, "topic")
        if not source or not target or source == target:
            continue
        relation = text_attr(record, "relation")
        key = (source, target)
        grouped[key]["value"] += RELATION_WEIGHTS.get(relation, 1.0)
        record_id = evidence_id(record, fallback=f"G{index}")
        if record_id:
            grouped[key]["evidence_ids"].append(record_id)

    if not grouped:
        grouped.update(_links_from_cards(evidence_cards or []))

    labels = dedupe([label for source, target in grouped for label in (source, target)])
    nodes = [FlowMapNodeSpec(id=label, label=label) for label in labels]
    links = [
        FlowMapLinkSpec(
            source=source,
            target=target,
            value=round(payload["value"], 2),
            evidence_ids=dedupe(payload["evidence_ids"]),
        )
        for (source, target), payload in sorted(grouped.items())
    ]
    return FlowMapSpec(nodes=nodes, links=links)


def _links_from_cards(cards: Iterable[Any]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"value": 0.0, "evidence_ids": []})
    for card in cards:
        source = text_attr(card, "company")
        target = text_attr(card, "topic")
        if not source or not target or source == target:
            continue
        weight = 1.5 if text_attr(card, "exposure_level") in {"core", "direct"} else 1.0
        key = (source, target)
        grouped[key]["value"] += weight
        grouped[key]["evidence_ids"].append(evidence_id(card))
    return grouped
