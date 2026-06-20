"""Evidence strength chart data."""

from __future__ import annotations

from typing import Any, Iterable

from aika.report.charts.common import text_attr
from aika.report.spec import EvidenceStrengthBarSpec


def build_evidence_strength_bar(
    evidence_cards: Iterable[Any],
    *,
    unsupported_claims: int = 0,
    no_evidence: int = 0,
) -> EvidenceStrengthBarSpec:
    counts = {
        "direct": 0,
        "indirect": 0,
        "mentioned": 0,
        "unsupported": max(0, int(unsupported_claims or 0)),
        "no_evidence": max(0, int(no_evidence or 0)),
    }
    for card in evidence_cards or []:
        level = _coverage_level(card)
        if level in counts:
            counts[level] += 1
    return EvidenceStrengthBarSpec(counts=counts)


def _coverage_level(card: Any) -> str:
    exposure = text_attr(card, "exposure_level").casefold()
    claim_type = text_attr(card, "claim_type").casefold()
    if exposure in {"core", "direct"} or claim_type == "company_exposure":
        return "direct"
    if exposure == "indirect":
        return "indirect"
    if exposure == "mentioned":
        return "mentioned"
    if claim_type in {"mechanism", "indicator", "supply_chain", "trend", "policy"}:
        return "indirect"
    return "mentioned"
