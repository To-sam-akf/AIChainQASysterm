"""Evidence-card normalization helpers."""

from __future__ import annotations

from typing import Any, Iterable

from src.aika_core.models import ClaimRecord, EvidenceCard


def standardize_evidence_card(value: Any) -> EvidenceCard:
    return EvidenceCard.from_any(value)


def standardize_evidence_cards(values: Iterable[Any]) -> list[EvidenceCard]:
    return [standardize_evidence_card(value) for value in values]


def evidence_cards_from_claims(claims: Iterable[ClaimRecord], *, start_index: int = 1) -> list[EvidenceCard]:
    return [claim.to_evidence_card(citation_id=f"E{index}") for index, claim in enumerate(claims, start=start_index)]
