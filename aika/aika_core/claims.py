"""Claim loading and lightweight CSV search helpers."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable

from aika.aika_core.data_paths import DEFAULT_CLAIMS_CSV
from aika.aika_core.models import ClaimRecord


def load_claims(path: str | Path = DEFAULT_CLAIMS_CSV) -> list[ClaimRecord]:
    claims_path = Path(path)
    if claims_path.is_dir():
        claims_path = claims_path / "claims.csv"
    if not claims_path.exists():
        return []
    with claims_path.open(newline="", encoding="utf-8") as file:
        return [ClaimRecord.from_row(row) for row in csv.DictReader(file)]


def search_claim_records(
    claims: Iterable[ClaimRecord],
    query: str,
    *,
    top_k: int = 8,
    **filters: Any,
) -> list[ClaimRecord]:
    limit = max(0, int(top_k))
    if limit == 0:
        return []
    scored = []
    for claim in claims:
        score = score_claim_record(claim, query, **filters)
        if score <= 0:
            continue
        scored.append((score, claim))
    scored.sort(key=lambda item: (-item[0], item[1].topic, item[1].claim_type, item[1].claim_id))
    return [
        ClaimRecord.from_row({**claim.to_dict(), "score": round(score, 4)}, score=round(score, 4))
        for score, claim in scored[:limit]
    ]


def score_claim_record(claim: ClaimRecord, query: str, **filters: Any) -> float:
    if not _passes_filters(claim, filters):
        return 0.0
    text = " ".join(
        (
            claim.claim_id,
            claim.claim_type,
            claim.topic,
            claim.claim_text,
            " ".join(claim.companies),
            claim.mechanism,
            claim.metric,
            claim.source_title,
            claim.section,
            claim.evidence_span,
        )
    )
    normalized_text = _normalize(text)
    normalized_query = _normalize(query)
    score = 0.0
    if normalized_query and normalized_query in normalized_text:
        score += 12.0
    for term in _query_terms(query):
        term_norm = _normalize(term)
        if term_norm and term_norm in normalized_text:
            score += 5.0 if len(term_norm) >= 2 else 1.0
    if claim.topic and _normalize(claim.topic) in normalized_query:
        score += 6.0
    for company in claim.companies:
        if company and _normalize(company) in normalized_query:
            score += 4.0
    score += {
        "company_exposure": 2.0,
        "mechanism": 1.6,
        "bottleneck": 1.6,
        "indicator": 1.4,
        "risk": 1.2,
    }.get(claim.claim_type, 0.4)
    if claim.source_tier == "1":
        score += 0.5
    try:
        score += min(float(claim.confidence), 1.0)
    except ValueError:
        score += 0.5
    if not query.strip() and filters:
        score += 1.0
    return score


def _passes_filters(claim: ClaimRecord, filters: dict[str, Any]) -> bool:
    companies = _filter_values(filters.get("company") or filters.get("companies"))
    if companies and not any(_contains_any(company, companies) for company in claim.companies):
        claim_text = f"{claim.claim_text} {claim.evidence_span}"
        if not _contains_any(claim_text, companies):
            return False
    topics = _filter_values(filters.get("topic") or filters.get("topics"))
    if topics and not _contains_any(claim.topic, topics):
        return False
    claim_types = _filter_values(filters.get("claim_type") or filters.get("claim_types"))
    if claim_types and claim.claim_type not in claim_types:
        return False
    source_tiers = _filter_values(filters.get("source_tier") or filters.get("source_tiers"))
    if source_tiers and claim.source_tier not in source_tiers:
        return False
    return True


def _query_terms(query: str) -> list[str]:
    text = str(query or "").strip()
    if not text:
        return []
    tokens = [token for token in re.split(r"\s+|[，,。；;？?]+", text) if token]
    if len(text) <= 12 and text not in tokens:
        tokens.append(text)
    return _dedupe(tokens)


def _filter_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _contains_any(text: str, values: Iterable[str]) -> bool:
    normalized = _normalize(text)
    return any(_normalize(value) in normalized or normalized in _normalize(value) for value in values if _normalize(value))


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _dedupe(values: Iterable[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        key = _normalize(value)
        if key and key not in seen:
            seen.add(key)
            output.append(str(value))
    return output
