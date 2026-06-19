"""Evidence-card normalization helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime
from typing import Any

from aika.aika_core.models import ClaimRecord, ConclusionCard, EvidenceCard, EvidenceLink, EvidenceGap


COUNTER_CLAIM_TYPES = {"risk", "bottleneck", "constraint", "uncertainty", "decline"}
COUNTER_TEXT_PATTERNS = (
    "风险",
    "瓶颈",
    "约束",
    "不确定",
    "下降",
    "承压",
    "替代",
    "反证",
    "conflict",
    "risk",
    "bottleneck",
    "constraint",
    "uncertainty",
    "decline",
)


def standardize_evidence_card(value: Any) -> EvidenceCard:
    return EvidenceCard.from_any(value)


def standardize_evidence_cards(values: Iterable[Any]) -> list[EvidenceCard]:
    return [standardize_evidence_card(value) for value in values]


def evidence_cards_from_claims(claims: Iterable[ClaimRecord], *, start_index: int = 1) -> list[EvidenceCard]:
    return [claim.to_evidence_card(citation_id=f"E{index}") for index, claim in enumerate(claims, start=start_index)]


def freshness_status(date_text: Any, today: date | datetime | str | None = None) -> str:
    source_date = _parse_date(date_text)
    if source_date is None:
        return "unknown"
    current = _parse_date(today) if today is not None else date.today()
    if current is None:
        current = date.today()
    months = (current.year - source_date.year) * 12 + (current.month - source_date.month)
    if current.day < source_date.day:
        months -= 1
    if months <= 18:
        return "fresh"
    if months <= 36:
        return "aging"
    return "stale"


def build_evidence_ux_bundle(
    subject: str,
    evidence_cards: Iterable[Any],
    *,
    claims: Iterable[Any] | None = None,
    gaps: Iterable[Any] | None = None,
    max_conclusions: int = 6,
    llm_counter_audit: Mapping[str, Any] | Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    cards = [_card_dict(card, index) for index, card in enumerate(evidence_cards or [], start=1)]
    claim_records = [claim if isinstance(claim, ClaimRecord) else ClaimRecord.from_row(_dict_from_any(claim)) for claim in list(claims or [])]
    counter_ids = _counter_evidence_ids(cards)
    llm_audit = _resolve_llm_audit(llm_counter_audit, subject=subject, cards=cards, claims=claim_records)
    counter_ids.update(_llm_counter_ids(llm_audit, cards))

    conclusions = _build_conclusions(
        subject=subject,
        cards=cards,
        claims=claim_records,
        counter_ids=counter_ids,
        max_conclusions=max_conclusions,
    )
    links = _build_links(conclusions, cards, counter_ids)
    _attach_conclusion_ids(cards, links)
    counter_evidence = [_counter_payload(card, llm_audit) for card in cards if card["evidence_id"] in counter_ids]
    if not counter_evidence and cards:
        counter_evidence = [
            {
                "status": "none",
                "summary": "当前样本未检出反证。",
                "source": "rules",
                "evidence_ids": [],
            }
        ]
    return {
        "conclusions": [conclusion.to_dict() for conclusion in conclusions],
        "evidence_cards": cards,
        "evidence_links": [link.to_dict() for link in links],
        "counter_evidence": counter_evidence,
        "evidence_ux_meta": {
            "subject": str(subject or "").strip(),
            "conclusion_count": len(conclusions),
            "evidence_count": len(cards),
            "gap_count": len(list(gaps or [])),
            "counter_evidence_count": len(counter_ids),
            "llm_counter_audit": _compact_llm_meta(llm_audit),
        },
    }


def _build_conclusions(
    *,
    subject: str,
    cards: list[dict[str, Any]],
    claims: list[ClaimRecord],
    counter_ids: set[str],
    max_conclusions: int,
) -> list[ConclusionCard]:
    conclusions: list[ConclusionCard] = []
    seen: set[str] = set()
    for claim in claims:
        text = claim.claim_text.strip()
        if not text or text in seen:
            continue
        evidence_ids = _matching_evidence_ids(claim, cards)
        counter_for_claim = [evidence_id for evidence_id in evidence_ids if evidence_id in counter_ids]
        conclusion_id = f"C{len(conclusions) + 1}"
        conclusions.append(
            ConclusionCard(
                conclusion_id=conclusion_id,
                conclusion_text=text,
                conclusion_type=_conclusion_type(claim.claim_type),
                confidence=claim.confidence,
                evidence_ids=evidence_ids,
                counter_evidence_ids=counter_for_claim,
                evidence_status="supported" if evidence_ids else "insufficient",
                counter_evidence_status="possible" if counter_for_claim else "none",
                counter_evidence_summary=_counter_summary(counter_for_claim),
            )
        )
        seen.add(text)
        if len(conclusions) >= max(1, int(max_conclusions or 1)):
            return conclusions

    for card in cards:
        text = str(card.get("evidence") or card.get("evidence_span") or "").strip()
        if not text or text in seen:
            continue
        evidence_id = str(card.get("evidence_id") or "")
        counter_for_card = [evidence_id] if evidence_id in counter_ids else []
        conclusion_id = f"C{len(conclusions) + 1}"
        conclusions.append(
            ConclusionCard(
                conclusion_id=conclusion_id,
                conclusion_text=text,
                conclusion_type=_conclusion_type(str(card.get("claim_type") or "")),
                confidence=str(card.get("confidence") or ""),
                evidence_ids=[evidence_id] if evidence_id else [],
                counter_evidence_ids=counter_for_card,
                evidence_status="supported" if evidence_id else "insufficient",
                counter_evidence_status="possible" if counter_for_card else "none",
                counter_evidence_summary=_counter_summary(counter_for_card),
            )
        )
        seen.add(text)
        if len(conclusions) >= max(1, int(max_conclusions or 1)):
            return conclusions

    if not conclusions:
        conclusions.append(
            ConclusionCard(
                conclusion_id="C1",
                conclusion_text=f"{str(subject or '当前问题').strip()} 当前证据不足，无法形成可验证结论。",
                conclusion_type="gap",
                evidence_status="insufficient",
                counter_evidence_status="unknown",
                counter_evidence_summary="缺少可审计证据卡片。",
            )
        )
    return conclusions


def _build_links(conclusions: list[ConclusionCard], cards: list[dict[str, Any]], counter_ids: set[str]) -> list[EvidenceLink]:
    available = {str(card.get("evidence_id") or "") for card in cards}
    links: list[EvidenceLink] = []
    for conclusion in conclusions:
        for evidence_id in conclusion.evidence_ids:
            if evidence_id not in available:
                continue
            support_type = "contradicts" if evidence_id in counter_ids else "supports"
            links.append(
                EvidenceLink(
                    conclusion_id=conclusion.conclusion_id,
                    evidence_id=evidence_id,
                    support_type=support_type,
                    rationale="risk/bottleneck evidence" if support_type == "contradicts" else "direct claim/evidence support",
                )
            )
    return links


def _card_dict(value: Any, index: int) -> dict[str, Any]:
    card = EvidenceCard.from_any(value).to_dict()
    published_at = str(card.get("published_at") or card.get("as_of_date") or "").strip()
    card["published_at"] = published_at
    card["freshness_status"] = card.get("freshness_status") or freshness_status(published_at)
    card["source_title"] = str(card.get("source") or card.get("title") or "").strip()
    card["evidence_id"] = str(card.get("citation_id") or card.get("claim_id") or f"UXE{index}").strip()
    card["counter_evidence_status"] = card.get("counter_evidence_status") or "none"
    card["counter_evidence_summary"] = card.get("counter_evidence_summary") or "当前样本未检出反证。"
    card["supported_conclusion_ids"] = list(card.get("supported_conclusion_ids") or [])
    card["contradicted_conclusion_ids"] = list(card.get("contradicted_conclusion_ids") or [])
    return card


def _matching_evidence_ids(claim: ClaimRecord, cards: list[dict[str, Any]]) -> list[str]:
    matches: list[str] = []
    for card in cards:
        evidence_id = str(card.get("evidence_id") or "")
        if not evidence_id:
            continue
        if claim.claim_id and claim.claim_id == str(card.get("claim_id") or ""):
            matches.append(evidence_id)
            continue
        if claim.claim_text and claim.claim_text == str(card.get("evidence") or ""):
            matches.append(evidence_id)
    if matches:
        return _dedupe(matches)
    for card in cards:
        evidence_id = str(card.get("evidence_id") or "")
        if evidence_id:
            return [evidence_id]
    return []


def _counter_evidence_ids(cards: list[dict[str, Any]]) -> set[str]:
    output: set[str] = set()
    for card in cards:
        claim_type = str(card.get("claim_type") or "").strip().casefold()
        text = " ".join(str(card.get(key) or "") for key in ("evidence", "evidence_span", "title", "topic")).casefold()
        if claim_type in COUNTER_CLAIM_TYPES or any(pattern.casefold() in text for pattern in COUNTER_TEXT_PATTERNS):
            evidence_id = str(card.get("evidence_id") or "")
            if evidence_id:
                card["counter_evidence_status"] = "possible"
                card["counter_evidence_summary"] = "规则识别到风险、瓶颈或不确定性证据。"
                output.add(evidence_id)
    return output


def _resolve_llm_audit(
    audit: Mapping[str, Any] | Callable[[dict[str, Any]], Mapping[str, Any]] | None,
    *,
    subject: str,
    cards: list[dict[str, Any]],
    claims: list[ClaimRecord],
) -> dict[str, Any]:
    if audit is None:
        return {"status": "not_configured", "source": "rules"}
    if isinstance(audit, Mapping):
        return dict(audit)
    try:
        return dict(audit({"subject": subject, "evidence_cards": cards, "claims": [claim.to_dict() for claim in claims]}))
    except Exception as exc:  # pragma: no cover - defensive user-supplied hook.
        return {"status": "error", "source": "llm", "error": str(exc)}


def _llm_counter_ids(audit: Mapping[str, Any], cards: list[dict[str, Any]]) -> set[str]:
    if str(audit.get("status") or "").casefold() not in {"completed", "ok", "pass", "warning"}:
        return set()
    card_ids = {str(card.get("evidence_id") or "") for card in cards}
    output: set[str] = set()
    for key in ("counter_evidence_ids", "contradicted_evidence_ids", "risk_evidence_ids"):
        value = audit.get(key)
        if isinstance(value, str):
            candidates = [item.strip() for item in re.split(r"[,，;；|、]", value) if item.strip()]
        elif isinstance(value, Iterable):
            candidates = [str(item).strip() for item in value if str(item).strip()]
        else:
            candidates = []
        output.update(item for item in candidates if item in card_ids)
    for card in cards:
        if str(card.get("evidence_id") or "") in output:
            card["counter_evidence_status"] = "possible"
            card["counter_evidence_summary"] = str(audit.get("summary") or "LLM 反证审计标记为可能反证。")
    return output


def _attach_conclusion_ids(cards: list[dict[str, Any]], links: list[EvidenceLink]) -> None:
    by_id = {str(card.get("evidence_id") or ""): card for card in cards}
    for link in links:
        card = by_id.get(link.evidence_id)
        if card is None:
            continue
        key = "contradicted_conclusion_ids" if link.support_type == "contradicts" else "supported_conclusion_ids"
        ids = list(card.get(key) or [])
        if link.conclusion_id not in ids:
            ids.append(link.conclusion_id)
        card[key] = ids


def _counter_payload(card: dict[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": card.get("counter_evidence_status") or "possible",
        "summary": card.get("counter_evidence_summary") or str(audit.get("summary") or "可能存在反证或风险边界。"),
        "source": "llm" if str(audit.get("source") or "") == "llm" else "rules",
        "evidence_id": card.get("evidence_id"),
        "citation_id": card.get("citation_id"),
        "evidence": card.get("evidence"),
        "claim_type": card.get("claim_type"),
        "source_title": card.get("source_title"),
        "page": card.get("page"),
    }


def _compact_llm_meta(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": str(audit.get("status") or "not_configured"),
        "source": str(audit.get("source") or "rules"),
        "error": str(audit.get("error") or ""),
        "summary": str(audit.get("summary") or ""),
    }


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    year_match = re.search(r"(19|20)\d{2}", text)
    if not year_match:
        return None
    year = int(year_match.group(0))
    month = 1
    day = 1
    month_match = re.search(r"(?:19|20)\d{2}[-/.年](\d{1,2})", text)
    if month_match:
        month = max(1, min(12, int(month_match.group(1))))
    day_match = re.search(r"(?:19|20)\d{2}[-/.年]\d{1,2}[-/.月](\d{1,2})", text)
    if day_match:
        day = max(1, min(28, int(day_match.group(1))))
    try:
        return date(year, month, day)
    except ValueError:
        return date(year, 1, 1)


def _conclusion_type(claim_type: str) -> str:
    value = str(claim_type or "").casefold()
    if value in {"risk", "bottleneck", "constraint", "uncertainty", "decline"}:
        return "risk"
    if value in {"indicator", "metric"}:
        return "indicator"
    if value in {"company_exposure", "supply_chain"}:
        return "fact"
    if value in {"trend", "mechanism"}:
        return value
    return "fact"


def _counter_summary(counter_ids: list[str]) -> str:
    if counter_ids:
        return "存在风险、瓶颈或不确定性证据，需要与正向结论一起阅读。"
    return "当前样本未检出反证。"


def _dict_from_any(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return dict(value.to_dict())
    return {}


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output
