"""Deterministic metrics for QA benchmark reports."""

from __future__ import annotations

import re
from typing import Any

from aika.eval.dataset import EvalCase


CITATION_PATTERN = re.compile(r"(?<![A-Za-z0-9])E\d+(?![A-Za-z0-9])")
REFUSAL_TERMS = ("当前知识库", "证据不足", "未找到", "无法给出", "不能", "不应", "缺少证据")
SENTENCE_SPLIT_PATTERN = re.compile(r"[\n。！？；;]+")
RELATION_CLAIM_TYPE_MAP = {
    "DISCLOSES_RISK": "risk",
    "HAS_METRIC": "indicator",
    "HAS_INDICATOR": "indicator",
    "HAS_EXPOSURE": "company_exposure",
    "HAS_PRODUCT": "company_exposure",
    "USES_TECHNOLOGY": "mechanism",
    "BELONGS_TO_CHAIN": "supply_chain",
    "CONSTRAINS": "bottleneck",
    "ENABLES": "mechanism",
    "DRIVES": "supply_chain",
    "DEPENDS_ON": "supply_chain",
}


def score_case(result: dict[str, Any], case: EvalCase, *, k: int = 6) -> dict[str, Any]:
    cards = list(result.get("evidence_cards") or result.get("evidence") or [])
    top_cards = cards[: max(1, int(k or 6))]
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    answer = str(result.get("answer") or "")
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    checks = verification.get("checks") if isinstance(verification.get("checks"), dict) else {}
    metrics = {
        "claim_recall@k": round(claim_recall_at_k(case, top_cards), 4),
        "evidence_precision@k": round(evidence_precision_at_k(case, top_cards), 4),
        "citation_validity": round(citation_validity(answer, top_cards, verification, case), 4),
        "answer_groundedness": round(answer_groundedness(answer, result, case), 4),
        "unsupported_claim_rate": round(unsupported_claim_rate(answer, diagnostics, checks), 4),
        "human_score": case.human_score,
    }
    auto_score = automatic_score(metrics)
    failures = failure_reasons(case, result, metrics)
    return {
        "metrics": metrics,
        "score": round(auto_score, 4),
        "status": "pass" if auto_score >= 0.72 and not failures else "warn" if auto_score >= 0.45 else "fail",
        "failures": failures,
        "evidence_gaps": collect_evidence_gaps(result),
    }


def claim_recall_at_k(case: EvalCase, cards: list[dict[str, Any]]) -> float:
    if case.refusal_expected:
        return 1.0 if answer_refusal_from_cards(cards) else 0.0
    if case.expected_claim_ids:
        found = {card_text(card, "claim_id") for card in cards}
        expected = set(case.expected_claim_ids)
        return len(expected & found) / max(len(expected), 1)
    slots = expected_slots(case)
    if not slots:
        return 1.0 if cards else 0.0
    hits = sum(1 for slot_type, value in slots if any(card_hits_slot(card, slot_type, value) for card in cards))
    return hits / len(slots)


def evidence_precision_at_k(case: EvalCase, cards: list[dict[str, Any]]) -> float:
    if not cards:
        return 1.0 if case.refusal_expected else 0.0
    if case.refusal_expected:
        return 1.0 if not cards or all(not is_forbidden_evidence(card, case) for card in cards) else 0.0
    return sum(1 for card in cards if card_matches_case(card, case)) / len(cards)


def citation_validity(answer: str, cards: list[dict[str, Any]], verification: dict[str, Any], case: EvalCase) -> float:
    checks = verification.get("checks") if isinstance(verification.get("checks"), dict) else {}
    citation_check = checks.get("citation_validity") if isinstance(checks.get("citation_validity"), dict) else {}
    missing = list(citation_check.get("missing_citations") or checks.get("missing_citations") or [])
    cited = set(CITATION_PATTERN.findall(answer))
    available = {card_text(card, "citation_id") for card in cards if card_text(card, "citation_id")}
    if missing or (cited - available):
        return 0.0
    if case.refusal_expected:
        return 1.0
    if cards and not cited:
        return 0.0
    return 1.0


def answer_groundedness(answer: str, result: dict[str, Any], case: EvalCase) -> float:
    if case.refusal_expected:
        has_refusal = contains_any(answer, REFUSAL_TERMS)
        has_forbidden = contains_any(answer, case.forbidden_terms)
        if has_refusal and not has_forbidden:
            return 1.0
        return 0.4 if has_refusal else 0.0
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    status = str(verification.get("status") or "").casefold()
    score = {"pass": 1.0, "warn": 0.6, "fail": 0.0}.get(status, 0.5)
    support_text = " ".join(
        [
            answer,
            *[card_full_text(card) for card in list(result.get("evidence_cards") or [])[:6]],
        ]
    )
    if case.expected_companies and not all(term in support_text for term in case.expected_companies):
        score -= 0.15
    if case.expected_topics and not any(term in support_text for term in case.expected_topics):
        score -= 0.1
    if contains_any(answer, case.forbidden_terms):
        score -= 0.2
    return max(0.0, min(1.0, score))


def unsupported_claim_rate(answer: str, diagnostics: dict[str, Any], checks: dict[str, Any]) -> float:
    unsupported_terms = list(diagnostics.get("unsupported_terms") or checks.get("unsupported_terms") or [])
    numeric_check = checks.get("numeric_support") if isinstance(checks.get("numeric_support"), dict) else {}
    unsupported_terms.extend(str(item) for item in list(numeric_check.get("unsupported") or []))
    unsupported_count = len({str(item) for item in unsupported_terms if str(item).strip()})
    return min(1.0, unsupported_count / max(claim_like_sentence_count(answer), 1))


def automatic_score(metrics: dict[str, Any]) -> float:
    values = [
        float(metrics.get("claim_recall@k") or 0.0),
        float(metrics.get("evidence_precision@k") or 0.0),
        float(metrics.get("citation_validity") or 0.0),
        float(metrics.get("answer_groundedness") or 0.0),
        1.0 - float(metrics.get("unsupported_claim_rate") or 0.0),
    ]
    return sum(values) / len(values)


def failure_reasons(case: EvalCase, result: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if case.expected_answer_type and not case.refusal_expected and result.get("answer_type") != case.expected_answer_type:
        failures.append(f"answer_type:{result.get('answer_type')}!=expected:{case.expected_answer_type}")
    if case.refusal_expected and not contains_any(str(result.get("answer") or ""), REFUSAL_TERMS):
        failures.append("refusal_expected_but_not_detected")
    for term in case.forbidden_terms:
        if term and term in str(result.get("answer") or ""):
            failures.append(f"forbidden_term:{term}")
    if float(metrics.get("claim_recall@k") or 0.0) < 0.5 and not case.refusal_expected:
        failures.append("low_claim_recall@k")
    if float(metrics.get("evidence_precision@k") or 0.0) < 0.4 and not case.refusal_expected:
        failures.append("low_evidence_precision@k")
    if float(metrics.get("citation_validity") or 0.0) < 1.0:
        failures.append("citation_invalid_or_missing")
    if float(metrics.get("answer_groundedness") or 0.0) < 0.6:
        failures.append("low_answer_groundedness")
    if float(metrics.get("unsupported_claim_rate") or 0.0) > 0.2:
        failures.append("unsupported_claim_rate_high")
    return failures


def collect_evidence_gaps(result: dict[str, Any]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for source in (
        result.get("verification", {}),
        result.get("research_outputs", {}),
    ):
        if not isinstance(source, dict):
            continue
        rows = source.get("evidence_gaps")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    text = str(row.get("gap") or row.get("reason") or "").strip()
                    if text:
                        output.append(
                            {
                                "gap": text,
                                "priority": str(row.get("priority") or ""),
                                "suggested_source": str(row.get("suggested_source") or ""),
                            }
                        )
    seen: set[str] = set()
    deduped = []
    for row in output:
        if row["gap"] in seen:
            continue
        seen.add(row["gap"])
        deduped.append(row)
    return deduped[:8]


def expected_slots(case: EvalCase) -> list[tuple[str, str]]:
    slots: list[tuple[str, str]] = []
    slots.extend(("company", company) for company in case.expected_companies)
    slots.extend(("topic", topic) for topic in case.expected_topics)
    slots.extend(("claim_type", claim_type) for claim_type in case.required_claim_types)
    return slots


def card_matches_case(card: dict[str, Any], case: EvalCase) -> bool:
    if case.expected_claim_ids:
        return card_text(card, "claim_id") in set(case.expected_claim_ids)
    checks: list[bool] = []
    if case.expected_companies:
        checks.append(any(card_hits_slot(card, "company", company) for company in case.expected_companies))
    if case.expected_topics:
        checks.append(any(card_hits_slot(card, "topic", topic) for topic in case.expected_topics))
    if case.required_claim_types:
        checks.append(any(card_hits_slot(card, "claim_type", claim_type) for claim_type in case.required_claim_types))
    return all(checks) if checks else bool(card)


def card_hits_slot(card: dict[str, Any], slot_type: str, value: str) -> bool:
    if slot_type == "claim_type":
        return normalized_claim_type(card) == value
    if slot_type == "company":
        return value in card_full_text(card)
    if slot_type == "topic":
        return value.casefold() in card_full_text(card).casefold()
    return False


def normalized_claim_type(card: dict[str, Any]) -> str:
    claim_type = card_text(card, "claim_type")
    if claim_type:
        return claim_type
    return RELATION_CLAIM_TYPE_MAP.get(card_text(card, "relation"), "")


def is_forbidden_evidence(card: dict[str, Any], case: EvalCase) -> bool:
    text = card_full_text(card)
    return any(term and term in text for term in case.forbidden_terms)


def answer_refusal_from_cards(cards: list[dict[str, Any]]) -> bool:
    return not cards


def claim_like_sentence_count(answer: str) -> int:
    parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(str(answer or "")) if part.strip()]
    if not parts:
        return 1
    return len([part for part in parts if len(part) >= 8]) or len(parts)


def card_text(card: dict[str, Any], key: str) -> str:
    value = card.get(key, "")
    if value is None:
        return ""
    return str(value)


def card_full_text(card: dict[str, Any]) -> str:
    return " ".join(
        card_text(card, key)
        for key in (
            "claim_id",
            "citation_id",
            "title",
            "evidence",
            "evidence_span",
            "source",
            "section",
            "company",
            "topic",
            "target",
            "claim_type",
            "relation",
            "exposure_level",
        )
    )


def contains_any(text: str, terms: list[str] | tuple[str, ...]) -> bool:
    return any(term and term in text for term in terms)
