"""Shared helpers for report chart builders."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any, Iterable


def as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return dict(value.to_dict())
    if is_dataclass(value):
        return asdict(value)
    return {}


def text_attr(value: Any, name: str) -> str:
    row = as_dict(value)
    if row:
        candidate = row.get(name, "")
        if name == "source" and not candidate:
            candidate = row.get("source_title", "")
        if name == "evidence" and not candidate:
            candidate = row.get("text") or row.get("evidence_span", "")
        if name == "published_at" and not candidate:
            candidate = row.get("as_of_date", "")
        return str(candidate or "").strip()
    return str(getattr(value, name, "") or "").strip()


def evidence_id(value: Any, fallback: str = "") -> str:
    return (
        text_attr(value, "citation_id")
        or text_attr(value, "evidence_id")
        or text_attr(value, "claim_id")
        or fallback
    )


def normalize_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").casefold())


def dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = normalize_key(text)
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def parse_year(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"(20\d{2}|19\d{2})", text)
    return match.group(1) if match else ""


def parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m", "%Y/%m", "%Y.%m", "%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def freshness_status(value: Any, *, today: date | None = None) -> str:
    text = str(value or "").strip().lower()
    if text in {"fresh", "aging", "stale", "unknown"}:
        return text
    parsed = parse_date(text)
    if parsed is None:
        return "unknown"
    current = today or date.today()
    months = (current.year - parsed.year) * 12 + (current.month - parsed.month)
    if current.day < parsed.day:
        months -= 1
    if months <= 18:
        return "fresh"
    if months <= 36:
        return "aging"
    return "stale"


def source_key(value: Any) -> str:
    return text_attr(value, "source_report_id") or text_attr(value, "source") or text_attr(value, "title")
