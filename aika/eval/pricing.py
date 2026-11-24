"""Optional provider pricing, supplied by the operator rather than guessed."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any


def validate_prices(prices: dict[str, Any] | None) -> None:
    if prices is None:
        return
    if not prices.get("currency") or not prices.get("source"):
        raise ValueError("Pricing requires currency and source attribution")
    date.fromisoformat(prices["effective_date"])
    for rates in prices["models"].values():
        for value in rates.values():
            if Decimal(str(value)) < 0:
                raise ValueError("Prices cannot be negative")


def billing_summary(calls: list[dict[str, Any]], prices: dict[str, Any] | None = None) -> dict[str, Any]:
    unique = {call["call_id"]: call for call in calls}
    known = Decimal(0)
    unknown = 0
    input_tokens = output_tokens = cached_tokens = 0
    for call in unique.values():
        usage = call.get("usage") or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens", 0 if call.get("kind") == "embedding" else None)
        cached = usage.get("prompt_cache_hit_tokens", (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        if prompt is not None:
            input_tokens += prompt
        if completion is not None:
            output_tokens += completion
        cached_tokens += cached
        model = call.get("response_model") or call.get("request", {}).get("model")
        rates = (prices or {}).get("models", {}).get(model)
        if rates is None or prompt is None or completion is None or cached > prompt:
            unknown += 1
            continue
        if cached and "input_cached_per_million" not in rates:
            unknown += 1
            continue
        if "input_per_million" not in rates or (completion and "output_per_million" not in rates):
            unknown += 1
            continue
        known += ((Decimal(prompt - cached) * Decimal(str(rates["input_per_million"])))
                  + Decimal(cached) * Decimal(str(rates.get("input_cached_per_million", 0)))
                  + Decimal(completion) * Decimal(str(rates.get("output_per_million", 0)))) / Decimal(1000000)
    return {
        "unique_attempts": len(unique), "unknown_usage_attempts": sum(call.get("usage") is None for call in unique.values()),
        "input_tokens_known": input_tokens, "output_tokens_known": output_tokens, "cached_input_tokens_known": cached_tokens,
        "unknown_cost_attempts": unknown, "amount": None if unknown else str(known), "known_amount": str(known),
        "currency": (prices or {}).get("currency"), "complete": unknown == 0, "pricing": prices,
    }
