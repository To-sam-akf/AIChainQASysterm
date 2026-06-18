"""Optional LLM evidence reranking with deterministic fallback."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


RERANK_MODES = {"auto", "heuristic", "llm"}


@dataclass(frozen=True)
class RerankResult:
    cards: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidenceReranker:
    def __init__(self, *, mode: str = "auto") -> None:
        self.mode = normalize_rerank_mode(mode)

    @classmethod
    def from_env(cls) -> "EvidenceReranker":
        return cls(mode=os.getenv("QA_RERANK_MODE", "auto"))

    def rerank(
        self,
        *,
        question: str,
        cards: list[Any],
        limit: int,
        llm_client: Any | None = None,
        llm_options: dict[str, Any] | None = None,
        use_llm: bool = False,
    ) -> RerankResult:
        cards = list(cards or [])
        limit = max(0, int(limit or 0))
        if not cards or limit <= 0:
            return RerankResult([], {"mode": self.mode, "source": "empty", "candidate_count": len(cards)})

        if not self._should_call_llm(llm_client, use_llm):
            return RerankResult(cards[:limit], {"mode": self.mode, "source": "heuristic", "candidate_count": len(cards)})

        try:
            ranked_ids = self._llm_rank_ids(question, cards, llm_client, llm_options or {})
            reordered = reorder_cards(cards, ranked_ids)
            if not reordered:
                raise ValueError("LLM rerank returned no usable candidate ids")
            return RerankResult(
                fill_remaining(reordered, cards)[:limit],
                {
                    "mode": self.mode,
                    "source": "llm",
                    "candidate_count": len(cards),
                    "ranked_ids": ranked_ids,
                },
            )
        except Exception as exc:
            return RerankResult(
                cards[:limit],
                {
                    "mode": self.mode,
                    "source": "heuristic",
                    "candidate_count": len(cards),
                    "error": str(exc),
                },
            )

    def _should_call_llm(self, llm_client: Any | None, use_llm: bool) -> bool:
        if self.mode == "heuristic":
            return False
        if self.mode == "auto" and not use_llm:
            return False
        return llm_client is not None and (hasattr(llm_client, "chat_json") or hasattr(llm_client, "chat_text"))

    def _llm_rank_ids(
        self,
        question: str,
        cards: list[Any],
        llm_client: Any,
        llm_options: dict[str, Any],
    ) -> list[str]:
        payload = {
            "question": question,
            "candidates": [
                {
                    "id": candidate_id(index),
                    "kind": card_attr(card, "kind"),
                    "title": card_attr(card, "title"),
                    "company": card_attr(card, "company"),
                    "topic": card_attr(card, "topic"),
                    "claim_type": card_attr(card, "claim_type"),
                    "exposure_level": card_attr(card, "exposure_level"),
                    "evidence": shorten(card_attr(card, "evidence"), 500),
                    "score": card_attr(card, "score"),
                }
                for index, card in enumerate(cards, start=1)
            ],
        }
        system_prompt = (
            "你是证据精排器，只能输出候选 id 的 JSON 排序。"
            "不要生成新事实，不要修改证据。"
        )
        user_prompt = (
            "请按回答问题所需证据价值排序，优先保留能支撑主题、公司、指标和风险的证据。"
            "输出 JSON：{\"ranked_ids\":[\"C1\",\"C2\"]}。\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        if hasattr(llm_client, "chat_json"):
            kwargs = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": 0.0,
            }
            kwargs.update(llm_options)
            response = llm_client.chat_json(**kwargs)
            return extract_ranked_ids(response)
        kwargs = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": 0.0,
        }
        kwargs.update(llm_options)
        response = llm_client.chat_text(**kwargs)
        return extract_ranked_ids(response)


def normalize_rerank_mode(value: str) -> str:
    mode = str(value or "auto").strip().casefold()
    return mode if mode in RERANK_MODES else "auto"


def candidate_id(index: int) -> str:
    return f"C{index}"


def card_attr(card: Any, name: str) -> str:
    if isinstance(card, dict):
        value = card.get(name, "")
    else:
        value = getattr(card, name, "")
    return str(value or "")


def shorten(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def extract_ranked_ids(response: Any) -> list[str]:
    if isinstance(response, dict):
        value = response.get("ranked_ids") or response.get("ids") or response.get("ranking")
        return normalize_ids(value)
    text = str(response or "")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return extract_ranked_ids(parsed)
        return normalize_ids(parsed)
    except json.JSONDecodeError:
        return normalize_ids(re.findall(r"\bC\d+\b", text))


def normalize_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        values = re.findall(r"\bC\d+\b", value)
    elif isinstance(value, list):
        values = [str(item) for item in value]
    else:
        values = []
    output: list[str] = []
    for item in values:
        item = item.strip().upper()
        if re.fullmatch(r"C\d+", item) and item not in output:
            output.append(item)
    return output


def reorder_cards(cards: list[Any], ranked_ids: list[str]) -> list[Any]:
    by_id = {candidate_id(index): card for index, card in enumerate(cards, start=1)}
    return [by_id[item] for item in ranked_ids if item in by_id]


def fill_remaining(selected: list[Any], cards: list[Any]) -> list[Any]:
    output = list(selected)
    seen = {id(card) for card in output}
    for card in cards:
        if id(card) in seen:
            continue
        seen.add(id(card))
        output.append(card)
    return output
