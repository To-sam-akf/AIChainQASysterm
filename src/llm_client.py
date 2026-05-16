"""OpenAI-compatible LLM client used by the extraction pipeline."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from src.extraction_schema import SchemaError, parse_json_object


ROOT_DIR = Path(__file__).resolve().parents[1]


def env_bool(names: str | tuple[str, ...], default: bool = False) -> bool:
    if isinstance(names, str):
        names = (names,)
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        return value.strip().casefold() not in {"0", "false", "no", "off", "disabled"}
    return default


def load_dotenv(path: Path = ROOT_DIR / ".env") -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class ChatTextResult:
    content: str
    reasoning_content: str = ""
    model: str = ""
    usage: dict[str, Any] | None = None


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        thinking_default = "deepseek" in self.base_url.casefold()
        self.thinking_enabled = env_bool(("LLM_THINKING_ENABLED", "LLM_ENABLE_THINKING"), thinking_default)
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "high").strip()
        self._thinking_runtime_disabled = False
        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("LLM_API_KEY, LLM_BASE_URL and LLM_MODEL must be configured")

    @property
    def chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        payload = self._build_payload(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                data = self._post_chat(payload)
                content = data["choices"][0]["message"]["content"]
                return parse_json_object(content)
            except (requests.RequestException, KeyError, IndexError, SchemaError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")

    def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return self.chat_text_with_metadata(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
        ).content

    def chat_text_with_metadata(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatTextResult:
        return self.chat_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
        )

    def chat_messages(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatTextResult:
        payload = self._build_payload(
            messages=messages,
            temperature=temperature,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                data = self._post_chat(payload)
                message = data["choices"][0]["message"]
                return ChatTextResult(
                    content=str(message.get("content") or "").strip(),
                    reasoning_content=str(message.get("reasoning_content") or "").strip(),
                    model=str(data.get("model") or self.model),
                    usage=data.get("usage"),
                )
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")

    def _build_payload(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        response_format: dict[str, str] | None = None,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        use_thinking = self.thinking_enabled if thinking_enabled is None else thinking_enabled
        explicit_thinking = thinking_enabled is not None
        if use_thinking and (explicit_thinking or not self._thinking_runtime_disabled):
            payload["thinking"] = {"type": "enabled"}
            effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
            if effort:
                payload["reasoning_effort"] = effort
        else:
            payload["temperature"] = temperature
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        allow_response_format = "response_format" in payload
        allow_thinking = "thinking" in payload or "reasoning_effort" in payload
        for attempt in range(self.max_retries):
            try:
                request_payload = dict(payload)
                if not allow_response_format:
                    request_payload.pop("response_format", None)
                if not allow_thinking:
                    request_payload.pop("thinking", None)
                    request_payload.pop("reasoning_effort", None)
                response = requests.post(self.chat_url, headers=headers, json=request_payload, timeout=self.timeout)
                if response.status_code == 400 and allow_response_format:
                    allow_response_format = False
                    continue
                if response.status_code == 400 and allow_thinking:
                    allow_thinking = False
                    self._thinking_runtime_disabled = True
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")


class MockLLMClient:
    """Deterministic client for local integration tests and demos without API keys."""

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        del thinking_enabled, reasoning_effort
        del system_prompt, temperature
        company = "浪潮信息" if "浪潮信息" in user_prompt else "中科曙光" if "中科曙光" in user_prompt else ""
        if not company:
            company = "样例公司"
        text = user_prompt.split("文本：", 1)[-1]
        if "AI服务器" in text or "服务器" in text:
            tech = "AI服务器"
        elif "算力" in text:
            tech = "算力"
        else:
            return {"entities": [], "relations": []}
        evidence = next((line.strip() for line in text.splitlines() if tech in line or "算力" in line), "")[:120]
        evidence = evidence or f"{company}涉及{tech}"
        return {
            "entities": [
                {"type": "Company", "name": company},
                {"type": "Technology", "name": tech},
            ],
            "relations": [
                {
                    "head_type": "Company",
                    "head": company,
                    "relation": "USES_TECHNOLOGY",
                    "tail_type": "Technology",
                    "tail": tech,
                    "evidence": evidence,
                    "confidence": 0.6,
                }
            ],
        }

    def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        del thinking_enabled, reasoning_effort
        del system_prompt, temperature
        if "当前知识库中未找到相关证据" in user_prompt:
            return "当前知识库中未找到相关证据。"
        return "基于检索证据，当前知识库中找到了相关事实，具体请查看证据链。"

    def chat_text_with_metadata(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatTextResult:
        return ChatTextResult(
            content=self.chat_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            )
        )
