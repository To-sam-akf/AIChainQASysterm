"""OpenAI-compatible LLM client used by the extraction pipeline."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

from src.extraction_schema import SchemaError, parse_json_object


ROOT_DIR = Path(__file__).resolve().parents[1]


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
        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("LLM_API_KEY, LLM_BASE_URL and LLM_MODEL must be configured")

    @property
    def chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def chat_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        allow_response_format = True
        for attempt in range(self.max_retries):
            try:
                request_payload = dict(payload)
                if not allow_response_format:
                    request_payload.pop("response_format", None)
                response = requests.post(self.chat_url, headers=headers, json=request_payload, timeout=self.timeout)
                if response.status_code == 400 and allow_response_format:
                    allow_response_format = False
                    continue
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return parse_json_object(content)
            except (requests.RequestException, KeyError, IndexError, SchemaError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")


class MockLLMClient:
    """Deterministic client for local integration tests and demos without API keys."""

    def chat_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict[str, Any]:
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
