"""OpenAI-compatible embedding client used by the local semantic index."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from src.llm_client import load_dotenv


DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_EMBEDDING_TIMEOUT = 60
DEFAULT_EMBEDDING_RETRIES = 2


def embedding_configured() -> bool:
    """Return true only when an embedding model is configured."""

    load_dotenv()
    return bool(os.getenv("EMBEDDING_MODEL", "").strip())


class OpenAICompatibleEmbeddingClient:
    """Minimal client for OpenAI-compatible ``/embeddings`` endpoints."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        dimensions: int | None = None,
        timeout: int = DEFAULT_EMBEDDING_TIMEOUT,
        max_retries: int = DEFAULT_EMBEDDING_RETRIES,
    ) -> None:
        load_dotenv()
        self.base_url = (
            base_url
            or os.getenv("EMBEDDING_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or ""
        ).strip().rstrip("/")
        self.api_key = (
            api_key
            or os.getenv("EMBEDDING_API_KEY")
            or os.getenv("LLM_API_KEY")
            or ""
        ).strip()
        self.model = (model or os.getenv("EMBEDDING_MODEL") or "").strip()
        self.batch_size = int(batch_size or os.getenv("EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE)))
        self.dimensions = dimensions if dimensions is not None else parse_optional_int(os.getenv("EMBEDDING_DIMENSIONS", ""))
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))

        if not self.model:
            raise ValueError("EMBEDDING_MODEL is not configured")
        if not self.base_url:
            raise ValueError("EMBEDDING_BASE_URL or LLM_BASE_URL is not configured")
        if not self.api_key:
            raise ValueError("EMBEDDING_API_KEY or LLM_API_KEY is not configured")
        if self.batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be positive")

    @property
    def embeddings_url(self) -> str:
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        return f"{self.base_url}/embeddings"

    def embed_texts(self, texts: list[str], progress_callback: Any | None = None) -> list[list[float]]:
        if not texts:
            return []
        output: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [text if str(text).strip() else " " for text in texts[start : start + self.batch_size]]
            output.extend(self._embed_batch(batch))
            if progress_callback is not None:
                progress_callback(min(start + len(batch), len(texts)), len(texts))
        return output

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.embeddings_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return self._parse_embeddings(response.json(), expected_count=len(texts))
            except Exception as exc:  # pragma: no cover - retry timing is environment-sensitive.
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Embedding request failed: {last_error}") from last_error

    def _parse_embeddings(self, payload: dict[str, Any], *, expected_count: int) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Embedding response missing data list")
        rows = sorted(data, key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0)
        vectors: list[list[float]] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
                raise ValueError("Embedding response row missing embedding")
            vector = [float(value) for value in row["embedding"]]
            if not vector:
                raise ValueError("Embedding vector is empty")
            vectors.append(vector)
        if len(vectors) != expected_count:
            raise ValueError(f"Embedding response count mismatch: expected {expected_count}, got {len(vectors)}")
        return vectors


def parse_optional_int(value: str) -> int | None:
    value = str(value or "").strip()
    if not value:
        return None
    return int(value)
