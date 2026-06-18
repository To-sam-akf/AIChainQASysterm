"""Base interfaces for AIQASYS agent implementations."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class BaseAgent:
    """Minimal synchronous and streaming interface shared by agents."""

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def run_stream(self, *args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
        raise NotImplementedError
