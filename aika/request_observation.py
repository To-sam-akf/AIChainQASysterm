"""Optional request telemetry. Credentials and response bodies are never recorded."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator


RequestObserver = Callable[[dict[str, Any]], None]
REQUEST_ROLE: ContextVar[str] = ContextVar("aika_request_role", default="unspecified")


@contextmanager
def request_role(role: str) -> Iterator[None]:
    token = REQUEST_ROLE.set(role)
    try:
        yield
    finally:
        REQUEST_ROLE.reset(token)
