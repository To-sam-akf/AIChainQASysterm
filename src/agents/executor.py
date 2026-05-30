"""Execution boundary for agent tools with telemetry and simple budgets."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ToolBudget:
    max_tool_calls: int = 64
    max_llm_calls: int = 4
    tool_calls: int = 0
    llm_calls: int = 0

    @property
    def exhausted(self) -> bool:
        return self.tool_calls >= self.max_tool_calls or self.llm_calls >= self.max_llm_calls

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def consume(self, *, requires_llm: bool = False) -> "ToolBudget":
        return ToolBudget(
            max_tool_calls=self.max_tool_calls,
            max_llm_calls=self.max_llm_calls,
            tool_calls=self.tool_calls + 1,
            llm_calls=self.llm_calls + (1 if requires_llm else 0),
        )


@dataclass(frozen=True)
class ToolExecution:
    tool: str
    args: dict[str, Any]
    result: Any = None
    result_count: int = 0
    elapsed_ms: float = 0.0
    error: str = ""
    budget_exhausted: bool = False
    budget: dict[str, Any] | None = None

    def to_call_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "result_count": self.result_count,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "budget_exhausted": self.budget_exhausted,
        }


class ToolExecutor:
    """Runs read-only agent tools and normalizes success/error telemetry."""

    def __init__(self, *, errors: list[str] | None = None, budget: ToolBudget | None = None) -> None:
        self.errors = errors if errors is not None else []
        self.budget = budget or ToolBudget()

    def execute(
        self,
        tool: str,
        args: dict[str, Any],
        func: Callable[[], Any],
        count_result: Callable[[Any], int],
        *,
        requires_llm: bool = False,
    ) -> ToolExecution:
        started_at = time.perf_counter()
        if self.budget.exhausted:
            return ToolExecution(
                tool=tool,
                args=dict(args),
                result=None,
                elapsed_ms=0.0,
                error="Tool budget exhausted",
                budget_exhausted=True,
                budget=self.budget.to_dict(),
            )

        error = ""
        before_error_count = len(self.errors)
        result: Any = None
        try:
            result = func()
        except Exception as exc:  # pragma: no cover - defensive executor boundary.
            error = str(exc)
            self.errors.append(f"{tool} failed: {exc}")
        if not error and len(self.errors) > before_error_count:
            error = "; ".join(self.errors[before_error_count:])

        result_count = 0
        if result is not None:
            try:
                result_count = int(count_result(result))
            except Exception:
                result_count = 0

        self.budget = self.budget.consume(requires_llm=requires_llm)
        return ToolExecution(
            tool=tool,
            args=dict(args),
            result=result,
            result_count=result_count,
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
            error=error,
            budget_exhausted=False,
            budget=self.budget.to_dict(),
        )


def result_count(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, (str, bytes)):
        return 1 if result else 0
    if isinstance(result, dict):
        return len(result)
    if isinstance(result, (list, tuple, set)):
        return len(result)
    return 1
