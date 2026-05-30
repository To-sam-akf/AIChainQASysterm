"""QA agent wrapper around the first-phase ReAct-style runner."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from src.agents.base import BaseAgent


class QAAgent(BaseAgent):
    def __init__(self, engine: Any, *, max_steps: int | None = None) -> None:
        self.engine = engine
        self.max_steps = max_steps

    def run(
        self,
        question: str,
        *,
        conversation_history: list[dict[str, str]] | None = None,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        if self._can_use_agent_runner():
            from src.agent_runner import AgentRunner

            return AgentRunner(self.engine, max_steps=self._agent_max_steps()).run(
                question,
                conversation_history=conversation_history,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            )
        return self.engine.answer_question(
            question,
            conversation_history=conversation_history,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
        )

    def run_stream(
        self,
        question: str,
        *,
        conversation_history: list[dict[str, str]] | None = None,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        if self._can_use_agent_runner():
            from src.agent_runner import AgentRunner

            yield from AgentRunner(self.engine, max_steps=self._agent_max_steps()).run_stream(
                question,
                conversation_history=conversation_history,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            )
            return
        if hasattr(self.engine, "answer_question_stream"):
            yield from self.engine.answer_question_stream(
                question,
                conversation_history=conversation_history,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            )
            return
        yield {
            "type": "final",
            "result": self.run(
                question,
                conversation_history=conversation_history,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            ),
        }

    def _agent_max_steps(self) -> int:
        if self.max_steps is not None:
            return self.max_steps
        return int(getattr(self.engine, "agent_max_steps", 4) or 4)

    def _can_use_agent_runner(self) -> bool:
        required_attrs = (
            "_contextualize_question",
            "_generate_display_cypher",
            "_query_graph",
            "_search_rag",
            "_search_research",
            "_generate_answer",
        )
        return all(hasattr(self.engine, attr) for attr in required_attrs)
