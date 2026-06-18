"""QA agent wrapper around the first-phase ReAct-style runner."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from aika.agents.base import BaseAgent


class QAAgent(BaseAgent):
    def __init__(self, engine: Any, *, max_steps: int | None = None) -> None:
        self.engine = engine
        self.max_steps = max_steps

    """
    非流式
    QAEngine.answer_question()
    → QAAgent.run()
    → LangGraphAgentRunner.run()      # 默认，中心化并行

    流式
    QAEngine.answer_question_stream()
    → QAAgent.run_stream()
    → LangGraphAgentRunner.run_stream()  # 默认，与非流式共用中心 runner
    """

    def run(
        self,
        question: str,
        *,
        conversation_history: list[dict[str, str]] | None = None,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        if self._can_use_agent_runner():
            if self._agent_runner_name() == "legacy":
                from aika.agent_runner import AgentRunner

                result = AgentRunner(self.engine, max_steps=self._agent_max_steps()).run(
                    question,
                    conversation_history=conversation_history,
                    thinking_enabled=thinking_enabled,
                    reasoning_effort=reasoning_effort,
                )
                self._mark_runner(result, runner="legacy", langgraph_enabled=False)
                return result
            from aika.langgraph_runner import LangGraphAgentRunner

            return LangGraphAgentRunner(self.engine, max_steps=self._agent_max_steps()).run(
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
            if self._agent_runner_name() == "legacy":
                from aika.agent_runner import AgentRunner

                for event in AgentRunner(self.engine, max_steps=self._agent_max_steps()).run_stream(
                    question,
                    conversation_history=conversation_history,
                    thinking_enabled=thinking_enabled,
                    reasoning_effort=reasoning_effort,
                ):
                    if event.get("type") == "final" and isinstance(event.get("result"), dict):
                        self._mark_runner(event["result"], runner="legacy_stream", langgraph_enabled=False)
                    yield event
                return
            from aika.langgraph_runner import LangGraphAgentRunner

            yield from LangGraphAgentRunner(self.engine, max_steps=self._agent_max_steps()).run_stream(
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

    def _agent_runner_name(self) -> str:
        runner = str(getattr(self.engine, "agent_runner", "langgraph") or "langgraph").strip().casefold()
        return runner if runner in {"langgraph", "legacy"} else "langgraph"

    @staticmethod
    def _mark_runner(result: dict[str, Any], *, runner: str, langgraph_enabled: bool) -> None:
        diagnostics = result.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            result["diagnostics"] = diagnostics
        diagnostics["agent_runner"] = runner
        diagnostics["langgraph_enabled"] = langgraph_enabled

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
