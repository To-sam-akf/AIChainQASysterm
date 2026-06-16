"""Centralized LangGraph orchestration for parallel multi-agent retrieval."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Iterator, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent_runner import (
    AgentRetrievalState,
    AgentRunner,
    AgentTraceStep,
    build_supplemental_question,
    should_refuse_metric_answer,
)
from src.agent_tools import AgentToolCall, AgentTools
from src.agents.centralized import (
    AgentAssignment,
    AgentResult,
    CentralizedCoordinator,
    CentralLLMBudget,
    EvidenceCandidate,
    ReviewReport,
)
from src.agents.coverage import CoverageReport
from src.langchain_tools import build_langchain_tools
from src.professional_qa import EvidenceCard


ProgressCallback = Callable[[dict[str, Any]], None]


class LangGraphAgentState(TypedDict, total=False):
    question: str
    conversation_history: list[dict[str, str]] | None
    thinking_enabled: bool | None
    reasoning_effort: str | None
    streaming: bool
    progress_callback: ProgressCallback | None
    total_start: float
    timings_ms: dict[str, float]
    errors: list[str]
    llm_budget: CentralLLMBudget
    llm_client: Any | None
    llm_options: dict[str, Any]
    tools: AgentTools
    coordinator: CentralizedCoordinator
    langchain_tool_names: list[str]
    history: list[dict[str, str]]
    history_memory: dict[str, Any]
    contextual_question: str
    plan: Any
    task_plan: Any
    planner_source: str
    generated: Any
    hyde_query: Any
    trace: list[AgentTraceStep]
    retrieval_state: AgentRetrievalState
    assignments: list[AgentAssignment]
    wave_results: list[AgentResult]
    candidate_entries: list[tuple[EvidenceCard, str]]
    candidates: list[EvidenceCandidate]
    review_report: ReviewReport
    coverage_report: CoverageReport
    round_index: int
    all_tool_calls: list[AgentToolCall]
    raw_cards: list[EvidenceCard]
    evidence_cards: list[EvidenceCard]
    summary_metadata: dict[str, Any]
    result: dict[str, Any]


class LangGraphAgentRunner(AgentRunner):
    """Central supervisor with parallel query agents and deterministic fallbacks."""

    def __init__(self, engine: Any, *, max_steps: int = 4) -> None:
        super().__init__(engine, max_steps=max_steps)
        self._graph = self._build_graph()

    def run(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        final_state = self._invoke(
            question,
            conversation_history,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
            streaming=False,
            progress_callback=None,
        )
        return dict(final_state.get("result") or {})

    def run_stream(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        events: queue.Queue[dict[str, Any]] = queue.Queue()

        def worker() -> None:
            try:
                final_state = self._invoke(
                    question,
                    conversation_history,
                    thinking_enabled=thinking_enabled,
                    reasoning_effort=reasoning_effort,
                    streaming=True,
                    progress_callback=events.put,
                )
                events.put({"type": "final", "result": dict(final_state.get("result") or {})})
            except Exception as exc:  # pragma: no cover - defensive streaming boundary.
                events.put({"type": "_runner_error", "error": str(exc)})

        thread = threading.Thread(target=worker, name="qa-centralized-stream", daemon=True)
        thread.start()
        while True:
            event = events.get()
            if event.get("type") == "_runner_error":
                thread.join()
                raise RuntimeError(str(event.get("error") or "centralized runner failed"))
            yield event
            if event.get("type") == "final":
                thread.join()
                return

    def _invoke(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None,
        *,
        thinking_enabled: bool | None,
        reasoning_effort: str | None,
        streaming: bool,
        progress_callback: ProgressCallback | None,
    ) -> LangGraphAgentState:
        from src.qa_engine import build_llm_options

        errors: list[str] = []
        llm_options = build_llm_options(
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
        )
        llm_budget = CentralLLMBudget(
            self.engine.llm_client,
            max_calls=int(getattr(self.engine, "multi_agent_max_llm_calls", 12)),
        )
        planning_client = llm_budget.client("planning")
        tools = AgentTools(
            self.engine,
            errors=errors,
            llm_options=llm_options,
            llm_client=planning_client,
        )
        coordinator = CentralizedCoordinator(
            self.engine,
            llm_budget=llm_budget,
            llm_options=llm_options,
        )
        initial_state: LangGraphAgentState = {
            "question": question.strip(),
            "conversation_history": conversation_history,
            "thinking_enabled": thinking_enabled,
            "reasoning_effort": reasoning_effort,
            "streaming": streaming,
            "progress_callback": progress_callback,
            "total_start": time.perf_counter(),
            "timings_ms": {},
            "errors": errors,
            "llm_budget": llm_budget,
            "llm_client": planning_client,
            "llm_options": llm_options,
            "tools": tools,
            "coordinator": coordinator,
            "langchain_tool_names": [tool.name for tool in build_langchain_tools(tools)],
            "trace": [],
            "retrieval_state": AgentRetrievalState(),
            "round_index": 0,
            "candidate_entries": [],
            "all_tool_calls": [],
        }
        return self._graph.invoke(initial_state)

    def _build_graph(self) -> Any:
        builder = StateGraph(LangGraphAgentState)
        builder.add_node("plan", self._plan_node)
        builder.add_node("dispatch", self._dispatch_node)
        builder.add_node("query_wave", self._query_wave_node)
        builder.add_node("graphrag_join", self._graphrag_join_node)
        builder.add_node("review", self._review_node)
        builder.add_node("summarize", self._summarize_node)
        builder.add_node("verify_answer", self._verify_answer_node)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "dispatch")
        builder.add_edge("dispatch", "query_wave")
        builder.add_edge("query_wave", "graphrag_join")
        builder.add_edge("graphrag_join", "review")
        builder.add_conditional_edges(
            "review",
            self._route_after_review,
            {"supplement": "dispatch", "summarize": "summarize"},
        )
        builder.add_edge("summarize", "verify_answer")
        builder.add_edge("verify_answer", END)
        return builder.compile()

    def _plan_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        self._progress(state, "agent_plan", "中心调度 Agent 正在规划问题和证据目标")
        history, history_memory, contextual_question, plan, task_plan, planner_source, generated, hyde_query, trace = self._plan(
            str(state.get("question") or ""),
            state.get("conversation_history"),
            state["tools"],
            state["timings_ms"],
        )
        return {
            "history": history,
            "history_memory": history_memory,
            "contextual_question": contextual_question,
            "plan": plan,
            "task_plan": task_plan,
            "planner_source": planner_source,
            "generated": generated,
            "hyde_query": hyde_query,
            "trace": trace,
            "retrieval_state": AgentRetrievalState(generated=generated),
        }

    def _dispatch_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        round_index = int(state.get("round_index") or 0)
        review = state.get("review_report")
        gaps = list(review.coverage.gaps) if review is not None else []
        question = str(state.get("contextual_question") or "")
        if gaps:
            question = build_supplemental_question(question, gaps[0].to_dict())
        assignments = state["coordinator"].dispatch(
            question=question,
            plan=state["plan"],
            task_plan=state["task_plan"],
            round_index=round_index,
            gaps=gaps,
        )
        self._progress(
            state,
            "agent_dispatch",
            f"第 {round_index + 1} 轮已派发 {len(assignments)} 个查询 Agent",
        )
        return {"assignments": assignments}

    def _query_wave_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        round_index = int(state.get("round_index") or 0)
        started_at = time.perf_counter()
        results = state["coordinator"].execute_wave(
            list(state.get("assignments") or []),
            plan=state["plan"],
            hyde_query=state.get("hyde_query"),
        )
        self._record_timing(state["timings_ms"], f"query_wave_{round_index}", started_at)
        completed = sum(1 for result in results if result.status == "completed")
        self._progress(
            state,
            "agent_query_wave",
            f"第 {round_index + 1} 轮查询完成：{completed}/{len(results)} 路成功",
        )
        return {"wave_results": results}

    def _graphrag_join_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        retrieval_state, entries, calls, errors = state["coordinator"].join_results(
            state.get("retrieval_state") or AgentRetrievalState(generated=state.get("generated")),
            list(state.get("wave_results") or []),
            plan=state["plan"],
            existing_entries=list(state.get("candidate_entries") or []),
        )
        state["errors"].extend(errors)
        self._progress(
            state,
            "agent_join",
            (
                f"已汇合图谱 {len(retrieval_state.graph_records)} 条、RAG {len(retrieval_state.rag_hits)} 条、"
                f"投研证据 {len(retrieval_state.research_hits)} 条、语义证据 {len(retrieval_state.semantic_hits)} 条"
            ),
        )
        return {
            "retrieval_state": retrieval_state,
            "candidate_entries": entries,
            "all_tool_calls": [*list(state.get("all_tool_calls") or []), *calls],
        }

    def _review_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        candidates = state["coordinator"].build_candidates(
            list(state.get("candidate_entries") or []),
            plan=state["plan"],
        )
        round_index = int(state.get("round_index") or 0)
        report = state["coordinator"].review(
            question=str(state.get("contextual_question") or ""),
            plan=state["plan"],
            task_plan=state["task_plan"],
            retrieval_state=state["retrieval_state"],
            candidates=candidates,
            round_index=round_index,
        )
        state["retrieval_state"].coverage_report = report.coverage
        missing = "、".join(report.coverage.missing) or "无"
        self._progress(
            state,
            "agent_review",
            f"审核完成：接受 {len(report.accepted_ids)} 条候选，证据缺口 {missing}",
        )
        updates: dict[str, Any] = {
            "candidates": candidates,
            "review_report": report,
            "coverage_report": report.coverage,
        }
        if report.coverage.should_continue:
            updates["round_index"] = round_index + 1
        return updates

    @staticmethod
    def _route_after_review(state: LangGraphAgentState) -> Literal["supplement", "summarize"]:
        report = state.get("review_report")
        if report is not None and report.coverage.should_continue:
            return "supplement"
        return "summarize"

    def _summarize_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        started_at = time.perf_counter()
        report = state["review_report"]
        raw_cards, evidence_cards, drafts, metadata = state["coordinator"].summarize(
            question=str(state.get("contextual_question") or ""),
            plan=state["plan"],
            candidates=list(state.get("candidates") or []),
            accepted_ids=report.accepted_ids,
        )
        self._record_timing(state["timings_ms"], "evidence", started_at)
        state["retrieval_state"].reranker = dict(metadata)
        trace = list(state.get("trace") or [])
        all_calls = list(state.get("all_tool_calls") or [])
        trace.append(
            AgentTraceStep(
                step=len(trace) + 1,
                phase="retrieve",
                thought="中心调度 Agent 并行执行多路查询，并在汇合点完成 GraphRAG 路径组装。",
                action="dispatch -> parallel_query_wave -> graphrag_join",
                tool_calls=all_calls,
                observation=(
                    f"完成 {len(state['coordinator'].diagnostics()['rounds'])} 轮查询，"
                    f"形成 {len(state.get('candidates') or [])} 条候选证据。"
                ),
            )
        )
        trace.append(
            AgentTraceStep(
                step=len(trace) + 1,
                phase="supplement",
                thought="审核 Agent 检查相关性、冲突与覆盖度，并决定是否发起下一轮并行补检。",
                action=(
                    "supplemental_retrieve"
                    if int(state.get("round_index") or 0) > 0
                    else "skip_supplement"
                ),
                tool_calls=[],
                observation=(
                    f"stop_reason={report.coverage.stop_reason}，"
                    f"missing={'、'.join(report.coverage.missing) or '无'}，"
                    f"review_source={report.source}。"
                ),
            )
        )
        self._progress(
            state,
            "agent_summarize",
            f"归纳 Agent 已生成 {len(evidence_cards)} 张可追溯证据卡",
        )
        return {
            "raw_cards": raw_cards,
            "evidence_cards": evidence_cards,
            "summary_metadata": {**metadata, "draft_count": len(drafts)},
            "trace": trace,
        }

    def _verify_answer_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        from src.agents.coverage import metric_gap_answer
        from src.agents.verification import build_evidence_limited_answer

        answer_tools = AgentTools(
            self.engine,
            errors=state["errors"],
            llm_options=state["llm_options"],
            llm_client=state["llm_budget"].client("answer"),
        )
        evidence_cards = list(state.get("evidence_cards") or [])
        raw_cards = list(state.get("raw_cards") or [])
        coverage = state["coverage_report"]
        refused_metric = should_refuse_metric_answer(coverage)
        started_at = time.perf_counter()
        answer = ""
        reasoning_content = ""
        if refused_metric:
            answer = metric_gap_answer(coverage)
            answer_call = AgentToolCall(
                tool="generate_answer",
                args={"answer_type": state["plan"].answer_type, "evidence_cards": len(evidence_cards), "refused": True},
                result_count=1,
            )
            if state.get("streaming"):
                self._emit_answer(state, answer)
        elif state.get("streaming"):
            for event in self.engine._generate_answer_stream(
                str(state.get("question") or ""),
                str(state.get("contextual_question") or ""),
                state.get("history") or [],
                state["plan"],
                state["retrieval_state"].graph_records,
                evidence_cards,
                state["errors"],
                state["llm_options"],
                answer_tools.llm_client,
                thinking_enabled=bool(state.get("thinking_enabled")),
            ):
                event_type = event.get("type")
                if event_type in {"answer_delta", "progress"}:
                    self._callback(state, event)
                elif event_type == "answer_complete":
                    answer = str(event.get("answer") or "")
                    reasoning_content = str(event.get("reasoning_content") or "")
            answer_call = AgentToolCall(
                tool="generate_answer_stream",
                args={"answer_type": state["plan"].answer_type, "evidence_cards": len(evidence_cards)},
                result_count=1 if answer else 0,
                elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
        else:
            (answer, reasoning_content), answer_call = answer_tools.generate_answer(
                str(state.get("question") or ""),
                str(state.get("contextual_question") or ""),
                state.get("history") or [],
                state["plan"],
                state["retrieval_state"].graph_records,
                evidence_cards,
            )
        self._record_timing(state["timings_ms"], "answer", started_at)

        verification, verify_call = answer_tools.verify_answer_support(
            answer,
            state["plan"],
            evidence_cards,
            raw_cards,
            str(state.get("contextual_question") or ""),
        )
        tool_calls = [
            AgentToolCall(
                tool="summarize_evidence",
                args={"candidate_count": len(state.get("candidates") or [])},
                result_count=len(evidence_cards),
                error="" if state.get("summary_metadata", {}).get("source") == "llm" else "deterministic fallback",
            ),
            answer_call,
            verify_call,
        ]
        if verification.get("status") == "fail" and not refused_metric:
            answer = build_evidence_limited_answer(state["plan"], evidence_cards, verification, coverage)
            reasoning_content = ""
            verification, second_verify_call = answer_tools.verify_answer_support(
                answer,
                state["plan"],
                evidence_cards,
                raw_cards,
                str(state.get("contextual_question") or ""),
            )
            tool_calls.append(second_verify_call)

        trace = list(state.get("trace") or [])
        trace.append(
            AgentTraceStep(
                step=len(trace) + 1,
                phase="verify_answer",
                thought="归纳证据卡后生成答案，并执行引用、数值、风险和公司覆盖验证。",
                action="summarize_evidence -> generate_answer -> verify_answer_support",
                tool_calls=tool_calls,
                observation=f"verification={verification.get('status')}，证据卡 {len(evidence_cards)} 条。",
            )
        )
        result = self._build_result(
            str(state.get("question") or ""),
            str(state.get("contextual_question") or ""),
            answer,
            reasoning_content,
            state["plan"],
            state["task_plan"],
            str(state.get("planner_source") or ""),
            state.get("generated"),
            state["retrieval_state"],
            evidence_cards,
            verification,
            state.get("hyde_query"),
            trace,
            state["timings_ms"],
            state["errors"],
            state.get("llm_client"),
            thinking_enabled=state.get("thinking_enabled"),
            reasoning_effort=state.get("reasoning_effort"),
            total_start=float(state.get("total_start") or time.perf_counter()),
            history_count=len(state.get("history") or []),
            history_memory=state.get("history_memory") or {},
        )
        diagnostics = result.setdefault("diagnostics", {})
        diagnostics["agent_runner"] = "langgraph"
        diagnostics["langgraph_enabled"] = True
        diagnostics["langchain_tools"] = list(state.get("langchain_tool_names") or [])
        diagnostics["llm_calls"] = state["llm_budget"].calls
        diagnostics["multi_agent"] = {
            **state["coordinator"].diagnostics(),
            "review": state["review_report"].to_dict(),
            "summary": dict(state.get("summary_metadata") or {}),
        }
        diagnostics["agent_budget"]["max_llm_calls"] = state["llm_budget"].max_calls
        return {"result": result, "trace": trace}

    def _progress(self, state: LangGraphAgentState, stage: str, message: str) -> None:
        if not state.get("thinking_enabled"):
            return
        self._callback(state, {"type": "progress", "stage": stage, "message": message})

    @staticmethod
    def _callback(state: LangGraphAgentState, event: dict[str, Any]) -> None:
        callback = state.get("progress_callback")
        if callback is not None:
            callback(event)

    def _emit_answer(self, state: LangGraphAgentState, answer: str) -> None:
        from src.qa_engine import chunk_text

        for chunk in chunk_text(answer):
            self._callback(state, {"type": "answer_delta", "content": chunk})
