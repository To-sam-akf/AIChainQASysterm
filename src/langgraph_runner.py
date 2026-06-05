"""LangGraph orchestration for the evidence-driven QA agent."""

from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent_runner import (
    AgentRetrievalState,
    AgentRunner,
    AgentTraceStep,
    build_supplemental_question,
    coverage_tool_call,
    merge_graph_records,
    merge_rag_hits,
    merge_research_hits,
    merge_semantic_hits,
)
from src.agent_tools import AgentToolCall, AgentTools
from src.agents.coverage import CoverageReport, EvidenceCoverageChecker, next_supplement_gap
from src.langchain_tools import build_langchain_tools


class LangGraphAgentState(TypedDict, total=False):
    question: str
    conversation_history: list[dict[str, str]] | None
    thinking_enabled: bool | None
    reasoning_effort: str | None
    total_start: float
    timings_ms: dict[str, float]
    errors: list[str]
    llm_client: Any | None
    llm_options: dict[str, Any]
    tools: AgentTools
    langchain_tool_names: list[str]
    history: list[dict[str, str]]
    contextual_question: str
    plan: Any
    task_plan: Any
    planner_source: str
    generated: Any
    trace: list[AgentTraceStep]
    retrieval_state: AgentRetrievalState
    coverage_report: CoverageReport
    supplement_tool_calls: list[AgentToolCall]
    supplement_rounds: int
    supplement_observations: list[str]
    force_finalize_supplement: bool
    result: dict[str, Any]


class LangGraphAgentRunner(AgentRunner):
    """StateGraph-backed runner with the same public result contract as AgentRunner."""

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
        from src.qa_engine import CountingLLMClient, build_llm_options

        errors: list[str] = []
        llm_client = CountingLLMClient(self.engine.llm_client) if self.engine.llm_client is not None else None
        llm_options = build_llm_options(thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort)
        tools = AgentTools(self.engine, errors=errors, llm_options=llm_options, llm_client=llm_client)
        langchain_tools = build_langchain_tools(tools)
        initial_state: LangGraphAgentState = {
            "question": question.strip(),
            "conversation_history": conversation_history,
            "thinking_enabled": thinking_enabled,
            "reasoning_effort": reasoning_effort,
            "total_start": time.perf_counter(),
            "timings_ms": {},
            "errors": errors,
            "llm_client": llm_client,
            "llm_options": llm_options,
            "tools": tools,
            "langchain_tool_names": [tool.name for tool in langchain_tools],
            "trace": [],
            "supplement_tool_calls": [],
            "supplement_rounds": 0,
            "supplement_observations": [],
            "force_finalize_supplement": False,
        }
        final_state = self._graph.invoke(initial_state)
        return dict(final_state.get("result") or {})

    def _build_graph(self) -> Any:
        builder = StateGraph(LangGraphAgentState)
        builder.add_node("plan", self._plan_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("coverage_check", self._coverage_check_node)
        builder.add_node("supplement_round", self._supplement_round_node)
        builder.add_node("finalize_supplement", self._finalize_supplement_node)
        builder.add_node("verify_answer", self._verify_answer_node)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "retrieve")
        builder.add_edge("retrieve", "coverage_check")
        builder.add_conditional_edges(
            "coverage_check",
            self._route_after_coverage,
            {"supplement": "supplement_round", "finalize": "finalize_supplement"},
        )
        builder.add_edge("supplement_round", "coverage_check")
        builder.add_edge("finalize_supplement", "verify_answer")
        builder.add_edge("verify_answer", END)
        return builder.compile()

    def _plan_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        history, contextual_question, plan, task_plan, planner_source, generated, trace = self._plan(
            str(state.get("question") or ""),
            state.get("conversation_history"),
            state["tools"],
            state["timings_ms"],
        )
        return {
            "history": history,
            "contextual_question": contextual_question,
            "plan": plan,
            "task_plan": task_plan,
            "planner_source": planner_source,
            "generated": generated,
            "trace": trace,
        }

    def _retrieve_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        retrieval_state = self._retrieve(
            state["generated"],
            str(state.get("contextual_question") or ""),
            state["plan"],
            state["tools"],
            state["trace"],
            state["timings_ms"],
        )
        return {"retrieval_state": retrieval_state, "trace": state["trace"]}

    def _coverage_check_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        coverage = EvidenceCoverageChecker().check(
            state["plan"],
            state["task_plan"],
            state["retrieval_state"],
            retrieval_round=int(state.get("supplement_rounds") or 0),
        )
        tool_calls = list(state.get("supplement_tool_calls") or [])
        tool_calls.append(coverage_tool_call(coverage))
        return {"coverage_report": coverage, "supplement_tool_calls": tool_calls}

    def _route_after_coverage(self, state: LangGraphAgentState) -> Literal["supplement", "finalize"]:
        if state.get("force_finalize_supplement"):
            return "finalize"
        coverage = state.get("coverage_report")
        if coverage is not None and coverage.should_continue:
            return "supplement"
        return "finalize"

    def _supplement_round_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        coverage = state.get("coverage_report")
        if coverage is None:
            return {"force_finalize_supplement": True}
        gap = next_supplement_gap(coverage)
        if gap is None:
            return {"force_finalize_supplement": True}

        tools = state["tools"]
        timings_ms = state["timings_ms"]
        retrieval_state = state["retrieval_state"]
        round_index = int(state.get("supplement_rounds") or 0) + 1
        supplemental_question = build_supplemental_question(str(state.get("contextual_question") or ""), gap.to_dict())
        supplement_plan = tools.supplemental_plan(state["plan"], supplemental_question, gap.companies or None)
        round_calls: list[AgentToolCall] = []

        stage_start = time.perf_counter()
        generated, cypher_call = tools.prepare_cypher(supplemental_question, supplement_plan)
        graph_records, graph_call = tools.query_graph(generated, supplement_plan)
        self._record_timing(timings_ms, f"supplement_{round_index}_graph", stage_start)
        round_calls.extend([cypher_call, graph_call])

        stage_start = time.perf_counter()
        rag_hits, rag_call = tools.search_rag(supplemental_question, supplement_plan)
        self._record_timing(timings_ms, f"supplement_{round_index}_rag", stage_start)
        round_calls.append(rag_call)

        stage_start = time.perf_counter()
        research_hits, research_call = tools.search_research(supplemental_question, supplement_plan)
        self._record_timing(timings_ms, f"supplement_{round_index}_research", stage_start)
        round_calls.append(research_call)

        stage_start = time.perf_counter()
        semantic_hits, semantic_call = tools.search_semantic(supplemental_question, supplement_plan)
        self._record_timing(timings_ms, f"supplement_{round_index}_semantic", stage_start)
        round_calls.append(semantic_call)

        retrieval_state.graph_records = merge_graph_records(retrieval_state.graph_records, graph_records or [])
        retrieval_state.rag_hits = merge_rag_hits(retrieval_state.rag_hits, rag_hits or [])
        retrieval_state.research_hits = merge_research_hits(retrieval_state.research_hits, research_hits or [])
        retrieval_state.semantic_hits = merge_semantic_hits(retrieval_state.semantic_hits, semantic_hits or [])
        observations = list(state.get("supplement_observations") or [])
        observations.append(
            f"第 {round_index} 轮补检 {gap.coverage}：累计图谱 {len(retrieval_state.graph_records)} 条、"
            f"RAG {len(retrieval_state.rag_hits)} 条、投研证据 {len(retrieval_state.research_hits)} 条、"
            f"语义证据 {len(retrieval_state.semantic_hits)} 条。"
        )
        return {
            "retrieval_state": retrieval_state,
            "supplement_rounds": round_index,
            "supplement_tool_calls": [*list(state.get("supplement_tool_calls") or []), *round_calls],
            "supplement_observations": observations,
            "force_finalize_supplement": False,
        }

    def _finalize_supplement_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        coverage = state["coverage_report"]
        retrieval_state = state["retrieval_state"]
        retrieval_state.coverage_report = coverage
        rounds = int(state.get("supplement_rounds") or 0)
        missing_text = "、".join(coverage.missing) if coverage.missing else "无"
        if rounds == 0 and coverage.sufficient:
            thought = "检查证据结构后，当前证据已经覆盖主要问题要素。"
            action = "skip_supplement"
            observation = str(coverage.stop_reason)
        else:
            thought = "根据证据覆盖检查结果，多轮补检缺失的公司、风险、指标、敞口或机理证据。"
            action = "supplemental_retrieve" if rounds else "skip_supplement"
            observations = "；".join(state.get("supplement_observations") or [])
            observation = (
                (observations + "；" if observations else "")
                + f"stop_reason={coverage.stop_reason}，missing={missing_text}。"
            )
        trace.append(
            AgentTraceStep(
                step=len(trace) + 1,
                phase="supplement",
                thought=thought,
                action=action,
                tool_calls=list(state.get("supplement_tool_calls") or []),
                observation=observation,
            )
        )
        return {"trace": trace, "retrieval_state": retrieval_state}

    def _verify_answer_node(self, state: LangGraphAgentState) -> dict[str, Any]:
        result = self._verify_and_answer(
            str(state.get("question") or ""),
            str(state.get("contextual_question") or ""),
            state.get("history") or [],
            state["plan"],
            state["task_plan"],
            str(state.get("planner_source") or ""),
            state["retrieval_state"],
            state["coverage_report"],
            state["tools"],
            state["trace"],
            state["timings_ms"],
            state["errors"],
            state.get("llm_client"),
            thinking_enabled=state.get("thinking_enabled"),
            reasoning_effort=state.get("reasoning_effort"),
            total_start=float(state.get("total_start") or time.perf_counter()),
        )
        diagnostics = result.setdefault("diagnostics", {})
        diagnostics["agent_runner"] = "langgraph"
        diagnostics["langgraph_enabled"] = True
        diagnostics["langchain_tools"] = list(state.get("langchain_tool_names") or [])
        return {"result": result}
