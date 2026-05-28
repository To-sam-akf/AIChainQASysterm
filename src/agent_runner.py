"""Rule-based ReAct-style runner for the first agentic QA phase."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from src.agent_tools import AgentToolCall, AgentTools
from src.professional_qa import EvidenceCard, assign_citation_ids, legacy_evidence_rows
from src.question_planner import QuestionPlan
from src.rag_index import RagHit
from src.research_claims import ResearchHit


MAX_AGENT_STEPS = 4


@dataclass
class AgentTraceStep:
    step: int
    phase: str
    thought: str
    action: str
    tool_calls: list[AgentToolCall] = field(default_factory=list)
    observation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "phase": self.phase,
            "thought": self.thought,
            "action": self.action,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "observation": self.observation,
        }


@dataclass
class AgentRetrievalState:
    graph_records: list[dict[str, Any]] = field(default_factory=list)
    rag_hits: list[RagHit] = field(default_factory=list)
    research_hits: list[ResearchHit] = field(default_factory=list)
    generated: Any | None = None


class AgentRunner:
    def __init__(self, engine: Any, *, max_steps: int = MAX_AGENT_STEPS) -> None:
        self.engine = engine
        self.max_steps = max(1, min(int(max_steps or MAX_AGENT_STEPS), MAX_AGENT_STEPS))

    def run(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        from src.qa_engine import CountingLLMClient, build_llm_options

        total_start = time.perf_counter()
        timings_ms: dict[str, float] = {}
        errors: list[str] = []
        question = question.strip()
        llm_client = CountingLLMClient(self.engine.llm_client) if self.engine.llm_client is not None else None
        llm_options = build_llm_options(thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort)
        tools = AgentTools(self.engine, errors=errors, llm_options=llm_options, llm_client=llm_client)

        history, contextual_question, plan, planner_source, generated, trace = self._plan(
            question,
            conversation_history,
            tools,
            timings_ms,
        )
        retrieval_state = self._retrieve(generated, contextual_question, plan, tools, trace, timings_ms)
        self._supplement(contextual_question, plan, tools, retrieval_state, trace, timings_ms)
        result = self._verify_and_answer(
            question,
            contextual_question,
            history,
            plan,
            planner_source,
            retrieval_state,
            tools,
            trace,
            timings_ms,
            errors,
            llm_client,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
            total_start=total_start,
        )
        return result

    def run_stream(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        from src.qa_engine import CountingLLMClient, build_llm_options, stream_progress

        total_start = time.perf_counter()
        timings_ms: dict[str, float] = {}
        errors: list[str] = []
        question = question.strip()
        llm_client = CountingLLMClient(self.engine.llm_client) if self.engine.llm_client is not None else None
        llm_options = build_llm_options(thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort)
        tools = AgentTools(self.engine, errors=errors, llm_options=llm_options, llm_client=llm_client)

        if thinking_enabled:
            yield stream_progress("agent_plan", "Agent 正在规划问题、上下文和检索路径")
        history, contextual_question, plan, planner_source, generated, trace = self._plan(
            question,
            conversation_history,
            tools,
            timings_ms,
        )
        if thinking_enabled:
            yield stream_progress("agent_plan", trace[-1].observation)

        if thinking_enabled:
            yield stream_progress("agent_retrieve", "Agent 正在调用图谱、RAG 和投研证据工具")
        retrieval_state = self._retrieve(generated, contextual_question, plan, tools, trace, timings_ms)
        if thinking_enabled:
            yield stream_progress("agent_retrieve", trace[-1].observation)

        if thinking_enabled:
            yield stream_progress("agent_supplement", "Agent 正在检查是否需要补充检索")
        self._supplement(contextual_question, plan, tools, retrieval_state, trace, timings_ms)
        if thinking_enabled:
            yield stream_progress("agent_supplement", trace[-1].observation)

        stage_start = time.perf_counter()
        raw_cards, evidence_cards, rank_call = tools.rank_evidence(
            retrieval_state.research_hits,
            retrieval_state.graph_records,
            retrieval_state.rag_hits,
            plan,
        )
        evidence_cards = assign_citation_ids(evidence_cards)
        self._record_timing(timings_ms, "evidence", stage_start)

        answer = ""
        reasoning_content = ""
        stage_start = time.perf_counter()
        for event in self.engine._generate_answer_stream(
            question,
            contextual_question,
            history,
            plan,
            retrieval_state.graph_records,
            evidence_cards,
            errors,
            llm_options,
            llm_client,
            thinking_enabled=bool(thinking_enabled),
        ):
            if event.get("type") in {"answer_delta", "progress"}:
                yield event
                continue
            if event.get("type") == "answer_complete":
                answer = str(event.get("answer") or "")
                reasoning_content = str(event.get("reasoning_content") or "")
        self._record_timing(timings_ms, "answer", stage_start)
        verification, verify_call = tools.verify_answer_support(answer, plan, evidence_cards, raw_cards)
        trace.append(
            AgentTraceStep(
                step=len(trace) + 1,
                phase="verify_answer",
                thought="用筛选后的证据生成答案，并检查引用、公司覆盖和无支撑表述。",
                action="rank_evidence -> generate_answer_stream -> verify_answer_support",
                tool_calls=[
                    rank_call,
                    AgentToolCall(
                        tool="generate_answer_stream",
                        args={"answer_type": plan.answer_type, "evidence_cards": len(evidence_cards)},
                        result_count=1 if answer else 0,
                        elapsed_ms=timings_ms.get("answer", 0),
                        error="",
                    ),
                    verify_call,
                ],
                observation=f"生成答案，verification={verification.get('status')}，证据卡 {len(evidence_cards)} 条。",
            )
        )
        result = self._build_result(
            question,
            contextual_question,
            answer,
            reasoning_content,
            plan,
            planner_source,
            generated,
            retrieval_state,
            evidence_cards,
            verification,
            trace,
            timings_ms,
            errors,
            llm_client,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
            total_start=total_start,
            history_count=len(history),
        )
        yield {"type": "final", "result": result}

    def _plan(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None,
        tools: AgentTools,
        timings_ms: dict[str, float],
    ) -> tuple[list[dict[str, str]], str, QuestionPlan, str, Any, list[AgentTraceStep]]:
        from src.qa_engine import describe_plan_progress, normalize_conversation_history

        trace: list[AgentTraceStep] = []
        stage_start = time.perf_counter()
        history = normalize_conversation_history(
            conversation_history,
            max_turns=self.engine.history_max_turns,
            max_chars=self.engine.history_max_chars,
        )
        self._record_timing(timings_ms, "history", stage_start)

        stage_start = time.perf_counter()
        contextual_question, contextualize_call = tools.contextualize_question(question, history)
        self._record_timing(timings_ms, "contextualize", stage_start)

        stage_start = time.perf_counter()
        plan, planner_source, plan_call = tools.plan_question(contextual_question)
        self._record_timing(timings_ms, "plan", stage_start)

        stage_start = time.perf_counter()
        generated, cypher_call = tools.prepare_cypher(contextual_question, plan)
        self._record_timing(timings_ms, "cypher", stage_start)

        trace.append(
            AgentTraceStep(
                step=1,
                phase="plan",
                thought="先把问题改写为可检索问题，并确定答案类型、公司、主题和图谱查询。",
                action="contextualize_question -> plan_question -> prepare_cypher",
                tool_calls=[contextualize_call, plan_call, cypher_call],
                observation=describe_plan_progress(plan),
            )
        )
        return history, contextual_question, plan, planner_source, generated, trace

    def _retrieve(
        self,
        generated: Any,
        contextual_question: str,
        plan: QuestionPlan,
        tools: AgentTools,
        trace: list[AgentTraceStep],
        timings_ms: dict[str, float],
    ) -> AgentRetrievalState:
        stage_start = time.perf_counter()
        graph_records, graph_call = tools.query_graph(generated, plan)
        self._record_timing(timings_ms, "graph", stage_start)

        stage_start = time.perf_counter()
        rag_hits, rag_call = tools.search_rag(contextual_question, plan)
        self._record_timing(timings_ms, "rag", stage_start)

        stage_start = time.perf_counter()
        research_hits, research_call = tools.search_research(contextual_question, plan)
        self._record_timing(timings_ms, "research", stage_start)

        trace.append(
            AgentTraceStep(
                step=len(trace) + 1,
                phase="retrieve",
                thought="同时利用结构化图谱、原文 RAG 和投研 Claim/Dossier 做第一轮召回。",
                action="query_graph -> search_rag -> search_research",
                tool_calls=[graph_call, rag_call, research_call],
                observation=f"召回图谱 {len(graph_records)} 条、RAG {len(rag_hits)} 条、投研证据 {len(research_hits)} 条。",
            )
        )
        return AgentRetrievalState(
            graph_records=graph_records,
            rag_hits=rag_hits,
            research_hits=research_hits,
            generated=generated,
        )

    def _supplement(
        self,
        contextual_question: str,
        plan: QuestionPlan,
        tools: AgentTools,
        state: AgentRetrievalState,
        trace: list[AgentTraceStep],
        timings_ms: dict[str, float],
    ) -> None:
        decision = supplement_decision(plan, state)
        tool_calls: list[AgentToolCall] = []
        if not decision["needed"]:
            trace.append(
                AgentTraceStep(
                    step=len(trace) + 1,
                    phase="supplement",
                    thought="检查证据结构后，当前证据已经覆盖主要问题要素。",
                    action="skip_supplement",
                    tool_calls=[],
                    observation=str(decision["reason"]),
                )
            )
            return

        supplemental_question = f"{contextual_question} {decision['query_suffix']}".strip()
        supplement_plan = tools.supplemental_plan(plan, supplemental_question, decision.get("companies"))
        stage_start = time.perf_counter()
        generated, cypher_call = tools.prepare_cypher(supplemental_question, supplement_plan)
        graph_records, graph_call = tools.query_graph(generated, supplement_plan)
        self._record_timing(timings_ms, "supplement_graph", stage_start)
        tool_calls.extend([cypher_call, graph_call])

        stage_start = time.perf_counter()
        rag_hits, rag_call = tools.search_rag(supplemental_question, supplement_plan)
        self._record_timing(timings_ms, "supplement_rag", stage_start)
        tool_calls.append(rag_call)

        stage_start = time.perf_counter()
        research_hits, research_call = tools.search_research(supplemental_question, supplement_plan)
        self._record_timing(timings_ms, "supplement_research", stage_start)
        tool_calls.append(research_call)

        state.graph_records = merge_graph_records(state.graph_records, graph_records)
        state.rag_hits = merge_rag_hits(state.rag_hits, rag_hits)
        state.research_hits = merge_research_hits(state.research_hits, research_hits)
        trace.append(
            AgentTraceStep(
                step=len(trace) + 1,
                phase="supplement",
                thought=str(decision["reason"]),
                action="supplemental_retrieve",
                tool_calls=tool_calls,
                observation=(
                    f"补检后累计图谱 {len(state.graph_records)} 条、RAG {len(state.rag_hits)} 条、"
                    f"投研证据 {len(state.research_hits)} 条。"
                ),
            )
        )

    def _verify_and_answer(
        self,
        question: str,
        contextual_question: str,
        history: list[dict[str, str]],
        plan: QuestionPlan,
        planner_source: str,
        state: AgentRetrievalState,
        tools: AgentTools,
        trace: list[AgentTraceStep],
        timings_ms: dict[str, float],
        errors: list[str],
        llm_client: Any | None,
        *,
        thinking_enabled: bool | None,
        reasoning_effort: str | None,
        total_start: float,
    ) -> dict[str, Any]:
        stage_start = time.perf_counter()
        raw_cards, evidence_cards, rank_call = tools.rank_evidence(
            state.research_hits,
            state.graph_records,
            state.rag_hits,
            plan,
        )
        evidence_cards = assign_citation_ids(evidence_cards)
        self._record_timing(timings_ms, "evidence", stage_start)

        stage_start = time.perf_counter()
        (answer, reasoning_content), answer_call = tools.generate_answer(
            question,
            contextual_question,
            history,
            plan,
            state.graph_records,
            evidence_cards,
        )
        self._record_timing(timings_ms, "answer", stage_start)

        verification, verify_call = tools.verify_answer_support(answer, plan, evidence_cards, raw_cards)
        trace.append(
            AgentTraceStep(
                step=len(trace) + 1,
                phase="verify_answer",
                thought="筛选最终证据，生成答案，并对引用、公司覆盖和无支撑表述做自检。",
                action="rank_evidence -> generate_answer -> verify_answer_support",
                tool_calls=[rank_call, answer_call, verify_call],
                observation=f"生成答案，verification={verification.get('status')}，证据卡 {len(evidence_cards)} 条。",
            )
        )
        return self._build_result(
            question,
            contextual_question,
            answer,
            reasoning_content,
            plan,
            planner_source,
            state.generated,
            state,
            evidence_cards,
            verification,
            trace,
            timings_ms,
            errors,
            llm_client,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
            total_start=total_start,
            history_count=len(history),
        )

    def _build_result(
        self,
        question: str,
        contextual_question: str,
        answer: str,
        reasoning_content: str,
        plan: QuestionPlan,
        planner_source: str,
        generated: Any,
        state: AgentRetrievalState,
        evidence_cards: list[EvidenceCard],
        verification: dict[str, Any],
        trace: list[AgentTraceStep],
        timings_ms: dict[str, float],
        errors: list[str],
        llm_client: Any | None,
        *,
        thinking_enabled: bool | None,
        reasoning_effort: str | None,
        total_start: float,
        history_count: int,
    ) -> dict[str, Any]:
        from src.qa_engine import answer_subgraph, detect_unsupported_terms

        stage_start = time.perf_counter()
        evidence = legacy_evidence_rows(evidence_cards)
        rag_hit_rows = [hit.to_dict() for hit in state.rag_hits]
        research_hit_rows = [hit.to_dict() for hit in state.research_hits]
        evidence_card_rows = [card.to_dict() for card in evidence_cards]
        subgraph = answer_subgraph(state.graph_records, evidence_cards)
        unsupported_terms = detect_unsupported_terms(answer, evidence_cards)
        self._record_timing(timings_ms, "render_payload", stage_start)

        diagnostics = {
            "graph_backend": self.engine.status.graph_backend,
            "graph_records": len(state.graph_records),
            "rag_hits": len(state.rag_hits),
            "research_hits": len(state.research_hits),
            "evidence_cards": len(evidence_cards),
            "rerank_top_n": self.engine.rerank_top_n,
            "history_messages": history_count,
            "contextualized": contextual_question != question,
            "contextualizer_mode": self.engine.contextualizer_mode,
            "planner_source": planner_source,
            "enable_llm_cypher": self.engine.enable_llm_cypher,
            "enable_llm_planner": self.engine.enable_llm_planner,
            "thinking_enabled": thinking_enabled,
            "reasoning_effort": reasoning_effort or "",
            "graph_error": self.engine.status.graph_error,
            "rag_error": self.engine.status.rag_error,
            "research_error": self.engine.status.research_error,
            "llm_error": self.engine.status.llm_error,
            "unsupported_terms": unsupported_terms,
            "agent_enabled": True,
            "agent_max_steps": self.max_steps,
            "agent_steps": min(len(trace), self.max_steps),
            "agent_trace": [step.to_dict() for step in trace[: self.max_steps]],
            "agent_verification": verification,
        }
        timings_ms["total"] = round((time.perf_counter() - total_start) * 1000, 2)
        diagnostics["timings_ms"] = timings_ms
        diagnostics["llm_calls"] = llm_client.calls if llm_client is not None else {"total": 0}

        return {
            "question": question,
            "contextual_question": contextual_question,
            "answer": answer,
            "reasoning_content": reasoning_content,
            "answer_type": plan.answer_type,
            "plan": plan.to_dict(),
            "cypher": generated.cypher,
            "cypher_params": generated.params,
            "cypher_source": generated.source,
            "graph_records": state.graph_records,
            "rag_hits": rag_hit_rows,
            "research_hits": research_hit_rows,
            "evidence_cards": evidence_card_rows,
            "evidence": evidence,
            "subgraph": subgraph,
            "diagnostics": diagnostics,
            "errors": errors,
        }

    @staticmethod
    def _record_timing(timings_ms: dict[str, float], name: str, started_at: float) -> None:
        timings_ms[name] = round((time.perf_counter() - started_at) * 1000, 2)


def supplement_decision(plan: QuestionPlan, state: AgentRetrievalState) -> dict[str, Any]:
    if plan.answer_type == "risk_analysis" and not has_risk_evidence(state):
        return {
            "needed": True,
            "reason": "风险分析问题缺少明确风险证据，需要补检风险、反证和不确定性。",
            "query_suffix": "风险 反证 不确定性",
        }
    if plan.answer_type == "company_compare":
        missing = missing_companies(plan, state)
        if missing:
            return {
                "needed": True,
                "reason": f"公司对比证据未覆盖：{'、'.join(missing)}，需要按缺失公司补检。",
                "query_suffix": "业务 差异 指标 风险",
                "companies": missing,
            }
    if plan.answer_type == "topic_to_company" and not has_company_exposure(state):
        return {
            "needed": True,
            "reason": "主题到公司问题缺少公司敞口证据，需要补检公司敞口和受益公司。",
            "query_suffix": "公司敞口 受益 上市公司",
        }
    if plan.answer_type in {"industry_bottleneck", "thematic_research"} and not has_mechanism_evidence(state):
        return {
            "needed": True,
            "reason": "主题研究缺少技术机理或瓶颈证据，需要补检技术机理、产业传导和指标。",
            "query_suffix": "技术机理 瓶颈 传导 指标",
        }
    return {"needed": False, "reason": "证据结构已满足当前答案类型的最低要求。", "query_suffix": ""}


def has_risk_evidence(state: AgentRetrievalState) -> bool:
    if any(record.get("relation") == "DISCLOSES_RISK" for record in state.graph_records):
        return True
    if any(hit.claim_type == "risk" for hit in state.research_hits):
        return True
    risk_terms = ("风险", "不确定", "波动", "不及预期")
    return any(any(term in hit.snippet for term in risk_terms) for hit in state.rag_hits)


def has_company_exposure(state: AgentRetrievalState) -> bool:
    if any(record.get("company") for record in state.graph_records):
        return True
    return any(hit.claim_type == "company_exposure" and hit.company for hit in state.research_hits)


def has_mechanism_evidence(state: AgentRetrievalState) -> bool:
    mechanism_types = {"mechanism", "bottleneck", "supply_chain", "trend", "indicator"}
    if any(hit.claim_type in mechanism_types for hit in state.research_hits):
        return True
    mechanism_relations = {"CONSTRAINS", "ENABLES", "DRIVES", "DEPENDS_ON", "HAS_INDICATOR"}
    if any(record.get("relation") in mechanism_relations for record in state.graph_records):
        return True
    mechanism_terms = ("机理", "瓶颈", "传导", "指标", "带宽", "功耗", "散热")
    return any(any(term in hit.snippet for term in mechanism_terms) for hit in state.rag_hits)


def missing_companies(plan: QuestionPlan, state: AgentRetrievalState) -> list[str]:
    if not plan.companies:
        return []
    covered = {
        str(record.get("company") or "")
        for record in state.graph_records
        if record.get("company")
    }
    covered.update(hit.company for hit in state.research_hits if hit.company)
    covered.update(hit.company for hit in state.rag_hits if hit.company)
    return [company for company in plan.companies if company not in covered]


def merge_graph_records(existing: list[dict[str, Any]], additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = list(existing)
    seen = {
        (
            row.get("company", ""),
            row.get("relation", ""),
            row.get("target", ""),
            row.get("evidence", ""),
        )
        for row in output
    }
    for row in additions:
        key = (row.get("company", ""), row.get("relation", ""), row.get("target", ""), row.get("evidence", ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def merge_rag_hits(existing: list[RagHit], additions: list[RagHit]) -> list[RagHit]:
    output = list(existing)
    seen = {hit.chunk_id or f"{hit.source_title}:{hit.page}:{hit.snippet[:80]}" for hit in output}
    for hit in additions:
        key = hit.chunk_id or f"{hit.source_title}:{hit.page}:{hit.snippet[:80]}"
        if key in seen:
            continue
        seen.add(key)
        output.append(hit)
    return output


def merge_research_hits(existing: list[ResearchHit], additions: list[ResearchHit]) -> list[ResearchHit]:
    output = list(existing)
    seen = {(hit.kind, hit.title, hit.company, hit.text[:120]) for hit in output}
    for hit in additions:
        key = (hit.kind, hit.title, hit.company, hit.text[:120])
        if key in seen:
            continue
        seen.add(key)
        output.append(hit)
    return output
