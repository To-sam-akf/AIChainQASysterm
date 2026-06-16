"""Centralized multi-agent coordination for parallel evidence retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from src.agent_runner import (
    AgentRetrievalState,
    merge_graph_records,
    merge_rag_hits,
    merge_research_hits,
    merge_semantic_hits,
)
from src.agent_tools import (
    AgentToolCall,
    AgentTools,
    cards_from_graphrag_result,
    cards_from_semantic_hits,
)
from src.agents.coverage import CoverageReport, EvidenceCoverageChecker, EvidenceGap
from src.agents.planner import AgentTaskPlan, query_suffix_for_coverage
from src.extraction_schema import parse_json_object
from src.graphrag import (
    GraphRagResult,
    dedupe_hits,
    finalize_graphrag,
    retrieve_graphrag,
)
from src.professional_qa import (
    EvidenceCard,
    assign_citation_ids,
    card_identity,
    cards_from_graph_records,
    cards_from_rag_hits,
    cards_from_research_hits,
    rank_evidence_cards,
)
from src.question_planner import QuestionPlan


AGENT_TYPES = ("graph", "rag", "research", "semantic", "graphrag")
LLM_METHODS = {
    "chat_json",
    "chat_text",
    "chat_text_with_metadata",
    "chat_messages",
    "stream_chat_messages",
}


class LLMBudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentAssignment:
    assignment_id: str
    agent_type: str
    query: str
    filters: dict[str, Any] = field(default_factory=dict)
    coverage_goals: list[str] = field(default_factory=list)
    round_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentResult:
    assignment: AgentAssignment
    status: str
    output: Any = None
    tool_calls: list[AgentToolCall] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str = ""
    degraded: bool = False
    timed_out: bool = False

    def diagnostic(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment.assignment_id,
            "agent_type": self.assignment.agent_type,
            "status": self.status,
            "result_count": result_count(self.output),
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "degraded": self.degraded,
            "timed_out": self.timed_out,
        }


@dataclass(frozen=True)
class EvidenceCandidate:
    candidate_id: str
    card: EvidenceCard
    retrieval_agent: str

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.card.kind,
            "title": self.card.title,
            "evidence": shorten(self.card.evidence, 500),
            "source": self.card.source,
            "page": self.card.page,
            "company": self.card.company,
            "topic": self.card.topic,
            "claim_type": self.card.claim_type,
            "relation": self.card.relation,
            "score": self.card.score,
            "retrieval_agent": self.retrieval_agent,
        }


@dataclass(frozen=True)
class ReviewReport:
    accepted_ids: list[str]
    rejected_ids: list[str]
    conflicts: list[dict[str, Any]]
    coverage: CoverageReport
    source: str
    degraded: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_ids": list(self.accepted_ids),
            "rejected_ids": list(self.rejected_ids),
            "conflicts": list(self.conflicts),
            "coverage": self.coverage.to_dict(),
            "source": self.source,
            "degraded": self.degraded,
            "error": self.error,
        }


@dataclass(frozen=True)
class EvidenceCardDraft:
    candidate_ids: list[str]
    title: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CentralLLMBudget:
    """Thread-safe call budget shared by all centralized agents."""

    def __init__(self, client: Any | None, *, max_calls: int) -> None:
        self._client = client
        self.max_calls = max(0, int(max_calls or 0))
        self._lock = threading.Lock()
        self._used = 0
        self._method_calls: dict[str, int] = {}
        self._role_calls: dict[str, int] = {}
        self._fallbacks: dict[str, int] = {}
        self._errors: list[str] = []

    def client(self, role: str) -> "RoleLLMClient | None":
        if self._client is None:
            return None
        return RoleLLMClient(self, role)

    def reserve(self, role: str, method: str) -> Any:
        with self._lock:
            if self._client is None:
                raise LLMBudgetExhausted("LLM is not configured")
            if self._used >= self.max_calls:
                raise LLMBudgetExhausted(f"LLM call budget exhausted ({self.max_calls})")
            self._used += 1
            self._method_calls[method] = self._method_calls.get(method, 0) + 1
            self._role_calls[role] = self._role_calls.get(role, 0) + 1
            return getattr(self._client, method)

    def note_fallback(self, role: str, error: str = "") -> None:
        with self._lock:
            self._fallbacks[role] = self._fallbacks.get(role, 0) + 1
            if error:
                self._errors.append(f"{role}: {error}")

    @property
    def calls(self) -> dict[str, int]:
        with self._lock:
            return {"total": self._used, **dict(self._method_calls)}

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_calls": self.max_calls,
                "used_calls": self._used,
                "remaining_calls": max(0, self.max_calls - self._used),
                "calls_by_role": dict(self._role_calls),
                "fallbacks_by_role": dict(self._fallbacks),
                "fallback_count": sum(self._fallbacks.values()),
                "errors": list(self._errors),
            }


class RoleLLMClient:
    def __init__(self, budget: CentralLLMBudget, role: str) -> None:
        self._budget = budget
        self._role = role

    @property
    def calls(self) -> dict[str, int]:
        return self._budget.calls

    def __getattr__(self, name: str) -> Any:
        underlying = self._budget._client
        if underlying is None:
            raise AttributeError(name)
        attr = getattr(underlying, name)
        if name not in LLM_METHODS or not callable(attr):
            return attr

        def budgeted(*args: Any, **kwargs: Any) -> Any:
            method = self._budget.reserve(self._role, name)
            return method(*args, **kwargs)

        return budgeted


class CentralizedCoordinator:
    def __init__(
        self,
        engine: Any,
        *,
        llm_budget: CentralLLMBudget,
        llm_options: dict[str, Any],
    ) -> None:
        self.engine = engine
        self.llm_budget = llm_budget
        self.llm_options = dict(llm_options)
        self.max_workers = max(1, int(getattr(engine, "multi_agent_max_workers", 5) or 5))
        self.timeout_seconds = max(
            1.0,
            float(getattr(engine, "multi_agent_task_timeout_seconds", 90.0) or 90.0),
        )
        self._runtime_lock = threading.Lock()
        self._active_workers = 0
        self._peak_concurrency = 0
        self._timeouts = 0
        self._rounds: list[dict[str, Any]] = []

    def dispatch(
        self,
        *,
        question: str,
        plan: QuestionPlan,
        task_plan: AgentTaskPlan,
        round_index: int,
        gaps: list[EvidenceGap] | None = None,
    ) -> list[AgentAssignment]:
        available = self.available_agents()
        fallback = self._fallback_assignments(
            question=question,
            task_plan=task_plan,
            round_index=round_index,
            available=available,
            gaps=gaps or [],
        )
        if not available:
            return []
        payload = self._safe_json(
            role="supervisor",
            system_prompt=(
                "你是中心调度 Agent。只输出严格 JSON，负责选择检索 Agent 和改写各路查询。"
                "只能使用给定 agent_type，不回答用户问题。"
            ),
            user_prompt=json.dumps(
                {
                    "question": question,
                    "question_plan": plan.to_dict(),
                    "required_coverage": task_plan.required_coverage,
                    "review_gaps": [gap.to_dict() for gap in gaps or []],
                    "available_agents": available,
                    "schema": {
                        "assignments": [
                            {
                                "agent_type": "graph|rag|research|semantic|graphrag",
                                "query": "string",
                                "filters": {},
                                "coverage_goals": ["string"],
                            }
                        ]
                    },
                },
                ensure_ascii=False,
            ),
        )
        assignments = validate_assignments(
            payload,
            available=available,
            question=question,
            round_index=round_index,
            max_assignments=self.max_workers,
        )
        if assignments:
            return assignments
        self.llm_budget.note_fallback("supervisor", "invalid or empty dispatch payload")
        return fallback

    def execute_wave(
        self,
        assignments: list[AgentAssignment],
        *,
        plan: QuestionPlan,
        hyde_query: Any | None = None,
    ) -> list[AgentResult]:
        if not assignments:
            return []
        started_at = time.perf_counter()
        executor = ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(assignments)),
            thread_name_prefix="qa-query-agent",
        )
        futures: dict[Future[AgentResult], AgentAssignment] = {
            executor.submit(self._run_assignment_with_metrics, assignment, plan, hyde_query): assignment
            for assignment in assignments
        }
        done, pending = wait(futures, timeout=self.timeout_seconds)
        by_id: dict[str, AgentResult] = {}
        for future in done:
            assignment = futures[future]
            try:
                by_id[assignment.assignment_id] = future.result()
            except Exception as exc:
                by_id[assignment.assignment_id] = AgentResult(
                    assignment=assignment,
                    status="error",
                    error=str(exc),
                )
        for future in pending:
            assignment = futures[future]
            future.cancel()
            with self._runtime_lock:
                self._timeouts += 1
            by_id[assignment.assignment_id] = AgentResult(
                assignment=assignment,
                status="timeout",
                elapsed_ms=round(self.timeout_seconds * 1000, 2),
                error=f"query agent timed out after {self.timeout_seconds:g}s",
                degraded=True,
                timed_out=True,
            )
        executor.shutdown(wait=False, cancel_futures=True)
        results = [
            by_id.get(
                assignment.assignment_id,
                AgentResult(assignment=assignment, status="error", error="missing agent result"),
            )
            for assignment in assignments
        ]
        with self._runtime_lock:
            self._rounds.append(
                {
                    "round": assignments[0].round_index,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    "assignments": [assignment.to_dict() for assignment in assignments],
                    "results": [result.diagnostic() for result in results],
                }
            )
        return results

    def join_results(
        self,
        state: AgentRetrievalState,
        results: list[AgentResult],
        *,
        plan: QuestionPlan,
        existing_entries: list[tuple[EvidenceCard, str]],
    ) -> tuple[AgentRetrievalState, list[tuple[EvidenceCard, str]], list[AgentToolCall], list[str]]:
        entries = list(existing_entries)
        calls: list[AgentToolCall] = []
        errors: list[str] = []
        partials: list[GraphRagResult] = []
        for result in results:
            calls.extend(result.tool_calls)
            if result.error:
                errors.append(f"{result.assignment.agent_type} agent failed: {result.error}")
            agent_type = result.assignment.agent_type
            output = result.output
            if agent_type == "graph" and isinstance(output, dict):
                records = list(output.get("records") or [])
                state.graph_records = merge_graph_records(state.graph_records, records)
                generated = output.get("generated")
                if generated is not None:
                    state.generated = generated
                entries.extend((card, "graph") for card in cards_from_graph_records(records, plan))
            elif agent_type == "rag":
                hits = list(output or [])
                state.rag_hits = merge_rag_hits(state.rag_hits, hits)
                entries.extend((card, "rag") for card in cards_from_rag_hits(hits, plan))
            elif agent_type == "research":
                hits = list(output or [])
                state.research_hits = merge_research_hits(state.research_hits, hits)
                entries.extend((card, "research") for card in cards_from_research_hits(hits, plan))
            elif agent_type == "semantic":
                hits = list(output or [])
                state.semantic_hits = merge_semantic_hits(state.semantic_hits, hits)
                entries.extend((card, "semantic") for card in cards_from_semantic_hits(hits, plan))
            elif agent_type == "graphrag" and isinstance(output, GraphRagResult):
                partials.append(output)
                state.research_hits = merge_research_hits(
                    state.research_hits,
                    [*output.global_hits, *output.local_hits],
                )
                entries.extend(
                    (card, "graphrag")
                    for card in cards_from_research_hits([*output.global_hits, *output.local_hits], plan)
                )

        if partials:
            partial = merge_graphrag_results(state.graphrag, partials)
            state.graphrag = finalize_graphrag(
                partial,
                plan=plan,
                graph_records=state.graph_records,
                path_top_k=int(getattr(self.engine, "graph_path_top_k", 6)),
            )
            entries.extend((card, "graphrag") for card in cards_from_graphrag_result(state.graphrag, plan))
        elif state.graphrag is not None:
            state.graphrag = finalize_graphrag(
                state.graphrag,
                plan=plan,
                graph_records=state.graph_records,
                path_top_k=int(getattr(self.engine, "graph_path_top_k", 6)),
            )
        return state, entries, calls, errors

    def build_candidates(
        self,
        entries: list[tuple[EvidenceCard, str]],
        *,
        plan: QuestionPlan,
    ) -> list[EvidenceCandidate]:
        by_identity: dict[tuple[str, str, str, str], tuple[EvidenceCard, str]] = {}
        for card, agent in entries:
            if not card.evidence.strip():
                continue
            by_identity.setdefault(card_identity(card), (card, agent))
        cards = [item[0] for item in by_identity.values()]
        deterministic = rank_evidence_cards(cards, limit=max(len(cards), 1), plan=plan)
        agent_by_identity = {identity: value[1] for identity, value in by_identity.items()}
        return [
            EvidenceCandidate(
                candidate_id=candidate_id(card),
                card=card,
                retrieval_agent=agent_by_identity.get(card_identity(card), ""),
            )
            for card in deterministic
        ]

    def review(
        self,
        *,
        question: str,
        plan: QuestionPlan,
        task_plan: AgentTaskPlan,
        retrieval_state: AgentRetrievalState,
        candidates: list[EvidenceCandidate],
        round_index: int,
    ) -> ReviewReport:
        deterministic_coverage = EvidenceCoverageChecker().check(
            plan,
            task_plan,
            retrieval_state,
            retrieval_round=round_index,
        )
        candidate_ids = {candidate.candidate_id for candidate in candidates}
        fallback_ids = [candidate.candidate_id for candidate in candidates]
        payload = self._safe_json(
            role="reviewer",
            system_prompt=(
                "你是证据审核 Agent。检查相关性、来源质量、冲突和覆盖度。"
                "只能引用候选 ID，只输出严格 JSON，不生成新证据。"
            ),
            user_prompt=json.dumps(
                {
                    "question": question,
                    "question_plan": plan.to_dict(),
                    "required_coverage": task_plan.required_coverage,
                    "deterministic_coverage": deterministic_coverage.to_dict(),
                    "candidates": [candidate.prompt_payload() for candidate in candidates[:40]],
                    "schema": {
                        "accepted_ids": ["candidate_id"],
                        "rejected_ids": ["candidate_id"],
                        "missing_coverage": ["required coverage name"],
                        "conflicts": [{"candidate_ids": ["candidate_id"], "reason": "string"}],
                    },
                },
                ensure_ascii=False,
            ),
        )
        if not isinstance(payload, dict):
            self.llm_budget.note_fallback("reviewer", "invalid review payload")
            return ReviewReport(
                accepted_ids=fallback_ids,
                rejected_ids=[],
                conflicts=[],
                coverage=deterministic_coverage,
                source="deterministic",
                degraded=True,
                error="invalid review payload",
            )
        accepted = clean_ids(payload.get("accepted_ids"), candidate_ids)
        rejected = clean_ids(payload.get("rejected_ids"), candidate_ids)
        if candidates and not accepted:
            self.llm_budget.note_fallback("reviewer", "review rejected every deterministic candidate")
            accepted = fallback_ids
            rejected = []
            source = "deterministic"
            degraded = True
        else:
            accepted = [item for item in accepted if item not in rejected]
            source = "llm"
            degraded = False
        conflicts = validate_conflicts(payload.get("conflicts"), candidate_ids)
        missing = [
            value
            for value in clean_strings(payload.get("missing_coverage"))
            if value in task_plan.required_coverage
        ]
        coverage = merge_review_coverage(deterministic_coverage, missing, plan)
        return ReviewReport(
            accepted_ids=accepted,
            rejected_ids=rejected,
            conflicts=conflicts,
            coverage=coverage,
            source=source,
            degraded=degraded,
        )

    def summarize(
        self,
        *,
        question: str,
        plan: QuestionPlan,
        candidates: list[EvidenceCandidate],
        accepted_ids: list[str],
    ) -> tuple[list[EvidenceCard], list[EvidenceCard], list[EvidenceCardDraft], dict[str, Any]]:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        accepted = [by_id[item] for item in accepted_ids if item in by_id]
        raw_cards = [
            replace(
                candidate.card,
                candidate_ids=[candidate.candidate_id],
                retrieval_agent=candidate.retrieval_agent,
            )
            for candidate in candidates
        ]
        payload = self._safe_json(
            role="summarizer",
            system_prompt=(
                "你是证据归纳 Agent。只能选择候选、生成简短标题和选择理由。"
                "不得改写 evidence，不得生成候选之外的事实，只输出严格 JSON。"
            ),
            user_prompt=json.dumps(
                {
                    "question": question,
                    "question_plan": plan.to_dict(),
                    "max_cards": int(getattr(self.engine, "evidence_top_n", 6)),
                    "candidates": [candidate.prompt_payload() for candidate in accepted[:40]],
                    "schema": {
                        "cards": [
                            {
                                "candidate_ids": ["candidate_id"],
                                "title": "short string",
                                "reason": "short string",
                            }
                        ]
                    },
                },
                ensure_ascii=False,
            ),
        )
        drafts = validate_drafts(payload, {candidate.candidate_id for candidate in accepted})
        source = "llm"
        degraded = False
        if not drafts:
            self.llm_budget.note_fallback("summarizer", "invalid or empty summary payload")
            ranked = rank_evidence_cards(
                [candidate.card for candidate in accepted],
                limit=int(getattr(self.engine, "evidence_top_n", 6)),
                plan=plan,
            )
            candidate_by_identity = {card_identity(candidate.card): candidate for candidate in accepted}
            drafts = [
                EvidenceCardDraft(
                    candidate_ids=[candidate_by_identity[card_identity(card)].candidate_id],
                )
                for card in ranked
                if card_identity(card) in candidate_by_identity
            ]
            source = "deterministic"
            degraded = True

        cards: list[EvidenceCard] = []
        seen: set[str] = set()
        for draft in drafts:
            primary_id = next((item for item in draft.candidate_ids if item in by_id), "")
            if not primary_id or primary_id in seen:
                continue
            seen.add(primary_id)
            candidate = by_id[primary_id]
            cards.append(
                replace(
                    candidate.card,
                    title=shorten(draft.title, 160) or candidate.card.title,
                    reason=shorten(draft.reason, 200) or candidate.card.reason,
                    candidate_ids=list(draft.candidate_ids),
                    retrieval_agent=candidate.retrieval_agent,
                )
            )
            if len(cards) >= int(getattr(self.engine, "evidence_top_n", 6)):
                break
        if plan.answer_type == "risk_analysis" and not any(card.relation == "DISCLOSES_RISK" or card.claim_type == "risk" for card in cards):
            risk_candidate = next(
                (
                    candidate
                    for candidate in accepted
                    if candidate.card.relation == "DISCLOSES_RISK" or candidate.card.claim_type == "risk"
                ),
                None,
            )
            if risk_candidate is not None:
                risk_card = replace(
                    risk_candidate.card,
                    candidate_ids=[risk_candidate.candidate_id],
                    retrieval_agent=risk_candidate.retrieval_agent,
                )
                if len(cards) >= int(getattr(self.engine, "evidence_top_n", 6)):
                    cards[-1] = risk_card
                else:
                    cards.append(risk_card)
        return (
            raw_cards,
            assign_citation_ids(cards),
            drafts,
            {"source": source, "degraded": degraded, "drafts": [draft.to_dict() for draft in drafts]},
        )

    def diagnostics(self) -> dict[str, Any]:
        with self._runtime_lock:
            return {
                "mode": "centralized",
                "max_workers": self.max_workers,
                "peak_concurrency": self._peak_concurrency,
                "task_timeout_seconds": self.timeout_seconds,
                "timeout_count": self._timeouts,
                "rounds": list(self._rounds),
                "llm": self.llm_budget.diagnostics(),
            }

    def available_agents(self) -> list[str]:
        agents: list[str] = []
        if getattr(self.engine, "graph_client", None) is not None or getattr(self.engine, "csv_graph", None) is not None:
            agents.append("graph")
        if getattr(self.engine, "rag_index", None) is not None:
            agents.append("rag")
        if getattr(self.engine, "research_memory", None) is not None:
            agents.extend(["research", "graphrag"])
        if getattr(self.engine, "semantic_index", None) is not None:
            agents.append("semantic")
        return agents

    def _fallback_assignments(
        self,
        *,
        question: str,
        task_plan: AgentTaskPlan,
        round_index: int,
        available: list[str],
        gaps: list[EvidenceGap],
    ) -> list[AgentAssignment]:
        coverage = [gap.coverage for gap in gaps] or list(task_plan.required_coverage)
        return [
            AgentAssignment(
                assignment_id=f"r{round_index}-{agent_type}",
                agent_type=agent_type,
                query=question,
                coverage_goals=coverage,
                round_index=round_index,
            )
            for agent_type in available[: self.max_workers]
        ]

    def _run_assignment_with_metrics(
        self,
        assignment: AgentAssignment,
        plan: QuestionPlan,
        hyde_query: Any | None = None,
    ) -> AgentResult:
        with self._runtime_lock:
            self._active_workers += 1
            self._peak_concurrency = max(self._peak_concurrency, self._active_workers)
        try:
            return self._run_assignment(assignment, plan, hyde_query)
        finally:
            with self._runtime_lock:
                self._active_workers -= 1

    def _run_assignment(
        self,
        assignment: AgentAssignment,
        plan: QuestionPlan,
        hyde_query: Any | None = None,
    ) -> AgentResult:
        started_at = time.perf_counter()
        local_errors: list[str] = []
        role = f"query_{assignment.agent_type}"
        client = self.llm_budget.client(role)
        query, filters, degraded = self._rewrite_query(assignment, plan)
        companies = clean_strings(filters.get("companies"))
        if not companies and filters.get("company"):
            companies = [str(filters["company"])]
        scoped_plan = replace(
            plan,
            question=query,
            companies=companies or plan.companies,
        )
        retrieval_query = (
            query
            if assignment.agent_type == "graph"
            else self.engine._hyde_retrieval_query(query, hyde_query)
        )
        tools = AgentTools(
            self.engine,
            errors=local_errors,
            llm_options=self.llm_options,
            llm_client=client,
        )
        tool_calls: list[AgentToolCall] = []
        output: Any = None
        try:
            if assignment.agent_type == "graph":
                generated, prepare_call = tools.prepare_cypher(query, scoped_plan)
                records, query_call = tools.query_graph(generated, scoped_plan)
                tool_calls.extend([prepare_call, query_call])
                output = {"generated": generated, "records": records or []}
            elif assignment.agent_type == "rag":
                output, call = tools.search_rag(retrieval_query, scoped_plan)
                tool_calls.append(call)
            elif assignment.agent_type == "research":
                output, call = tools.search_research(retrieval_query, scoped_plan)
                tool_calls.append(call)
            elif assignment.agent_type == "semantic":
                output, call = tools.search_semantic(retrieval_query, scoped_plan)
                tool_calls.append(call)
            elif assignment.agent_type == "graphrag":
                call_started = time.perf_counter()
                output = retrieve_graphrag(
                    question=retrieval_query,
                    plan=scoped_plan,
                    research_memory=getattr(self.engine, "research_memory", None),
                    max_subquestions=int(getattr(self.engine, "drift_max_subquestions", 6)),
                    global_top_k=int(getattr(self.engine, "global_dossier_top_k", 3)),
                    local_top_k=int(getattr(self.engine, "local_claim_top_k", 12)),
                    path_top_k=int(getattr(self.engine, "graph_path_top_k", 6)),
                )
                tool_calls.append(
                    AgentToolCall(
                        tool="graphrag_retrieve",
                        args={"question": retrieval_query[:160], "answer_type": scoped_plan.answer_type},
                        result_count=len(output.global_hits) + len(output.local_hits),
                        elapsed_ms=round((time.perf_counter() - call_started) * 1000, 2),
                    )
                )
            else:
                raise ValueError(f"unsupported query agent: {assignment.agent_type}")
        except Exception as exc:
            local_errors.append(str(exc))
        error = "; ".join(error for error in local_errors if error)
        return AgentResult(
            assignment=replace(assignment, query=query, filters=filters),
            status="error" if error and result_count(output) == 0 else "completed",
            output=output,
            tool_calls=tool_calls,
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
            error=error,
            degraded=degraded or bool(error),
        )

    def _rewrite_query(
        self,
        assignment: AgentAssignment,
        plan: QuestionPlan,
    ) -> tuple[str, dict[str, Any], bool]:
        payload = self._safe_json(
            role=f"query_{assignment.agent_type}",
            system_prompt=(
                f"你是 {assignment.agent_type} 查询 Agent。"
                "只改写适合本数据源的检索查询和过滤条件，只输出严格 JSON，不回答问题。"
            ),
            user_prompt=json.dumps(
                {
                    "question": assignment.query,
                    "question_plan": plan.to_dict(),
                    "coverage_goals": assignment.coverage_goals,
                    "schema": {"query": "string", "filters": {"companies": ["string"]}},
                },
                ensure_ascii=False,
            ),
        )
        if isinstance(payload, dict):
            query = " ".join(str(payload.get("query") or "").split())[:600]
            filters = payload.get("filters")
            if query and isinstance(filters, dict):
                return query, dict(filters), False
        self.llm_budget.note_fallback(f"query_{assignment.agent_type}", "invalid query rewrite payload")
        return assignment.query, dict(assignment.filters), True

    def _safe_json(self, *, role: str, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        client = self.llm_budget.client(role)
        if client is None:
            return None
        kwargs = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": 0.0,
            **self.llm_options,
        }
        try:
            if hasattr(client, "chat_json"):
                payload = client.chat_json(**kwargs)
                return payload if isinstance(payload, dict) else None
            if hasattr(client, "chat_text"):
                return parse_json_object(str(client.chat_text(**kwargs)))
        except Exception as exc:
            self.llm_budget.note_fallback(role, str(exc))
        return None


def validate_assignments(
    payload: Any,
    *,
    available: list[str],
    question: str,
    round_index: int,
    max_assignments: int,
) -> list[AgentAssignment]:
    if not isinstance(payload, dict) or not isinstance(payload.get("assignments"), list):
        return []
    output: list[AgentAssignment] = []
    seen: set[str] = set()
    for item in payload["assignments"]:
        if not isinstance(item, dict):
            continue
        agent_type = str(item.get("agent_type") or "").strip().casefold()
        if agent_type not in AGENT_TYPES or agent_type not in available or agent_type in seen:
            continue
        query = " ".join(str(item.get("query") or question).split())[:600]
        if not query:
            continue
        filters = item.get("filters")
        goals = clean_strings(item.get("coverage_goals"))
        output.append(
            AgentAssignment(
                assignment_id=f"r{round_index}-{agent_type}",
                agent_type=agent_type,
                query=query,
                filters=dict(filters) if isinstance(filters, dict) else {},
                coverage_goals=goals,
                round_index=round_index,
            )
        )
        seen.add(agent_type)
        if len(output) >= max_assignments:
            break
    return output


def merge_graphrag_results(
    existing: GraphRagResult | None,
    additions: list[GraphRagResult],
) -> GraphRagResult:
    results = ([existing] if existing is not None else []) + list(additions)
    latest = results[-1]
    global_hits = dedupe_hits([hit for result in results for hit in result.global_hits])
    local_hits = dedupe_hits([hit for result in results for hit in result.local_hits])
    subquestions = []
    seen_subquestions: set[str] = set()
    for result in results:
        for subquestion in result.subquestions:
            if subquestion.question in seen_subquestions:
                continue
            seen_subquestions.add(subquestion.question)
            subquestions.append(subquestion)
    return GraphRagResult(
        route=latest.route,
        subquestions=subquestions,
        global_hits=global_hits,
        local_hits=local_hits,
        paths=[],
        company_rankings=[],
        edges=[],
    )


def merge_review_coverage(
    deterministic: CoverageReport,
    llm_missing: list[str],
    plan: QuestionPlan,
) -> CoverageReport:
    missing = list(deterministic.missing)
    for coverage in llm_missing:
        if coverage not in missing:
            missing.append(coverage)
    if missing == deterministic.missing:
        return deterministic
    satisfied = [coverage for coverage in deterministic.required if coverage not in missing]
    gaps_by_coverage = {gap.coverage: gap for gap in deterministic.gaps}
    gaps = []
    for coverage in missing:
        gap = gaps_by_coverage.get(coverage)
        if gap is None:
            gap = EvidenceGap(
                coverage=coverage,
                reason=f"审核 Agent 判断 {coverage} 证据仍不充分。",
                query_suffix=query_suffix_for_coverage(coverage),
                companies=list(plan.companies) if coverage == "company_coverage" else [],
            )
        gaps.append(gap)
    if deterministic.retrieval_round >= deterministic.max_retrieval_rounds:
        status = "fail" if "metric_evidence" in missing else "warn"
        stop_reason = "max_retrieval_rounds_reached"
    else:
        status = "warn"
        stop_reason = "needs_supplement"
    return CoverageReport(
        status=status,
        required=list(deterministic.required),
        satisfied=satisfied,
        missing=missing,
        gaps=gaps,
        stop_reason=stop_reason,
        retrieval_round=deterministic.retrieval_round,
        max_retrieval_rounds=deterministic.max_retrieval_rounds,
    )


def validate_drafts(payload: Any, allowed_ids: set[str]) -> list[EvidenceCardDraft]:
    if not isinstance(payload, dict) or not isinstance(payload.get("cards"), list):
        return []
    drafts: list[EvidenceCardDraft] = []
    seen: set[str] = set()
    for item in payload["cards"]:
        if not isinstance(item, dict):
            continue
        ids = clean_ids(item.get("candidate_ids"), allowed_ids)
        if not ids or ids[0] in seen:
            continue
        seen.add(ids[0])
        drafts.append(
            EvidenceCardDraft(
                candidate_ids=ids,
                title=str(item.get("title") or ""),
                reason=str(item.get("reason") or ""),
            )
        )
    return drafts


def validate_conflicts(value: Any, allowed_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    output: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        ids = clean_ids(item.get("candidate_ids"), allowed_ids)
        if len(ids) < 2:
            continue
        output.append({"candidate_ids": ids, "reason": shorten(str(item.get("reason") or ""), 240)})
    return output


def candidate_id(card: EvidenceCard) -> str:
    identity = "\x1f".join(card_identity(card))
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10].upper()
    return f"C{digest}"


def clean_ids(value: Any, allowed: set[str]) -> list[str]:
    return [item for item in clean_strings(value) if item in allowed]


def clean_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in output:
            output.append(text)
    return output


def shorten(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def result_count(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, GraphRagResult):
        return len(result.global_hits) + len(result.local_hits) + len(result.paths)
    if isinstance(result, dict):
        if "records" in result:
            return len(result.get("records") or [])
        return len(result)
    if isinstance(result, (list, tuple, set)):
        return len(result)
    return 1
