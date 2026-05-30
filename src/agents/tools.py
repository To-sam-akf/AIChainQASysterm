"""Tool registry metadata for agent workflows."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

from src.agents.models import AgentToolSpec


class ToolRegistry:
    def __init__(self, specs: Iterable[AgentToolSpec] | None = None) -> None:
        self._tools: dict[str, AgentToolSpec] = {}
        self._executors: dict[str, Callable[[dict[str, Any]], Any]] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: AgentToolSpec) -> None:
        if not spec.name:
            raise ValueError("Tool name cannot be empty")
        self._tools[spec.name] = spec

    def get(self, name: str) -> AgentToolSpec:
        return self._tools[name]

    def register_executor(self, name: str, executor: Callable[[dict[str, Any]], Any]) -> None:
        if name not in self._tools:
            raise KeyError(name)
        self._executors[name] = executor

    def execute(self, name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = self.get(name)
        del spec
        args = dict(payload or {})
        started_at = time.perf_counter()
        result: Any = None
        error = ""
        executor = self._executors.get(name)
        if executor is None:
            error = f"No executor registered for tool: {name}"
        else:
            try:
                result = executor(args)
            except Exception as exc:  # pragma: no cover - defensive executor boundary.
                error = str(exc)
        return {
            "tool": name,
            "args": args,
            "result": result,
            "result_count": result_count(result),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "error": error,
        }

    def list(self) -> list[dict]:
        return [self._tools[name].to_dict() for name in sorted(self._tools)]

    def names(self) -> list[str]:
        return sorted(self._tools)


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


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            AgentToolSpec(
                name="contextualize_question",
                description="Rewrite a follow-up question into an independently searchable question.",
                input_schema={"question": "string", "history": "messages[]"},
                output_schema={"contextual_question": "string"},
                requires_llm=True,
            ),
            AgentToolSpec(
                name="plan_question",
                description="Classify answer type, companies, topics, relations, and evidence needs.",
                input_schema={"question": "string"},
                output_schema={"plan": "QuestionPlan"},
                requires_llm=True,
            ),
            AgentToolSpec(
                name="query_graph",
                description="Retrieve structured company, topic, metric, and risk relations from Neo4j or CSV graph.",
                input_schema={"cypher": "string", "plan": "QuestionPlan"},
                output_schema={"records": "object[]"},
            ),
            AgentToolSpec(
                name="search_rag",
                description="Search local BM25 report chunks.",
                input_schema={"query": "string", "plan": "QuestionPlan"},
                output_schema={"hits": "RagHit[]"},
            ),
            AgentToolSpec(
                name="search_research_claims",
                description="Search curated Claim and Segment Dossier research memory.",
                input_schema={"query": "string", "plan": "QuestionPlan"},
                output_schema={"hits": "ResearchHit[]"},
            ),
            AgentToolSpec(
                name="search_segment_dossiers",
                description="Retrieve topic-level dossier evidence from the curated research layer.",
                input_schema={"topics": "string[]"},
                output_schema={"dossiers": "object[]"},
            ),
            AgentToolSpec(
                name="search_semantic_index",
                description="Search dense embedding index when configured.",
                input_schema={"query": "string", "top_k": "number"},
                output_schema={"hits": "SemanticHit[]"},
                requires_llm=True,
            ),
            AgentToolSpec(
                name="rank_evidence",
                description="Merge, budget, and rank graph, RAG, claim, dossier, and semantic evidence cards.",
                input_schema={"raw_cards": "EvidenceCard[]", "plan": "QuestionPlan"},
                output_schema={"evidence_cards": "EvidenceCard[]"},
            ),
            AgentToolSpec(
                name="verify_answer_support",
                description="Check citation ids, company coverage, risk evidence, and unsupported answer terms.",
                input_schema={"answer": "string", "evidence_cards": "EvidenceCard[]"},
                output_schema={"verification": "object"},
            ),
            AgentToolSpec(
                name="detect_evidence_gaps",
                description="Check whether retrieved evidence covers required companies, risks, metrics, mechanisms, and exposure.",
                input_schema={"plan": "QuestionPlan", "retrieval_state": "AgentRetrievalState"},
                output_schema={"coverage": "object", "gaps": "object[]", "stop_reason": "string"},
            ),
            AgentToolSpec(
                name="build_research_outputs",
                description="Build report, company table, risk checklist, and evidence gaps from selected evidence.",
                input_schema={"question": "string", "evidence_cards": "EvidenceCard[]"},
                output_schema={"research_outputs": "object"},
            ),
        ]
    )
