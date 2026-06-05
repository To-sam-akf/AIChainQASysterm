"""LangChain Core tool adapters for the AIQASYS deterministic agent tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from src.agent_tools import AgentTools
from src.cypher_generator import GeneratedCypher
from src.question_planner import QuestionPlan, heuristic_plan_question


def build_langchain_tools(agent_tools: AgentTools) -> list[StructuredTool]:
    """Expose existing deterministic AgentTools as LangChain Core tools.

    These tools are intentionally structured wrappers around the local business
    logic. The LangGraph runner still drives tool order deterministically; this
    adapter makes the same capabilities available through LangChain's standard
    tool abstraction for future agent/tool-selection work.
    """

    def contextualize_question(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Rewrite a follow-up question into a standalone retrieval query."""

        result, call = agent_tools.contextualize_question(question, history or [])
        return {"question": result, "tool_call": call.to_dict()}

    def plan_question(question: str) -> dict[str, Any]:
        """Plan answer type, companies, topics, and relations for a question."""

        plan, source, call = agent_tools.plan_question(question)
        return {"plan": plan.to_dict(), "source": source, "tool_call": call.to_dict()}

    def prepare_cypher(question: str, plan: dict[str, Any]) -> dict[str, Any]:
        """Prepare a display Cypher query or CSV pseudo-query for graph search."""

        generated, call = agent_tools.prepare_cypher(question, question_plan_from_payload(plan, question))
        return {"generated": generated.to_dict(), "tool_call": call.to_dict()}

    def query_graph(generated: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        """Query the configured graph backend with generated graph parameters."""

        rows, call = agent_tools.query_graph(generated_cypher_from_payload(generated), question_plan_from_payload(plan))
        return {"records": rows, "tool_call": call.to_dict()}

    def search_rag(question: str, plan: dict[str, Any]) -> dict[str, Any]:
        """Search local BM25 report chunks."""

        hits, call = agent_tools.search_rag(question, question_plan_from_payload(plan, question))
        return {"hits": [hit.to_dict() for hit in hits], "tool_call": call.to_dict()}

    def search_research_claims(question: str, plan: dict[str, Any]) -> dict[str, Any]:
        """Search curated Claim and Segment Dossier research memory."""

        hits, call = agent_tools.search_research(question, question_plan_from_payload(plan, question))
        return {"hits": [hit.to_dict() for hit in hits], "tool_call": call.to_dict()}

    def search_semantic_index(question: str, plan: dict[str, Any]) -> dict[str, Any]:
        """Search the local embedding semantic index when configured."""

        hits, call = agent_tools.search_semantic(question, question_plan_from_payload(plan, question))
        return {"hits": [hit.to_dict() for hit in hits], "tool_call": call.to_dict()}

    def graphrag_retrieve(question: str, plan: dict[str, Any], graph_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Run Query Router, DRIFT subquestions, multi-hop paths, and rankings."""

        result, call = agent_tools.run_graphrag(question, question_plan_from_payload(plan, question), graph_records)
        return {"graphrag": result.to_dict(), "tool_call": call.to_dict()}

    return [
        StructuredTool.from_function(contextualize_question, name="contextualize_question"),
        StructuredTool.from_function(plan_question, name="plan_question"),
        StructuredTool.from_function(prepare_cypher, name="prepare_cypher"),
        StructuredTool.from_function(query_graph, name="query_graph"),
        StructuredTool.from_function(search_rag, name="search_rag"),
        StructuredTool.from_function(search_research_claims, name="search_research_claims"),
        StructuredTool.from_function(search_semantic_index, name="search_semantic_index"),
        StructuredTool.from_function(graphrag_retrieve, name="graphrag_retrieve"),
    ]


def question_plan_from_payload(payload: dict[str, Any] | QuestionPlan | None, fallback_question: str = "") -> QuestionPlan:
    if isinstance(payload, QuestionPlan):
        return payload
    payload = dict(payload or {})
    question = str(payload.get("question") or fallback_question or "").strip()
    if not payload:
        return heuristic_plan_question(question)
    return QuestionPlan(
        question=question,
        answer_type=str(payload.get("answer_type") or "thematic_research"),
        companies=[str(item) for item in list(payload.get("companies") or [])],
        topics=[str(item) for item in list(payload.get("topics") or [])],
        expanded_topics=[str(item) for item in list(payload.get("expanded_topics") or [])],
        relations=[str(item) for item in list(payload.get("relations") or [])],
        core_companies_only=bool(payload.get("core_companies_only", True)),
        needs_comparison=bool(payload.get("needs_comparison", False)),
        needs_risk=bool(payload.get("needs_risk", False)),
        needs_metrics=bool(payload.get("needs_metrics", False)),
        needs_chain=bool(payload.get("needs_chain", False)),
    )


def generated_cypher_from_payload(payload: dict[str, Any] | GeneratedCypher | None) -> GeneratedCypher:
    if isinstance(payload, GeneratedCypher):
        return payload
    payload = dict(payload or {})
    return GeneratedCypher(
        cypher=str(payload.get("cypher") or ""),
        params=dict(payload.get("params") or {}),
        source=str(payload.get("source") or "langchain_tool_payload"),
        error=str(payload.get("error") or ""),
    )
