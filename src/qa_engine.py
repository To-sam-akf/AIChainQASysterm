"""Professional CSV/Neo4j + local RAG + LLM QA orchestration."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.cypher_generator import GeneratedCypher, generate_cypher
from src.curated_graph import DEFAULT_CURATED_DIR
from src.extraction_schema import normalize_name
from src.frontend_data import LocalKnowledgeGraph, RELATION_LABELS, subgraph_edges
from src.llm_client import OpenAICompatibleClient, env_bool, load_dotenv
from src.neo4j_client import Neo4jReadClient
from src.professional_qa import (
    build_professional_answer_prompt,
    cards_from_graph_records,
    cards_from_rag_hits,
    fallback_professional_answer,
    legacy_evidence_rows,
    pseudo_cypher_for_plan,
    rank_evidence_cards,
    search_csv_graph,
)
from src.question_planner import QuestionPlan, extract_companies, heuristic_plan_question, plan_question
from src.rag_index import DEFAULT_RAG_DIR, LocalRagIndex, RagHit
from src.web_search import DuckDuckGoSearchClient, WebSearchResponse


NO_EVIDENCE_ANSWER = "当前知识库中未找到相关证据。"

ANSWER_SYSTEM_PROMPT = """你是资深中国 AI 算力产业链研究分析师，擅长把图谱关系、研报原文和公开信息组织成有洞察的投研回答。
回答前先审视证据强弱、来源层级、时间性和可能冲突；最终只输出清晰结论与依据，不展示冗长推理草稿。
知识库证据（Neo4j/CSV 图谱、本地 RAG 原文）是事实主锚点；联网结果只能作为“联网补充”用于最新背景、线索或交叉验证。
你可以积极归纳、比较和提出研究假设，但每个事实判断都必须被输入证据支撑；证据不足时要明确边界和待核验问题。
禁止编造证据外事实、股票买卖建议、目标价或收益预测。"""

CONTEXTUALIZER_SYSTEM_PROMPT = """你是中国 AI 算力产业链问答系统的追问改写器。
根据历史对话，把用户当前问题改写成可独立检索的中文问题。
只输出改写后的问题，不解释，不回答问题，不引入历史对话中没有出现的新公司或主题。"""

TEMPLATE_RELATIONS = {
    "USES_TECHNOLOGY",
    "HAS_PRODUCT",
    "BELONGS_TO_CHAIN",
    "HAS_METRIC",
    "DISCLOSES_RISK",
    "SUPPORTED_BY_POLICY",
    "CONSTRAINS",
    "ENABLES",
}


@dataclass
class QAEngineStatus:
    neo4j_enabled: bool
    rag_enabled: bool
    llm_enabled: bool
    csv_graph_enabled: bool = False
    graph_backend: str = "neo4j"
    graph_data_dir: str = ""
    graph_error: str = ""
    rag_error: str = ""
    llm_error: str = ""


class CountingLLMClient:
    """Proxy that records remote LLM calls while preserving hasattr behavior."""

    TRACKED_METHODS = {"chat_json", "chat_text", "chat_text_with_metadata", "chat_messages", "stream_chat_messages"}

    def __init__(self, client: Any) -> None:
        self._client = client
        self.calls: dict[str, int] = {"total": 0}

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if name not in self.TRACKED_METHODS or not callable(attr):
            return attr

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls["total"] = self.calls.get("total", 0) + 1
            self.calls[name] = self.calls.get(name, 0) + 1
            return attr(*args, **kwargs)

        return counted


class QAEngine:
    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        graph_client: Any | None = None,
        csv_graph: LocalKnowledgeGraph | None = None,
        rag_index: LocalRagIndex | None = None,
        enable_llm_cypher: bool = False,
        enable_llm_planner: bool = False,
        contextualizer_mode: str = "auto",
        rag_top_k: int = 6,
        graph_limit: int = 50,
        rerank_top_n: int = 12,
        evidence_top_n: int = 6,
        core_companies_only: bool = True,
        history_max_turns: int = 3,
        history_max_chars: int = 4000,
        web_search_client: Any | None = None,
        web_search_enabled: bool = False,
        web_search_top_k: int = 5,
        status: QAEngineStatus | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.graph_client = graph_client
        self.csv_graph = csv_graph
        self.rag_index = rag_index
        self.enable_llm_cypher = enable_llm_cypher
        self.enable_llm_planner = enable_llm_planner
        self.contextualizer_mode = normalize_contextualizer_mode(contextualizer_mode)
        self.rag_top_k = rag_top_k
        self.graph_limit = graph_limit
        self.rerank_top_n = rerank_top_n
        self.evidence_top_n = evidence_top_n
        self.core_companies_only = core_companies_only
        self.history_max_turns = history_max_turns
        self.history_max_chars = history_max_chars
        self.web_search_client = web_search_client
        self.web_search_enabled = web_search_enabled
        self.web_search_top_k = max(web_search_top_k, 0)
        self.status = status or QAEngineStatus(
            neo4j_enabled=graph_client is not None,
            csv_graph_enabled=csv_graph is not None,
            rag_enabled=rag_index is not None,
            llm_enabled=llm_client is not None,
            graph_backend="neo4j" if graph_client is not None else "csv" if csv_graph is not None else "none",
        )

    @classmethod
    def from_env(cls) -> "QAEngine":
        load_dotenv()
        rag_top_k = int(os.getenv("RAG_TOP_K", "6"))
        graph_limit = int(os.getenv("QA_GRAPH_LIMIT", "50"))
        rerank_top_n = int(os.getenv("QA_RERANK_TOP_N", "12"))
        evidence_top_n = int(os.getenv("QA_EVIDENCE_TOP_N", "6"))
        core_companies_only = os.getenv("QA_CORE_COMPANIES_ONLY", "true").casefold() != "false"
        enable_llm_cypher = os.getenv("QA_ENABLE_LLM_CYPHER", "false").casefold() != "false"
        enable_llm_planner = os.getenv("QA_ENABLE_LLM_PLANNER", "false").casefold() != "false"
        contextualizer_mode = normalize_contextualizer_mode(os.getenv("QA_CONTEXTUALIZER_MODE", "auto"))
        history_max_turns = int(os.getenv("QA_HISTORY_MAX_TURNS", "3"))
        history_max_chars = int(os.getenv("QA_HISTORY_MAX_CHARS", "4000"))
        graph_backend = os.getenv("QA_GRAPH_BACKEND", "auto").casefold()
        web_search_default = "deepseek" in os.getenv("LLM_BASE_URL", "").casefold()
        web_search_enabled = env_bool("QA_WEB_SEARCH_ENABLED", web_search_default)
        web_search_top_k = int(os.getenv("QA_WEB_SEARCH_TOP_K", "5"))
        web_search_timeout = float(os.getenv("QA_WEB_SEARCH_TIMEOUT", "5"))

        llm_client = None
        llm_error = ""
        try:
            llm_client = OpenAICompatibleClient()
        except Exception as exc:
            llm_error = str(exc)

        rag_index = None
        rag_error = ""
        try:
            rag_index_dir = Path(os.getenv("RAG_INDEX_DIR", str(DEFAULT_RAG_DIR)))
            rag_index = LocalRagIndex.load(rag_index_dir)
        except Exception as exc:
            rag_error = str(exc)

        csv_graph = None
        graph_error = ""
        graph_data_dir = Path(os.getenv("KG_DATA_DIR", str(DEFAULT_CURATED_DIR)))
        if not (graph_data_dir / "entities.csv").exists() or not (graph_data_dir / "relations.csv").exists():
            graph_data_dir = Path(__file__).resolve().parents[1] / "data" / "verified"
        try:
            csv_graph = LocalKnowledgeGraph.from_dir(graph_data_dir)
        except Exception as exc:
            graph_error = f"CSV graph unavailable: {exc}"

        graph_client = None
        neo4j_enabled = False
        selected_backend = "csv" if csv_graph is not None else "none"
        if graph_backend in {"auto", "neo4j"}:
            try:
                candidate = Neo4jReadClient()
                ok, message = candidate.check_connection()
                if ok:
                    graph_client = candidate
                    neo4j_enabled = True
                    selected_backend = "neo4j"
                else:
                    candidate.close()
                    if graph_backend == "neo4j":
                        graph_error = message
            except Exception as exc:
                if graph_backend == "neo4j":
                    graph_error = str(exc)
        if graph_backend == "csv":
            selected_backend = "csv" if csv_graph is not None else "none"

        status = QAEngineStatus(
            neo4j_enabled=neo4j_enabled,
            csv_graph_enabled=csv_graph is not None,
            rag_enabled=rag_index is not None,
            llm_enabled=llm_client is not None,
            graph_backend=selected_backend,
            graph_data_dir=str(graph_data_dir),
            graph_error=graph_error,
            rag_error=rag_error,
            llm_error=llm_error,
        )
        web_search_client = (
            DuckDuckGoSearchClient(timeout=web_search_timeout, top_k=web_search_top_k)
            if web_search_enabled
            else None
        )
        return cls(
            llm_client=llm_client,
            graph_client=graph_client,
            csv_graph=csv_graph,
            rag_index=rag_index,
            enable_llm_cypher=enable_llm_cypher,
            enable_llm_planner=enable_llm_planner,
            contextualizer_mode=contextualizer_mode,
            rag_top_k=rag_top_k,
            graph_limit=graph_limit,
            rerank_top_n=rerank_top_n,
            evidence_top_n=evidence_top_n,
            core_companies_only=core_companies_only,
            history_max_turns=history_max_turns,
            history_max_chars=history_max_chars,
            web_search_client=web_search_client,
            web_search_enabled=web_search_enabled,
            web_search_top_k=web_search_top_k,
            status=status,
        )

    def close(self) -> None:
        if self.graph_client is not None and hasattr(self.graph_client, "close"):
            self.graph_client.close()

    def answer_question(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
        web_search_enabled: bool | None = None,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        timings_ms: dict[str, float] = {}
        question = question.strip()
        errors: list[str] = []
        llm_client = CountingLLMClient(self.llm_client) if self.llm_client is not None else None
        llm_options = build_llm_options(thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort)
        stage_start = time.perf_counter()
        history = normalize_conversation_history(
            conversation_history,
            max_turns=self.history_max_turns,
            max_chars=self.history_max_chars,
        )
        record_timing(timings_ms, "history", stage_start)

        stage_start = time.perf_counter()
        contextual_question = self._contextualize_question(question, history, errors, llm_options, llm_client)
        record_timing(timings_ms, "contextualize", stage_start)

        stage_start = time.perf_counter()
        plan = heuristic_plan_question(contextual_question, core_companies_only=self.core_companies_only)
        planner_source = "heuristic"
        if self._should_use_llm_planner(plan, llm_client):
            plan = plan_question(
                contextual_question,
                client=llm_client,
                core_companies_only=self.core_companies_only,
                llm_options=llm_options,
            )
            planner_source = "llm"
        record_timing(timings_ms, "plan", stage_start)

        stage_start = time.perf_counter()
        generated = self._generate_display_cypher(contextual_question, plan, errors, llm_options, llm_client)
        if generated.error:
            errors.append(generated.error)
        record_timing(timings_ms, "cypher", stage_start)

        stage_start = time.perf_counter()
        graph_records = self._query_graph(generated, plan, errors)
        record_timing(timings_ms, "graph", stage_start)

        stage_start = time.perf_counter()
        rag_hits = self._search_rag(contextual_question, plan, errors)
        record_timing(timings_ms, "rag", stage_start)

        stage_start = time.perf_counter()
        raw_cards = [*cards_from_graph_records(graph_records, plan), *cards_from_rag_hits(rag_hits, plan)]
        evidence_cards = rank_evidence_cards(raw_cards, limit=self.evidence_top_n)
        if plan.answer_type == "risk_analysis":
            evidence_cards = ensure_relation_cards(evidence_cards, raw_cards, "DISCLOSES_RISK", limit=self.evidence_top_n)
        record_timing(timings_ms, "evidence", stage_start)

        stage_start = time.perf_counter()
        web_search_hits, web_search_error = self._search_web(contextual_question, plan, web_search_enabled)
        record_timing(timings_ms, "web_search", stage_start)

        stage_start = time.perf_counter()
        answer, reasoning_content = self._generate_answer(
            question,
            contextual_question,
            history,
            plan,
            graph_records,
            evidence_cards,
            web_search_hits,
            errors,
            llm_options,
            llm_client,
        )
        record_timing(timings_ms, "answer", stage_start)

        stage_start = time.perf_counter()
        evidence = legacy_evidence_rows(evidence_cards)
        rag_hit_rows = [hit.to_dict() for hit in rag_hits]
        evidence_card_rows = [card.to_dict() for card in evidence_cards]
        subgraph = graph_records_to_subgraph(graph_records)
        record_timing(timings_ms, "render_payload", stage_start)

        diagnostics = {
            "graph_backend": self.status.graph_backend,
            "graph_records": len(graph_records),
            "rag_hits": len(rag_hits),
            "evidence_cards": len(evidence_cards),
            "rerank_top_n": self.rerank_top_n,
            "history_messages": len(history),
            "contextualized": contextual_question != question,
            "contextualizer_mode": self.contextualizer_mode,
            "planner_source": planner_source,
            "enable_llm_cypher": self.enable_llm_cypher,
            "enable_llm_planner": self.enable_llm_planner,
            "thinking_enabled": thinking_enabled,
            "reasoning_effort": reasoning_effort or "",
            "web_search_enabled": self._effective_web_search_enabled(web_search_enabled),
            "web_search_hits": len(web_search_hits),
            "web_search_error": web_search_error,
            "graph_error": self.status.graph_error,
            "rag_error": self.status.rag_error,
            "llm_error": self.status.llm_error,
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
            "graph_records": graph_records,
            "rag_hits": rag_hit_rows,
            "web_search_hits": web_search_hits,
            "evidence_cards": evidence_card_rows,
            "evidence": evidence,
            "subgraph": subgraph,
            "diagnostics": diagnostics,
            "errors": errors,
        }

    def answer_question_stream(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
        web_search_enabled: bool | None = None,
    ) -> Iterator[dict[str, Any]]:
        total_start = time.perf_counter()
        timings_ms: dict[str, float] = {}
        question = question.strip()
        errors: list[str] = []
        llm_client = CountingLLMClient(self.llm_client) if self.llm_client is not None else None
        llm_options = build_llm_options(thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort)

        if thinking_enabled:
            yield stream_progress("history", "正在结合历史对话理解当前问题")
        stage_start = time.perf_counter()
        history = normalize_conversation_history(
            conversation_history,
            max_turns=self.history_max_turns,
            max_chars=self.history_max_chars,
        )
        record_timing(timings_ms, "history", stage_start)

        if thinking_enabled:
            yield stream_progress("contextualize", "正在判断是否需要补全追问上下文")
        stage_start = time.perf_counter()
        contextual_question = self._contextualize_question(question, history, errors, llm_options, llm_client)
        record_timing(timings_ms, "contextualize", stage_start)
        if thinking_enabled and contextual_question != question:
            yield stream_progress("contextualize", "已将追问改写为可独立检索的问题")

        if thinking_enabled:
            yield stream_progress("plan", "正在识别公司、主题和答案类型")
        stage_start = time.perf_counter()
        plan = heuristic_plan_question(contextual_question, core_companies_only=self.core_companies_only)
        planner_source = "heuristic"
        if self._should_use_llm_planner(plan, llm_client):
            plan = plan_question(
                contextual_question,
                client=llm_client,
                core_companies_only=self.core_companies_only,
                llm_options=llm_options,
            )
            planner_source = "llm"
        record_timing(timings_ms, "plan", stage_start)
        if thinking_enabled:
            yield stream_progress("plan", describe_plan_progress(plan))

        if thinking_enabled:
            yield stream_progress("cypher", "正在准备图谱查询条件")
        stage_start = time.perf_counter()
        generated = self._generate_display_cypher(contextual_question, plan, errors, llm_options, llm_client)
        if generated.error:
            errors.append(generated.error)
        record_timing(timings_ms, "cypher", stage_start)

        if thinking_enabled:
            yield stream_progress("graph", "正在检索产业链图谱关系")
        stage_start = time.perf_counter()
        graph_records = self._query_graph(generated, plan, errors)
        record_timing(timings_ms, "graph", stage_start)

        if thinking_enabled:
            yield stream_progress("rag", "正在召回本地研报与原文片段")
        stage_start = time.perf_counter()
        rag_hits = self._search_rag(contextual_question, plan, errors)
        record_timing(timings_ms, "rag", stage_start)

        if thinking_enabled:
            yield stream_progress("evidence", "正在筛选可支撑答案的证据")
        stage_start = time.perf_counter()
        raw_cards = [*cards_from_graph_records(graph_records, plan), *cards_from_rag_hits(rag_hits, plan)]
        evidence_cards = rank_evidence_cards(raw_cards, limit=self.evidence_top_n)
        if plan.answer_type == "risk_analysis":
            evidence_cards = ensure_relation_cards(evidence_cards, raw_cards, "DISCLOSES_RISK", limit=self.evidence_top_n)
        record_timing(timings_ms, "evidence", stage_start)
        if thinking_enabled:
            yield stream_progress("evidence", f"已保留 {len(evidence_cards)} 条高相关证据，开始组织答案")

        use_web_search = self._effective_web_search_enabled(web_search_enabled)
        if use_web_search:
            yield stream_progress("web_search", "正在联网检索最新公开信息")
        stage_start = time.perf_counter()
        web_search_hits, web_search_error = self._search_web(contextual_question, plan, web_search_enabled)
        record_timing(timings_ms, "web_search", stage_start)
        if use_web_search:
            if web_search_hits:
                yield stream_progress("web_search", f"已找到 {len(web_search_hits)} 条联网补充证据")
            else:
                yield stream_progress("web_search", "联网检索未返回可用补充证据，将继续使用知识库证据")

        stage_start = time.perf_counter()
        answer = ""
        reasoning_content = ""
        for event in self._generate_answer_stream(
            question,
            contextual_question,
            history,
            plan,
            graph_records,
            evidence_cards,
            web_search_hits,
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
        record_timing(timings_ms, "answer", stage_start)

        stage_start = time.perf_counter()
        evidence = legacy_evidence_rows(evidence_cards)
        rag_hit_rows = [hit.to_dict() for hit in rag_hits]
        evidence_card_rows = [card.to_dict() for card in evidence_cards]
        subgraph = graph_records_to_subgraph(graph_records)
        record_timing(timings_ms, "render_payload", stage_start)

        diagnostics = {
            "graph_backend": self.status.graph_backend,
            "graph_records": len(graph_records),
            "rag_hits": len(rag_hits),
            "evidence_cards": len(evidence_cards),
            "rerank_top_n": self.rerank_top_n,
            "history_messages": len(history),
            "contextualized": contextual_question != question,
            "contextualizer_mode": self.contextualizer_mode,
            "planner_source": planner_source,
            "enable_llm_cypher": self.enable_llm_cypher,
            "enable_llm_planner": self.enable_llm_planner,
            "thinking_enabled": thinking_enabled,
            "reasoning_effort": reasoning_effort or "",
            "web_search_enabled": use_web_search,
            "web_search_hits": len(web_search_hits),
            "web_search_error": web_search_error,
            "graph_error": self.status.graph_error,
            "rag_error": self.status.rag_error,
            "llm_error": self.status.llm_error,
        }
        timings_ms["total"] = round((time.perf_counter() - total_start) * 1000, 2)
        diagnostics["timings_ms"] = timings_ms
        diagnostics["llm_calls"] = llm_client.calls if llm_client is not None else {"total": 0}

        yield {
            "type": "final",
            "result": {
                "question": question,
                "contextual_question": contextual_question,
                "answer": answer,
                "reasoning_content": reasoning_content,
                "answer_type": plan.answer_type,
                "plan": plan.to_dict(),
                "cypher": generated.cypher,
                "cypher_params": generated.params,
                "cypher_source": generated.source,
                "graph_records": graph_records,
                "rag_hits": rag_hit_rows,
                "web_search_hits": web_search_hits,
                "evidence_cards": evidence_card_rows,
                "evidence": evidence,
                "subgraph": subgraph,
                "diagnostics": diagnostics,
                "errors": errors,
            },
        }

    def _should_use_llm_planner(self, plan: QuestionPlan, llm_client: Any | None) -> bool:
        if not self.enable_llm_planner or llm_client is None or not hasattr(llm_client, "chat_json"):
            return False
        return plan.answer_type == "thematic_research" and not plan.companies and not plan.topics

    def _generate_display_cypher(
        self,
        question: str,
        plan: QuestionPlan,
        errors: list[str],
        llm_options: dict[str, Any],
        llm_client: Any | None,
    ) -> GeneratedCypher:
        if self.status.graph_backend == "csv" or self.graph_client is None:
            return GeneratedCypher(
                cypher=pseudo_cypher_for_plan(plan, limit=self.graph_limit),
                params={
                    "companies": plan.companies,
                    "topic": plan.expanded_topics[0] if plan.expanded_topics else "",
                },
                source="question_plan_csv",
            )
        if not self.enable_llm_cypher or llm_client is None:
            return template_cypher_for_plan(plan, limit=self.graph_limit)
        try:
            return generate_cypher(
                question,
                client=llm_client,
                enable_llm=self.enable_llm_cypher,
                limit=self.graph_limit,
                llm_options=llm_options,
            )
        except Exception as exc:
            errors.append(f"Cypher generation failed: {exc}")
            return template_cypher_for_plan(plan, limit=self.graph_limit, error=str(exc))

    def _query_graph(
        self,
        generated: GeneratedCypher,
        plan: QuestionPlan,
        errors: list[str],
    ) -> list[dict[str, Any]]:
        if self.status.graph_backend == "csv" and self.csv_graph is not None:
            return search_csv_graph(self.csv_graph, plan, limit=self.graph_limit)
        if self.graph_client is None:
            if self.csv_graph is not None:
                return search_csv_graph(self.csv_graph, plan, limit=self.graph_limit)
            errors.append("Graph backend is not configured.")
            return []
        try:
            rows = self.graph_client.run_read_query(generated.cypher, generated.params, limit=self.graph_limit)
            if rows:
                return rows
            if self.csv_graph is not None:
                return search_csv_graph(self.csv_graph, plan, limit=self.graph_limit)
            return []
        except Exception as exc:
            errors.append(f"Neo4j query failed: {exc}")
            if self.csv_graph is not None:
                return search_csv_graph(self.csv_graph, plan, limit=self.graph_limit)
            return []

    def _search_rag(self, question: str, plan: QuestionPlan, errors: list[str]) -> list[RagHit]:
        if self.rag_index is None:
            if self.status.rag_error:
                errors.append(f"RAG index unavailable: {self.status.rag_error}")
            return []
        try:
            filters = {}
            if len(plan.companies) == 1 and plan.answer_type in {"risk_analysis", "company_profile"}:
                filters["company"] = plan.companies[0]
            query = " ".join([question, *plan.expanded_topics])
            return self.rag_index.search(query, top_k=max(self.rag_top_k, min(self.rerank_top_n, 20)), filters=filters)
        except Exception as exc:
            errors.append(f"RAG search failed: {exc}")
            return []

    def _effective_web_search_enabled(self, requested: bool | None) -> bool:
        return self.web_search_enabled if requested is None else requested

    def _search_web(
        self,
        question: str,
        plan: QuestionPlan,
        requested: bool | None,
    ) -> tuple[list[dict[str, Any]], str]:
        if not self._effective_web_search_enabled(requested):
            return [], ""
        if self.web_search_client is None:
            return [], "Web search client is not configured."
        query = build_web_search_query(question, plan)
        try:
            response = self.web_search_client.search(query, top_k=self.web_search_top_k)
        except Exception as exc:
            return [], str(exc)
        if isinstance(response, WebSearchResponse):
            return web_search_hits_to_dicts(response.hits), response.error
        if isinstance(response, tuple) and len(response) == 2:
            hits, error = response
            return web_search_hits_to_dicts(hits), str(error or "")
        return web_search_hits_to_dicts(response), ""

    def _contextualize_question(
        self,
        question: str,
        history: list[dict[str, str]],
        errors: list[str],
        llm_options: dict[str, Any],
        llm_client: Any | None,
    ) -> str:
        fallback = heuristic_contextual_question(question, history)
        if not history or llm_client is None or self.contextualizer_mode == "heuristic":
            return fallback
        if self.contextualizer_mode == "auto" and not question_needs_context(question):
            return fallback
        prompt = build_contextualizer_prompt(question)
        try:
            if hasattr(llm_client, "chat_messages"):
                response = llm_client.chat_messages(
                    messages=[
                        {"role": "system", "content": CONTEXTUALIZER_SYSTEM_PROMPT},
                        *history,
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    **llm_options,
                )
                return sanitize_contextual_question(response.content) or fallback
            if hasattr(llm_client, "chat_text"):
                content = llm_client.chat_text(
                    system_prompt=CONTEXTUALIZER_SYSTEM_PROMPT,
                    user_prompt=format_history_for_prompt(history) + "\n\n" + prompt,
                    temperature=0.0,
                    **llm_options,
                )
                return sanitize_contextual_question(content) or fallback
        except Exception as exc:
            errors.append(f"Question contextualization failed: {exc}")
        return fallback

    def _generate_answer(
        self,
        question: str,
        contextual_question: str,
        history: list[dict[str, str]],
        plan: QuestionPlan,
        graph_records: list[dict[str, Any]],
        evidence_cards: list[Any],
        web_search_hits: list[dict[str, Any]],
        errors: list[str],
        llm_options: dict[str, Any],
        llm_client: Any | None,
    ) -> tuple[str, str]:
        if not evidence_cards:
            return no_evidence_answer_with_web_clues(web_search_hits), ""
        prompt_question = question
        if contextual_question != question:
            prompt_question = f"用户当前追问：{question}\n结合历史对话改写后的检索问题：{contextual_question}"
        user_prompt = build_professional_answer_prompt(
            prompt_question,
            plan,
            graph_records,
            evidence_cards,
            web_search_hits=web_search_hits,
        )
        if llm_client is not None and hasattr(llm_client, "chat_text"):
            try:
                if history and hasattr(llm_client, "chat_messages"):
                    response = llm_client.chat_messages(
                        messages=[
                            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                            *history,
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                        **llm_options,
                    )
                    return response.content, response.reasoning_content
                if hasattr(llm_client, "chat_text_with_metadata"):
                    response = llm_client.chat_text_with_metadata(
                        system_prompt=ANSWER_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        temperature=0.2,
                        **llm_options,
                    )
                    return response.content, response.reasoning_content
                return llm_client.chat_text(
                    system_prompt=ANSWER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.2,
                    **llm_options,
                ), ""
            except Exception as exc:
                errors.append(f"LLM answer failed: {exc}")
        return fallback_professional_answer(plan, evidence_cards, graph_records), ""

    def _generate_answer_stream(
        self,
        question: str,
        contextual_question: str,
        history: list[dict[str, str]],
        plan: QuestionPlan,
        graph_records: list[dict[str, Any]],
        evidence_cards: list[Any],
        web_search_hits: list[dict[str, Any]],
        errors: list[str],
        llm_options: dict[str, Any],
        llm_client: Any | None,
        *,
        thinking_enabled: bool,
    ) -> Iterator[dict[str, Any]]:
        if not evidence_cards:
            answer = no_evidence_answer_with_web_clues(web_search_hits)
            for chunk in chunk_text(answer):
                yield {"type": "answer_delta", "content": chunk}
            yield {"type": "answer_complete", "answer": answer, "reasoning_content": ""}
            return

        prompt_question = question
        if contextual_question != question:
            prompt_question = f"用户当前追问：{question}\n结合历史对话改写后的检索问题：{contextual_question}"
        user_prompt = build_professional_answer_prompt(
            prompt_question,
            plan,
            graph_records,
            evidence_cards,
            web_search_hits=web_search_hits,
        )

        if thinking_enabled:
            yield {"type": "progress", "stage": "answer", "message": "正在生成结论、证据和风险边界"}

        if llm_client is not None and hasattr(llm_client, "stream_chat_messages"):
            chunks: list[str] = []
            try:
                messages = [
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    *history,
                    {"role": "user", "content": user_prompt},
                ]
                for chunk in llm_client.stream_chat_messages(
                    messages=messages,
                    temperature=0.2,
                    **llm_options,
                ):
                    content = str(getattr(chunk, "content", "") or "")
                    if not content:
                        continue
                    chunks.append(content)
                    yield {"type": "answer_delta", "content": content}
                answer = "".join(chunks).strip()
                if answer:
                    yield {"type": "answer_complete", "answer": answer, "reasoning_content": ""}
                    return
            except Exception as exc:
                errors.append(f"LLM answer stream failed: {exc}")
                if chunks:
                    yield {"type": "answer_complete", "answer": "".join(chunks).strip(), "reasoning_content": ""}
                    return

        answer, reasoning_content = self._generate_answer(
            question,
            contextual_question,
            history,
            plan,
            graph_records,
            evidence_cards,
            web_search_hits,
            errors,
            llm_options,
            llm_client,
        )
        for chunk in chunk_text(answer):
            yield {"type": "answer_delta", "content": chunk}
        yield {"type": "answer_complete", "answer": answer, "reasoning_content": reasoning_content}


def build_answer_prompt(question: str, graph_records: list[dict[str, Any]], rag_hits: list[RagHit]) -> str:
    payload = {
        "question": question,
        "neo4j_records": graph_records[:30],
        "rag_hits": [hit.to_dict() for hit in rag_hits[:8]],
    }
    return "请基于以下证据回答用户问题，不要使用证据外信息：\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def normalize_contextualizer_mode(value: str) -> str:
    mode = str(value or "auto").strip().casefold()
    if mode not in {"auto", "heuristic", "llm"}:
        return "auto"
    return mode


def record_timing(timings_ms: dict[str, float], name: str, started_at: float) -> None:
    timings_ms[name] = round((time.perf_counter() - started_at) * 1000, 2)


def template_cypher_for_plan(plan: QuestionPlan, *, limit: int = 50, error: str = "") -> GeneratedCypher:
    relations = [relation for relation in plan.relations if relation in TEMPLATE_RELATIONS]
    if plan.answer_type == "industry_bottleneck":
        for relation in ("CONSTRAINS", "DISCLOSES_RISK", "SUPPORTED_BY_POLICY", "ENABLES"):
            if relation not in relations:
                relations.append(relation)
    if not relations:
        relations = ["USES_TECHNOLOGY", "HAS_PRODUCT", "BELONGS_TO_CHAIN"]
    relation_clause = "|".join(relations)

    where: list[str] = ["type(r) <> 'MENTIONED_IN'"]
    params: dict[str, Any] = {}
    if plan.companies:
        params["company_norms"] = [normalize_name(company, "Company") for company in plan.companies]
        where.append("c.normalized_name IN $company_norms")
    if plan.expanded_topics:
        topics = plan.expanded_topics[:12]
        params["topics"] = topics
        params["topic_norms"] = [normalize_name(topic) for topic in topics]
        where.append(
            "(x.normalized_name IN $topic_norms OR any(topic IN $topics "
            "WHERE x.name CONTAINS topic OR r.evidence CONTAINS topic OR r.section CONTAINS topic))"
        )

    cypher = (
        f"MATCH (c:Company)-[r:{relation_clause}]->(x)\n"
        f"WHERE {' AND '.join(where)}\n"
        "RETURN c.name AS company, labels(c) AS company_labels, type(r) AS relation, "
        "x.name AS target, labels(x) AS target_labels, r.evidence AS evidence, "
        "r.source_title AS source, r.source_tier AS source_tier, r.page AS page, "
        "r.section AS section, r.source_report_id AS report_id\n"
        f"LIMIT {limit}"
    )
    return GeneratedCypher(cypher=cypher, params=params, source="template", error=error)


def build_llm_options(*, thinking_enabled: bool | None, reasoning_effort: str | None) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if thinking_enabled is not None:
        options["thinking_enabled"] = thinking_enabled
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort
    return options


def build_web_search_query(question: str, plan: QuestionPlan) -> str:
    parts = [question.strip()]
    parts.extend(plan.companies[:4])
    parts.extend(plan.topics[:4])
    parts.append("AI 算力 产业链")
    seen = set()
    tokens = []
    for part in parts:
        value = str(part or "").strip()
        if value and value not in seen:
            seen.add(value)
            tokens.append(value)
    return " ".join(tokens)[:300]


def web_search_hits_to_dicts(hits: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit in hits or []:
        if hasattr(hit, "to_dict"):
            row = hit.to_dict()
        elif isinstance(hit, dict):
            row = dict(hit)
        else:
            row = {
                "title": str(getattr(hit, "title", "") or ""),
                "url": str(getattr(hit, "url", "") or ""),
                "snippet": str(getattr(hit, "snippet", "") or ""),
            }
        title = str(row.get("title") or "").strip()
        url = str(row.get("url") or "").strip()
        snippet = str(row.get("snippet") or "").strip()
        if title and url:
            rows.append({"title": title, "url": url, "snippet": snippet})
    return rows


def no_evidence_answer_with_web_clues(web_search_hits: list[dict[str, Any]]) -> str:
    if not web_search_hits:
        return NO_EVIDENCE_ANSWER
    clues = []
    for index, hit in enumerate(web_search_hits[:3], start=1):
        title = str(hit.get("title") or "").strip()
        url = str(hit.get("url") or "").strip()
        snippet = str(hit.get("snippet") or "").strip()
        source = f"[{title}]({url})" if title and url else title or url
        if snippet:
            clues.append(f"{index}. {source}：{snippet}")
        elif source:
            clues.append(f"{index}. {source}")
    if not clues:
        return NO_EVIDENCE_ANSWER
    return (
        f"{NO_EVIDENCE_ANSWER}\n\n"
        "联网补充线索（未入库，仅供后续核验）：\n"
        + "\n".join(clues)
    )


def stream_progress(stage: str, message: str) -> dict[str, str]:
    return {"type": "progress", "stage": stage, "message": message}


def describe_plan_progress(plan: QuestionPlan) -> str:
    type_labels = {
        "topic_to_company": "主题到公司检索",
        "company_compare": "公司对比",
        "risk_analysis": "风险分析",
        "industry_bottleneck": "产业瓶颈分析",
        "company_profile": "公司画像",
        "thematic_research": "主题研究",
    }
    parts = [f"问题规划完成：{type_labels.get(plan.answer_type, plan.answer_type)}"]
    if plan.companies:
        parts.append(f"核心公司为{'、'.join(plan.companies[:4])}")
    if plan.topics:
        parts.append(f"关注主题为{'、'.join(plan.topics[:4])}")
    return "，".join(parts)


def chunk_text(text: str, *, size: int = 18) -> Iterator[str]:
    value = str(text or "")
    if not value:
        return
    for index in range(0, len(value), size):
        yield value[index : index + size]


def normalize_conversation_history(
    messages: list[dict[str, str]] | None,
    *,
    max_turns: int,
    max_chars: int,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages or []:
        role = str(message.get("role") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content[:3000]})

    if max_turns <= 0:
        return []
    normalized = normalized[-max_turns * 2 :]
    if max_chars <= 0:
        return normalized

    selected: list[dict[str, str]] = []
    used = 0
    for message in reversed(normalized):
        content = message["content"]
        size = len(content)
        if selected and used + size > max_chars:
            break
        if size > max_chars:
            content = content[-max_chars:]
            size = len(content)
        selected.append({"role": message["role"], "content": content})
        used += size
    return list(reversed(selected))


def build_contextualizer_prompt(question: str) -> str:
    return f"""请把下面当前问题改写成一个不依赖上下文也能检索的单轮问题。
如果当前问题已经完整，原样输出。

当前问题：{question}
"""


def heuristic_contextual_question(question: str, history: list[dict[str, str]]) -> str:
    question = question.strip()
    if not history or not question_needs_context(question):
        return question
    last_user = next((message["content"] for message in reversed(history) if message["role"] == "user"), "")
    if not last_user:
        return question
    companies = extract_companies(last_user)
    if companies:
        return f"{'和'.join(companies)} {question}"
    return f"{last_user}；追问：{question}"


def question_needs_context(question: str) -> bool:
    question = question.strip()
    context_terms = (
        "它",
        "其",
        "他们",
        "这些",
        "这类",
        "上述",
        "上面",
        "前面",
        "刚才",
        "继续",
        "进一步",
        "再展开",
        "风险呢",
        "差异呢",
        "还有呢",
    )
    if any(term in question for term in context_terms):
        return True
    if "主要风险" in question and not extract_companies(question):
        return True
    independent_terms = ("哪些公司", "上市公司", "有哪些", "是什么", "为什么", "如何", "多少")
    if any(term in question for term in independent_terms):
        return False
    return len(question) <= 18


def sanitize_contextual_question(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    text = text.replace("```", "").strip()
    if text.startswith(("改写后的问题：", "问题：")):
        text = text.split("：", 1)[-1].strip()
    text = text.strip("\"'“”‘’ \n")
    return text[:500]


def format_history_for_prompt(history: list[dict[str, str]]) -> str:
    lines = ["历史对话："]
    for message in history:
        role = "用户" if message["role"] == "user" else "助手"
        lines.append(f"{role}：{message['content']}")
    return "\n".join(lines)


def build_evidence(graph_records: list[dict[str, Any]], rag_hits: list[RagHit]) -> list[dict[str, Any]]:
    evidence = []
    for record in graph_records:
        text = str(record.get("evidence") or "")
        if text:
            evidence.append(
                {
                    "kind": "graph",
                    "source": record.get("source", ""),
                    "source_tier": record.get("source_tier", ""),
                    "page": record.get("page", ""),
                    "section": record.get("section", ""),
                    "evidence": text,
                    "score": "",
                }
            )
    for hit in rag_hits:
        evidence.append(
            {
                "kind": "rag",
                "source": hit.source_title,
                "source_tier": hit.source_tier,
                "page": hit.page,
                "section": hit.section,
                "evidence": hit.snippet,
                "score": hit.score,
            }
        )
    return evidence


def graph_records_to_subgraph(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    relation_rows = []
    for record in records:
        company = str(record.get("company") or "")
        target = str(record.get("target") or "")
        relation = str(record.get("relation") or "")
        if not company or not target or not relation:
            continue
        target_labels = record.get("target_labels") or []
        target_type = target_labels[0] if isinstance(target_labels, list) and target_labels else ""
        relation_rows.append(
            {
                "head_type": "Company",
                "head_name": company,
                "relation": relation,
                "tail_type": target_type,
                "tail_name": target,
            }
        )
    return subgraph_edges(relation_rows)


def ensure_relation_cards(cards: list[Any], raw_cards: list[Any], relation: str, *, limit: int) -> list[Any]:
    if any(getattr(card, "relation", "") == relation for card in cards):
        return cards
    additions = [card for card in raw_cards if getattr(card, "relation", "") == relation]
    if not additions:
        return cards
    merged = [additions[0], *cards]
    deduped = []
    seen = set()
    for card in merged:
        key = (getattr(card, "kind", ""), getattr(card, "source", ""), getattr(card, "page", ""), getattr(card, "evidence", "")[:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)
    return deduped[:limit]


def fallback_answer(graph_records: list[dict[str, Any]], rag_hits: list[RagHit]) -> str:
    if graph_records:
        facts = []
        for record in graph_records[:8]:
            company = record.get("company", "")
            relation = RELATION_LABELS.get(str(record.get("relation", "")), str(record.get("relation", "")))
            target = record.get("target", "")
            source = record.get("source", "")
            page = record.get("page", "")
            if company and target:
                facts.append(f"{company} {relation} {target}（{source} p.{page}）")
        if facts:
            return "根据当前知识库证据，" + "；".join(facts) + "。"
    if rag_hits:
        source_bits = [f"{hit.source_title} p.{hit.page}" for hit in rag_hits[:3] if hit.source_title]
        return "根据本地文档检索，找到相关原文证据：" + "；".join(source_bits) + "。"
    return NO_EVIDENCE_ANSWER
