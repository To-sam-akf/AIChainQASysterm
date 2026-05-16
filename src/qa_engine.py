"""Professional CSV/Neo4j + local RAG + LLM QA orchestration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.cypher_generator import GeneratedCypher, generate_cypher
from src.curated_graph import DEFAULT_CURATED_DIR
from src.frontend_data import LocalKnowledgeGraph, RELATION_LABELS, subgraph_edges
from src.llm_client import OpenAICompatibleClient, load_dotenv
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
from src.question_planner import QuestionPlan, plan_question
from src.rag_index import DEFAULT_RAG_DIR, LocalRagIndex, RagHit


NO_EVIDENCE_ANSWER = "当前知识库中未找到相关证据。"

ANSWER_SYSTEM_PROMPT = """你是中国 AI 算力产业链专业投研问答助手。
只能根据提供的 Neo4j/CSV 图谱结果和本地 RAG 原文片段回答，不要编造证据外信息。
答案用中文，面向资深投资者，按“结论、证据、研究要点、风险与边界”组织。
可以给事实归纳、产业链位置、催化因素、风险和跟踪指标；禁止给股票买卖建议、目标价或收益预测。"""


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


class QAEngine:
    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        graph_client: Any | None = None,
        csv_graph: LocalKnowledgeGraph | None = None,
        rag_index: LocalRagIndex | None = None,
        enable_llm_cypher: bool = True,
        rag_top_k: int = 6,
        graph_limit: int = 50,
        rerank_top_n: int = 40,
        evidence_top_n: int = 10,
        core_companies_only: bool = True,
        status: QAEngineStatus | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.graph_client = graph_client
        self.csv_graph = csv_graph
        self.rag_index = rag_index
        self.enable_llm_cypher = enable_llm_cypher
        self.rag_top_k = rag_top_k
        self.graph_limit = graph_limit
        self.rerank_top_n = rerank_top_n
        self.evidence_top_n = evidence_top_n
        self.core_companies_only = core_companies_only
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
        rerank_top_n = int(os.getenv("QA_RERANK_TOP_N", "40"))
        evidence_top_n = int(os.getenv("QA_EVIDENCE_TOP_N", "10"))
        core_companies_only = os.getenv("QA_CORE_COMPANIES_ONLY", "true").casefold() != "false"
        enable_llm_cypher = os.getenv("QA_ENABLE_LLM_CYPHER", "true").casefold() != "false"
        graph_backend = os.getenv("QA_GRAPH_BACKEND", "auto").casefold()

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
        return cls(
            llm_client=llm_client,
            graph_client=graph_client,
            csv_graph=csv_graph,
            rag_index=rag_index,
            enable_llm_cypher=enable_llm_cypher,
            rag_top_k=rag_top_k,
            graph_limit=graph_limit,
            rerank_top_n=rerank_top_n,
            evidence_top_n=evidence_top_n,
            core_companies_only=core_companies_only,
            status=status,
        )

    def close(self) -> None:
        if self.graph_client is not None and hasattr(self.graph_client, "close"):
            self.graph_client.close()

    def answer_question(self, question: str) -> dict[str, Any]:
        question = question.strip()
        errors: list[str] = []
        plan = plan_question(question, client=self.llm_client, core_companies_only=self.core_companies_only)
        generated = self._generate_display_cypher(question, plan, errors)
        if generated.error:
            errors.append(generated.error)
        graph_records = self._query_graph(generated, plan, errors)
        rag_hits = self._search_rag(question, plan, errors)
        raw_cards = [*cards_from_graph_records(graph_records, plan), *cards_from_rag_hits(rag_hits, plan)]
        evidence_cards = rank_evidence_cards(raw_cards, limit=self.evidence_top_n)
        if plan.answer_type == "risk_analysis":
            evidence_cards = ensure_relation_cards(evidence_cards, raw_cards, "DISCLOSES_RISK", limit=self.evidence_top_n)
        evidence = legacy_evidence_rows(evidence_cards)
        answer = self._generate_answer(question, plan, graph_records, evidence_cards, errors)
        return {
            "question": question,
            "answer": answer,
            "answer_type": plan.answer_type,
            "plan": plan.to_dict(),
            "cypher": generated.cypher,
            "cypher_params": generated.params,
            "cypher_source": generated.source,
            "graph_records": graph_records,
            "rag_hits": [hit.to_dict() for hit in rag_hits],
            "evidence_cards": [card.to_dict() for card in evidence_cards],
            "evidence": evidence,
            "subgraph": graph_records_to_subgraph(graph_records),
            "diagnostics": {
                "graph_backend": self.status.graph_backend,
                "graph_records": len(graph_records),
                "rag_hits": len(rag_hits),
                "evidence_cards": len(evidence_cards),
                "rerank_top_n": self.rerank_top_n,
                "graph_error": self.status.graph_error,
                "rag_error": self.status.rag_error,
                "llm_error": self.status.llm_error,
            },
            "errors": errors,
        }

    def _generate_display_cypher(self, question: str, plan: QuestionPlan, errors: list[str]) -> GeneratedCypher:
        if self.status.graph_backend == "csv" or self.graph_client is None:
            return GeneratedCypher(
                cypher=pseudo_cypher_for_plan(plan, limit=self.graph_limit),
                params={
                    "companies": plan.companies,
                    "topic": plan.expanded_topics[0] if plan.expanded_topics else "",
                },
                source="question_plan_csv",
            )
        try:
            return generate_cypher(
                question,
                client=self.llm_client,
                enable_llm=self.enable_llm_cypher,
                limit=self.graph_limit,
            )
        except Exception as exc:
            errors.append(f"Cypher generation failed: {exc}")
            return GeneratedCypher(cypher=pseudo_cypher_for_plan(plan, limit=self.graph_limit), params={}, source="fallback")

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

    def _generate_answer(
        self,
        question: str,
        plan: QuestionPlan,
        graph_records: list[dict[str, Any]],
        evidence_cards: list[Any],
        errors: list[str],
    ) -> str:
        if not evidence_cards:
            return NO_EVIDENCE_ANSWER
        if self.llm_client is not None and hasattr(self.llm_client, "chat_text"):
            try:
                return self.llm_client.chat_text(
                    system_prompt=ANSWER_SYSTEM_PROMPT,
                    user_prompt=build_professional_answer_prompt(question, plan, graph_records, evidence_cards),
                    temperature=0.2,
                )
            except Exception as exc:
                errors.append(f"LLM answer failed: {exc}")
        return fallback_professional_answer(plan, evidence_cards, graph_records)


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
