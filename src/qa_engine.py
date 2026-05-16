"""Unified Neo4j + local RAG + LLM QA orchestration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.cypher_generator import GeneratedCypher, generate_cypher
from src.frontend_data import RELATION_LABELS, subgraph_edges
from src.llm_client import OpenAICompatibleClient, load_dotenv
from src.neo4j_client import Neo4jReadClient
from src.rag_index import DEFAULT_RAG_DIR, LocalRagIndex, RagHit


NO_EVIDENCE_ANSWER = "当前知识库中未找到相关证据。"

ANSWER_SYSTEM_PROMPT = """你是 AI 算力产业链知识图谱问答系统。
只能根据提供的 Neo4j 图谱结果和本地 RAG 原文片段回答。
不要编造没有证据的信息。答案用中文，简洁说明结论，并标注来源报告或页码。
遇到投资、股票买卖、目标价、收益预测类问题，只回答知识库事实，不给投资建议。"""


@dataclass
class QAEngineStatus:
    neo4j_enabled: bool
    rag_enabled: bool
    llm_enabled: bool
    rag_error: str = ""
    llm_error: str = ""


class QAEngine:
    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        graph_client: Any | None = None,
        rag_index: LocalRagIndex | None = None,
        enable_llm_cypher: bool = True,
        rag_top_k: int = 6,
        graph_limit: int = 50,
        status: QAEngineStatus | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.graph_client = graph_client
        self.rag_index = rag_index
        self.enable_llm_cypher = enable_llm_cypher
        self.rag_top_k = rag_top_k
        self.graph_limit = graph_limit
        self.status = status or QAEngineStatus(
            neo4j_enabled=graph_client is not None,
            rag_enabled=rag_index is not None,
            llm_enabled=llm_client is not None,
        )

    @classmethod
    def from_env(cls) -> "QAEngine":
        load_dotenv()
        rag_top_k = int(os.getenv("RAG_TOP_K", "6"))
        graph_limit = int(os.getenv("QA_GRAPH_LIMIT", "50"))
        enable_llm_cypher = os.getenv("QA_ENABLE_LLM_CYPHER", "true").casefold() != "false"

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

        graph_client = Neo4jReadClient()
        status = QAEngineStatus(
            neo4j_enabled=True,
            rag_enabled=rag_index is not None,
            llm_enabled=llm_client is not None,
            rag_error=rag_error,
            llm_error=llm_error,
        )
        return cls(
            llm_client=llm_client,
            graph_client=graph_client,
            rag_index=rag_index,
            enable_llm_cypher=enable_llm_cypher,
            rag_top_k=rag_top_k,
            graph_limit=graph_limit,
            status=status,
        )

    def close(self) -> None:
        if self.graph_client is not None and hasattr(self.graph_client, "close"):
            self.graph_client.close()

    def answer_question(self, question: str) -> dict[str, Any]:
        question = question.strip()
        errors: list[str] = []
        generated = generate_cypher(
            question,
            client=self.llm_client,
            enable_llm=self.enable_llm_cypher,
            limit=self.graph_limit,
        )
        if generated.error:
            errors.append(generated.error)
        graph_records = self._query_graph(generated, errors)
        rag_hits = self._search_rag(question, errors)
        evidence = build_evidence(graph_records, rag_hits)
        answer = self._generate_answer(question, graph_records, rag_hits, evidence, errors)
        return {
            "question": question,
            "answer": answer,
            "cypher": generated.cypher,
            "cypher_params": generated.params,
            "cypher_source": generated.source,
            "graph_records": graph_records,
            "rag_hits": [hit.to_dict() for hit in rag_hits],
            "evidence": evidence,
            "subgraph": graph_records_to_subgraph(graph_records),
            "errors": errors,
        }

    def _query_graph(self, generated: GeneratedCypher, errors: list[str]) -> list[dict[str, Any]]:
        if self.graph_client is None:
            errors.append("Neo4j client is not configured.")
            return []
        try:
            return self.graph_client.run_read_query(
                generated.cypher,
                generated.params,
                limit=self.graph_limit,
            )
        except Exception as exc:
            errors.append(f"Neo4j query failed: {exc}")
            return []

    def _search_rag(self, question: str, errors: list[str]) -> list[RagHit]:
        if self.rag_index is None:
            if self.status.rag_error:
                errors.append(f"RAG index unavailable: {self.status.rag_error}")
            return []
        try:
            return self.rag_index.search(question, top_k=self.rag_top_k)
        except Exception as exc:
            errors.append(f"RAG search failed: {exc}")
            return []

    def _generate_answer(
        self,
        question: str,
        graph_records: list[dict[str, Any]],
        rag_hits: list[RagHit],
        evidence: list[dict[str, Any]],
        errors: list[str],
    ) -> str:
        if not evidence:
            return NO_EVIDENCE_ANSWER
        if self.llm_client is not None and hasattr(self.llm_client, "chat_text"):
            try:
                return self.llm_client.chat_text(
                    system_prompt=ANSWER_SYSTEM_PROMPT,
                    user_prompt=build_answer_prompt(question, graph_records, rag_hits),
                    temperature=0.2,
                )
            except Exception as exc:
                errors.append(f"LLM answer failed: {exc}")
        return fallback_answer(graph_records, rag_hits)


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
