"""Professional CSV/Neo4j + PostgreSQL RAG + LLM QA orchestration.

QA 引擎核心模块，协调多源数据检索和 LLM 生成的全流程。
支持两种图谱后端（Neo4j / CSV）和多种检索方式（RAG、研究报告、语义向量），
最终由 LLM 整合生成专业投研答案。

核心流程（workflow）：
1. history — 较早对话 LLM 压缩 + 最近对话原文保留
2. contextualize — 追问上下文补全
3. plan — 问题规划（启发式 / LLM）
4. cypher — 生成图谱查询语句
5. graph — 执行图查询获取结构化关系
6. rag — 语义检索非结构化文本
7. research — 检索研究报告声明
8. evidence — 证据排序与筛选
9. answer — LLM 生成最终答案
10. verify — 答案事实一致性验证
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.cypher_generator import GeneratedCypher, generate_cypher
from src.curated_graph import DEFAULT_CURATED_DIR
from src.extraction_schema import normalize_name
from src.frontend_data import LocalKnowledgeGraph, RELATION_LABELS, subgraph_edges
from src.llm_client import OpenAICompatibleClient, load_dotenv
from src.neo4j_client import Neo4jReadClient
from src.agents.verification import build_evidence_limited_answer, verify_answer_support
from src.professional_qa import (
    build_professional_answer_prompt,
    cards_from_graph_records,
    cards_from_rag_hits,
    cards_from_research_hits,
    assign_citation_ids,
    fallback_professional_answer,
    legacy_evidence_rows,
    pseudo_cypher_for_plan,
    rank_evidence_cards,
    search_csv_graph,
)
from src.postgres_retrieval import (
    EMBEDDING_DIMENSIONS,
    PostgresRagIndex,
    PostgresResearchMemory,
    PostgresRetrievalStore,
    PostgresSemanticIndex,
)
from src.question_planner import QuestionPlan, extract_companies, heuristic_plan_question, plan_question
from src.rag_index import RagHit
from src.reranker import normalize_rerank_mode
from src.research_agent import build_research_outputs
from src.research_claims import ResearchHit


# ============================================================
# 常量与模板
# ============================================================

NO_EVIDENCE_ANSWER = "当前知识库中未找到相关证据。"

ANSWER_SYSTEM_PROMPT_TEMPLATE = """你是中国 AI 算力产业链专业投研问答助手。
只能根据提供的 Neo4j/CSV 图谱结果和本地 RAG 原文片段回答，不要编造证据外信息。
当前日期是 {current_date}。回答中不得臆造当前日期或把已给定报告年份误判为未来。
答案用中文，面向资深投资者，按"核心判断、技术机理、产业传导、公司排序、领先指标、反证/边界、证据"组织。
不要泛泛总结；涉及"哪些公司/谁受益"时必须按 core/direct/indirect/mentioned 敞口分层，直接敞口优先。
涉及"为什么/瓶颈/趋势"时必须解释技术机理和商业传导；缺少证据的栏目明确写"当前证据不足"。
可以给事实归纳、产业链位置、催化因素、风险和跟踪指标；禁止给股票买卖建议、目标价或收益预测。"""

CONTEXTUALIZER_SYSTEM_PROMPT = """你是中国 AI 算力产业链问答系统的追问改写器。
根据历史对话，把用户当前问题改写成可独立检索的中文问题。
只输出改写后的问题，不解释，不回答问题，不引入历史对话中没有出现的新公司或主题。"""

HISTORY_COMPRESSION_SYSTEM_PROMPT = """你是对话上下文压缩器。
请把较早的多轮对话压缩为可供后续问答使用的短期记忆。
必须保留用户目标、公司与主题、关键事实和结论、约束与偏好、纠正信息、尚未解决的问题。
删除寒暄、重复表述和无关细节，不得补充原对话中没有的信息。
只输出摘要正文，不要回答最新问题，不要使用 Markdown 标题。"""

HISTORY_SUMMARY_PREFIX = "历史对话压缩摘要（仅作上下文，不是对当前问题的回答）：\n"

# 模板化 Cypher 查询中支持的关系类型集合
TEMPLATE_RELATIONS = {
    "USES_TECHNOLOGY",
    "HAS_PRODUCT",
    "BELONGS_TO_CHAIN",
    "HAS_METRIC",
    "DISCLOSES_RISK",
    "SUPPORTED_BY_POLICY",
    "CONSTRAINS",
    "ENABLES",
    "DRIVES",
    "DEPENDS_ON",
    "RELIEVES",
    "HAS_EXPOSURE",
    "HAS_INDICATOR",
    "BENEFITS_FROM",
}


def answer_system_prompt(current_date: str | None = None) -> str:
    """生成带当前日期的答案系统提示词。

    Args:
        current_date: 当前日期（可选，默认使用 datetime.now()）

    Returns:
        格式化后的系统提示词
    """
    return ANSWER_SYSTEM_PROMPT_TEMPLATE.format(current_date=current_date or datetime.now().date().isoformat())


@dataclass
class QAEngineStatus:
    """QA 引擎状态信息，记录各组件的可用性和错误状态。

    Attributes:
        neo4j_enabled: Neo4j 图数据库是否可用
        rag_enabled: RAG 索引是否可用
        llm_enabled: LLM 客户端是否可用
        research_enabled: 研究报告索引是否可用
        embedding_enabled: 语义向量索引是否可用
        csv_graph_enabled: CSV 图谱是否可用
        graph_backend: 当前使用的图谱后端（"neo4j" / "csv" / "none"）
        graph_data_dir: 图谱数据目录路径
        graph_error: 图谱组件的错误信息
        rag_error: RAG 索引的错误信息
        research_error: 研究报告索引的错误信息
        embedding_error: 语义索引的错误信息
        llm_error: LLM 客户端的错误信息
    """
    neo4j_enabled: bool
    rag_enabled: bool
    llm_enabled: bool
    research_enabled: bool = False
    embedding_enabled: bool = False
    csv_graph_enabled: bool = False
    graph_backend: str = "neo4j"
    graph_data_dir: str = ""
    graph_error: str = ""
    rag_error: str = ""
    research_error: str = ""
    embedding_error: str = ""
    llm_error: str = ""


class CountingLLMClient:
    """LLM 客户端代理，在保持 hasattr 行为的同时记录远程调用次数。

    用于诊断和调试，可在 diagnostics 中输出每次 QA 流程的 LLM 调用量。
    只代理 TRACKED_METHODS 中的方法，其余属性和方法透传。
    """

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
    """QA 引擎主类，协调知识图谱、RAG、研究报告、语义检索和 LLM 生成的全流程。

    支持两种回答模式：
    1. Agent 模式（enable_agent=True）— 使用 QAAgent 进行多步工具调用
    2. Workflow 模式 — 按固定流水线依次执行各阶段

    数据源支持：
    - 知识图谱：Neo4j 或 CSV 文件
    - RAG 索引：PostgreSQL 中的文档向量索引
    - 研究报告：PostgreSQL 中的声明/摘要索引
    - 语义索引：PostgreSQL 中的多类型语义向量索引
    """

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        graph_client: Any | None = None,
        csv_graph: LocalKnowledgeGraph | None = None,
        rag_index: Any | None = None,
        research_memory: Any | None = None,
        semantic_index: Any | None = None,
        retrieval_store: PostgresRetrievalStore | None = None,
        enable_llm_cypher: bool = False,
        enable_llm_planner: bool = False,
        contextualizer_mode: str = "auto",
        rag_top_k: int = 6,
        semantic_top_k: int = 8,
        graph_limit: int = 50,
        rerank_top_n: int = 12,
        evidence_top_n: int = 6,
        core_companies_only: bool = True,
        history_max_turns: int = 3,
        history_max_chars: int = 4000,
        history_compression_enabled: bool = True,
        history_summary_max_chars: int = 1600,
        history_compression_chunk_chars: int = 12000,
        enable_agent: bool = True,
        agent_max_steps: int = 4,
        agent_runner: str = "langgraph",
        rerank_mode: str = "auto",
        drift_max_subquestions: int = 6,
        global_dossier_top_k: int = 3,
        local_claim_top_k: int = 12,
        graph_path_top_k: int = 6,
        status: QAEngineStatus | None = None,
    ) -> None:
        """初始化 QA 引擎。

        Args:
            llm_client: LLM 客户端（如 OpenAICompatibleClient）
            graph_client: Neo4j 图数据库客户端
            csv_graph: CSV 图谱数据（LocalKnowledgeGraph）
            rag_index: RAG 索引（PostgresRagIndex）
            research_memory: 研究报告索引（PostgresResearchMemory）
            semantic_index: 语义向量索引（PostgresSemanticIndex）
            retrieval_store: PostgreSQL 检索存储
            enable_llm_cypher: 是否启用 LLM 生成 Cypher 查询
            enable_llm_planner: 是否启用 LLM 问题规划
            contextualizer_mode: 追问上下文补全模式（"auto" / "heuristic" / "llm"）
            rag_top_k: RAG 检索返回的候选数
            semantic_top_k: 语义检索返回的候选数
            graph_limit: 图谱查询返回的记录数上限
            rerank_top_n: 重排序的候选数
            evidence_top_n: 最终保留的证据卡片数
            core_companies_only: 是否只查询核心上市公司
            history_max_turns: 不压缩、直接保留原文的最近对话轮数
            history_max_chars: 压缩摘要与最近原文的总字符预算
            history_compression_enabled: 是否使用 LLM 压缩较早对话
            history_summary_max_chars: 压缩摘要的最大字符数
            history_compression_chunk_chars: 单次送入压缩器的历史字符数
            enable_agent: 是否启用 Agent 模式
            agent_max_steps: Agent 最大步数（1-4）
            agent_runner: Agent 运行器（"langgraph" / "legacy"）
            rerank_mode: 重排序模式（"auto" / "llm" / "none"）
            drift_max_subquestions: GraphRAG DRIFT 最大子问题数
            global_dossier_top_k: 全局档案检索 top-k
            local_claim_top_k: 本地声明检索 top-k
            graph_path_top_k: 图谱路径检索 top-k
            status: 引擎状态对象（可选，不传则自动生成）
        """
        self.llm_client = llm_client
        self.graph_client = graph_client
        self.csv_graph = csv_graph
        self.rag_index = rag_index
        self.research_memory = research_memory
        self.semantic_index = semantic_index
        self.retrieval_store = retrieval_store
        self.enable_llm_cypher = enable_llm_cypher
        self.enable_llm_planner = enable_llm_planner
        self.contextualizer_mode = normalize_contextualizer_mode(contextualizer_mode)
        self.rag_top_k = rag_top_k
        self.semantic_top_k = semantic_top_k
        self.graph_limit = graph_limit
        self.rerank_top_n = rerank_top_n
        self.evidence_top_n = evidence_top_n
        self.core_companies_only = core_companies_only
        self.history_max_turns = max(0, int(history_max_turns or 0))
        self.history_max_chars = max(0, int(history_max_chars or 0))
        self.history_compression_enabled = bool(history_compression_enabled)
        self.history_summary_max_chars = max(200, int(history_summary_max_chars or 1600))
        self.history_compression_chunk_chars = max(1000, int(history_compression_chunk_chars or 12000))
        self.enable_agent = enable_agent
        self.agent_max_steps = normalize_agent_max_steps(agent_max_steps)
        self.agent_runner = normalize_agent_runner(agent_runner)
        self.rerank_mode = normalize_rerank_mode(rerank_mode)
        self.drift_max_subquestions = max(1, int(drift_max_subquestions or 6))
        self.global_dossier_top_k = max(0, int(global_dossier_top_k or 3))
        self.local_claim_top_k = max(0, int(local_claim_top_k or 12))
        self.graph_path_top_k = max(0, int(graph_path_top_k or 6))
        self.status = status or QAEngineStatus(
            neo4j_enabled=graph_client is not None,
            csv_graph_enabled=csv_graph is not None,
            rag_enabled=rag_index is not None,
            research_enabled=research_memory is not None,
            embedding_enabled=semantic_index is not None,
            llm_enabled=llm_client is not None,
            graph_backend="neo4j" if graph_client is not None else "csv" if csv_graph is not None else "none",
        )

    @classmethod
    def from_env(cls) -> "QAEngine":
        """从环境变量构造 QAEngine 实例。

        读取 .env 文件或系统环境变量中的配置，自动初始化所有组件。
        组件初始化失败时记录错误到 status，不影响引擎启动（优雅降级）。

        Returns:
            配置完成的 QAEngine 实例
        """
        load_dotenv()
        rag_top_k = int(os.getenv("RAG_TOP_K", "6"))
        semantic_top_k = int(os.getenv("SEMANTIC_TOP_K", "8"))
        graph_limit = int(os.getenv("QA_GRAPH_LIMIT", "50"))
        rerank_top_n = int(os.getenv("QA_RERANK_TOP_N", "12"))
        evidence_top_n = int(os.getenv("QA_EVIDENCE_TOP_N", "6"))
        core_companies_only = os.getenv("QA_CORE_COMPANIES_ONLY", "true").casefold() != "false"
        enable_llm_cypher = os.getenv("QA_ENABLE_LLM_CYPHER", "false").casefold() != "false"
        enable_llm_planner = os.getenv("QA_ENABLE_LLM_PLANNER", "false").casefold() != "false"
        contextualizer_mode = normalize_contextualizer_mode(os.getenv("QA_CONTEXTUALIZER_MODE", "auto"))
        history_max_turns = int(os.getenv("QA_HISTORY_MAX_TURNS", "3"))
        history_max_chars = int(os.getenv("QA_HISTORY_MAX_CHARS", "4000"))
        history_compression_enabled = os.getenv("QA_HISTORY_COMPRESSION_ENABLED", "true").casefold() != "false"
        history_summary_max_chars = int(os.getenv("QA_HISTORY_SUMMARY_MAX_CHARS", "1600"))
        history_compression_chunk_chars = int(os.getenv("QA_HISTORY_COMPRESSION_CHUNK_CHARS", "12000"))
        enable_agent = os.getenv("QA_ENABLE_AGENT", "true").casefold() != "false"
        agent_max_steps = normalize_agent_max_steps(os.getenv("QA_AGENT_MAX_STEPS", "4"))
        agent_runner = normalize_agent_runner(os.getenv("QA_AGENT_RUNNER", "langgraph"))
        rerank_mode = normalize_rerank_mode(os.getenv("QA_RERANK_MODE", "auto"))
        drift_max_subquestions = int(os.getenv("QA_DRIFT_MAX_SUBQUESTIONS", "6"))
        global_dossier_top_k = int(os.getenv("QA_GLOBAL_DOSSIER_TOP_K", "3"))
        local_claim_top_k = int(os.getenv("QA_LOCAL_CLAIM_TOP_K", "12"))
        graph_path_top_k = int(os.getenv("QA_GRAPH_PATH_TOP_K", "6"))
        graph_backend = os.getenv("QA_GRAPH_BACKEND", "auto").casefold()

        # PostgreSQL 检索存储
        retrieval_store = PostgresRetrievalStore.from_env()
        try:
            retrieval_store.ensure_ready()
        except Exception:
            retrieval_store.close()
            raise
        rag_index = PostgresRagIndex(retrieval_store)
        research_memory = PostgresResearchMemory(retrieval_store)
        rag_error = ""
        research_error = ""

        # LLM 客户端
        llm_client = None
        llm_error = ""
        try:
            llm_client = OpenAICompatibleClient()
        except Exception as exc:
            llm_error = str(exc)

        # 语义向量索引
        semantic_index = None
        embedding_error = ""
        try:
            from src.embedding_client import OpenAICompatibleEmbeddingClient, embedding_configured

            if embedding_configured():
                embedding_client = OpenAICompatibleEmbeddingClient(dimensions=EMBEDDING_DIMENSIONS)
                semantic_index = PostgresSemanticIndex(
                    retrieval_store,
                    embedding_client=embedding_client,
                )
        except Exception as exc:
            embedding_error = str(exc)

        # CSV 图谱
        csv_graph = None
        graph_error = ""
        graph_data_dir = Path(os.getenv("KG_DATA_DIR", str(DEFAULT_CURATED_DIR)))
        if not (graph_data_dir / "entities.csv").exists() or not (graph_data_dir / "relations.csv").exists():
            graph_data_dir = Path(__file__).resolve().parents[1] / "data" / "verified"
        try:
            csv_graph = LocalKnowledgeGraph.from_dir(graph_data_dir)
        except Exception as exc:
            graph_error = f"CSV graph unavailable: {exc}"

        # Neo4j 图数据库
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
            research_enabled=research_memory is not None,
            embedding_enabled=semantic_index is not None,
            llm_enabled=llm_client is not None,
            graph_backend=selected_backend,
            graph_data_dir=str(graph_data_dir),
            graph_error=graph_error,
            rag_error=rag_error,
            research_error=research_error,
            embedding_error=embedding_error,
            llm_error=llm_error,
        )
        return cls(
            llm_client=llm_client,
            graph_client=graph_client,
            csv_graph=csv_graph,
            rag_index=rag_index,
            research_memory=research_memory,
            semantic_index=semantic_index,
            retrieval_store=retrieval_store,
            enable_llm_cypher=enable_llm_cypher,
            enable_llm_planner=enable_llm_planner,
            contextualizer_mode=contextualizer_mode,
            rag_top_k=rag_top_k,
            semantic_top_k=semantic_top_k,
            graph_limit=graph_limit,
            rerank_top_n=rerank_top_n,
            evidence_top_n=evidence_top_n,
            core_companies_only=core_companies_only,
            history_max_turns=history_max_turns,
            history_max_chars=history_max_chars,
            history_compression_enabled=history_compression_enabled,
            history_summary_max_chars=history_summary_max_chars,
            history_compression_chunk_chars=history_compression_chunk_chars,
            enable_agent=enable_agent,
            agent_max_steps=agent_max_steps,
            agent_runner=agent_runner,
            rerank_mode=rerank_mode,
            drift_max_subquestions=drift_max_subquestions,
            global_dossier_top_k=global_dossier_top_k,
            local_claim_top_k=local_claim_top_k,
            graph_path_top_k=graph_path_top_k,
            status=status,
        )

    def close(self) -> None:
        """释放资源：关闭图数据库连接和 PostgreSQL 连接。"""
        if self.graph_client is not None and hasattr(self.graph_client, "close"):
            self.graph_client.close()
        if self.retrieval_store is not None:
            self.retrieval_store.close()

    # ============================================================
    # 公开入口：同步回答 + 流式回答
    # ============================================================

    def answer_question(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """回答用户问题（同步入口）。

        根据 enable_agent 配置选择使用 Agent 模式还是 Workflow 模式。
        Agent 模式支持多步工具调用，Workflow 模式按固定流水线执行。

        Args:
            question: 用户问题
            conversation_history: 对话历史（可选）
            thinking_enabled: 是否显示思考过程
            reasoning_effort: 推理努力程度

        Returns:
            包含 answer、evidence、verification、subgraph、diagnostics 等的完整结果字典
        """
        if self.enable_agent:
            from src.agents.qa_agent import QAAgent

            return QAAgent(self, max_steps=self.agent_max_steps).run(
                question,
                conversation_history=conversation_history,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            )
        return self._answer_question_workflow(
            question,
            conversation_history=conversation_history,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
        )

    def answer_question_stream(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """流式回答用户问题。

        逐步产生 progress（进度）、answer_delta（答案片段）、final（最终结果）事件。
        支持思考过程的可视化展示。

        Args:
            question: 用户问题
            conversation_history: 对话历史
            thinking_enabled: 是否显示思考过程
            reasoning_effort: 推理努力程度

        Yields:
            事件字典：{"type": "progress", "stage": ..., "message": ...}
            或 {"type": "answer_delta", "content": ...}
            或 {"type": "answer_complete", "answer": ..., "reasoning_content": ...}
            或 {"type": "final", "result": {...}}
        """
        if self.enable_agent:
            from src.agents.qa_agent import QAAgent

            yield from QAAgent(self, max_steps=self.agent_max_steps).run_stream(
                question,
                conversation_history=conversation_history,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            )
            return
        yield from self._answer_question_stream_workflow(
            question,
            conversation_history=conversation_history,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
        )

    # ============================================================
    # Workflow 内部方法
    # ============================================================

    def _prepare_conversation_history(
        self,
        conversation_history: list[dict[str, str]] | None,
        *,
        errors: list[str],
        llm_options: dict[str, Any],
        llm_client: Any | None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Build short-term memory from an LLM summary plus recent verbatim turns."""
        normalized = normalize_conversation_messages(conversation_history)
        fallback = normalize_conversation_history(
            normalized,
            max_turns=self.history_max_turns,
            max_chars=self.history_max_chars,
        )
        metadata = {
            "history_strategy": "recent_window",
            "history_source_messages": len(normalized),
            "history_summarized_messages": 0,
            "history_summary_chars": 0,
            "history_compressed": False,
        }
        if not normalized:
            metadata["history_strategy"] = "empty"
            return [], metadata
        recent_start = max(0, len(normalized) - self.history_max_turns * 2)
        older = normalized[:recent_start]
        recent_candidates = normalized[recent_start:]
        if not older:
            metadata["history_strategy"] = "full_recent"
            return trim_history_to_chars(recent_candidates, self.history_max_chars), metadata
        if not self.history_compression_enabled or llm_client is None:
            return fallback, metadata

        summary_budget = self.history_summary_max_chars
        if self.history_max_chars > 0:
            available_summary_chars = self.history_max_chars - len(HISTORY_SUMMARY_PREFIX)
            if available_summary_chars <= 0:
                return fallback, metadata
            summary_budget = min(summary_budget, max(1, available_summary_chars // 2))
            recent_budget = max(0, self.history_max_chars - summary_budget - len(HISTORY_SUMMARY_PREFIX))
        else:
            recent_budget = 0
        recent = (
            []
            if self.history_max_chars > 0 and recent_budget <= 0
            else trim_history_to_chars(recent_candidates, recent_budget)
        )
        dropped_recent_count = max(0, len(recent_candidates) - len(recent))
        if dropped_recent_count:
            older = [*older, *recent_candidates[:dropped_recent_count]]
        try:
            summary = compress_history_with_llm(
                older,
                llm_client=llm_client,
                llm_options=llm_options,
                max_summary_chars=summary_budget,
                chunk_chars=self.history_compression_chunk_chars,
            )
        except Exception as exc:
            errors.append(f"Conversation history compression failed: {exc}")
            return fallback, metadata
        if not summary:
            errors.append("Conversation history compression failed: empty summary")
            return fallback, metadata

        summary_message = {
            "role": "assistant",
            "content": HISTORY_SUMMARY_PREFIX + summary,
        }
        history = [summary_message, *recent]
        if self.history_max_chars > 0:
            history = fit_summary_and_recent_to_budget(history, self.history_max_chars)
        metadata.update(
            {
                "history_strategy": "llm_compression",
                "history_summarized_messages": len(older),
                "history_summary_chars": len(summary),
                "history_compressed": True,
            }
        )
        return history, metadata

    def _answer_question_workflow(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Workflow 模式下的非流式问答流水线。

        按顺序执行以下阶段，收集各阶段耗时并输出 diagnostics：
        history → contextualize → plan → cypher → graph → rag → research → evidence → answer → verify

        Args:
            question: 用户问题
            conversation_history: 对话历史
            thinking_enabled: 是否显示思考过程（workflow 模式下暂不输出进度事件）
            reasoning_effort: 推理努力程度

        Returns:
            完整的问答结果字典
        """
        total_start = time.perf_counter()
        timings_ms: dict[str, float] = {}
        question = question.strip()
        errors: list[str] = []
        llm_client = CountingLLMClient(self.llm_client) if self.llm_client is not None else None
        llm_options = build_llm_options(thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort)

        # 阶段 1: 对话历史归一化
        stage_start = time.perf_counter()
        history, history_memory = self._prepare_conversation_history(
            conversation_history,
            errors=errors,
            llm_options=llm_options,
            llm_client=llm_client,
        )
        record_timing(timings_ms, "history", stage_start)

        # 阶段 2: 追问上下文补全
        stage_start = time.perf_counter()
        contextual_question = self._contextualize_question(question, history, errors, llm_options, llm_client)
        record_timing(timings_ms, "contextualize", stage_start)

        # 阶段 3: 问题规划
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

        # 阶段 4: 生成 Cypher 查询
        stage_start = time.perf_counter()
        generated = self._generate_display_cypher(contextual_question, plan, errors, llm_options, llm_client)
        if generated.error:
            errors.append(generated.error)
        record_timing(timings_ms, "cypher", stage_start)

        # 阶段 5: 执行图查询
        stage_start = time.perf_counter()
        graph_records = self._query_graph(generated, plan, errors)
        record_timing(timings_ms, "graph", stage_start)

        # 阶段 6: RAG 检索
        stage_start = time.perf_counter()
        rag_hits = self._search_rag(contextual_question, plan, errors)
        record_timing(timings_ms, "rag", stage_start)

        # 阶段 7: 研究报告检索
        stage_start = time.perf_counter()
        research_hits = self._search_research(contextual_question, plan, errors)
        record_timing(timings_ms, "research", stage_start)

        # 阶段 8: 证据合并与排序
        stage_start = time.perf_counter()
        raw_cards = [
            *cards_from_research_hits(research_hits, plan),
            *cards_from_graph_records(graph_records, plan),
            *cards_from_rag_hits(rag_hits, plan),
        ]
        evidence_cards = rank_evidence_cards(raw_cards, limit=self.evidence_top_n, plan=plan)
        if plan.answer_type == "risk_analysis":
            evidence_cards = ensure_relation_cards(evidence_cards, raw_cards, "DISCLOSES_RISK", limit=self.evidence_top_n)
        evidence_cards = assign_citation_ids(evidence_cards)
        record_timing(timings_ms, "evidence", stage_start)

        # 阶段 9: LLM 生成答案
        stage_start = time.perf_counter()
        answer, reasoning_content = self._generate_answer(
            question,
            contextual_question,
            history,
            plan,
            graph_records,
            evidence_cards,
            errors,
            llm_options,
            llm_client,
        )
        record_timing(timings_ms, "answer", stage_start)

        # 阶段 10: 答案事实一致性验证
        verification = verify_answer_support(answer, plan, evidence_cards, raw_cards, question=contextual_question)
        if verification.get("status") == "fail":
            answer = build_evidence_limited_answer(plan, evidence_cards, verification)
            reasoning_content = ""
            verification = verify_answer_support(answer, plan, evidence_cards, raw_cards, question=contextual_question)

        # 渲染最终输出
        stage_start = time.perf_counter()
        evidence = legacy_evidence_rows(evidence_cards)
        rag_hit_rows = [hit.to_dict() for hit in rag_hits]
        research_hit_rows = [hit.to_dict() for hit in research_hits]
        evidence_card_rows = [card.to_dict() for card in evidence_cards]
        subgraph = answer_subgraph(graph_records, evidence_cards)
        unsupported_terms = list(verification.get("checks", {}).get("unsupported_terms") or [])
        research_outputs = build_research_outputs(
            question=contextual_question,
            plan=plan,
            evidence_cards=evidence_cards,
            graph_records=graph_records,
            verification=verification,
        )
        record_timing(timings_ms, "render_payload", stage_start)

        diagnostics = {
            "graph_backend": self.status.graph_backend,
            "graph_records": len(graph_records),
            "rag_hits": len(rag_hits),
            "research_hits": len(research_hits),
            "embedding_hits": 0,
            "embedding_enabled": bool(getattr(self.status, "embedding_enabled", False)),
            "evidence_cards": len(evidence_cards),
            "rerank_top_n": self.rerank_top_n,
            "history_messages": len(history),
            **history_memory,
            "contextualized": contextual_question != question,
            "contextualizer_mode": self.contextualizer_mode,
            "planner_source": planner_source,
            "enable_llm_cypher": self.enable_llm_cypher,
            "enable_llm_planner": self.enable_llm_planner,
            "thinking_enabled": thinking_enabled,
            "reasoning_effort": reasoning_effort or "",
            "graph_error": self.status.graph_error,
            "rag_error": self.status.rag_error,
            "research_error": self.status.research_error,
            "embedding_error": getattr(self.status, "embedding_error", ""),
            "llm_error": self.status.llm_error,
            "unsupported_terms": unsupported_terms,
            "agent_enabled": False,
            "agent_runner": "workflow",
            "langgraph_enabled": False,
            "agent_max_steps": self.agent_max_steps,
            "agent_steps": 0,
            "agent_trace": [],
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
            "graph_records": graph_records,
            "rag_hits": rag_hit_rows,
            "research_hits": research_hit_rows,
            "evidence_cards": evidence_card_rows,
            "evidence": evidence,
            "research_outputs": research_outputs,
            "verification": verification,
            "subgraph": subgraph,
            "diagnostics": diagnostics,
            "errors": errors,
        }

    def _answer_question_stream_workflow(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Workflow 模式下的流式问答流水线。

        与 _answer_question_workflow 逻辑相同，但：
        - 在关键阶段 yield progress 事件（thinking_enabled 时）
        - answer 阶段逐 token 流式输出
        - 最终 yield final 事件携带完整结果

        Args:
            question: 用户问题
            conversation_history: 对话历史
            thinking_enabled: 是否显示思考过程
            reasoning_effort: 推理努力程度

        Yields:
            事件字典（progress / answer_delta / answer_complete / final）
        """
        total_start = time.perf_counter()
        timings_ms: dict[str, float] = {}
        question = question.strip()
        errors: list[str] = []
        llm_client = CountingLLMClient(self.llm_client) if self.llm_client is not None else None
        llm_options = build_llm_options(thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort)

        if thinking_enabled:
            yield stream_progress("history", "正在结合历史对话理解当前问题")
        stage_start = time.perf_counter()
        history, history_memory = self._prepare_conversation_history(
            conversation_history,
            errors=errors,
            llm_options=llm_options,
            llm_client=llm_client,
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
            yield stream_progress("research", "正在召回投研 Claim 与产业链摘要")
        stage_start = time.perf_counter()
        research_hits = self._search_research(contextual_question, plan, errors)
        record_timing(timings_ms, "research", stage_start)

        if thinking_enabled:
            yield stream_progress("evidence", "正在筛选可支撑答案的证据")
        stage_start = time.perf_counter()
        raw_cards = [
            *cards_from_research_hits(research_hits, plan),
            *cards_from_graph_records(graph_records, plan),
            *cards_from_rag_hits(rag_hits, plan),
        ]
        evidence_cards = rank_evidence_cards(raw_cards, limit=self.evidence_top_n, plan=plan)
        if plan.answer_type == "risk_analysis":
            evidence_cards = ensure_relation_cards(evidence_cards, raw_cards, "DISCLOSES_RISK", limit=self.evidence_top_n)
        evidence_cards = assign_citation_ids(evidence_cards)
        record_timing(timings_ms, "evidence", stage_start)
        if thinking_enabled:
            yield stream_progress("evidence", f"已保留 {len(evidence_cards)} 条高相关证据，开始组织答案")

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

        # 验证 + 重试（如果验证失败则使用 evidence-limited 答案）
        verification = verify_answer_support(answer, plan, evidence_cards, raw_cards, question=contextual_question)
        if verification.get("status") == "fail":
            answer = build_evidence_limited_answer(plan, evidence_cards, verification)
            reasoning_content = ""
            verification = verify_answer_support(answer, plan, evidence_cards, raw_cards, question=contextual_question)

        stage_start = time.perf_counter()
        evidence = legacy_evidence_rows(evidence_cards)
        rag_hit_rows = [hit.to_dict() for hit in rag_hits]
        research_hit_rows = [hit.to_dict() for hit in research_hits]
        evidence_card_rows = [card.to_dict() for card in evidence_cards]
        subgraph = answer_subgraph(graph_records, evidence_cards)
        unsupported_terms = list(verification.get("checks", {}).get("unsupported_terms") or [])
        research_outputs = build_research_outputs(
            question=contextual_question,
            plan=plan,
            evidence_cards=evidence_cards,
            graph_records=graph_records,
            verification=verification,
        )
        record_timing(timings_ms, "render_payload", stage_start)

        diagnostics = {
            "graph_backend": self.status.graph_backend,
            "graph_records": len(graph_records),
            "rag_hits": len(rag_hits),
            "research_hits": len(research_hits),
            "embedding_hits": 0,
            "embedding_enabled": bool(getattr(self.status, "embedding_enabled", False)),
            "evidence_cards": len(evidence_cards),
            "rerank_top_n": self.rerank_top_n,
            "history_messages": len(history),
            **history_memory,
            "contextualized": contextual_question != question,
            "contextualizer_mode": self.contextualizer_mode,
            "planner_source": planner_source,
            "enable_llm_cypher": self.enable_llm_cypher,
            "enable_llm_planner": self.enable_llm_planner,
            "thinking_enabled": thinking_enabled,
            "reasoning_effort": reasoning_effort or "",
            "graph_error": self.status.graph_error,
            "rag_error": self.status.rag_error,
            "research_error": self.status.research_error,
            "embedding_error": getattr(self.status, "embedding_error", ""),
            "llm_error": self.status.llm_error,
            "unsupported_terms": unsupported_terms,
            "agent_enabled": False,
            "agent_runner": "workflow",
            "langgraph_enabled": False,
            "agent_max_steps": self.agent_max_steps,
            "agent_steps": 0,
            "agent_trace": [],
            "agent_verification": verification,
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
                "research_hits": research_hit_rows,
                "evidence_cards": evidence_card_rows,
                "evidence": evidence,
                "research_outputs": research_outputs,
                "verification": verification,
                "subgraph": subgraph,
                "diagnostics": diagnostics,
                "errors": errors,
            },
        }

    # ============================================================
    # 各阶段私有方法
    # ============================================================

    def _should_use_llm_planner(self, plan: QuestionPlan, llm_client: Any | None) -> bool:
        """判断是否应该使用 LLM 进行问题规划优化。

        当满足以下条件时使用 LLM planner：
        1. enable_llm_planner 为 True
        2. LLM 客户端可用且支持 chat_json
        3. 问题的 answer_type 为 thematic_research 且未识别出公司和主题

        Args:
            plan: 启发式规则生成的 QuestionPlan
            llm_client: LLM 客户端

        Returns:
            是否启用 LLM 规划
        """
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
        """根据问题规划生成 Cypher 查询语句（三级降级策略）。

        1. CSV 图谱模式 → 使用 pseudo_cypher_for_plan（基于计划直接构建伪查询）
        2. 模板模式（LLM Cypher 未启用或无 LLM）→ template_cypher_for_plan
        3. LLM 模式 → generate_cypher 让 LLM 动态生成，失败时回退到模板

        Args:
            question: 用户问题
            plan: 查询计划
            errors: 错误收集列表
            llm_options: LLM 调用参数
            llm_client: LLM 客户端

        Returns:
            GeneratedCypher（包含 cypher 语句、参数、来源标记）
        """
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
        """执行图查询，获取结构化关系数据。

        支持 CSV 和 Neo4j 两种后端。Neo4j 查询失败时自动降级到 CSV 图谱。

        Args:
            generated: 生成的 Cypher 查询
            plan: 查询计划
            errors: 错误收集列表

        Returns:
            图查询结果记录列表
        """
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
        """在 RAG 索引中检索非结构化文本证据。

        当问题只涉及一家公司且为风险分析或公司画像时，按公司名过滤检索。
        检索查询 = 问题 + 扩展主题词。

        Args:
            question: 用户问题
            plan: 查询计划
            errors: 错误收集列表

        Returns:
            RagHit 列表
        """
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

    def _search_research(self, question: str, plan: QuestionPlan, errors: list[str]) -> list[ResearchHit]:
        """在研究报告索引中检索结构化投研声明和摘要。

        Args:
            question: 用户问题
            plan: 查询计划
            errors: 错误收集列表

        Returns:
            ResearchHit 列表
        """
        if self.research_memory is None:
            return []
        try:
            return self.research_memory.search(
                question,
                plan,
                limit=max(self.evidence_top_n, min(self.rerank_top_n, 20)),
            )
        except Exception as exc:
            errors.append(f"Research search failed: {exc}")
            return []

    def _contextualize_question(
        self,
        question: str,
        history: list[dict[str, str]],
        errors: list[str],
        llm_options: dict[str, Any],
        llm_client: Any | None,
    ) -> str:
        """基于对话历史对用户问题进行上下文补全。

        策略（按 contextualizer_mode 决定）：
        - heuristic: 仅使用启发式规则（提取上一轮的公司名拼接到当前问题）
        - auto: 如果问题需要上下文（question_needs_context）则用 LLM，否则启发式
        - llm: 始终用 LLM 改写

        Args:
            question: 用户当前问题
            history: 归一化后的对话历史
            errors: 错误收集列表
            llm_options: LLM 调用参数
            llm_client: LLM 客户端

        Returns:
            补全后的上下文无关问题文本
        """
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
        errors: list[str],
        llm_options: dict[str, Any],
        llm_client: Any | None,
    ) -> tuple[str, str]:
        """基于多源证据生成最终答案（非流式）。

        无证据时返回 NO_EVIDENCE_ANSWER。
        有 LLM 时使用专业答案 prompt 调用 LLM 生成。
        无 LLM 时使用 fallback_professional_answer 做模板化回答。

        Args:
            question: 用户原始问题
            contextual_question: 上下文补全后的问题
            history: 对话历史
            plan: 查询计划
            graph_records: 图查询结果
            evidence_cards: 排序筛选后的证据卡片
            errors: 错误收集列表
            llm_options: LLM 参数
            llm_client: LLM 客户端

        Returns:
            (answer_text, reasoning_content) 元组
        """
        if not evidence_cards:
            return NO_EVIDENCE_ANSWER, ""
        prompt_question = question
        if contextual_question != question:
            prompt_question = f"用户当前追问：{question}\n结合历史对话改写后的检索问题：{contextual_question}"
        user_prompt = build_professional_answer_prompt(prompt_question, plan, graph_records, evidence_cards)
        if llm_client is not None and hasattr(llm_client, "chat_text"):
            try:
                if history and hasattr(llm_client, "chat_messages"):
                    response = llm_client.chat_messages(
                        messages=[
                            {"role": "system", "content": answer_system_prompt()},
                            *history,
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                        **llm_options,
                    )
                    return response.content, response.reasoning_content
                if hasattr(llm_client, "chat_text_with_metadata"):
                    response = llm_client.chat_text_with_metadata(
                        system_prompt=answer_system_prompt(),
                        user_prompt=user_prompt,
                        temperature=0.2,
                        **llm_options,
                    )
                    return response.content, response.reasoning_content
                return llm_client.chat_text(
                    system_prompt=answer_system_prompt(),
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
        errors: list[str],
        llm_options: dict[str, Any],
        llm_client: Any | None,
        *,
        thinking_enabled: bool,
    ) -> Iterator[dict[str, Any]]:
        """流式生成答案。

        优先使用 LLM 的 stream_chat_messages 实现逐 token 输出。
        如果流式调用失败，回退到 _generate_answer 一次性生成后 chunk 输出。

        Args:
            question: 用户原始问题
            contextual_question: 上下文补全后的问题
            history: 对话历史
            plan: 查询计划
            graph_records: 图查询结果
            evidence_cards: 证据卡片列表
            errors: 错误收集列表
            llm_options: LLM 参数
            llm_client: LLM 客户端
            thinking_enabled: 是否显示思考过程

        Yields:
            answer_delta 事件（逐 chunk）或 answer_complete 事件
        """
        if not evidence_cards:
            for chunk in chunk_text(NO_EVIDENCE_ANSWER):
                yield {"type": "answer_delta", "content": chunk}
            yield {"type": "answer_complete", "answer": NO_EVIDENCE_ANSWER, "reasoning_content": ""}
            return

        prompt_question = question
        if contextual_question != question:
            prompt_question = f"用户当前追问：{question}\n结合历史对话改写后的检索问题：{contextual_question}"
        user_prompt = build_professional_answer_prompt(prompt_question, plan, graph_records, evidence_cards)

        if thinking_enabled:
            yield {"type": "progress", "stage": "answer", "message": "正在生成结论、证据和风险边界"}

        if llm_client is not None and hasattr(llm_client, "stream_chat_messages"):
            chunks: list[str] = []
            try:
                messages = [
                    {"role": "system", "content": answer_system_prompt()},
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

        # 流式失败时的降级：用非流式生成后 chunk 输出
        answer, reasoning_content = self._generate_answer(
            question,
            contextual_question,
            history,
            plan,
            graph_records,
            evidence_cards,
            errors,
            llm_options,
            llm_client,
        )
        for chunk in chunk_text(answer):
            yield {"type": "answer_delta", "content": chunk}
        yield {"type": "answer_complete", "answer": answer, "reasoning_content": reasoning_content}


# ============================================================
# 模块级辅助函数
# ============================================================


def build_answer_prompt(question: str, graph_records: list[dict[str, Any]], rag_hits: list[RagHit]) -> str:
    """构建答案生成的 prompt（旧版格式，用于兼容）。

    Args:
        question: 用户问题
        graph_records: 图谱查询结果
        rag_hits: RAG 检索结果

    Returns:
        JSON 格式的 prompt 字符串
    """
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
    """归一化上下文补全模式配置。

    Args:
        value: 原始模式值

    Returns:
        合法模式值（"auto" / "heuristic" / "llm"），非法值返回 "auto"
    """
    mode = str(value or "auto").strip().casefold()
    if mode not in {"auto", "heuristic", "llm"}:
        return "auto"
    return mode


def normalize_agent_max_steps(value: Any) -> int:
    """归一化 Agent 最大步数（限制在 1-4 范围内）。

    Args:
        value: 原始步数值

    Returns:
        归一化后的步数（1-4）
    """
    try:
        steps = int(value)
    except (TypeError, ValueError):
        steps = 4
    return max(1, min(steps, 4))


def normalize_agent_runner(value: Any) -> str:
    """归一化 Agent 运行器配置。

    Args:
        value: 原始运行器值

    Returns:
        合法运行器值（"langgraph" / "legacy"），非法值返回 "langgraph"
    """
    runner = str(value or "langgraph").strip().casefold()
    if runner not in {"langgraph", "legacy"}:
        return "langgraph"
    return runner


def record_timing(timings_ms: dict[str, float], name: str, started_at: float) -> None:
    """记录阶段耗时到 timings 字典。

    Args:
        timings_ms: 耗时字典（可变）
        name: 阶段名称
        started_at: 阶段开始时间（time.perf_counter）
    """
    timings_ms[name] = round((time.perf_counter() - started_at) * 1000, 2)


def template_cypher_for_plan(plan: QuestionPlan, *, limit: int = 50, error: str = "") -> GeneratedCypher:
    """根据查询计划生成模板化 Cypher 查询。

    自动处理关系类型选择、公司名归一化、主题匹配等逻辑。
    对于 industry_bottleneck 类型问题，自动补充 CONSTRAINS 等瓶颈相关关系。

    Args:
        plan: 查询计划
        limit: 返回记录数上限
        error: 错误信息（可选，传递上游错误）

    Returns:
        GeneratedCypher 对象
    """
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
    """构建 LLM 调用参数选项。

    Args:
        thinking_enabled: 是否启用推理展示
        reasoning_effort: 推理努力程度

    Returns:
        LLM 参数字典（只包含非 None 的配置项）
    """
    options: dict[str, Any] = {}
    if thinking_enabled is not None:
        options["thinking_enabled"] = thinking_enabled
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort
    return options


def stream_progress(stage: str, message: str) -> dict[str, str]:
    """构造流式进度事件。

    Args:
        stage: 阶段名称
        message: 进度消息

    Returns:
        进度事件字典
    """
    return {"type": "progress", "stage": stage, "message": message}


def describe_plan_progress(plan: QuestionPlan) -> str:
    """生成问题规划进度的中文描述文本。

    Args:
        plan: 查询计划

    Returns:
        中文描述字符串，如 "问题规划完成：主题研究，关注主题为AI芯片、光模块"
    """
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
    """将文本按指定大小分块。

    用于流式输出时将完整答案切分为多个 answer_delta 事件。

    Args:
        text: 待分块的文本
        size: 每块字符数（默认 18）

    Yields:
        文本块
    """
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
    """归一化对话历史，控制大小和内容。

    裁剪规则：
    1. 只保留 role 为 "user" 或 "assistant" 的消息
    2. 每条消息内容截断到 3000 字符
    3. 按 max_turns 限制轮数（最近 N 轮）
    4. 按 max_chars 限制总字符数（从最近开始累加）

    Args:
        messages: 原始对话历史
        max_turns: 最大保留轮数（≤0 时不保留历史）
        max_chars: 最大保留字符数（≤0 时不限制）

    Returns:
        归一化后的对话历史
    """
    normalized = normalize_conversation_messages(messages)

    if max_turns <= 0:
        return []
    normalized = normalized[-max_turns * 2 :]
    return trim_history_to_chars(normalized, max_chars)


def normalize_conversation_messages(
    messages: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """Normalize supported conversation messages without discarding old turns."""
    normalized: list[dict[str, str]] = []
    for message in messages or []:
        role = str(message.get("role") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content[:3000]})
    return normalized


def trim_history_to_chars(
    messages: list[dict[str, str]],
    max_chars: int,
) -> list[dict[str, str]]:
    """Keep the newest messages within a character budget."""
    if max_chars <= 0:
        return list(messages)

    selected: list[dict[str, str]] = []
    used = 0
    for message in reversed(messages):
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


def fit_summary_and_recent_to_budget(
    messages: list[dict[str, str]],
    max_chars: int,
) -> list[dict[str, str]]:
    """Keep a synthetic summary and as much recent verbatim history as possible."""
    if max_chars <= 0 or not messages:
        return messages
    summary = dict(messages[0])
    recent_budget = max(0, max_chars - len(summary["content"]))
    recent = [] if recent_budget <= 0 else trim_history_to_chars(messages[1:], recent_budget)
    remaining = max_chars - sum(len(message["content"]) for message in recent)
    if remaining <= 0:
        return recent
    summary["content"] = summary["content"][:remaining]
    return [summary, *recent]


def compress_history_with_llm(
    messages: list[dict[str, str]],
    *,
    llm_client: Any,
    llm_options: dict[str, Any],
    max_summary_chars: int,
    chunk_chars: int,
) -> str:
    """Compress older messages, using hierarchical summaries for long histories."""
    chunks = history_chunks(messages, max_chars=chunk_chars)
    summaries = [
        request_history_summary(
            chunk,
            llm_client=llm_client,
            llm_options=llm_options,
            max_summary_chars=max_summary_chars,
        )
        for chunk in chunks
    ]
    summaries = [summary for summary in summaries if summary]
    while len(summaries) > 1:
        summary_messages = [
            {"role": "assistant", "content": f"分段摘要 {index}：{summary}"}
            for index, summary in enumerate(summaries, start=1)
        ]
        merge_chunk_chars = max(chunk_chars, max_summary_chars * 2 + 100)
        chunks = history_chunks(summary_messages, max_chars=merge_chunk_chars)
        summaries = [
            request_history_summary(
                chunk,
                llm_client=llm_client,
                llm_options=llm_options,
                max_summary_chars=max_summary_chars,
            )
            for chunk in chunks
        ]
        summaries = [summary for summary in summaries if summary]
    return sanitize_history_summary(summaries[0] if summaries else "", max_chars=max_summary_chars)


def history_chunks(
    messages: list[dict[str, str]],
    *,
    max_chars: int,
) -> list[list[dict[str, str]]]:
    """Split history on message boundaries before LLM compression."""
    chunks: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    used = 0
    for message in messages:
        size = len(message["content"])
        if current and used + size > max_chars:
            chunks.append(current)
            current = []
            used = 0
        current.append(message)
        used += size
    if current:
        chunks.append(current)
    return chunks


def request_history_summary(
    messages: list[dict[str, str]],
    *,
    llm_client: Any,
    llm_options: dict[str, Any],
    max_summary_chars: int,
) -> str:
    prompt = (
        format_history_for_prompt(messages)
        + f"\n\n请压缩为不超过 {max_summary_chars} 个中文字符的连续摘要。"
    )
    options = dict(llm_options)
    options["thinking_enabled"] = False
    options.pop("reasoning_effort", None)
    if hasattr(llm_client, "chat_messages"):
        response = llm_client.chat_messages(
            messages=[
                {"role": "system", "content": HISTORY_COMPRESSION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            **options,
        )
        return sanitize_history_summary(response.content, max_chars=max_summary_chars)
    if hasattr(llm_client, "chat_text"):
        content = llm_client.chat_text(
            system_prompt=HISTORY_COMPRESSION_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.0,
            **options,
        )
        return sanitize_history_summary(content, max_chars=max_summary_chars)
    raise TypeError("LLM client does not support conversation history compression")


def sanitize_history_summary(content: str, *, max_chars: int) -> str:
    text = str(content or "").replace("```", "").strip()
    for prefix in ("摘要：", "对话摘要：", "历史摘要："):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return text[:max_chars]


def build_contextualizer_prompt(question: str) -> str:
    """构建追问改写器的 user prompt。

    Args:
        question: 用户当前问题

    Returns:
        prompt 字符串
    """
    return f"""请把下面当前问题改写成一个不依赖上下文也能检索的单轮问题。
如果当前问题已经完整，原样输出。

当前问题：{question}
"""


def heuristic_contextual_question(question: str, history: list[dict[str, str]]) -> str:
    """基于启发式规则的追问上下文补全。

    将上一轮用户问题中的公司名提取出来，拼接到当前问题前。
    如上一轮提到"寒武纪"，当前问"它的风险呢" → "寒武纪 它的风险呢"

    Args:
        question: 用户当前问题
        history: 对话历史

    Returns:
        补全后的问题
    """
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
    """判断当前问题是否需要上下文补全。

    需要上下文的情况：
    - 包含代词（它、其、他们、这些等）
    - 包含承接词（继续、进一步、再展开等）
    - "主要风险"不带公司名
    - 问题很短（≤18 字符）且无独立关键词

    Args:
        question: 用户当前问题

    Returns:
        是否需要上下文补全
    """
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
    """清洗 LLM 输出的上下文补全结果。

    移除 Markdown 代码块标记、前缀（"改写后的问题："等）、引号。

    Args:
        content: LLM 输出的原始文本

    Returns:
        清洗后的问题文本（最长 500 字符）
    """
    text = str(content or "").strip()
    if not text:
        return ""
    text = text.replace("```", "").strip()
    if text.startswith(("改写后的问题：", "问题：")):
        text = text.split("：", 1)[-1].strip()
    text = text.strip("\"'“”‘’ \n")
    return text[:500]


def format_history_for_prompt(history: list[dict[str, str]]) -> str:
    """将对话历史格式化为 prompt 可读文本。

    用于旧版 LLM 接口（chat_text），将历史对话拼接到 prompt 中。

    Args:
        history: 对话历史

    Returns:
        格式化后的历史文本
    """
    lines = ["历史对话："]
    for message in history:
        role = "用户" if message["role"] == "user" else "助手"
        lines.append(f"{role}：{message['content']}")
    return "\n".join(lines)


def build_evidence(graph_records: list[dict[str, Any]], rag_hits: list[RagHit]) -> list[dict[str, Any]]:
    """合并图谱和 RAG 证据为统一格式（旧版兼容）。

    Args:
        graph_records: 图谱查询结果
        rag_hits: RAG 检索结果

    Returns:
        合并后的证据列表
    """
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
    """将图谱查询结果转换为前端子图边列表。

    Args:
        records: 图谱查询记录

    Returns:
        子图边列表（head_type, head_name, relation, tail_type, tail_name）
    """
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


def answer_subgraph(records: list[dict[str, Any]], evidence_cards: list[Any]) -> list[dict[str, str]]:
    """从图谱记录和证据卡片构建答案的可视化子图。

    支持多种卡片类型：
    - graph: 图谱关系边
    - dossier: 公司档案（解析敞口、技术机理、瓶颈等行）
    - claim / rag: 研究声明/文本摘要（映射到公司和主题的关系）

    Args:
        records: 图谱查询记录
        evidence_cards: 证据卡片列表

    Returns:
        子图边列表（去重，最多 100 条）
    """
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add_edge(
        source: str,
        target: str,
        label: str,
        *,
        source_type: str = "",
        target_type: str = "",
        source_kind: str = "",
        citation_id: str = "",
        claim_type: str = "",
        exposure_level: str = "",
    ) -> None:
        source = str(source or "").strip()
        target = str(target or "").strip()
        label = str(label or "").strip()
        if not source or not target or not label:
            return
        key = (source, target, label, citation_id)
        if key in seen:
            return
        seen.add(key)
        edges.append(
            {
                "source": source,
                "target": target,
                "label": label,
                "source_type": source_type,
                "target_type": target_type,
                "source_kind": source_kind,
                "citation_id": citation_id,
                "claim_type": claim_type,
                "exposure_level": exposure_level,
            }
        )

    # 图谱关系边
    for record in records:
        company = str(record.get("company") or "")
        target = str(record.get("target") or "")
        relation = str(record.get("relation") or "")
        target_labels = record.get("target_labels") or []
        target_type = target_labels[0] if isinstance(target_labels, list) and target_labels else ""
        add_edge(
            company,
            target,
            RELATION_LABELS.get(relation, relation),
            source_type="Company",
            target_type=target_type,
            source_kind="graph",
        )

    # 证据卡片边
    for card in evidence_cards:
        kind = str(getattr(card, "kind", "") or "")
        citation_id = str(getattr(card, "citation_id", "") or "")
        claim_type = str(getattr(card, "claim_type", "") or "")
        exposure_level = str(getattr(card, "exposure_level", "") or "")
        topic = str(getattr(card, "topic", "") or "")
        company = str(getattr(card, "company", "") or "")
        relation = str(getattr(card, "relation", "") or "")
        target = str(getattr(card, "target", "") or "")

        if kind == "dossier":
            add_dossier_edges(card, add_edge)
            continue
        if kind == "graph" and company and target:
            add_edge(
                company,
                target,
                RELATION_LABELS.get(relation, relation),
                source_type="Company",
                target_type=target_type_for_claim(claim_type, relation),
                source_kind=kind,
                citation_id=citation_id,
                claim_type=claim_type,
                exposure_level=exposure_level,
            )
            continue
        if company:
            claim_target = target_for_claim(card)
            add_edge(
                company,
                claim_target,
                label_for_claim(card),
                source_type="Company",
                target_type=target_type_for_claim(claim_type, relation),
                source_kind=kind,
                citation_id=citation_id,
                claim_type=claim_type,
                exposure_level=exposure_level,
            )
            continue
        if topic:
            add_edge(
                topic,
                target_for_claim(card),
                label_for_claim(card),
                source_type="ValueChainSegment",
                target_type=target_type_for_claim(claim_type, relation),
                source_kind=kind,
                citation_id=citation_id,
                claim_type=claim_type,
                exposure_level=exposure_level,
            )
    return edges[:100]


def add_dossier_edges(card: Any, add_edge: Any) -> None:
    """解析公司档案（dossier）卡片中的结构化行，生成子图边。

    解析的证据行类型：
    - "公司敞口：" → 按层级（core/direct/indirect/mentioned）生成敞口边
    - "技术机理：" → 生成"支撑"边
    - "瓶颈：" → 生成"约束"边
    - "领先指标：" → 生成"指标"边
    - "风险与反证：" → 生成"风险"边

    Args:
        card: dossier 类型的证据卡片
        add_edge: 边添加回调函数
    """
    citation_id = str(getattr(card, "citation_id", "") or "")
    topic = str(getattr(card, "topic", "") or getattr(card, "title", "") or "")
    evidence = str(getattr(card, "evidence", "") or "")
    exposure_labels = {"core": "核心敞口", "direct": "直接敞口", "indirect": "间接敞口", "mentioned": "仅提及"}
    for line in evidence.splitlines():
        line = line.strip()
        if line.startswith("公司敞口："):
            for part in line.split("：", 1)[-1].split("；"):
                if ":" not in part:
                    continue
                level, names = part.split(":", 1)
                label = exposure_labels.get(level, "敞口")
                for name in names.split("、")[:10]:
                    add_edge(
                        name,
                        topic,
                        label,
                        source_type="Company",
                        target_type="ValueChainSegment",
                        source_kind="dossier",
                        citation_id=citation_id,
                        exposure_level=level,
                    )
        elif line.startswith("技术机理："):
            add_edge(topic, "技术机理", "支撑", source_type="ValueChainSegment", target_type="IndustryConcept", source_kind="dossier", citation_id=citation_id, claim_type="mechanism")
        elif line.startswith("瓶颈："):
            add_edge(topic, "瓶颈", "约束", source_type="ValueChainSegment", target_type="Risk", source_kind="dossier", citation_id=citation_id, claim_type="bottleneck")
        elif line.startswith("领先指标："):
            add_edge(topic, "领先指标", "指标", source_type="ValueChainSegment", target_type="Metric", source_kind="dossier", citation_id=citation_id, claim_type="indicator")
        elif line.startswith("风险与反证："):
            add_edge(topic, "风险与反证", "风险", source_type="ValueChainSegment", target_type="Risk", source_kind="dossier", citation_id=citation_id, claim_type="risk")


def label_for_claim(card: Any) -> str:
    """根据证据卡片的 claim_type 和 exposure_level 确定子图边标签。

    Args:
        card: 证据卡片

    Returns:
        中文字图边标签
    """
    claim_type = str(getattr(card, "claim_type", "") or "")
    relation = str(getattr(card, "relation", "") or "")
    exposure_level = str(getattr(card, "exposure_level", "") or "")
    if claim_type == "company_exposure":
        return {"core": "核心敞口", "direct": "直接敞口", "indirect": "间接敞口", "mentioned": "仅提及"}.get(exposure_level, "敞口")
    if claim_type == "risk":
        return "风险"
    if claim_type == "indicator":
        return "指标"
    if claim_type == "bottleneck":
        return "约束"
    if claim_type in {"mechanism", "supply_chain", "trend"}:
        return "支撑"
    if relation:
        return RELATION_LABELS.get(relation, relation)
    return "证据支持"


def target_for_claim(card: Any) -> str:
    """根据证据卡片的 claim_type 生成子图中的目标节点名称。

    Args:
        card: 证据卡片

    Returns:
        目标节点名称
    """
    claim_type = str(getattr(card, "claim_type", "") or "")
    target = str(getattr(card, "target", "") or "")
    topic = str(getattr(card, "topic", "") or "")
    if target:
        return target
    if claim_type == "risk":
        return f"{topic}风险" if topic else "风险"
    if claim_type == "indicator":
        return f"{topic}指标" if topic else "指标"
    if claim_type == "bottleneck":
        return f"{topic}瓶颈" if topic else "瓶颈"
    return topic or str(getattr(card, "title", "") or "证据")


def target_type_for_claim(claim_type: str, relation: str = "") -> str:
    """根据 claim_type 和 relation 确定子图中目标节点的类型。

    Args:
        claim_type: 声明类型
        relation: 关系类型

    Returns:
        节点类型标签（Risk / Metric / ValueChainSegment / IndustryConcept）
    """
    if relation == "DISCLOSES_RISK" or claim_type in {"risk", "bottleneck"}:
        return "Risk"
    if relation == "HAS_METRIC" or claim_type == "indicator":
        return "Metric"
    if claim_type == "company_exposure":
        return "ValueChainSegment"
    return "IndustryConcept"


def detect_unsupported_terms(answer: str, evidence_cards: list[Any]) -> list[str]:
    """检测答案中无证据支撑的术语。

    检查：
    - 答案中提到的公司名是否在证据中出现
    - 数字、百分比、金额等是否在证据中出现
    - 投研关键词（订单、毛利率、产能等）是否在证据中出现

    Args:
        answer: 待检查的答案文本
        evidence_cards: 证据卡片列表

    Returns:
        无证据支撑的术语列表（最多 20 个）
    """
    evidence_text = normalize_for_support(
        " ".join(
            " ".join(
                str(getattr(card, key, "") or "")
                for key in ("title", "evidence", "source", "section", "company", "target", "topic")
            )
            for card in evidence_cards
        )
    )
    unsupported: list[str] = []
    for company in extract_companies(answer):
        if normalize_for_support(company) not in evidence_text:
            unsupported.append(company)
    for term in re.findall(r"20\d{2}|[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|亿元|万元|元|万只|台|GB/s|Tb/s|GT/s|kW|MW|PUE)", answer, flags=re.I):
        normalized = normalize_for_support(term)
        if normalized and normalized not in evidence_text:
            unsupported.append(term.strip())
    for term in ("订单", "合同负债", "产能", "毛利率", "ASP", "客户结构", "资本开支", "PUE", "功率密度", "端口速率", "渗透率"):
        if term in answer and normalize_for_support(term) not in evidence_text:
            unsupported.append(term)
    return unique_texts(unsupported)[:20]


def normalize_for_support(value: str) -> str:
    """用于证据支撑检查的文本归一化：去除所有空白字符并转小写。

    Args:
        value: 原始文本

    Returns:
        归一化后的文本
    """
    return re.sub(r"\s+", "", str(value or "").casefold())


def unique_texts(values: list[str]) -> list[str]:
    """字符串列表去重（保留首次出现顺序）。

    Args:
        values: 原始字符串列表

    Returns:
        去重后的列表
    """
    output = []
    seen = set()
    for value in values:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def ensure_relation_cards(cards: list[Any], raw_cards: list[Any], relation: str, *, limit: int) -> list[Any]:
    """确保排序后的证据卡片中包含指定关系类型的卡片。

    如果排序后的 cards 中没有指定 relation 的卡片，
    从 raw_cards 中提取第一条该关系的卡片插入到最前面。

    Args:
        cards: 排序后的证据卡片
        raw_cards: 原始未排序卡片
        relation: 需要确保存在的关系类型
        limit: 总卡片数上限

    Returns:
        确保包含指定关系卡片后的列表
    """
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
    """构造无 LLM 时的兜底答案。

    优先使用图谱记录生成结构化事实摘要，其次使用 RAG 命中来源信息。

    Args:
        graph_records: 图谱查询结果
        rag_hits: RAG 检索结果

    Returns:
        兜底答案文本
    """
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
