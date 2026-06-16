"""Tool wrappers used by the first-phase rule-based QA agent.

该模块为第一阶段基于规则的 QA Agent 提供工具包装层。
每个工具方法封装了 QAEngine 的对应功能，并通过 _call 统一管理：
- 执行耗时统计 (elapsed_ms)
- 结果计数 (result_count)
- 错误收集 (errors)
- 预算耗尽标记 (budget_exhausted)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

from src.agents.executor import ToolExecutor
from src.cypher_generator import GeneratedCypher
from src.graphrag import GraphRagResult, run_graphrag
from src.professional_qa import (
    EvidenceCard,
    cards_from_graph_records,
    cards_from_rag_hits,
    cards_from_research_hits,
    rank_evidence_cards,
)
from src.question_planner import QuestionPlan, heuristic_plan_question, plan_question
from src.rag_index import RagHit
from src.reranker import EvidenceReranker
from src.research_claims import ResearchHit
from src.semantic_index import SemanticHit


@dataclass(frozen=True)
class AgentToolCall:
    """一次工具调用的遥测记录。

    Attributes:
        tool: 工具名称（如 "plan_question", "search_rag"）
        args: 调用参数摘要（用于日志和调试）
        result_count: 结果数量（如命中数、证据卡片数）
        elapsed_ms: 执行耗时，单位毫秒
        error: 错误信息（如果有）
        budget_exhausted: 是否因预算耗尽而中断
    """
    tool: str
    args: dict[str, Any]
    result_count: int = 0
    elapsed_ms: float = 0.0
    error: str = ""
    budget_exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """将遥测记录序列化为字典，用于日志/调试输出。"""
        return asdict(self)


class AgentTools:
    """工具适配层，封装 QAEngine 方法并收集每次工具调用的遥测数据。

    每个工具方法都通过 _call 统一包装，自动记录：
    - 执行耗时
    - 结果数量
    - 错误信息
    - 预算状态

    这些遥测数据用于 Agent 的预算管理、调试和可观测性。
    """

    def __init__(
        self,
        engine: Any,
        *,
        errors: list[str],
        llm_options: dict[str, Any],
        llm_client: Any | None,
    ) -> None:
        """初始化 AgentTools。

        Args:
            engine: QAEngine 实例，提供实际的后端功能
            errors: 错误收集列表（调用方传入的共享列表）
            llm_options: LLM 调用参数（如 temperature, model 等）
            llm_client: LLM 客户端（用于 LLM 增强的规划/重排序）
        """
        self.engine = engine
        self.errors = errors
        self.llm_options = llm_options
        self.llm_client = llm_client
        self.executor = ToolExecutor(errors=errors)
        self.last_reranker_metadata: dict[str, Any] = {}

    def contextualize_question(self, question: str, history: list[dict[str, str]]) -> tuple[str, AgentToolCall]:
        """基于对话历史对问题进行上下文补全。

        如果 history 非空，将用户当前问题与历史上下文融合，
        生成一个独立可回答的上下文化问题。

        Args:
            question: 用户当前问题
            history: 对话历史（角色+消息的列表）

        Returns:
            (contextualized_question, tool_call_telemetry) 的元组
        """
        return self._call(
            "contextualize_question",
            {"has_history": bool(history), "mode": self.engine.contextualizer_mode},
            lambda: self.engine._contextualize_question(
                question,
                history,
                self.errors,
                self.llm_options,
                self.llm_client,
            ),
            lambda result: 1 if result else 0,
        )

    def plan_question(self, question: str) -> tuple[QuestionPlan, str, AgentToolCall]:
        """对用户问题进行结构化解析，生成查询计划。

        优先使用启发式规则（heuristic_plan_question）快速解析，
        如果计划质量不足且 LLM 可用，则调用 LLM 进行二次优化。
        返回 planner_source 标记最终计划来源（"heuristic" 或 "llm"）。

        Args:
            question: 用户原始问题

        Returns:
            (QuestionPlan, planner_source, tool_call_telemetry) 的元组
        """
        planner_source = "heuristic"

        def run() -> QuestionPlan:
            nonlocal planner_source
            # 第一步：启发式规则解析（快速、低成本）
            plan = heuristic_plan_question(question, core_companies_only=self.engine.core_companies_only)
            # 第二步：评估是否需要 LLM 优化（如计划中公司/主题为空）
            if self.engine._should_use_llm_planner(plan, self.llm_client):
                plan = plan_question(
                    question,
                    client=self.llm_client,
                    core_companies_only=self.engine.core_companies_only,
                    llm_options=self.llm_options,
                )
                planner_source = "llm"
            return plan

        plan, call = self._call(
            "plan_question",
            {"question": question[:160], "llm_planner_enabled": self.engine.enable_llm_planner},
            run,
            lambda result: len(result.companies) + len(result.topics) + 1,
        )
        return plan, planner_source, call

    def prepare_cypher(self, question: str, plan: QuestionPlan) -> tuple[GeneratedCypher, AgentToolCall]:
        """根据查询计划生成知识图谱的 Cypher 查询语句。

        Args:
            question: 用户问题
            plan: 查询计划（含公司、主题、关系等）

        Returns:
            (GeneratedCypher, tool_call_telemetry) 的元组
        """
        def run() -> GeneratedCypher:
            generated = self.engine._generate_display_cypher(
                question,
                plan,
                self.errors,
                self.llm_options,
                self.llm_client,
            )
            if generated.error:
                self.errors.append(generated.error)
            return generated

        return self._call(
            "prepare_cypher",
            {"answer_type": plan.answer_type, "companies": plan.companies, "topics": plan.topics},
            run,
            lambda result: 1 if result.cypher else 0,
        )

    def build_hyde_query(self, question: str, plan: QuestionPlan) -> tuple[Any, AgentToolCall]:
        """生成本轮文本检索使用的 HYDE query。"""
        return self._call(
            "build_hyde_query",
            {
                "enabled": bool(getattr(self.engine, "enable_hyde", True)),
                "query_mode": str(getattr(self.engine, "hyde_query_mode", "hybrid")),
                "answer_type": plan.answer_type,
            },
            lambda: self.engine._build_hyde_query(
                question,
                plan,
                self.errors,
                self.llm_options,
                self.llm_client,
            ),
            lambda result: 1 if getattr(result, "hypothetical_answer", "") else 0,
        )

    def query_graph(self, generated: GeneratedCypher, plan: QuestionPlan) -> tuple[list[dict[str, Any]], AgentToolCall]:
        """执行 Cypher 查询，从知识图谱中获取结构化数据。

        Args:
            generated: 已生成的 Cypher 查询
            plan: 查询计划

        Returns:
            (graph_records, tool_call_telemetry) 的元组
        """
        return self._call(
            "query_graph",
            {"source": generated.source, "answer_type": plan.answer_type},
            lambda: self.engine._query_graph(generated, plan, self.errors),
            len,
        )

    def search_rag(self, question: str, plan: QuestionPlan) -> tuple[list[RagHit], AgentToolCall]:
        """基于 RAG 索引进行语义检索，获取非结构化的文本证据。

        Args:
            question: 用户问题
            plan: 查询计划

        Returns:
            (rag_hits_list, tool_call_telemetry) 的元组
        """
        return self._call(
            "search_rag",
            {"question": question[:160], "answer_type": plan.answer_type},
            lambda: self.engine._search_rag(question, plan, self.errors),
            len,
        )

    def search_research(self, question: str, plan: QuestionPlan) -> tuple[list[ResearchHit], AgentToolCall]:
        """搜索研究报告索引，获取深度研究相关的证据。

        Args:
            question: 用户问题
            plan: 查询计划

        Returns:
            (research_hits_list, tool_call_telemetry) 的元组
        """
        return self._call(
            "search_research",
            {"question": question[:160], "answer_type": plan.answer_type},
            lambda: self.engine._search_research(question, plan, self.errors),
            len,
        )

    def search_semantic(self, question: str, plan: QuestionPlan) -> tuple[list[SemanticHit], AgentToolCall]:
        """基于语义向量索引进行检索。

        将问题和计划中的公司、主题等信息拼接为语义查询文本，
        再到 semantic_index 中进行向量检索。

        Args:
            question: 用户问题
            plan: 查询计划

        Returns:
            (semantic_hits_list, tool_call_telemetry) 的元组
        """
        def run() -> list[SemanticHit]:
            semantic_index = getattr(self.engine, "semantic_index", None)
            if semantic_index is None:
                return []
            query = semantic_query_text(question, plan)
            top_k = int(getattr(self.engine, "semantic_top_k", 8))
            return semantic_index.search(query, top_k=top_k)

        result, call = self._call(
            "search_semantic",
            {"question": question[:160], "answer_type": plan.answer_type},
            run,
            len,
        )
        return result or [], call

    def run_graphrag(
        self,
        question: str,
        plan: QuestionPlan,
        graph_records: list[dict[str, Any]],
    ) -> tuple[GraphRagResult, AgentToolCall]:
        """执行 GraphRAG（基于图的检索增强生成）推理。

        结合知识图谱记录和研究记忆，进行多跳推理（DRIFT 模式），
        获取公司排名、多跳路径等深度分析结果。

        Args:
            question: 用户问题
            plan: 查询计划
            graph_records: 知识图谱查询结果

        Returns:
            (GraphRagResult, tool_call_telemetry) 的元组
        """
        return self._call(
            "graphrag_retrieve",
            {"question": question[:160], "answer_type": plan.answer_type},
            lambda: run_graphrag(
                question=question,
                plan=plan,
                research_memory=getattr(self.engine, "research_memory", None),
                graph_records=graph_records,
                max_subquestions=int(getattr(self.engine, "drift_max_subquestions", 6)),
                global_top_k=int(getattr(self.engine, "global_dossier_top_k", 3)),
                local_top_k=int(getattr(self.engine, "local_claim_top_k", 12)),
                path_top_k=int(getattr(self.engine, "graph_path_top_k", 6)),
            ),
            lambda result: len(result.global_hits) + len(result.local_hits) + len(result.paths) + len(result.company_rankings),
        )

    def rank_evidence(
        self,
        research_hits: list[ResearchHit],
        graph_records: list[dict[str, Any]],
        rag_hits: list[RagHit],
        semantic_hits: list[SemanticHit],
        plan: QuestionPlan,
        graphrag_result: GraphRagResult | None = None,
    ) -> tuple[list[EvidenceCard], list[EvidenceCard], AgentToolCall]:
        """对从多个来源检索到的证据进行排序和重排序。

        流程：
        1. 将所有来源的证据转换为统一的 EvidenceCard
        2. 初排序（rank_evidence_cards）
        3. 风险分析类问题补充关系卡片（ensure_relation_cards）
        4. 使用 EvidenceReranker 进行重排序（可选 LLM 增强）
        5. 再次补充风险关系卡片

        Args:
            research_hits: 研究报告检索结果
            graph_records: 知识图谱查询结果
            rag_hits: RAG 检索结果
            semantic_hits: 语义向量检索结果
            plan: 查询计划
            graphrag_result: GraphRAG 推理结果（可选）

        Returns:
            (raw_cards, ranked_cards, tool_call_telemetry) 的元组
            raw_cards 用于验证环节，ranked_cards 用于最终答案生成
        """
        def run() -> tuple[list[EvidenceCard], list[EvidenceCard]]:
            from src.qa_engine import ensure_relation_cards

            # 1. 合并所有来源的证据
            raw_cards = [
                *cards_from_research_hits(research_hits, plan),
                *cards_from_graph_records(graph_records, plan),
                *cards_from_rag_hits(rag_hits, plan),
                *cards_from_semantic_hits(semantic_hits, plan),
                *cards_from_graphrag_result(graphrag_result, plan),
            ]
            # 2. 初排序
            candidate_limit = max(int(getattr(self.engine, "evidence_top_n", 6)), int(getattr(self.engine, "rerank_top_n", 12)))
            evidence_cards = rank_evidence_cards(raw_cards, limit=candidate_limit, plan=plan)
            # 3. 风险分析类问题：确保包含 DISCLOSES_RISK 关系卡片
            if plan.answer_type == "risk_analysis":
                evidence_cards = ensure_relation_cards(
                    evidence_cards,
                    raw_cards,
                    "DISCLOSES_RISK",
                    limit=candidate_limit,
                )
            # 4. 重排序（自动模式，GraphRAG 启用时使用 LLM 增强）
            reranker = EvidenceReranker(mode=str(getattr(self.engine, "rerank_mode", "auto")))
            use_llm_rerank = bool(graphrag_result and graphrag_result.route.use_drift)
            reranked = reranker.rerank(
                question=plan.question,
                cards=evidence_cards,
                limit=int(getattr(self.engine, "evidence_top_n", 6)),
                llm_client=self.llm_client,
                llm_options=self.llm_options,
                use_llm=use_llm_rerank,
            )
            self.last_reranker_metadata = reranked.metadata
            evidence_cards = reranked.cards
            # 5. 再次补充风险关系卡片（重排序后可能丢失）
            if plan.answer_type == "risk_analysis":
                evidence_cards = ensure_relation_cards(
                    evidence_cards,
                    raw_cards,
                    "DISCLOSES_RISK",
                    limit=int(getattr(self.engine, "evidence_top_n", 6)),
                )
            return raw_cards, evidence_cards

        result, call = self._call(
            "rank_evidence",
            {
                "answer_type": plan.answer_type,
                "evidence_top_n": self.engine.evidence_top_n,
                "graphrag": bool(graphrag_result),
            },
            run,
            lambda result: len(result[1]),
        )
        return result[0], result[1], call

    def generate_answer(
        self,
        question: str,
        contextual_question: str,
        history: list[dict[str, str]],
        plan: QuestionPlan,
        graph_records: list[dict[str, Any]],
        evidence_cards: list[EvidenceCard],
    ) -> tuple[tuple[str, str], AgentToolCall]:
        """基于所有已收集的证据生成最终答案。

        Args:
            question: 用户原始问题
            contextual_question: 上下文补全后的问题
            history: 对话历史
            plan: 查询计划
            graph_records: 知识图谱记录（用于表格/结构化数据展示）
            evidence_cards: 排序后的证据卡片

        Returns:
            ((answer_text, answer_type_label), tool_call_telemetry) 的元组
        """
        return self._call(
            "generate_answer",
            {"answer_type": plan.answer_type, "evidence_cards": len(evidence_cards)},
            lambda: self.engine._generate_answer(
                question,
                contextual_question,
                history,
                plan,
                graph_records,
                evidence_cards,
                self.errors,
                self.llm_options,
                self.llm_client,
            ),
            lambda result: 1 if result[0] else 0,
        )

    def verify_answer_support(
        self,
        answer: str,
        plan: QuestionPlan,
        evidence_cards: list[EvidenceCard],
        raw_cards: list[EvidenceCard],
        question: str = "",
    ) -> tuple[dict[str, Any], AgentToolCall]:
        """验证生成的答案是否基于证据，检查事实一致性。

        对答案中的每个事实陈述，检查是否有对应的证据卡片支持，
        并返回检查结果字典（含每个陈述的验证状态）。

        Args:
            answer: 已生成的答案文本
            plan: 查询计划
            evidence_cards: 排序后的证据卡片（用于验证）
            raw_cards: 原始未排序的证据卡片（用于兜底查找）
            question: 用户原始问题（可选，用于上下文）

        Returns:
            (verification_result_dict, tool_call_telemetry) 的元组
        """
        return self._call(
            "verify_answer_support",
            {"answer_type": plan.answer_type, "evidence_cards": len(evidence_cards)},
            lambda: verify_answer_support(answer, plan, evidence_cards, raw_cards, question=question),
            lambda result: len(result.get("checks", {})),
        )

    def supplemental_plan(self, plan: QuestionPlan, question: str, companies: list[str] | None = None) -> QuestionPlan:
        """生成一个补充性的查询计划，用于后续追问或修正。

        基于原计划替换问题文本和/或公司列表，其他字段保持不变。
        常用于 Agent 在对话过程中需要重新聚焦问题场景。

        Args:
            plan: 原始查询计划
            question: 新的问题文本
            companies: 新的公司列表（可选，不传则沿用原计划）

        Returns:
            更新后的 QuestionPlan
        """
        return replace(plan, question=question, companies=companies if companies is not None else plan.companies)

    def _call(
        self,
        tool: str,
        args: dict[str, Any],
        func: Callable[[], Any],
        count_result: Callable[[Any], int],
    ) -> tuple[Any, AgentToolCall]:
        """统一工具调用入口，包装执行、耗时统计和遥测记录。

        所有工具方法都通过此方法执行，确保：
        - 统一的执行接口
        - 自动的耗时统计
        - 错误捕获和记录
        - 结果计数
        - 预算状态追踪

        Args:
            tool: 工具名称
            args: 调用参数（摘要）
            func: 实际执行函数
            count_result: 结果计数函数（从执行结果中提取数量）

        Returns:
            (func_execution_result, AgentToolCall_telemetry) 的元组
        """
        execution = self.executor.execute(tool, args, func, count_result)
        call = AgentToolCall(
            tool=tool,
            args=args,
            result_count=execution.result_count,
            elapsed_ms=execution.elapsed_ms,
            error=execution.error,
            budget_exhausted=execution.budget_exhausted,
        )
        return execution.result, call


def semantic_query_text(question: str, plan: QuestionPlan) -> str:
    """拼接用于语义向量检索的查询文本。

    将问题原文、答案类型、公司列表、扩展主题等拼接成一个字符串，
    用于在语义索引中进行更准确的向量检索。

    Args:
        question: 用户问题
        plan: 查询计划

    Returns:
        拼接后的语义查询字符串
    """
    parts = [question, plan.answer_type, *plan.companies, *plan.expanded_topics]
    return " ".join(part for part in parts if part).strip()


def cards_from_semantic_hits(hits: list[SemanticHit], plan: QuestionPlan) -> list[EvidenceCard]:
    """将语义向量检索结果转换为统一的 EvidenceCard 列表。

    根据命中类型（claim/dossier/rag）设置不同的基础分值：
    - claim（研究声明）: 52 分
    - dossier（公司档案）: 55 分
    - rag（通用文本）: 12 分
    再加上语义相似度得分的加权。

    Args:
        hits: 语义向量检索结果
        plan: 查询计划（当前未使用）

    Returns:
        EvidenceCard 列表
    """
    del plan
    cards: list[EvidenceCard] = []
    for hit in hits:
        kind = hit.kind if hit.kind in {"rag", "claim", "dossier"} else "rag"
        base_score = {
            "claim": 52.0,
            "dossier": 55.0,
            "rag": 12.0,
        }.get(kind, 10.0)
        cards.append(
            EvidenceCard(
                citation_id="",
                kind=kind,
                title=hit.title,
                evidence=hit.text,
                claim_id=hit.ref_id if kind == "claim" else "",
                source=hit.source,
                page=hit.page,
                section=hit.section,
                company=hit.company,
                source_tier=hit.source_tier,
                score=round(base_score + hit.score * 10.0, 4),
                reason="语义向量召回",
                topic=hit.topic,
                claim_type=hit.claim_type,
                exposure_level=hit.exposure_level,
                confidence=hit.confidence,
                as_of_date=hit.as_of_date,
                semantic_score=hit.score,
                semantic_ref_id=hit.ref_id or hit.doc_id,
            )
        )
    return cards


def cards_from_graphrag_result(result: GraphRagResult | None, plan: QuestionPlan) -> list[EvidenceCard]:
    """将 GraphRAG 推理结果转换为 EvidenceCard 列表。

    包括两类卡片：
    1. 公司排序卡片（company_ranking）：包含综合排序理由、指标证据和风险证据
    2. 多跳路径卡片（graphrag_path）：包含供应链/技术路径的多跳解释

    Args:
        result: GraphRAG 推理结果（可能为 None）
        plan: 查询计划（当前未使用）

    Returns:
        EvidenceCard 列表
    """
    del plan
    if result is None:
        return []
    cards: list[EvidenceCard] = []
    # 公司排序卡片
    for ranking in result.company_rankings:
        evidence = (
            f"{ranking.company} 综合排序：{ranking.reason}。"
            f"指标证据：{ranking.indicator_evidence or '当前证据不足'}；"
            f"风险证据：{ranking.risk_evidence or '当前证据不足'}。"
        )
        cards.append(
            EvidenceCard(
                citation_id="",
                kind="company_ranking",
                title=f"{ranking.company} GraphRAG 公司排序",
                evidence=evidence,
                company=ranking.company,
                source="图谱推理公司排序",
                score=round(70.0 + ranking.score, 4),
                reason="GraphRAG 公司排序",
                topic=ranking.topic,
                claim_type="company_exposure",
                exposure_level=ranking.exposure_level,
            )
        )
    # 多跳路径卡片
    for path in result.paths:
        cards.append(
            EvidenceCard(
                citation_id="",
                kind="graphrag_path",
                title=f"{path.company} 多跳路径",
                evidence=path.explanation,
                company=path.company,
                source="图谱推理多跳路径",
                score=round(64.0 + path.score, 4),
                reason="GraphRAG 多跳路径",
                topic=path.topic,
                claim_type="supply_chain",
                exposure_level="",
            )
        )
    return cards


def verify_answer_support(
    answer: str,
    plan: QuestionPlan,
    evidence_cards: list[EvidenceCard],
    raw_cards: list[EvidenceCard],
    *,
    question: str = "",
) -> dict[str, Any]:
    """验证答案中的事实陈述是否有证据支撑。

    对答案进行事实分解，逐一检查每个陈述是否能在证据卡片中找到支持，
    返回包含每个检查项结果的字典。

    Args:
        answer: 待验证的答案文本
        plan: 查询计划
        evidence_cards: 排序后的证据卡片（主要验证依据）
        raw_cards: 原始未排序证据卡片（用于兜底匹配）
        question: 用户原始问题

    Returns:
        验证结果字典，包含 "checks" 等字段
    """
    from src.agents.verification import verify_answer_support as verify

    return verify(answer, plan, evidence_cards, raw_cards, question=question)
