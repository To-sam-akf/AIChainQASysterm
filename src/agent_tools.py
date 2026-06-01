"""Tool wrappers used by the first-phase rule-based QA agent."""

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
    tool: str
    args: dict[str, Any]
    result_count: int = 0
    elapsed_ms: float = 0.0
    error: str = ""
    budget_exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentTools:
    """Small adapter layer around QAEngine methods with tool-call telemetry."""

    def __init__(
        self,
        engine: Any,
        *,
        errors: list[str],
        llm_options: dict[str, Any],
        llm_client: Any | None,
    ) -> None:
        self.engine = engine
        self.errors = errors
        self.llm_options = llm_options
        self.llm_client = llm_client
        self.executor = ToolExecutor(errors=errors)
        self.last_reranker_metadata: dict[str, Any] = {}

    def contextualize_question(self, question: str, history: list[dict[str, str]]) -> tuple[str, AgentToolCall]:
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
        planner_source = "heuristic"

        def run() -> QuestionPlan:
            nonlocal planner_source
            plan = heuristic_plan_question(question, core_companies_only=self.engine.core_companies_only)
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

    def query_graph(self, generated: GeneratedCypher, plan: QuestionPlan) -> tuple[list[dict[str, Any]], AgentToolCall]:
        return self._call(
            "query_graph",
            {"source": generated.source, "answer_type": plan.answer_type},
            lambda: self.engine._query_graph(generated, plan, self.errors),
            len,
        )

    def search_rag(self, question: str, plan: QuestionPlan) -> tuple[list[RagHit], AgentToolCall]:
        return self._call(
            "search_rag",
            {"question": question[:160], "answer_type": plan.answer_type},
            lambda: self.engine._search_rag(question, plan, self.errors),
            len,
        )

    def search_research(self, question: str, plan: QuestionPlan) -> tuple[list[ResearchHit], AgentToolCall]:
        return self._call(
            "search_research",
            {"question": question[:160], "answer_type": plan.answer_type},
            lambda: self.engine._search_research(question, plan, self.errors),
            len,
        )

    def search_semantic(self, question: str, plan: QuestionPlan) -> tuple[list[SemanticHit], AgentToolCall]:
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
        def run() -> tuple[list[EvidenceCard], list[EvidenceCard]]:
            from src.qa_engine import ensure_relation_cards

            raw_cards = [
                *cards_from_research_hits(research_hits, plan),
                *cards_from_graph_records(graph_records, plan),
                *cards_from_rag_hits(rag_hits, plan),
                *cards_from_semantic_hits(semantic_hits, plan),
                *cards_from_graphrag_result(graphrag_result, plan),
            ]
            candidate_limit = max(int(getattr(self.engine, "evidence_top_n", 6)), int(getattr(self.engine, "rerank_top_n", 12)))
            evidence_cards = rank_evidence_cards(raw_cards, limit=candidate_limit, plan=plan)
            if plan.answer_type == "risk_analysis":
                evidence_cards = ensure_relation_cards(
                    evidence_cards,
                    raw_cards,
                    "DISCLOSES_RISK",
                    limit=candidate_limit,
                )
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
        return self._call(
            "verify_answer_support",
            {"answer_type": plan.answer_type, "evidence_cards": len(evidence_cards)},
            lambda: verify_answer_support(answer, plan, evidence_cards, raw_cards, question=question),
            lambda result: len(result.get("checks", {})),
        )

    def supplemental_plan(self, plan: QuestionPlan, question: str, companies: list[str] | None = None) -> QuestionPlan:
        return replace(plan, question=question, companies=companies if companies is not None else plan.companies)

    def _call(
        self,
        tool: str,
        args: dict[str, Any],
        func: Callable[[], Any],
        count_result: Callable[[Any], int],
    ) -> tuple[Any, AgentToolCall]:
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
    parts = [question, plan.answer_type, *plan.companies, *plan.expanded_topics]
    return " ".join(part for part in parts if part).strip()


def cards_from_semantic_hits(hits: list[SemanticHit], plan: QuestionPlan) -> list[EvidenceCard]:
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
    del plan
    if result is None:
        return []
    cards: list[EvidenceCard] = []
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
    from src.agents.verification import verify_answer_support as verify

    return verify(answer, plan, evidence_cards, raw_cards, question=question)
