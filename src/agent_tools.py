"""Tool wrappers used by the first-phase rule-based QA agent."""

from __future__ import annotations

import time
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

from src.cypher_generator import GeneratedCypher
from src.professional_qa import (
    EvidenceCard,
    cards_from_graph_records,
    cards_from_rag_hits,
    cards_from_research_hits,
    rank_evidence_cards,
)
from src.question_planner import QuestionPlan, heuristic_plan_question, plan_question
from src.rag_index import RagHit
from src.research_claims import ResearchHit
from src.semantic_index import SemanticHit


@dataclass(frozen=True)
class AgentToolCall:
    tool: str
    args: dict[str, Any]
    result_count: int = 0
    elapsed_ms: float = 0.0
    error: str = ""

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

    def rank_evidence(
        self,
        research_hits: list[ResearchHit],
        graph_records: list[dict[str, Any]],
        rag_hits: list[RagHit],
        semantic_hits: list[SemanticHit],
        plan: QuestionPlan,
    ) -> tuple[list[EvidenceCard], list[EvidenceCard], AgentToolCall]:
        def run() -> tuple[list[EvidenceCard], list[EvidenceCard]]:
            from src.qa_engine import ensure_relation_cards

            raw_cards = [
                *cards_from_research_hits(research_hits, plan),
                *cards_from_graph_records(graph_records, plan),
                *cards_from_rag_hits(rag_hits, plan),
                *cards_from_semantic_hits(semantic_hits, plan),
            ]
            evidence_cards = rank_evidence_cards(raw_cards, limit=self.engine.evidence_top_n, plan=plan)
            if plan.answer_type == "risk_analysis":
                evidence_cards = ensure_relation_cards(
                    evidence_cards,
                    raw_cards,
                    "DISCLOSES_RISK",
                    limit=self.engine.evidence_top_n,
                )
            return raw_cards, evidence_cards

        result, call = self._call(
            "rank_evidence",
            {"answer_type": plan.answer_type, "evidence_top_n": self.engine.evidence_top_n},
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
    ) -> tuple[dict[str, Any], AgentToolCall]:
        return self._call(
            "verify_answer_support",
            {"answer_type": plan.answer_type, "evidence_cards": len(evidence_cards)},
            lambda: verify_answer_support(answer, plan, evidence_cards, raw_cards),
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
        started_at = time.perf_counter()
        error = ""
        before_error_count = len(self.errors)
        try:
            result = func()
        except Exception as exc:  # pragma: no cover - defensive shell around tools.
            result = None
            error = str(exc)
            self.errors.append(f"{tool} failed: {exc}")
        if not error and len(self.errors) > before_error_count:
            error = "; ".join(self.errors[before_error_count:])
        result_count = 0
        if result is not None:
            try:
                result_count = int(count_result(result))
            except Exception:
                result_count = 0
        call = AgentToolCall(
            tool=tool,
            args=args,
            result_count=result_count,
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
            error=error,
        )
        return result, call


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


def verify_answer_support(
    answer: str,
    plan: QuestionPlan,
    evidence_cards: list[EvidenceCard],
    raw_cards: list[EvidenceCard],
) -> dict[str, Any]:
    from src.qa_engine import detect_unsupported_terms

    unsupported_terms = detect_unsupported_terms(answer, evidence_cards)
    citation_ids = {card.citation_id for card in evidence_cards if card.citation_id}
    cited_ids = set(re.findall(r"\[(E\d+)\]", answer))
    missing_citations = sorted(cited_ids - citation_ids)
    company_coverage = company_coverage_checks(plan, evidence_cards, raw_cards)
    risk_cards = [
        card
        for card in evidence_cards
        if card.claim_type == "risk" or card.relation == "DISCLOSES_RISK"
    ]
    checks = {
        "evidence_count": len(evidence_cards),
        "raw_evidence_count": len(raw_cards),
        "company_coverage": company_coverage,
        "risk_evidence_count": len(risk_cards),
        "citation_ids": sorted(citation_ids),
        "missing_citations": missing_citations,
        "unsupported_terms": unsupported_terms,
    }
    if not evidence_cards:
        status = "fail"
    elif unsupported_terms or missing_citations or company_coverage.get("missing"):
        status = "warn"
    else:
        status = "pass"
    return {"status": status, "checks": checks}


def company_coverage_checks(
    plan: QuestionPlan,
    evidence_cards: list[EvidenceCard],
    raw_cards: list[EvidenceCard],
) -> dict[str, Any]:
    if not plan.companies:
        return {"required": [], "covered": [], "missing": []}
    cards = evidence_cards or raw_cards
    covered = sorted({card.company for card in cards if card.company in set(plan.companies)})
    missing = [company for company in plan.companies if company not in covered]
    return {"required": plan.companies, "covered": covered, "missing": missing}
