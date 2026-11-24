from dataclasses import replace
from types import SimpleNamespace

import pytest

from aika.agent_runner import AgentRetrievalState, should_refuse_metric_answer
from aika.agents.centralized import CentralLLMBudget, EvidenceCandidate, LLMBudgetExhausted
from aika.agents.coverage import EvidenceCoverageChecker
from aika.agents.planner import TaskPlanner
from aika.professional_qa import cards_from_rag_hits
from aika.question_planner import QuestionPlan
from aika.rag_index import RagHit
from aika.semantic_index import SemanticHit


TABLE = """### 公司研发投入情况
| 项目 | 2025年 | 2024年 |
|---|---|---|
| 研发投入金额（万元） | 12,345.67 | 10,000.00 |
| 研发投入占营业收入比例 | 2.35% | 3.15% |
| 研发投入资本化的金额（万元） | 678.90 | 500.00 |
"""


def raw_hit(text=TABLE, company="浪潮信息"):
    return RagHit(chunk_id="chunk-table", report_id="report", source_title="2025年报", source_tier="1",
                  source_type="annual_report", page="15", section="研发投入", content_type="table", table_id="table-1",
                  company=company, text=text, snippet="公司研发投入情况 | 研发投入金额（万元） | 12,345...", score=10)


def plan():
    return QuestionPlan(question="浪潮信息2025年研发投入金额、资本化研发投入及占营收比例是多少？",
                        answer_type="company_profile", companies=["浪潮信息"], needs_metrics=True)


def check(hits):
    p = plan()
    return EvidenceCoverageChecker().check(p, TaskPlanner().plan(p.question, p),
                                           AgentRetrievalState(rag_hits=hits), retrieval_round=0)


def test_raw_numeric_table_satisfies_metric_coverage_without_extracted_labels():
    report = check([raw_hit()])
    assert report.sufficient and not report.should_continue
    assert report.satisfied == ["metric_evidence"]


@pytest.mark.parametrize("hit", [
    raw_hit("2025年研发投入情况，单位：万元。"),
    raw_hit("公司持续加大研发投入，2025年进一步扩大研发队伍。"),
    raw_hit("| 营业收入（万元） | 90,000.00 |", company="其他公司"),
    raw_hit("| 毛利率 | 30.00% |"),
])
def test_heading_year_or_unrelated_disclosure_does_not_satisfy_metrics(hit):
    assert check([hit]).missing == ["metric_evidence"]


def test_raw_semantic_table_is_also_recognized():
    p = plan()
    state = AgentRetrievalState(semantic_hits=[SemanticHit(doc_id="rag:chunk-table", kind="rag", title="年报",
                                 text=TABLE, score=.8, company="浪潮信息")])
    assert EvidenceCoverageChecker().check(p, TaskPlanner().plan(p.question, p), state, retrieval_round=0).sufficient


def test_full_table_survives_card_and_reviewer_payload():
    hit = raw_hit(TABLE + "\n" + "附注：完整原文。" * 90 + "\n尾部资本化金额678.90万元。")
    card = cards_from_rag_hits([hit], plan())[0]
    assert card.evidence == hit.text
    assert card.semantic_ref_id == hit.chunk_id
    assert EvidenceCandidate("candidate", card, "rag").prompt_payload()["evidence"] == hit.text


def test_budget_leaves_summary_and_answer_calls_and_prevents_another_full_wave():
    client = SimpleNamespace(chat_json=lambda: {}, chat_text=lambda: "answer")
    budget = CentralLLMBudget(client, max_calls=6)
    assert budget.can_start_retrieval_round(2)
    for _ in range(4):
        budget.client("query_rag").chat_json()
    assert not budget.can_start_retrieval_round(1)
    with pytest.raises(LLMBudgetExhausted):
        budget.client("query_rag").chat_json()
    budget.client("summarizer").chat_json()
    assert budget.client("answer").chat_text() == "answer"
    assert budget.calls["total"] == 6


def test_budget_stop_keeps_unresolved_metric_gap_and_refusal():
    report = replace(check([]), stop_reason="llm_budget_insufficient", status="fail")
    assert report.missing == ["metric_evidence"]
    assert not report.should_continue
    assert should_refuse_metric_answer(report)
