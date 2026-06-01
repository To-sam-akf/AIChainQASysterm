from src.agents.verification import build_evidence_limited_answer, detect_conflict_groups, verify_answer_support
from src.professional_qa import EvidenceCard
from src.question_planner import heuristic_plan_question


def card(
    citation_id: str,
    *,
    evidence: str,
    company: str = "英维克",
    topic: str = "液冷",
    claim_type: str = "company_exposure",
    exposure_level: str = "core",
    source: str = "测试报告",
    as_of_date: str = "2025",
) -> EvidenceCard:
    return EvidenceCard(
        citation_id=citation_id,
        kind="claim",
        title=f"{company} {topic} {claim_type}",
        evidence=evidence,
        claim_id=f"claim_{citation_id}",
        source=source,
        page="1",
        company=company,
        topic=topic,
        claim_type=claim_type,
        exposure_level=exposure_level,
        as_of_date=as_of_date,
    )


def test_numeric_and_year_without_source_fail_verification() -> None:
    plan = heuristic_plan_question("英维克液冷收入是多少？")
    evidence = [card("E1", evidence="英维克提供端到端液冷产品。", as_of_date="")]

    verification = verify_answer_support(
        "英维克 2026 年液冷收入达到 10亿元 [E1]。",
        plan,
        evidence,
        evidence,
        question=plan.question,
    )

    assert verification["status"] == "fail"
    unsupported = verification["checks"]["numeric_support"]["unsupported"]
    assert "2026" in unsupported
    assert "10亿元" in unsupported


def test_metric_question_requires_indicator_evidence() -> None:
    plan = heuristic_plan_question("英维克液冷订单指标是什么？")
    evidence = [card("E1", evidence="英维克提供端到端液冷产品。")]

    verification = verify_answer_support("英维克有液冷产品证据 [E1]。", plan, evidence, evidence, question=plan.question)

    assert verification["status"] == "fail"
    assert verification["checks"]["metric_support"]["status"] == "fail"
    assert any("指标证据" in row["gap"] for row in verification["evidence_gaps"])


def test_exposure_level_mismatch_fails_verification() -> None:
    plan = heuristic_plan_question("英维克液冷业务画像")
    evidence = [
        card(
            "E1",
            evidence="英维克 对 液冷 的公司敞口为 direct：提供液冷产品。",
            exposure_level="direct",
        )
    ]

    verification = verify_answer_support("英维克是液冷核心敞口公司 [E1]。", plan, evidence, evidence, question=plan.question)

    assert verification["status"] == "fail"
    assert verification["checks"]["exposure_support"]["mismatches"][0]["answer_level"] == "核心敞口"


def test_risk_question_requires_risk_or_counter_evidence() -> None:
    plan = heuristic_plan_question("英维克液冷业务主要风险是什么？")
    evidence = [card("E1", evidence="英维克提供端到端液冷产品。")]

    verification = verify_answer_support("英维克液冷业务存在客户需求波动风险 [E1]。", plan, evidence, evidence, question=plan.question)

    assert verification["status"] == "fail"
    assert verification["checks"]["risk_support"]["status"] == "fail"
    assert any("风险" in row["gap"] for row in verification["evidence_gaps"])


def test_conflict_groups_pair_optimistic_claim_with_risk_claim() -> None:
    cards = [
        card("E1", evidence="英维克液冷业务受益于高功率密度需求提升。", claim_type="company_exposure"),
        card("E2", evidence="英维克液冷业务存在客户需求不及预期风险。", claim_type="risk"),
    ]

    groups = detect_conflict_groups(cards)

    assert groups
    assert groups[0]["conflict_type"] == "optimistic_vs_risk"
    assert groups[0]["claim_a"]["citation_id"] == "E1"
    assert groups[0]["claim_b"]["citation_id"] == "E2"


def test_evidence_limited_answer_outputs_gaps_when_verification_fails() -> None:
    plan = heuristic_plan_question("英维克液冷订单指标是什么？")
    evidence = [card("E1", evidence="英维克提供端到端液冷产品。")]
    verification = verify_answer_support("英维克订单为 10亿元 [E1]。", plan, evidence, evidence, question=plan.question)

    answer = build_evidence_limited_answer(plan, evidence, verification)

    assert "证据支持" in answer
    assert "证据缺口" in answer
    assert "英维克订单为 10亿元" not in answer
