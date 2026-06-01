from src.professional_qa import EvidenceCard
from src.question_planner import heuristic_plan_question
from src.research_agent import build_research_outputs


def test_research_outputs_include_report_table_risks_and_gaps() -> None:
    plan = heuristic_plan_question("中际旭创和新易盛在光模块业务上的差异、风险和跟踪指标是什么？")
    cards = [
        EvidenceCard(
            citation_id="E1",
            kind="claim",
            title="中际旭创 光模块 company_exposure",
            evidence="中际旭创 对 光模块 的公司敞口为 core：拥有高速光模块产品。",
            claim_id="c1",
            company="中际旭创",
            topic="光模块",
            claim_type="company_exposure",
            exposure_level="core",
            source="年报",
        ),
        EvidenceCard(
            citation_id="E2",
            kind="claim",
            title="新易盛 光模块 company_exposure",
            evidence="新易盛 对 光模块 的公司敞口为 direct：布局高速光模块。",
            claim_id="c2",
            company="新易盛",
            topic="光模块",
            claim_type="company_exposure",
            exposure_level="direct",
            source="年报",
        ),
        EvidenceCard(
            citation_id="E3",
            kind="claim",
            title="光模块 risk",
            evidence="光模块业务可能面临客户需求不及预期和价格竞争风险。",
            claim_id="c3",
            company="中际旭创",
            topic="光模块",
            claim_type="risk",
            source="年报",
        ),
    ]

    outputs = build_research_outputs(
        question=plan.question,
        plan=plan,
        evidence_cards=cards,
        graph_records=[],
        verification={"status": "pass", "checks": {}},
    )

    assert "投研简报" in outputs["report"]["title"]
    assert outputs["verification"]["status"] == "pass"
    assert len(outputs["company_compare_table"]["rows"]) == 2
    assert outputs["risk_checklist"][0]["priority"] == "高"
    assert any("领先指标" in row["gap"] for row in outputs["evidence_gaps"])


def test_research_outputs_merge_verification_gaps_and_conflicts() -> None:
    plan = heuristic_plan_question("英维克液冷业务主要风险是什么？")
    cards = [
        EvidenceCard(
            citation_id="E1",
            kind="claim",
            title="英维克 液冷 company_exposure",
            evidence="英维克液冷业务受益于高功率密度需求提升。",
            claim_id="c1",
            company="英维克",
            topic="液冷",
            claim_type="company_exposure",
            exposure_level="core",
            source="年报",
        )
    ]
    verification = {
        "status": "fail",
        "checks": {},
        "evidence_gaps": [{"gap": "缺少明确风险、反证或不确定性证据。", "priority": "高"}],
        "conflict_groups": [{"conflict_group_id": "conflict_1"}],
    }

    outputs = build_research_outputs(
        question=plan.question,
        plan=plan,
        evidence_cards=cards,
        graph_records=[],
        verification=verification,
    )

    assert outputs["verification"]["status"] == "fail"
    assert outputs["meta"]["conflict_group_count"] == 1
    assert any(row["gap"] == "缺少明确风险、反证或不确定性证据。" for row in outputs["evidence_gaps"])
