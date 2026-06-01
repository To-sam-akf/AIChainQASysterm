from src.graphrag import DriftPlanner, QueryRouter, run_graphrag
from src.frontend_data import LocalKnowledgeGraph
from src.professional_qa import EvidenceCard
from src.qa_engine import QAEngine
from src.question_planner import heuristic_plan_question
from src.reranker import EvidenceReranker
from src.research_claims import ResearchMemory


def claim(
    claim_id: str,
    *,
    topic: str = "液冷",
    company: str = "",
    claim_type: str,
    text: str,
    exposure_level: str = "",
    source_tier: str = "1",
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "claim_type": claim_type,
        "topic": topic,
        "claim_text": text,
        "companies": [company] if company else [],
        "source_title": "测试报告",
        "page": "1",
        "section": "正文",
        "source_tier": source_tier,
        "confidence": "0.9",
        "as_of_date": "2026",
        "exposure_level": exposure_level,
        "evidence_span": text,
        "review_status": "auto",
    }


def memory() -> ResearchMemory:
    return ResearchMemory(
        [
            claim(
                "c1",
                claim_type="company_exposure",
                company="英维克",
                text="英维克 对 液冷 的公司敞口为 core：提供 CDU、冷板和管路系统。",
                exposure_level="core",
            ),
            claim(
                "c2",
                claim_type="indicator",
                company="英维克",
                text="液冷可跟踪指标包括 CDU 交付、订单和高功率机柜渗透率。",
            ),
            claim(
                "c3",
                claim_type="risk",
                company="英维克",
                text="液冷业务存在客户资本开支波动和交付节奏不确定性风险。",
            ),
            claim(
                "c4",
                claim_type="company_exposure",
                company="高澜股份",
                text="高澜股份 对 液冷 的公司敞口为 direct：提供热管理产品。",
                exposure_level="direct",
            ),
            claim(
                "c5",
                claim_type="mechanism",
                text="液冷通过缓解高功率 AI 服务器散热约束支撑智算中心部署。",
            ),
        ],
        [
            {
                "topic": "液冷",
                "summary": "液冷用于缓解高功率机柜散热约束，直接敞口公司包括英维克。",
                "company_exposure": {"core": ["英维克"], "direct": ["高澜股份"]},
                "technology_mechanism": ["液冷缓解 AI 服务器功率密度提升后的散热瓶颈"],
                "leading_indicators": ["CDU 交付、订单、高功率机柜渗透率"],
                "risks": ["客户资本开支波动"],
                "gaps": [],
            }
        ],
    )


def test_query_router_routes_professional_question_types() -> None:
    router = QueryRouter()

    assert router.route("液冷产业链有哪些上市公司？", heuristic_plan_question("液冷产业链有哪些上市公司？")).kind == "company_exposure"
    assert router.route("中际旭创和新易盛在光模块业务上的差异是什么？", heuristic_plan_question("中际旭创和新易盛在光模块业务上的差异是什么？")).kind == "company_compare"
    assert router.route("浪潮信息AI服务器订单指标是什么？", heuristic_plan_question("浪潮信息AI服务器订单指标是什么？")).kind == "metric_only"
    assert router.route("英维克液冷业务主要风险是什么？", heuristic_plan_question("英维克液冷业务主要风险是什么？")).kind == "risk_review"
    assert router.route("AI算力产业链当前最大的瓶颈是什么？", heuristic_plan_question("AI算力产业链当前最大的瓶颈是什么？")).kind == "global_causal"


def test_drift_planner_splits_broad_question_into_global_and_local_subquestions() -> None:
    question = "AI算力产业链当前最大的瓶颈是什么？"
    plan = heuristic_plan_question(question)
    route = QueryRouter().route(question, plan)

    subquestions = DriftPlanner(max_subquestions=6).plan(question, plan, route)

    assert len(subquestions) >= 2
    assert any(item.focus == "global_mechanism" for item in subquestions)
    assert any(item.focus == "local_company_metric_risk" for item in subquestions)
    assert all(item.question for item in subquestions)


def test_graphrag_combines_global_dossier_local_claims_company_ranking_and_paths() -> None:
    question = "液冷产业链有哪些上市公司，各自处于什么环节？"
    plan = heuristic_plan_question(question)
    graph_records = [
        {
            "company": "英维克",
            "relation": "HAS_PRODUCT",
            "target": "液冷CDU",
            "target_labels": ["Product"],
            "evidence": "英维克提供 CDU、冷板和管路系统。",
            "source": "测试报告",
            "page": "1",
            "source_tier": "1",
        }
    ]

    result = run_graphrag(
        question=question,
        plan=plan,
        research_memory=memory(),
        graph_records=graph_records,
        max_subquestions=6,
        global_top_k=2,
        local_top_k=10,
        path_top_k=4,
    )

    assert result.global_hits
    assert {hit.claim_type for hit in result.local_hits} >= {"company_exposure", "indicator", "risk"}
    assert result.company_rankings[0].company == "英维克"
    assert result.company_rankings[0].indicator_evidence
    assert result.company_rankings[0].risk_evidence
    assert result.paths
    path = result.paths[0].to_dict()
    assert {"demand", "technology", "segment", "company", "indicator", "risk"} <= set(path)
    assert "需求驱动" in {edge["label"] for edge in result.edges}


class FakeRerankLLM:
    def chat_json(self, **kwargs):
        assert "ranked_ids" in kwargs["user_prompt"]
        return {"ranked_ids": ["C2", "C1"]}


class FailingRerankLLM:
    def chat_json(self, **kwargs):
        del kwargs
        raise RuntimeError("rerank unavailable")


def test_llm_reranker_reorders_and_falls_back_to_heuristic() -> None:
    cards = [
        EvidenceCard(citation_id="", kind="claim", title="A", evidence="低相关证据", score=10),
        EvidenceCard(citation_id="", kind="claim", title="B", evidence="高相关证据", score=9),
    ]

    reranked = EvidenceReranker(mode="llm").rerank(
        question="液冷风险是什么？",
        cards=cards,
        limit=2,
        llm_client=FakeRerankLLM(),
        use_llm=True,
    )
    fallback = EvidenceReranker(mode="llm").rerank(
        question="液冷风险是什么？",
        cards=cards,
        limit=2,
        llm_client=FailingRerankLLM(),
        use_llm=True,
    )

    assert [card.title for card in reranked.cards] == ["B", "A"]
    assert reranked.metadata["source"] == "llm"
    assert [card.title for card in fallback.cards] == ["A", "B"]
    assert fallback.metadata["source"] == "heuristic"
    assert "error" in fallback.metadata


def test_qa_result_exposes_graphrag_payload_without_new_api_endpoint() -> None:
    graph = LocalKnowledgeGraph(
        entities=[],
        relations=[
            {
                "head_type": "Company",
                "head_name": "英维克",
                "relation": "HAS_PRODUCT",
                "tail_type": "Product",
                "tail_name": "液冷CDU",
                "evidence": "英维克提供 CDU、冷板和管路系统。",
                "source_title": "测试报告",
                "page": "1",
                "source_tier": "1",
                "section": "液冷",
            }
        ],
    )
    engine = QAEngine(csv_graph=graph, rag_index=None, research_memory=memory(), llm_client=None)

    result = engine.answer_question("液冷产业链有哪些上市公司？")

    graphrag = result["research_outputs"]["graphrag"]
    assert graphrag["route"]["kind"] == "company_exposure"
    assert graphrag["company_rankings"]
    assert result["diagnostics"]["graphrag"]["company_rankings"] >= 1
    assert any(edge.get("source_kind") == "graphrag_path" for edge in result["subgraph"])
