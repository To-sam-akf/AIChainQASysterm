import json
import threading
import time

from aika.frontend_data import LocalKnowledgeGraph
from aika.llm_client import ChatTextResult
from aika.qa_engine import QAEngine
from aika.rag_index import RagHit
from aika.research_claims import ResearchHit


class ConcurrencyProbe:
    def __init__(self, delay: float = 0.12) -> None:
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def run(self) -> None:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1


class SlowGraphClient:
    def __init__(self, probe: ConcurrencyProbe) -> None:
        self.probe = probe

    def run_read_query(self, cypher, params, *, limit):
        del cypher, params, limit
        self.probe.run()
        return [
            {
                "company": "浪潮信息",
                "company_labels": ["Company"],
                "relation": "HAS_PRODUCT",
                "target": "AI服务器",
                "target_labels": ["Product"],
                "evidence": "浪潮信息布局AI服务器。",
                "source": "测试报告",
                "page": "1",
                "section": "主营业务",
                "source_tier": "1",
            }
        ]


class SlowRagIndex:
    def __init__(self, probe: ConcurrencyProbe) -> None:
        self.probe = probe

    def search(self, question, *, top_k, filters):
        del question, top_k, filters
        self.probe.run()
        return []


class SlowResearchMemory:
    def __init__(self, probe: ConcurrencyProbe) -> None:
        self.probe = probe

    def search(self, question, plan, *, limit):
        del question, plan, limit
        self.probe.run()
        return []


def test_centralized_runner_executes_three_retrievers_concurrently() -> None:
    probe = ConcurrencyProbe()
    engine = QAEngine(
        graph_client=SlowGraphClient(probe),
        rag_index=SlowRagIndex(probe),
        research_memory=SlowResearchMemory(probe),
        llm_client=None,
    )

    result = engine.answer_question("浪潮信息有哪些AI服务器产品？")

    assert probe.peak >= 3
    assert result["diagnostics"]["multi_agent"]["peak_concurrency"] >= 3
    assert result["diagnostics"]["multi_agent"]["rounds"][0]["elapsed_ms"] < 300
    assert result["evidence_cards"][0]["candidate_ids"]
    assert result["evidence_cards"][0]["retrieval_agent"] == "graph"


class SingleRagIndex:
    def search(self, question, *, top_k, filters):
        del question, top_k, filters
        return [
            RagHit(
                chunk_id="chunk-1",
                report_id="report-1",
                source_title="测试报告",
                source_tier="1",
                source_type="annual_report",
                page="8",
                section="主营业务",
                content_type="text",
                table_id="",
                company="浪潮信息",
                text="浪潮信息布局AI服务器。",
                snippet="浪潮信息布局AI服务器。",
                score=10.0,
            )
        ]


class StructuredMultiAgentLLM:
    def chat_json(self, *, system_prompt, user_prompt, temperature=0.0, **kwargs):
        del temperature, kwargs
        payload = json.loads(user_prompt)
        if "中心调度 Agent" in system_prompt:
            return {
                "assignments": [
                    {
                        "agent_type": "rag",
                        "query": payload["question"],
                        "filters": {},
                        "coverage_goals": payload["required_coverage"],
                    }
                ]
            }
        if "查询 Agent" in system_prompt:
            return {"query": payload["question"], "filters": {}}
        if "证据审核 Agent" in system_prompt:
            ids = [item["candidate_id"] for item in payload["candidates"]]
            return {
                "accepted_ids": ids,
                "rejected_ids": [],
                "missing_coverage": [],
                "conflicts": [],
            }
        if "证据归纳 Agent" in system_prompt:
            candidate = payload["candidates"][0]
            return {
                "cards": [
                    {
                        "candidate_ids": [candidate["candidate_id"]],
                        "title": "AI服务器原文证据",
                        "reason": "直接支撑问题",
                    }
                ]
            }
        return {}

    def chat_text(self, *, system_prompt, user_prompt, temperature=0.2, **kwargs):
        del system_prompt, user_prompt, temperature, kwargs
        return "核心判断：浪潮信息布局AI服务器 [E1]。"

    def chat_messages(self, *, messages, temperature=0.2, **kwargs):
        del messages, temperature, kwargs
        return ChatTextResult(content="核心判断：浪潮信息布局AI服务器 [E1]。")


def test_supervisor_dynamically_selects_agent_and_summary_keeps_original_evidence() -> None:
    engine = QAEngine(
        rag_index=SingleRagIndex(),
        llm_client=StructuredMultiAgentLLM(),
        multi_agent_max_llm_calls=12,
    )

    result = engine.answer_question("浪潮信息有哪些AI服务器产品？")

    assignments = result["diagnostics"]["multi_agent"]["rounds"][0]["assignments"]
    assert [item["agent_type"] for item in assignments] == ["rag"]
    assert result["evidence_cards"][0]["evidence"] == "浪潮信息布局AI服务器。"
    assert result["evidence_cards"][0]["candidate_ids"]
    assert result["evidence_cards"][0]["retrieval_agent"] == "rag"
    assert result["diagnostics"]["llm_calls"]["total"] <= 12


class ConditionalRiskMemory:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, question, plan, *, limit):
        del question, plan, limit
        self.calls += 1
        if self.calls == 1:
            return []
        return [
            ResearchHit(
                kind="claim",
                title="英维克风险",
                text="英维克液冷业务存在客户需求波动风险。",
                topic="液冷",
                company="英维克",
                claim_type="risk",
                source="测试报告",
                page="1",
                score=10.0,
            )
        ]


def test_review_gap_triggers_sequential_supplement_rounds() -> None:
    memory = ConditionalRiskMemory()
    engine = QAEngine(research_memory=memory, llm_client=None)

    result = engine.answer_question("英维克液冷业务主要风险是什么？")

    rounds = result["diagnostics"]["multi_agent"]["rounds"]
    assert len(rounds) >= 2
    assert [item["round"] for item in rounds[:2]] == [0, 1]
    assert result["diagnostics"]["agent_stop_reason"] == "evidence_sufficient"
