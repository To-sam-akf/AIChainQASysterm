from __future__ import annotations

import json
from pathlib import Path

import aika.llm_client as llm_client
from aika.aika_mcp import tools
from aika.aika_mcp.schemas import request_schema_catalog
from aika.aika_mcp.server import create_server, registered_tool_names


def test_request_schemas_are_json_serializable() -> None:
    catalog = request_schema_catalog()

    encoded = json.dumps(catalog, ensure_ascii=False)

    assert "search_evidence" in catalog
    assert "run_research_task" in catalog
    assert "query" in encoded


def test_missing_required_argument_returns_clear_error() -> None:
    result = tools.search_evidence({})

    assert result["status"] == "error"
    assert result["error"]["type"] == "validation_error"
    assert "query" in json.dumps(result["error"], ensure_ascii=False)


def test_search_evidence_returns_structured_cards_with_csv_fallback(tmp_path: Path) -> None:
    result = tools.search_evidence({"query": "液冷", "top_k": 4, "home": str(tmp_path / "missing")})

    assert result["status"] == "completed"
    assert result["tool"] == "search_evidence"
    assert result["meta"]["backend"] == "csv"
    assert result["meta"]["warnings"]
    assert 0 < len(result["evidence_cards"]) <= 4
    for card in result["evidence_cards"]:
        assert card["evidence"]
        assert card.get("citation_id") or card.get("citation_status") == "uncited"


def test_run_research_task_returns_report_evidence_and_verification(tmp_path: Path) -> None:
    result = tools.run_research_task({"topic": "液冷产业链", "home": str(tmp_path / "missing")})

    assert result["status"] == "completed"
    assert result["tool"] == "run_research_task"
    assert result["title"]
    assert result["report_type"] in {
        "evidence_coverage_audit",
        "preliminary_research_brief",
        "industry_research_report",
        "deep_research_report",
    }
    assert result["report_type_label"]
    assert isinstance(result["coverage"], dict)
    assert "coverage_score" in result["coverage"]
    assert "direct_claim_ratio" in result["coverage"]
    assert result["report_markdown"]
    assert result["report_markdown"].startswith("## 一页结论")
    assert "## 证据覆盖评级" in result["report_markdown"]
    assert "## 关键发现" in result["report_markdown"]
    assert "## Evidence Review" in result["report_markdown"]
    assert "## 证据附录" in result["report_markdown"]
    assert "结论与证据卡片" not in result["report_markdown"]
    assert result["evidence_cards"]
    assert result["conclusions"]
    assert result["evidence_links"]
    assert "counter_evidence" in result
    assert isinstance(result["verification"], dict)
    assert "checks" in result["verification"]
    assert isinstance(result["evidence_gaps"], list)
    assert result["agent_trace"]
    for card in result["evidence_cards"]:
        assert card.get("citation_id") or card.get("citation_status") == "uncited"
        assert card.get("source_title") or card.get("source")
        assert card.get("published_at") or card.get("freshness_status") == "unknown"
        assert "claim_type" in card
        assert "confidence" in card
        assert "counter_evidence_status" in card
    assert any(card.get("citation_id") and f"[{card['citation_id']}]" in result["report_markdown"] for card in result["evidence_cards"])
    first_cited = next(card for card in result["evidence_cards"] if card.get("citation_id"))
    appendix = result["report_markdown"].split("## 证据附录", 1)[1]
    assert f"[{first_cited['citation_id']}]" in appendix
    assert "Claim 类型" in appendix
    assert "置信度" in appendix
    assert "时效" in appendix
    assert "反证状态" in appendix


def test_build_research_brief_returns_report_type_and_coverage(tmp_path: Path) -> None:
    result = tools.build_research_brief({"topic": "液冷产业链", "home": str(tmp_path / "missing")})

    assert result["status"] == "completed"
    assert result["tool"] == "build_research_brief"
    assert result["title"]
    assert result["report_type"] in {
        "evidence_coverage_audit",
        "preliminary_research_brief",
        "industry_research_report",
        "deep_research_report",
    }
    assert result["report_type_label"]
    assert isinstance(result["coverage"], dict)
    assert "coverage_score" in result["coverage"]
    assert "direct_claim_ratio" in result["coverage"]


def test_llm_counter_audit_uses_configured_client(monkeypatch) -> None:
    class FakeClient:
        def chat_json(self, **kwargs):
            return {"status": "completed", "summary": "发现可能反证", "counter_evidence_ids": ["E1"]}

    monkeypatch.setenv("AIKA_EVIDENCE_UX_LLM_MODE", "auto")
    monkeypatch.setattr(llm_client, "OpenAICompatibleClient", FakeClient)
    context = tools.BackendContext(backend=object(), meta={}, warnings=[])

    result = tools._llm_counter_audit(
        context,
        subject="液冷",
        evidence_cards=[{"evidence_id": "E1", "evidence": "存在成本不确定性"}],
        claims=[],
    )

    assert result["source"] == "llm"
    assert result["counter_evidence_ids"] == ["E1"]


def test_llm_counter_audit_falls_back_on_auto_failure(monkeypatch) -> None:
    class BrokenClient:
        def __init__(self):
            raise ValueError("missing key")

    monkeypatch.setenv("AIKA_EVIDENCE_UX_LLM_MODE", "auto")
    monkeypatch.setattr(llm_client, "OpenAICompatibleClient", BrokenClient)
    context = tools.BackendContext(backend=object(), meta={}, warnings=[])

    result = tools._llm_counter_audit(
        context,
        subject="液冷",
        evidence_cards=[{"evidence_id": "E1", "evidence": "存在成本不确定性"}],
        claims=[],
    )

    assert result["status"] == "fallback"
    assert result["source"] == "rules"
    assert context.warnings


def test_mcp_server_registers_required_tools() -> None:
    names = registered_tool_names(create_server())

    assert "search_evidence" in names
    assert "search_claims" in names
    assert "run_research_task" in names
