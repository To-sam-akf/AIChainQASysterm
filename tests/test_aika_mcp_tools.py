from __future__ import annotations

import json
from pathlib import Path

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
    assert result["report_markdown"]
    assert result["evidence_cards"]
    assert isinstance(result["verification"], dict)
    assert "checks" in result["verification"]
    assert isinstance(result["evidence_gaps"], list)
    assert result["agent_trace"]
    for card in result["evidence_cards"]:
        assert card.get("citation_id") or card.get("citation_status") == "uncited"


def test_mcp_server_registers_required_tools() -> None:
    names = registered_tool_names(create_server())

    assert "search_evidence" in names
    assert "search_claims" in names
    assert "run_research_task" in names

