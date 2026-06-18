"""FastMCP server entrypoint for AIKA research tools."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import sys
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

import aika.aika_mcp.tools as tools
from aika.aika_core.backends.sqlite_backend import DEFAULT_PROFILE


Transport = Literal["stdio", "sse", "streamable-http"]


def create_server() -> FastMCP:
    server = FastMCP(
        "AIKA",
        instructions=(
            "AIKA exposes read-only, evidence-driven AI compute industry research tools. "
            "All tools return structured JSON envelopes with citations when evidence is available."
        ),
    )

    @server.tool(name="search_evidence", description="Search AIKA evidence spans and return citation-ready evidence cards.")
    def search_evidence_tool(
        query: str,
        top_k: int = 8,
        companies: list[str] | None = None,
        topics: list[str] | None = None,
        claim_types: list[str] | None = None,
        backend: str = "auto",
        home: str | None = None,
        profile: str = DEFAULT_PROFILE,
    ) -> dict[str, Any]:
        return tools.search_evidence(
            query=query,
            top_k=top_k,
            companies=companies or [],
            topics=topics or [],
            claim_types=claim_types or [],
            backend=backend,
            home=home,
            profile=profile,
        )

    @server.tool(name="search_claims", description="Search AIKA curated research claims and return structured claim records.")
    def search_claims_tool(
        query: str,
        top_k: int = 8,
        companies: list[str] | None = None,
        topics: list[str] | None = None,
        claim_types: list[str] | None = None,
        backend: str = "auto",
        home: str | None = None,
        profile: str = DEFAULT_PROFILE,
    ) -> dict[str, Any]:
        return tools.search_claims(
            query=query,
            top_k=top_k,
            companies=companies or [],
            topics=topics or [],
            claim_types=claim_types or [],
            backend=backend,
            home=home,
            profile=profile,
        )

    @server.tool(name="get_company_profile", description="Build a company profile from AIKA evidence and graph records.")
    def get_company_profile_tool(
        company: str,
        topic: str = "",
        backend: str = "auto",
        home: str | None = None,
        profile: str = DEFAULT_PROFILE,
    ) -> dict[str, Any]:
        return tools.get_company_profile(company=company, topic=topic, backend=backend, home=home, profile=profile)

    @server.tool(name="compare_companies", description="Compare companies on a topic using AIKA evidence.")
    def compare_companies_tool(
        companies: list[str],
        topic: str = "",
        backend: str = "auto",
        home: str | None = None,
        profile: str = DEFAULT_PROFILE,
    ) -> dict[str, Any]:
        return tools.compare_companies(companies=companies, topic=topic, backend=backend, home=home, profile=profile)

    @server.tool(name="query_industry_graph", description="Query AIKA local industry graph edges.")
    def query_industry_graph_tool(
        company: str = "",
        technology: str = "",
        relation_type: str = "",
        limit: int = 80,
        backend: str = "auto",
        home: str | None = None,
        profile: str = DEFAULT_PROFILE,
    ) -> dict[str, Any]:
        return tools.query_industry_graph(
            company=company,
            technology=technology,
            relation_type=relation_type,
            limit=limit,
            backend=backend,
            home=home,
            profile=profile,
        )

    @server.tool(name="build_research_brief", description="Build a deterministic AIKA research brief from local evidence.")
    def build_research_brief_tool(
        query: str = "",
        topic: str = "",
        backend: str = "auto",
        home: str | None = None,
        profile: str = DEFAULT_PROFILE,
    ) -> dict[str, Any]:
        return tools.build_research_brief(query=query, topic=topic, backend=backend, home=home, profile=profile)

    @server.tool(name="audit_evidence_gaps", description="Audit missing or weak evidence for a research subject.")
    def audit_evidence_gaps_tool(
        query: str = "",
        companies: list[str] | None = None,
        topic: str = "",
        backend: str = "auto",
        home: str | None = None,
        profile: str = DEFAULT_PROFILE,
    ) -> dict[str, Any]:
        return tools.audit_evidence_gaps(
            query=query,
            companies=companies or [],
            topic=topic,
            backend=backend,
            home=home,
            profile=profile,
        )

    @server.tool(name="run_research_task", description="Run AIKA's deterministic multi-stage research task pipeline.")
    def run_research_task_tool(
        topic: str,
        companies: list[str] | None = None,
        task_type: str = "research_brief",
        depth: str = "standard",
        require_citations: bool = True,
        backend: str = "auto",
        home: str | None = None,
        profile: str = DEFAULT_PROFILE,
    ) -> dict[str, Any]:
        return tools.run_research_task(
            task_type=task_type,
            topic=topic,
            companies=companies or [],
            depth=depth,
            require_citations=require_citations,
            backend=backend,
            home=home,
            profile=profile,
        )

    return server


def registered_tool_names(server: FastMCP) -> list[str]:
    manager = getattr(server, "_tool_manager", None)
    registered = getattr(manager, "_tools", {}) if manager is not None else {}
    return sorted(str(name) for name in registered)


def run_server(*, transport: Transport = "stdio") -> None:
    server = create_server()
    if transport == "stdio":
        _run_stdio_loop(server)
        return
    server.run(transport=transport)


def _run_stdio_loop(server: FastMCP) -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        _trace_stdio("in", line.rstrip("\n"))
        response = _handle_jsonrpc_message(server, line)
        if response is not None:
            _write_stdout_json(response)


def _handle_jsonrpc_message(server: FastMCP, line: str) -> dict[str, Any] | None:
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        return _jsonrpc_error(None, -32700, f"Parse error: {exc}")
    if not isinstance(message, dict):
        return _jsonrpc_error(None, -32600, "Invalid JSON-RPC message.")
    method = str(message.get("method") or "")
    request_id = message.get("id")
    if method.startswith("notifications/") or request_id is None:
        return None
    try:
        result = _dispatch_jsonrpc(server, method, message.get("params") or {})
    except ValueError as exc:
        return _jsonrpc_error(request_id, -32602, str(exc))
    except Exception as exc:  # pragma: no cover - defensive JSON-RPC boundary.
        return _jsonrpc_error(request_id, -32603, str(exc))
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _dispatch_jsonrpc(server: FastMCP, method: str, params: Any) -> Any:
    if method == "initialize":
        return _initialize_result(params)
    if method == "ping":
        return {}
    if method == "shutdown":
        return None
    if method == "prompts/list":
        return {"prompts": []}
    if method == "resources/list":
        return {"resources": []}
    if method == "tools/list":
        return {"tools": [_model_dump(tool) for tool in asyncio.run(server.list_tools())]}
    if method == "tools/call":
        if not isinstance(params, dict):
            raise ValueError("tools/call params must be an object.")
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if not name:
            raise ValueError("tools/call requires params.name.")
        if not isinstance(arguments, dict):
            raise ValueError("tools/call params.arguments must be an object.")
        return _call_tool_result(server, name, arguments)
    raise ValueError(f"Unsupported method: {method}")


def _initialize_result(params: Any) -> dict[str, Any]:
    requested_protocol = ""
    if isinstance(params, dict):
        requested_protocol = str(params.get("protocolVersion") or "")
    return {
        "protocolVersion": requested_protocol or "2025-06-18",
        "capabilities": {
            "experimental": {},
            "tools": {"listChanged": False},
        },
        "serverInfo": {"name": "AIKA", "version": _package_version()},
        "instructions": (
            "AIKA exposes read-only, evidence-driven AI compute industry research tools. "
            "All tools return structured JSON envelopes with citations when evidence is available."
        ),
    }


def _call_tool_result(server: FastMCP, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    content, structured = asyncio.run(server.call_tool(name, arguments))
    structured_payload = _jsonable(structured)
    return {
        "content": [_model_dump(item) for item in content],
        "structuredContent": structured_payload,
        "isError": isinstance(structured_payload, dict) and structured_payload.get("status") == "error",
    }


def _jsonrpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _write_stdout_json(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    _trace_stdio("out", line)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _trace_stdio(direction: str, payload: str) -> None:
    trace_path = os.environ.get("AIKA_MCP_TRACE")
    if not trace_path:
        return
    with open(trace_path, "a", encoding="utf-8") as handle:
        handle.write(f"{direction}: {payload}\n")


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return value
    return {"value": _jsonable(value)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump(by_alias=True, exclude_none=True))
    return str(value)


def _package_version() -> str:
    for distribution_name in ("aika-research-mcp", "aiqasys"):
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "0.1.0"


if __name__ == "__main__":
    run_server()
