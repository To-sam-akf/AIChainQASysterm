"""FastAPI backend for the React AIQASYS frontend."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.conversation_store import (
    ConversationNotFoundError,
    ConversationStore,
    InvalidConversationError,
    now_iso,
)
from src.curated_graph import DEFAULT_CURATED_DIR
from src.frontend_data import LocalKnowledgeGraph, RELATION_LABELS, render_svg_graph, subgraph_edges
from src.llm_client import load_dotenv
from src.qa_engine import QAEngine


REASONING_EFFORTS = ["low", "medium", "high"]
EXAMPLE_QUESTIONS = [
    "液冷产业链有哪些上市公司，各自处于什么环节？",
    "中际旭创和新易盛在光模块业务上的差异是什么？",
    "继续说它们的主要风险",
    "英维克液冷业务进展和主要风险是什么？",
    "AI算力产业链当前最大的瓶颈是什么？",
]


class ConversationCreateRequest(BaseModel):
    title: str = ""


class ConversationTitleRequest(BaseModel):
    title: str


class MessageCreateRequest(BaseModel):
    question: str
    thinking_enabled: bool | None = None
    reasoning_effort: str | None = None


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off", "disabled"}


def default_thinking_enabled() -> bool:
    model_name = os.getenv("LLM_MODEL", "").casefold()
    thinking_default = "deepseek" in os.getenv("LLM_BASE_URL", "").casefold() and (
        "reasoner" in model_name or "v4-pro" in model_name
    )
    return env_bool("LLM_THINKING_ENABLED", thinking_default)


def default_reasoning_effort() -> str:
    effort = os.getenv("LLM_REASONING_EFFORT", "low").strip() or "low"
    return effort if effort in REASONING_EFFORTS else "low"


@lru_cache(maxsize=1)
def _cached_conversation_store() -> ConversationStore:
    return ConversationStore()


@lru_cache(maxsize=1)
def _cached_qa_engine() -> QAEngine:
    return QAEngine.from_env()


@lru_cache(maxsize=1)
def _cached_knowledge_graph() -> LocalKnowledgeGraph:
    data_dir = Path(os.getenv("KG_DATA_DIR", str(DEFAULT_CURATED_DIR)))
    if not (data_dir / "entities.csv").exists() or not (data_dir / "relations.csv").exists():
        return LocalKnowledgeGraph.from_csvs()
    return LocalKnowledgeGraph.from_dir(data_dir)


async def get_conversation_store() -> ConversationStore:
    return _cached_conversation_store()


async def get_qa_engine() -> QAEngine:
    return _cached_qa_engine()


async def get_knowledge_graph() -> LocalKnowledgeGraph:
    return _cached_knowledge_graph()


def relation_label_options() -> dict[str, str]:
    return {"全部关系": "", **{label: relation for relation, label in RELATION_LABELS.items()}}


def http_error_from_store(exc: Exception) -> HTTPException:
    if isinstance(exc, ConversationNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    if isinstance(exc, InvalidConversationError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def public_stream_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    payload.pop("type", None)
    return payload


router = APIRouter(prefix="/api")


@router.get("/status")
async def api_status(
    engine: QAEngine = Depends(get_qa_engine),
    graph: LocalKnowledgeGraph = Depends(get_knowledge_graph),
) -> dict[str, Any]:
    entity_counts = graph.entity_counts()
    relation_counts = graph.relation_counts()
    return {
        "graph_backend": engine.status.graph_backend,
        "neo4j_enabled": engine.status.neo4j_enabled,
        "rag_enabled": engine.status.rag_enabled,
        "llm_enabled": engine.status.llm_enabled,
        "csv_graph_enabled": engine.status.csv_graph_enabled,
        "graph_data_dir": engine.status.graph_data_dir,
        "errors": {
            "graph": engine.status.graph_error,
            "rag": engine.status.rag_error,
            "llm": engine.status.llm_error,
        },
        "stats": {
            "companies": entity_counts.get("Company", 0),
            "reports": graph.reports_count(),
            "entities": len(graph.entities),
            "relations": len(graph.relations),
            "entity_counts": dict(entity_counts),
            "relation_counts": {RELATION_LABELS.get(key, key): value for key, value in relation_counts.items()},
        },
        "settings": {
            "thinking_enabled": default_thinking_enabled(),
            "reasoning_effort": default_reasoning_effort(),
            "reasoning_efforts": REASONING_EFFORTS,
        },
    }


@router.get("/examples")
async def api_examples() -> dict[str, list[str]]:
    return {"examples": EXAMPLE_QUESTIONS}


@router.get("/conversations")
async def list_conversations(
    limit: int = Query(50, ge=1, le=200),
    store: ConversationStore = Depends(get_conversation_store),
) -> dict[str, Any]:
    return {"conversations": store.list(limit=limit)}


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: ConversationCreateRequest,
    store: ConversationStore = Depends(get_conversation_store),
) -> dict[str, Any]:
    return store.create(title=request.title)


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),
) -> dict[str, Any]:
    try:
        return store.get(conversation_id)
    except Exception as exc:
        raise http_error_from_store(exc) from exc


@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: ConversationTitleRequest,
    store: ConversationStore = Depends(get_conversation_store),
) -> dict[str, Any]:
    try:
        return store.update_title(conversation_id, request.title)
    except Exception as exc:
        raise http_error_from_store(exc) from exc


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),
) -> Response:
    try:
        store.delete(conversation_id)
    except Exception as exc:
        raise http_error_from_store(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/conversations/{conversation_id}/messages")
async def append_message(
    conversation_id: str,
    request: MessageCreateRequest,
    store: ConversationStore = Depends(get_conversation_store),
    engine: QAEngine = Depends(get_qa_engine),
) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty")
    if request.reasoning_effort and request.reasoning_effort not in REASONING_EFFORTS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid reasoning effort")
    try:
        history = store.history_messages(conversation_id)
    except Exception as exc:
        raise http_error_from_store(exc) from exc

    thinking_enabled = default_thinking_enabled() if request.thinking_enabled is None else request.thinking_enabled
    reasoning_effort = request.reasoning_effort or (default_reasoning_effort() if thinking_enabled else "")
    result = engine.answer_question(
        question,
        conversation_history=history,
        thinking_enabled=thinking_enabled,
        reasoning_effort=reasoning_effort or None,
    )
    result["diagnostics"]["thinking_enabled"] = thinking_enabled
    result["diagnostics"]["reasoning_effort"] = reasoning_effort
    turn = {
        "created_at": now_iso(),
        "question": question,
        "answer": result["answer"],
        "thinking_enabled": thinking_enabled,
        "reasoning_effort": reasoning_effort,
        "result": result,
    }
    try:
        conversation = store.append_turn(conversation_id, turn)
    except Exception as exc:
        raise http_error_from_store(exc) from exc
    return {"conversation": conversation, "turn": turn}


@router.post("/conversations/{conversation_id}/messages/stream")
async def append_message_stream(
    conversation_id: str,
    request: MessageCreateRequest,
    store: ConversationStore = Depends(get_conversation_store),
    engine: QAEngine = Depends(get_qa_engine),
) -> StreamingResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty")
    if request.reasoning_effort and request.reasoning_effort not in REASONING_EFFORTS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid reasoning effort")
    try:
        history = store.history_messages(conversation_id)
    except Exception as exc:
        raise http_error_from_store(exc) from exc

    thinking_enabled = default_thinking_enabled() if request.thinking_enabled is None else request.thinking_enabled
    reasoning_effort = request.reasoning_effort or (default_reasoning_effort() if thinking_enabled else "")

    def generate_events() -> Any:
        result: dict[str, Any] | None = None
        try:
            if hasattr(engine, "answer_question_stream"):
                for event in engine.answer_question_stream(
                    question,
                    conversation_history=history,
                    thinking_enabled=thinking_enabled,
                    reasoning_effort=reasoning_effort or None,
                ):
                    event_type = str(event.get("type") or "message")
                    if event_type == "final":
                        result = dict(event.get("result") or {})
                        continue
                    if event_type in {"progress", "answer_delta"}:
                        yield sse_event(event_type, public_stream_payload(event))
            else:
                result = engine.answer_question(
                    question,
                    conversation_history=history,
                    thinking_enabled=thinking_enabled,
                    reasoning_effort=reasoning_effort or None,
                )
                yield sse_event("answer_delta", {"content": result.get("answer", "")})

            if result is None:
                raise RuntimeError("No answer result was generated")
            diagnostics = result.get("diagnostics")
            if not isinstance(diagnostics, dict):
                diagnostics = {}
                result["diagnostics"] = diagnostics
            diagnostics["thinking_enabled"] = thinking_enabled
            diagnostics["reasoning_effort"] = reasoning_effort
            turn = {
                "created_at": now_iso(),
                "question": question,
                "answer": result["answer"],
                "thinking_enabled": thinking_enabled,
                "reasoning_effort": reasoning_effort,
                "result": result,
            }
            conversation = store.append_turn(conversation_id, turn)
            yield sse_event("final", {"conversation": conversation, "turn": turn})
        except Exception as exc:
            yield sse_event("error", {"message": str(exc) or "生成失败"})

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: str,
    format: str = Query("md", pattern="^(md|json)$"),
    store: ConversationStore = Depends(get_conversation_store),
) -> Response:
    try:
        content, filename, media_type = store.export(conversation_id, format)
    except Exception as exc:
        raise http_error_from_store(exc) from exc
    ascii_fallback = "conversation.md" if format == "md" else "conversation.json"
    headers = {
        "Content-Disposition": f"attachment; filename={ascii_fallback}; filename*=UTF-8''{quote(filename)}"
    }
    return Response(content=content, media_type=media_type, headers=headers)


@router.get("/graph/summary")
async def graph_summary(graph: LocalKnowledgeGraph = Depends(get_knowledge_graph)) -> dict[str, Any]:
    entity_counts = graph.entity_counts()
    relation_counts = graph.relation_counts()
    return {
        "companies": entity_counts.get("Company", 0),
        "reports": graph.reports_count(),
        "entities": len(graph.entities),
        "relations": len(graph.relations),
        "entity_counts": dict(entity_counts),
        "relation_counts": {RELATION_LABELS.get(key, key): value for key, value in relation_counts.items()},
        "companies_options": graph.names_by_type("Company"),
        "technologies_options": graph.names_by_type("Technology"),
        "relation_options": relation_label_options(),
    }


@router.get("/graph/subgraph")
async def graph_subgraph(
    company: str = "",
    technology: str = "",
    relation_type: str = "",
    limit: int = Query(80, ge=1, le=200),
    graph: LocalKnowledgeGraph = Depends(get_knowledge_graph),
) -> dict[str, Any]:
    rows = graph.subgraph_relations(
        company=company,
        technology=technology,
        relation_type=relation_type,
        limit=limit,
    )
    edges = subgraph_edges(rows)
    return {
        "rows": rows,
        "edges": edges,
        "svg": render_svg_graph(edges),
    }


def create_app() -> FastAPI:
    load_dotenv()
    app = FastAPI(title="AIQASYS API", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()
