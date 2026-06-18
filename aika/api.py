"""FastAPI backend for the React AIQASYS frontend."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from aika.agents.research_agent import ResearchAgent
from aika.agents.store import AgentTaskNotFoundError, AgentTaskStore, InvalidAgentTaskError
from aika.conversation_store import (
    ConversationNotFoundError,
    ConversationStore,
    InvalidConversationError,
    now_iso,
)
from aika.curated_graph import DEFAULT_CURATED_DIR
from aika.eval.feedback import FeedbackStore, InvalidFeedbackError
from aika.eval.store import EvalRunNotFoundError, EvalRunStore, InvalidEvalRunError
from aika.frontend_data import LocalKnowledgeGraph, RELATION_LABELS, render_svg_graph, subgraph_edges
from aika.llm_client import load_dotenv
from aika.qa_engine import QAEngine, normalize_agent_max_steps, normalize_agent_runner
from aika.research_claims import normalize_claim_review


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


class AgentTaskCreateRequest(BaseModel):
    task_type: str = "research_brief"
    goal: str
    thinking_enabled: bool | None = None
    reasoning_effort: str | None = None


class ClaimReviewRequest(BaseModel):
    claim_text: str | None = None
    claim_type: str | None = None
    topic: str | None = None
    companies: list[str] | str | None = None
    mechanism: str | None = None
    direction: str | None = None
    horizon: str | None = None
    metric: str | None = None
    value: str | None = None
    unit: str | None = None
    evidence_span: str | None = None
    confidence: str | None = None
    as_of_date: str | None = None
    exposure_level: str | None = None
    review_status: str | None = None
    reviewer_note: str | None = None
    quality_flags: str | None = None
    conflict_group_id: str | None = None


class FeedbackCreateRequest(BaseModel):
    conversation_id: str = ""
    turn_index: int = 0
    question: str
    answer_hash: str = ""
    helpful: bool | None = None
    evidence_supported: bool | None = None
    missing_answer: bool | None = None
    human_score: int | None = None
    note: str = ""
    citation_ids: list[str] = []


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


def default_agent_enabled() -> bool:
    return env_bool("QA_ENABLE_AGENT", True)


def default_agent_max_steps() -> int:
    return normalize_agent_max_steps(os.getenv("QA_AGENT_MAX_STEPS", "4"))


def default_agent_runner() -> str:
    return normalize_agent_runner(os.getenv("QA_AGENT_RUNNER", "langgraph"))


@lru_cache(maxsize=1)
def _cached_conversation_store() -> ConversationStore:
    return ConversationStore()


@lru_cache(maxsize=1)
def _cached_agent_task_store() -> AgentTaskStore:
    return AgentTaskStore(Path(os.getenv("AGENT_TASK_DIR", "data/agent_tasks")))


@lru_cache(maxsize=1)
def _cached_eval_run_store() -> EvalRunStore:
    return EvalRunStore(Path(os.getenv("EVAL_RUN_DIR", "data/eval_runs")))


@lru_cache(maxsize=1)
def _cached_feedback_store() -> FeedbackStore:
    return FeedbackStore(Path(os.getenv("FEEDBACK_DIR", "data/feedback")))


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


async def get_agent_task_store() -> AgentTaskStore:
    return _cached_agent_task_store()


async def get_eval_run_store() -> EvalRunStore:
    return _cached_eval_run_store()


async def get_feedback_store() -> FeedbackStore:
    return _cached_feedback_store()


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


def http_error_from_agent_store(exc: Exception) -> HTTPException:
    if isinstance(exc, AgentTaskNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent task not found")
    if isinstance(exc, InvalidAgentTaskError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


def http_error_from_eval_store(exc: Exception) -> HTTPException:
    if isinstance(exc, EvalRunNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Eval run not found")
    if isinstance(exc, InvalidEvalRunError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


def http_error_from_feedback_store(exc: Exception) -> HTTPException:
    if isinstance(exc, InvalidFeedbackError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def public_stream_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    payload.pop("type", None)
    return payload


def agent_task_markdown(task: dict[str, Any]) -> str:
    final_outputs = task.get("final_outputs") if isinstance(task.get("final_outputs"), dict) else {}
    research_outputs = task.get("research_outputs") if isinstance(task.get("research_outputs"), dict) else {}
    report = research_outputs.get("report") if isinstance(research_outputs, dict) else {}
    if not isinstance(report, dict):
        report = {}
    title = str(final_outputs.get("report_title") or report.get("title") or task.get("title") or "Agent 研究任务")
    report_markdown = str(final_outputs.get("report_markdown") or report.get("markdown") or final_outputs.get("qa_answer") or "")
    task_type = str(final_outputs.get("task_type") or task.get("task_type") or "")
    task_label = str(final_outputs.get("task_label") or task_type or "Agent 任务")
    lines = [
        f"# {title}",
        "",
        f"- 任务 ID：{task.get('task_id', '')}",
        f"- 任务类型：{task_label}（{task_type}）",
        f"- 状态：{task.get('status', '')}",
        f"- 用户目标：{task.get('goal', '')}",
        f"- 更新时间：{task.get('updated_at', '')}",
        "",
    ]
    task_outputs = research_outputs.get("task_outputs") if isinstance(research_outputs, dict) else {}
    task_output_markdown = agent_task_outputs_markdown(task_outputs if isinstance(task_outputs, dict) else {})
    if task_output_markdown:
        lines.extend([task_output_markdown, ""])
    if report_markdown:
        lines.extend([report_markdown, ""])
    gaps = research_outputs.get("evidence_gaps") if isinstance(research_outputs, dict) else []
    if isinstance(gaps, list) and gaps:
        lines.extend(["## 证据缺口", ""])
        for gap in gaps:
            if isinstance(gap, dict):
                lines.append(f"- {gap.get('gap', '')}（{gap.get('priority', '')}）")
        lines.append("")
    return "\n".join(lines)


def agent_task_outputs_markdown(task_outputs: dict[str, Any]) -> str:
    if not task_outputs:
        return ""
    schema_type = str(task_outputs.get("schema_type") or "")
    lines = ["## 任务结构化输出", "", f"- Schema：{schema_type or 'unknown'}", ""]
    title_map = {
        "compare_table": "公司对比表",
        "common_drivers": "共同驱动",
        "differences": "差异点",
        "risk_differences": "风险差异",
        "profile": "公司画像",
        "business_position": "业务卡位",
        "technology_products": "产品/技术",
        "indicators": "指标证据",
        "risks": "风险",
        "risk_checklist": "风险清单",
        "counter_evidence": "反证/边界",
        "follow_up_indicators": "跟踪指标",
        "evidence_gaps": "证据缺口",
        "missing_companies": "缺失公司证据",
        "missing_metrics": "缺失指标证据",
        "missing_risks": "缺失风险证据",
        "suggested_sources": "建议补充来源",
        "evidence_index": "证据索引",
    }
    for key, title in title_map.items():
        value = task_outputs.get(key)
        block = markdown_value_preview(value)
        if block:
            lines.extend([f"### {title}", "", block, ""])
    return "\n".join(lines).strip()


def markdown_value_preview(value: Any) -> str:
    if isinstance(value, dict):
        rows = value.get("rows")
        if isinstance(rows, list):
            return markdown_value_preview(rows)
        items = [(str(key), str(item)) for key, item in value.items() if item not in (None, "", [], {})]
        return "\n".join(f"- {key}：{item}" for key, item in items[:8])
    if isinstance(value, list):
        lines: list[str] = []
        for item in value[:8]:
            if isinstance(item, dict):
                text = str(
                    item.get("gap")
                    or item.get("risk")
                    or item.get("risks")
                    or item.get("evidence")
                    or item.get("indicator")
                    or item.get("business_position")
                    or item.get("suggested_source")
                    or item.get("company")
                    or item.get("scope")
                    or item
                )
                extra = str(item.get("citation_id") or item.get("priority") or "")
                lines.append(f"- {text}" + (f"（{extra}）" if extra else ""))
            elif item:
                lines.append(f"- {item}")
        return "\n".join(lines)
    if isinstance(value, str):
        return value
    return ""


router = APIRouter(prefix="/api")


@router.get("/status")
async def api_status(
    engine: QAEngine = Depends(get_qa_engine),
    graph: LocalKnowledgeGraph = Depends(get_knowledge_graph),
) -> dict[str, Any]:
    entity_counts = graph.entity_counts()
    relation_counts = graph.relation_counts()
    research_stats: dict[str, Any] = {}
    research_memory = getattr(engine, "research_memory", None)
    if research_memory is not None and hasattr(research_memory, "claim_stats"):
        research_stats = research_memory.claim_stats()
    return {
        "graph_backend": engine.status.graph_backend,
        "neo4j_enabled": engine.status.neo4j_enabled,
        "rag_enabled": engine.status.rag_enabled,
        "research_enabled": getattr(engine.status, "research_enabled", False),
        "embedding_enabled": getattr(engine.status, "embedding_enabled", False),
        "llm_enabled": engine.status.llm_enabled,
        "csv_graph_enabled": engine.status.csv_graph_enabled,
        "graph_data_dir": engine.status.graph_data_dir,
        "errors": {
            "graph": engine.status.graph_error,
            "rag": engine.status.rag_error,
            "research": getattr(engine.status, "research_error", ""),
            "embedding": getattr(engine.status, "embedding_error", ""),
            "llm": engine.status.llm_error,
        },
        "stats": {
            "companies": entity_counts.get("Company", 0),
            "reports": graph.reports_count(),
            "entities": len(graph.entities),
            "relations": len(graph.relations),
            "entity_counts": dict(entity_counts),
            "relation_counts": {RELATION_LABELS.get(key, key): value for key, value in relation_counts.items()},
            "research": research_stats,
        },
        "settings": {
            "thinking_enabled": default_thinking_enabled(),
            "reasoning_effort": default_reasoning_effort(),
            "reasoning_efforts": REASONING_EFFORTS,
            "agent_enabled": getattr(engine, "enable_agent", default_agent_enabled()),
            "agent_max_steps": getattr(engine, "agent_max_steps", default_agent_max_steps()),
            "agent_runner": getattr(engine, "agent_runner", default_agent_runner()),
            "multi_agent_max_workers": getattr(engine, "multi_agent_max_workers", 5),
            "multi_agent_max_llm_calls": getattr(engine, "multi_agent_max_llm_calls", 12),
            "multi_agent_task_timeout_seconds": getattr(engine, "multi_agent_task_timeout_seconds", 90.0),
            "hyde_enabled": getattr(engine, "enable_hyde", True),
            "hyde_query_mode": getattr(engine, "hyde_query_mode", "hybrid"),
            "hyde_max_chars": getattr(engine, "hyde_max_chars", 700),
        },
    }


@router.get("/examples")
async def api_examples() -> dict[str, list[str]]:
    return {"examples": EXAMPLE_QUESTIONS}


@router.get("/eval/runs")
async def list_eval_runs(
    limit: int = Query(20, ge=1, le=200),
    store: EvalRunStore = Depends(get_eval_run_store),
) -> dict[str, Any]:
    return {"runs": store.list(limit=limit)}


@router.get("/eval/runs/{run_id}")
async def get_eval_run(
    run_id: str,
    store: EvalRunStore = Depends(get_eval_run_store),
) -> dict[str, Any]:
    try:
        return store.get(run_id)
    except Exception as exc:
        raise http_error_from_eval_store(exc) from exc


@router.get("/feedback")
async def list_feedback(
    limit: int = Query(100, ge=1, le=500),
    store: FeedbackStore = Depends(get_feedback_store),
) -> dict[str, Any]:
    return {"feedback": store.list(limit=limit)}


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
async def create_feedback(
    request: FeedbackCreateRequest,
    store: FeedbackStore = Depends(get_feedback_store),
) -> dict[str, Any]:
    try:
        feedback = store.save(request.model_dump())
    except Exception as exc:
        raise http_error_from_feedback_store(exc) from exc
    return {"feedback": feedback}


@router.get("/agent/tasks")
async def list_agent_tasks(
    limit: int = Query(50, ge=1, le=200),
    store: AgentTaskStore = Depends(get_agent_task_store),
) -> dict[str, Any]:
    return {"tasks": store.list(limit=limit)}


@router.post("/agent/tasks", status_code=status.HTTP_201_CREATED)
async def create_agent_task(
    request: AgentTaskCreateRequest,
    store: AgentTaskStore = Depends(get_agent_task_store),
    engine: QAEngine = Depends(get_qa_engine),
) -> dict[str, Any]:
    goal = request.goal.strip()
    if not goal:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Agent task goal cannot be empty")
    if request.reasoning_effort and request.reasoning_effort not in REASONING_EFFORTS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid reasoning effort")

    thinking_enabled = default_thinking_enabled() if request.thinking_enabled is None else request.thinking_enabled
    reasoning_effort = request.reasoning_effort or (default_reasoning_effort() if thinking_enabled else "")
    try:
        task = ResearchAgent(engine, store).run(
            task_type=request.task_type,
            goal=goal,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort or None,
        )
    except Exception as exc:
        raise http_error_from_agent_store(exc) from exc
    return {"task": task}


@router.get("/agent/tasks/{task_id}")
async def get_agent_task(
    task_id: str,
    store: AgentTaskStore = Depends(get_agent_task_store),
) -> dict[str, Any]:
    try:
        return store.get(task_id)
    except Exception as exc:
        raise http_error_from_agent_store(exc) from exc


@router.get("/agent/tasks/{task_id}/export")
async def export_agent_task(
    task_id: str,
    format: str = Query("md", pattern="^(md|json)$"),
    store: AgentTaskStore = Depends(get_agent_task_store),
) -> Response:
    try:
        task = store.get(task_id)
    except Exception as exc:
        raise http_error_from_agent_store(exc) from exc
    fmt = format.casefold()
    safe_title = str(task.get("title") or task_id).strip() or task_id
    safe_title = "".join(char if char.isalnum() or char in "._-\u4e00-\u9fff" else "_" for char in safe_title).strip("_")
    safe_title = safe_title or task_id
    if fmt == "json":
        content = json.dumps(task, ensure_ascii=False, indent=2)
        filename = f"{safe_title}.json"
        media_type = "application/json; charset=utf-8"
    else:
        content = agent_task_markdown(task)
        filename = f"{safe_title}.md"
        media_type = "text/markdown; charset=utf-8"
    ascii_fallback = "agent_task.md" if fmt == "md" else "agent_task.json"
    headers = {
        "Content-Disposition": f"attachment; filename={ascii_fallback}; filename*=UTF-8''{quote(filename)}"
    }
    return Response(content=content, media_type=media_type, headers=headers)


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


@router.post("/research/claims/{claim_id}/review")
async def review_claim(
    claim_id: str,
    request: ClaimReviewRequest,
    engine: QAEngine = Depends(get_qa_engine),
) -> dict[str, Any]:
    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No claim updates provided")
    research_memory = getattr(engine, "research_memory", None)
    if research_memory is None or not hasattr(research_memory, "get_claim"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Research memory is not available")
    try:
        research_memory.get_claim(claim_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found") from exc

    review = normalize_claim_review(claim_id, updates, reviewer="frontend")
    try:
        claim = research_memory.review_claim(claim_id, review)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found") from exc
    return {"claim": claim, "review": review}


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


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    engine = _cached_qa_engine()
    try:
        yield
    finally:
        engine.close()
        _cached_qa_engine.cache_clear()


def create_app() -> FastAPI:
    load_dotenv()
    app = FastAPI(title="AIQASYS API", version="0.1.0", lifespan=app_lifespan)
    app.include_router(router)
    return app


app = create_app()
