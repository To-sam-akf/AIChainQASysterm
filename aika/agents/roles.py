"""Public research roles and execution-scoped event instrumentation.

Only execution code emits these events. Source documents cannot select roles or tools.
"""
from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable

ROLES = [
    {"id": "chief", "name": "首席研究员", "avatar": "首", "responsibility": "理解任务、安排分工并汇总研究结论", "capabilities": ["任务规划", "角色调度", "综合回答"]},
    {"id": "librarian", "name": "资料管理员", "avatar": "资", "responsibility": "收录和管理资料，跟踪真实处理状态", "capabilities": ["文本与网址收录", "资料查询", "确认删除", "重新处理"]},
    {"id": "analyst", "name": "研报分析师", "avatar": "报", "responsibility": "检索和阅读原文，提取观点与数据并提供引用", "capabilities": ["原文检索", "语义检索", "研报解读"]},
    {"id": "industry", "name": "产业研究员", "avatar": "产", "responsibility": "结合产业链和公司关系进行分析与比较", "capabilities": ["图谱查询", "产业证据检索", "公司比较"]},
    {"id": "reviewer", "name": "证据审查员", "avatar": "审", "responsibility": "核查证据、数据口径、冲突和资料缺口", "capabilities": ["覆盖检查", "冲突审查", "引用与答案核查"]},
]
ROLE_BY_ID = {role["id"]: role for role in ROLES}
QUERY_ROLES = {"rag": "analyst", "semantic": "analyst", "graph": "industry", "research": "industry", "graphrag": "industry"}
_sink: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar("role_event_sink", default=None)
_fallback_tools: ContextVar[bool] = ContextVar("fallback_role_tools", default=False)
_lead: ContextVar[str] = ContextVar("role_lead", default="chief")


def resolve_role(question: str, role_id: str | None) -> tuple[str, str]:
    match = re.match(r"^@([^\s：:,，]+)[\s：:,，]*", question)
    mentioned = None
    if match:
        mentioned = next((role["id"] for role in ROLES if match[1] in (role["name"], role["id"])), None)
        if mentioned is None:
            raise ValueError("未知内置角色，请从角色列表选择")
        question = question[match.end():].strip()
    if role_id and role_id not in ROLE_BY_ID:
        raise ValueError("未知内置角色，请从角色列表选择")
    return question, role_id or mentioned or "chief"


@contextmanager
def role_context(sink: Callable[[dict[str, Any]], None], lead: str = "chief", *, fallback_tools: bool = False):
    sink_token, lead_token = _sink.set(sink), _lead.set(lead)
    tool_token = _fallback_tools.set(fallback_tools)
    try:
        yield
    finally:
        _fallback_tools.reset(tool_token)
        _lead.reset(lead_token)
        _sink.reset(sink_token)


def lead_role() -> str:
    return _lead.get()


def lead_question(question: str) -> str:
    role = ROLE_BY_ID[lead_role()]
    if role["id"] == "chief":
        return question
    return f"{question}\n回答要求：由{role['name']}牵头，重点履行以下职责：{role['responsibility']}。保持证据和引用要求。"


def emit_role(role_id: str, step_id: str, status: str, summary: str, **details: Any) -> None:
    sink = _sink.get()
    if sink is not None:
        sink({"type": "role_event", "role_id": role_id, "step_id": step_id, "status": status, "summary": summary, **details})


@contextmanager
def role_step(role_id: str, summary: str, step_id: str | None = None):
    step_id = step_id or uuid.uuid4().hex
    emit_role(role_id, step_id, "running", summary)
    outcome: dict[str, Any] = {"status": "completed", "summary": summary + "，已完成"}
    try:
        yield outcome
    except Exception:
        emit_role(role_id, step_id, "failed", summary + "失败，请查看任务错误")
        raise
    else:
        emit_role(role_id, step_id, **outcome)


def fallback_tool_events() -> bool:
    return _fallback_tools.get()
