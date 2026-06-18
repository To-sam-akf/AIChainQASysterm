"""Shared data models for persisted agent tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal


AgentTaskStatus = Literal["pending", "running", "completed", "failed"]
VALID_AGENT_TASK_STATUSES = {"pending", "running", "completed", "failed"}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def short_text(value: str, limit: int = 48) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)] + "..."


@dataclass(frozen=True)
class AgentToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    cost_level: str = "low"
    safety_level: str = "read_only"
    requires_llm: bool = False
    cacheable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentToolSpec":
        return cls(
            name=str(payload.get("name") or ""),
            description=str(payload.get("description") or ""),
            input_schema=dict(payload.get("input_schema") or {}),
            output_schema=dict(payload.get("output_schema") or {}),
            timeout=float(payload.get("timeout") or 30.0),
            cost_level=str(payload.get("cost_level") or "low"),
            safety_level=str(payload.get("safety_level") or "read_only"),
            requires_llm=bool(payload.get("requires_llm", False)),
            cacheable=bool(payload.get("cacheable", True)),
        )


@dataclass(frozen=True)
class AgentStep:
    step: int
    phase: str
    thought: str = ""
    action: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observation: str = ""
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentStep":
        return cls(
            step=int(payload.get("step") or 0),
            phase=str(payload.get("phase") or ""),
            thought=str(payload.get("thought") or ""),
            action=str(payload.get("action") or ""),
            tool_calls=list(payload.get("tool_calls") or []),
            observation=str(payload.get("observation") or ""),
            status=str(payload.get("status") or "completed"),
        )


@dataclass(frozen=True)
class AgentState:
    task_id: str
    task_type: str
    goal: str
    status: AgentTaskStatus
    created_at: str
    updated_at: str
    plan: dict[str, Any] = field(default_factory=dict)
    current_step: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    evidence_pool: list[dict[str, Any]] = field(default_factory=list)
    selected_evidence: list[dict[str, Any]] = field(default_factory=list)
    evidence_gaps: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    final_outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **updates: Any) -> "AgentState":
        payload = self.to_dict()
        payload.update(updates)
        payload["updated_at"] = updates.get("updated_at") or now_iso()
        return AgentState.from_dict(payload)

    @classmethod
    def new(cls, *, task_id: str, task_type: str, goal: str) -> "AgentState":
        created_at = now_iso()
        return cls(
            task_id=task_id,
            task_type=task_type,
            goal=goal,
            status="pending",
            created_at=created_at,
            updated_at=created_at,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentState":
        status = str(payload.get("status") or "pending")
        if status not in VALID_AGENT_TASK_STATUSES:
            status = "pending"
        return cls(
            task_id=str(payload.get("task_id") or ""),
            task_type=str(payload.get("task_type") or ""),
            goal=str(payload.get("goal") or ""),
            status=status,  # type: ignore[arg-type]
            created_at=str(payload.get("created_at") or now_iso()),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or now_iso()),
            plan=dict(payload.get("plan") or {}),
            current_step=int(payload.get("current_step") or 0),
            steps=list(payload.get("steps") or []),
            tool_calls=list(payload.get("tool_calls") or []),
            evidence_pool=list(payload.get("evidence_pool") or []),
            selected_evidence=list(payload.get("selected_evidence") or []),
            evidence_gaps=list(payload.get("evidence_gaps") or []),
            stop_reason=str(payload.get("stop_reason") or ""),
            budget=dict(payload.get("budget") or {}),
            verification=dict(payload.get("verification") or {}),
            final_outputs=dict(payload.get("final_outputs") or {}),
            errors=[str(error) for error in list(payload.get("errors") or [])],
        )


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    task_type: str
    goal: str
    title: str
    status: AgentTaskStatus
    created_at: str
    updated_at: str
    plan: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    evidence_cards: list[dict[str, Any]] = field(default_factory=list)
    research_outputs: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    final_outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **updates: Any) -> "AgentTask":
        payload = self.to_dict()
        payload.update(updates)
        payload["updated_at"] = updates.get("updated_at") or now_iso()
        return AgentTask.from_dict(payload)

    @classmethod
    def new(cls, *, task_id: str, task_type: str, goal: str) -> "AgentTask":
        created_at = now_iso()
        return cls(
            task_id=task_id,
            task_type=task_type,
            goal=goal,
            title=short_text(goal, 56) or task_type,
            status="pending",
            created_at=created_at,
            updated_at=created_at,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentTask":
        status = str(payload.get("status") or "pending")
        if status not in VALID_AGENT_TASK_STATUSES:
            status = "pending"
        return cls(
            task_id=str(payload.get("task_id") or ""),
            task_type=str(payload.get("task_type") or ""),
            goal=str(payload.get("goal") or ""),
            title=str(payload.get("title") or short_text(str(payload.get("goal") or ""), 56)),
            status=status,  # type: ignore[arg-type]
            created_at=str(payload.get("created_at") or now_iso()),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or now_iso()),
            plan=dict(payload.get("plan") or {}),
            steps=list(payload.get("steps") or []),
            tool_calls=list(payload.get("tool_calls") or []),
            evidence_cards=list(payload.get("evidence_cards") or []),
            research_outputs=dict(payload.get("research_outputs") or {}),
            diagnostics=dict(payload.get("diagnostics") or {}),
            errors=[str(error) for error in list(payload.get("errors") or [])],
            final_outputs=dict(payload.get("final_outputs") or {}),
        )

    def summary(self) -> "AgentTaskSummary":
        return AgentTaskSummary(
            task_id=self.task_id,
            task_type=self.task_type,
            title=self.title,
            goal=self.goal,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            evidence_card_count=len(self.evidence_cards),
            evidence_gap_count=int(self.final_outputs.get("evidence_gap_count") or 0),
            preview=short_text(str(self.final_outputs.get("qa_answer") or self.goal), 72),
        )


@dataclass(frozen=True)
class AgentTaskSummary:
    task_id: str
    task_type: str
    title: str
    goal: str
    status: AgentTaskStatus
    created_at: str
    updated_at: str
    evidence_card_count: int = 0
    evidence_gap_count: int = 0
    preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
