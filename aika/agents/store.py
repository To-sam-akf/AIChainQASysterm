"""JSONL persistence for agent task snapshots."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from aika.agents.models import AgentTask


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_AGENT_TASK_DIR = ROOT_DIR / "data" / "agent_tasks"
AGENT_TASKS_FILE = "agent_tasks.jsonl"
AGENT_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class AgentTaskNotFoundError(KeyError):
    """Raised when an agent task id cannot be resolved."""


class InvalidAgentTaskError(ValueError):
    """Raised when an agent task id or payload is invalid."""


def new_task_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class AgentTaskStore:
    def __init__(self, directory: Path | str = DEFAULT_AGENT_TASK_DIR) -> None:
        self.directory = Path(directory)
        self.path = self.directory / AGENT_TASKS_FILE

    def list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        tasks = list(self._latest_tasks().values())
        tasks.sort(key=lambda task: task.updated_at or task.created_at, reverse=True)
        if limit is not None:
            tasks = tasks[:limit]
        return [task.summary().to_dict() for task in tasks]

    def get(self, task_id: str) -> dict[str, Any]:
        self._validate_id(task_id)
        task = self._latest_tasks().get(task_id)
        if task is None:
            raise AgentTaskNotFoundError(task_id)
        return task.to_dict()

    def save(self, task: AgentTask | dict[str, Any]) -> dict[str, Any]:
        payload = task.to_dict() if isinstance(task, AgentTask) else dict(task)
        normalized = AgentTask.from_dict(payload)
        self._validate_id(normalized.task_id)
        if not normalized.task_type:
            raise InvalidAgentTaskError("Agent task type cannot be empty")
        if not normalized.goal.strip():
            raise InvalidAgentTaskError("Agent task goal cannot be empty")
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(normalized.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return normalized.to_dict()

    def create_pending(self, *, task_type: str, goal: str) -> AgentTask:
        task = AgentTask.new(task_id=new_task_id(), task_type=task_type, goal=goal)
        self.save(task)
        return task

    def _latest_tasks(self) -> dict[str, AgentTask]:
        latest: dict[str, AgentTask] = {}
        if not self.path.exists():
            return latest
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            task = AgentTask.from_dict(payload)
            if not task.task_id:
                continue
            latest[task.task_id] = task
        return latest

    @staticmethod
    def _validate_id(task_id: str) -> None:
        if not task_id or not AGENT_TASK_ID_RE.match(task_id):
            raise InvalidAgentTaskError("Invalid agent task id")
