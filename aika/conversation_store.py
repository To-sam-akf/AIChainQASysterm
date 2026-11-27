"""Persistent conversation storage for the web/API frontends."""

from __future__ import annotations

import json
import re
import uuid
import threading
from functools import wraps
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONVERSATION_DIR = ROOT_DIR / "data" / "conversations"
CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ConversationNotFoundError(KeyError):
    """Raised when a conversation id cannot be resolved to a stored file."""


class ConversationConflictError(ValueError):
    """An active task or a conflicting request prevents this operation."""


class InvalidConversationError(ValueError):
    """Raised when a conversation id or stored payload is invalid."""


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def conversation_locked(method):
    @wraps(method)
    def wrapped(self, conversation_id, *args, **kwargs):
        key = str(self.directory.resolve()) + "/" + conversation_id
        with _LOCKS_GUARD:
            lock = _LOCKS.setdefault(key, threading.RLock())
        with lock:
            return method(self, conversation_id, *args, **kwargs)
    return wrapped


@dataclass(frozen=True)
class ConversationSummary:
    id: str
    title: str
    created_at: str
    updated_at: str
    turn_count: int
    preview: str
    pinned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn_count": self.turn_count,
            "preview": self.preview,
            "pinned": self.pinned,
        }


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def short_text(value: str, limit: int = 42) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)] + "..."


def conversation_messages_from_turns(turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        if turn.get("status", "completed") != "completed":
            continue
        question = str(turn.get("question") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if question:
            messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})
    return messages


def conversation_markdown(conversation: dict[str, Any]) -> str:
    turns = list(conversation.get("turns") or [])
    lines = [
        "# AI算力产业链知识图谱问答记录",
        "",
        f"- 标题：{conversation.get('title') or title_from_turns(turns)}",
        f"- 保存时间：{now_iso()}",
        f"- 对话轮次：{len(turns)}",
        "",
    ]
    for index, turn in enumerate(turns, start=1):
        result = turn.get("result") or {}
        contextual_question = result.get("contextual_question", turn.get("question", ""))
        lines.extend(
            [
                f"## 第 {index} 轮",
                "",
                f"**用户问题**：{turn.get('question', '')}",
                "",
            ]
        )
        if contextual_question and contextual_question != turn.get("question"):
            lines.extend([f"**上下文改写**：{contextual_question}", ""])
        if turn.get("task_id"):
            lines.extend([f"**任务状态**：{turn.get('status', 'completed')}", ""])
            for event in turn.get("events", []):
                if event.get("type") == "role_event":
                    from aika.agents.roles import ROLE_BY_ID
                    name = ROLE_BY_ID.get(event["role_id"], {}).get("name", event["role_id"])
                    lines.append(f"- {name} · {event['status']}：{event.get('summary', '')}")
            lines.append("")
        lines.extend([f"**助手回答**：{turn.get('answer', '')}", ""])
    return "\n".join(lines)


def conversation_json(conversation: dict[str, Any]) -> str:
    payload = dict(conversation)
    payload["saved_at"] = now_iso()
    return json.dumps(payload, ensure_ascii=False, indent=2)


def title_from_turns(turns: list[dict[str, Any]]) -> str:
    first_question = next((str(turn.get("question") or "").strip() for turn in turns if turn.get("question")), "")
    return short_text(first_question, 28) if first_question else "新对话"


def preview_from_turns(turns: list[dict[str, Any]]) -> str:
    if not turns:
        return "还没有消息"
    latest = turns[-1]
    return short_text(str(latest.get("question") or latest.get("answer") or ""), 56)


class ConversationStore:
    def __init__(self, directory: Path | str = DEFAULT_CONVERSATION_DIR) -> None:
        self.directory = Path(directory)

    def list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        conversations = []
        for path in self._json_paths():
            try:
                conversations.append(self._load_path(path))
            except (InvalidConversationError, OSError, json.JSONDecodeError):
                continue
        conversations.sort(key=lambda item: (item.get("pinned") is True, str(item.get("updated_at") or item.get("created_at") or "")), reverse=True)
        if limit is not None:
            conversations = conversations[:limit]
        return [self._summary(item).to_dict() for item in conversations]

    def create(self, *, title: str = "") -> dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True)
        created_at = now_iso()
        conversation_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        conversation = {
            "id": conversation_id,
            "title": title.strip() or "新对话",
            "created_at": created_at,
            "updated_at": created_at,
            "pinned": False,
            "turns": [],
        }
        self._write(self._new_path(conversation_id), conversation)
        return conversation

    @conversation_locked
    def get(self, conversation_id: str) -> dict[str, Any]:
        path = self._path_for_id(conversation_id)
        if path is None:
            raise ConversationNotFoundError(conversation_id)
        return self._load_path(path)

    @conversation_locked
    def update_title(self, conversation_id: str, title: str) -> dict[str, Any]:
        title = title.strip()
        if not title:
            raise InvalidConversationError("Conversation title cannot be empty")
        conversation, path = self._load_with_path(conversation_id)
        conversation["title"] = short_text(title, 80)
        conversation["updated_at"] = now_iso()
        self._write(path, conversation)
        return conversation

    @conversation_locked
    def update_pinned(self, conversation_id: str, pinned: bool) -> dict[str, Any]:
        conversation, path = self._load_with_path(conversation_id)
        conversation["pinned"] = pinned
        self._write(path, conversation)
        return conversation

    @conversation_locked
    def delete(self, conversation_id: str) -> None:
        path = self._path_for_id(conversation_id)
        if path is None:
            raise ConversationNotFoundError(conversation_id)
        if any(t.get("status") in {"pending", "running"} for t in self.get(conversation_id)["turns"]):
            raise ConversationConflictError("任务仍在执行，完成后可删除会话")
        path.unlink()

    @conversation_locked
    def append_turn(self, conversation_id: str, turn: dict[str, Any]) -> dict[str, Any]:
        conversation, path = self._load_with_path(conversation_id)
        turns = list(conversation.get("turns") or [])
        turns.append(turn)
        conversation["turns"] = turns
        if not conversation.get("title") or conversation.get("title") == "新对话":
            conversation["title"] = title_from_turns(turns)
        conversation["updated_at"] = now_iso()
        self._write(path, conversation)
        return conversation

    @conversation_locked
    def begin_task(self, conversation_id: str, turn: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        conversation = self.get(conversation_id)
        for existing in conversation["turns"]:
            if existing.get("request_id") == turn["request_id"]:
                if existing.get("request_signature") != turn["request_signature"]:
                    raise ConversationConflictError("相同请求 ID 的内容不一致")
                return existing, False
        if any(t.get("status") in {"pending", "running"} for t in conversation["turns"]):
            raise ConversationConflictError("当前会话已有执行中的任务")
        self.append_turn(conversation_id, turn)
        return turn, True

    def task(self, conversation_id: str, task_id: str) -> dict[str, Any]:
        for turn in self.get(conversation_id)["turns"]:
            if turn.get("task_id") == task_id:
                return turn
        raise ConversationNotFoundError(task_id)

    @conversation_locked
    def update_task(self, conversation_id: str, task_id: str, *, event: dict[str, Any] | None = None, **updates: Any) -> dict[str, Any]:
        conversation, path = self._load_with_path(conversation_id)
        turn = next((t for t in conversation["turns"] if t.get("task_id") == task_id), None)
        if turn is None:
            raise ConversationNotFoundError(task_id)
        if turn.get("status") not in {"pending", "running"}:
            return turn
        if event is not None:
            events = turn.setdefault("events", [])
            if event.get("type") == "role_event":
                previous = next((e for e in reversed(events) if e.get("type") == "role_event" and e.get("step_id") == event["step_id"]), None)
                if previous and previous["status"] in {"completed", "failed", "degraded", "skipped"}:
                    return turn
            event = {**event, "event_id": uuid.uuid4().hex, "seq": len(events) + 1,
                     "conversation_id": conversation_id, "task_id": task_id, "created_at": now_iso()}
            events.append(event)
            if event["type"] == "answer_delta":
                turn["answer"] = turn.get("answer", "") + event["content"]
        turn.update(updates)
        conversation["updated_at"] = now_iso()
        self._write(path, conversation)
        return turn

    def recover_tasks(self) -> int:
        recovered = 0
        for summary in self.list():
            conversation_id = summary["id"]
            for turn in self.get(conversation_id)["turns"]:
                if turn.get("status") in {"pending", "running"}:
                    self.fail_task(conversation_id, turn["task_id"], "服务重启导致任务中断，请重新提交")
                    recovered += 1
        return recovered

    @conversation_locked
    def fail_task(self, conversation_id: str, task_id: str, message: str) -> None:
        turn = self.task(conversation_id, task_id)
        latest = {e["step_id"]: e for e in turn.get("events", []) if e.get("type") == "role_event"}
        for step_id, event in latest.items():
            if event["status"] in {"pending", "running"}:
                self.update_task(conversation_id, task_id, event={"type": "role_event", "role_id": event["role_id"], "step_id": step_id, "status": "failed", "summary": message})
        self.update_task(conversation_id, task_id, event={"type": "error", "message": message}, status="failed", error=message)

    def export(self, conversation_id: str, fmt: str) -> tuple[str, str, str]:
        conversation = self.get(conversation_id)
        safe_title = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", conversation.get("title") or conversation_id).strip("_")
        safe_title = safe_title or conversation_id
        fmt = fmt.casefold()
        if fmt == "md":
            return conversation_markdown(conversation), f"{safe_title}.md", "text/markdown; charset=utf-8"
        if fmt == "json":
            return conversation_json(conversation), f"{safe_title}.json", "application/json; charset=utf-8"
        raise InvalidConversationError("Export format must be md or json")

    def history_messages(self, conversation_id: str) -> list[dict[str, str]]:
        return conversation_messages_from_turns(self.get(conversation_id).get("turns") or [])

    def _load_with_path(self, conversation_id: str) -> tuple[dict[str, Any], Path]:
        path = self._path_for_id(conversation_id)
        if path is None:
            raise ConversationNotFoundError(conversation_id)
        return self._load_path(path), path

    def _summary(self, conversation: dict[str, Any]) -> ConversationSummary:
        turns = list(conversation.get("turns") or [])
        return ConversationSummary(
            id=str(conversation["id"]),
            title=str(conversation.get("title") or title_from_turns(turns)),
            created_at=str(conversation.get("created_at") or ""),
            updated_at=str(conversation.get("updated_at") or conversation.get("created_at") or ""),
            turn_count=len(turns),
            preview=preview_from_turns(turns),
            pinned=conversation.get("pinned") is True,
        )

    def _load_path(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidConversationError(f"Invalid conversation JSON: {path.name}") from exc
        if not isinstance(payload, dict):
            raise InvalidConversationError(f"Conversation payload must be an object: {path.name}")
        return self._normalize_payload(path, payload)

    def _normalize_payload(self, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        turns = payload.get("turns") or []
        if not isinstance(turns, list):
            raise InvalidConversationError(f"Conversation turns must be a list: {path.name}")
        stat = path.stat()
        fallback_time = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        conversation_id = str(payload.get("id") or self._id_from_path(path))
        created_at = str(
            payload.get("created_at")
            or next((turn.get("created_at") for turn in turns if isinstance(turn, dict) and turn.get("created_at")), "")
            or payload.get("saved_at")
            or fallback_time
        )
        updated_at = str(payload.get("updated_at") or payload.get("saved_at") or fallback_time)
        title = str(payload.get("title") or title_from_turns(turns))
        normalized = dict(payload)
        normalized.update(
            {
                "id": conversation_id,
                "title": title,
                "created_at": created_at,
                "updated_at": updated_at,
                "turns": turns,
                "pinned": payload.get("pinned") is True,
            }
        )
        return normalized

    def _json_paths(self) -> list[Path]:
        if not self.directory.exists():
            return []
        paths = []
        for path in self.directory.iterdir():
            if not path.is_file() or path.suffix != ".json":
                continue
            if ":Zone.Identifier" in path.name or path.name.endswith(".Zone.Identifier"):
                continue
            if path.name.startswith("."):
                continue
            paths.append(path)
        return paths

    def _path_for_id(self, conversation_id: str) -> Path | None:
        self._validate_id(conversation_id)
        candidates = [
            self.directory / f"conversation_{conversation_id}.json",
            self.directory / f"qa_conversation_{conversation_id}.json",
            self.directory / f"{conversation_id}.json",
        ]
        if conversation_id.startswith(("conversation_", "qa_conversation_")):
            candidates.insert(0, self.directory / f"{conversation_id}.json")
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _new_path(self, conversation_id: str) -> Path:
        self._validate_id(conversation_id)
        return self.directory / f"conversation_{conversation_id}.json"

    @staticmethod
    def _id_from_path(path: Path) -> str:
        stem = path.stem
        for prefix in ("conversation_", "qa_conversation_"):
            if stem.startswith(prefix):
                return stem[len(prefix) :]
        return stem

    @staticmethod
    def _validate_id(conversation_id: str) -> None:
        if not conversation_id or not CONVERSATION_ID_RE.match(conversation_id):
            raise InvalidConversationError("Invalid conversation id")

    @staticmethod
    def _write(path: Path, conversation: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(conversation_json(conversation), encoding="utf-8")
        tmp_path.replace(path)
