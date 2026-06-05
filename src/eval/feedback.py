"""JSONL persistence for user feedback."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FEEDBACK_DIR = ROOT_DIR / "data" / "feedback"
FEEDBACK_FILE = "feedback.jsonl"
FEEDBACK_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class InvalidFeedbackError(ValueError):
    """Raised when feedback payload is invalid."""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_feedback_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class FeedbackStore:
    def __init__(self, directory: Path | str = DEFAULT_FEEDBACK_DIR) -> None:
        self.directory = Path(directory)
        self.path = self.directory / FEEDBACK_FILE

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        feedback = normalize_feedback(payload)
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(feedback, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return feedback

    def list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        rows = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return rows[:limit] if limit is not None else rows


def normalize_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question") or "").strip()
    if not question:
        raise InvalidFeedbackError("Feedback question cannot be empty")
    human_score = payload.get("human_score")
    if human_score in ("", None):
        normalized_score = None
    else:
        try:
            normalized_score = int(human_score)
        except (TypeError, ValueError) as exc:
            raise InvalidFeedbackError("human_score must be an integer from 1 to 5") from exc
        if normalized_score < 1 or normalized_score > 5:
            raise InvalidFeedbackError("human_score must be an integer from 1 to 5")
    helpful = normalize_bool_or_none(payload.get("helpful"))
    evidence_supported = normalize_bool_or_none(payload.get("evidence_supported"))
    missing_answer = normalize_bool_or_none(payload.get("missing_answer"))
    note = str(payload.get("note") or "").strip()
    if helpful is None and evidence_supported is None and missing_answer is None and normalized_score is None and not note:
        raise InvalidFeedbackError("Feedback must include at least one rating or note")
    answer_hash = str(payload.get("answer_hash") or "").strip()
    if not answer_hash:
        answer_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]
    citation_ids = payload.get("citation_ids") or []
    if isinstance(citation_ids, str):
        citation_ids = [item.strip() for item in citation_ids.split(",") if item.strip()]
    if not isinstance(citation_ids, list):
        raise InvalidFeedbackError("citation_ids must be a list")
    feedback_id = str(payload.get("feedback_id") or new_feedback_id())
    if not FEEDBACK_ID_RE.match(feedback_id):
        raise InvalidFeedbackError("Invalid feedback id")
    return {
        "feedback_id": feedback_id,
        "created_at": str(payload.get("created_at") or now_iso()),
        "conversation_id": str(payload.get("conversation_id") or "").strip(),
        "turn_index": int(payload.get("turn_index") or 0),
        "question": question,
        "answer_hash": answer_hash,
        "helpful": helpful,
        "evidence_supported": evidence_supported,
        "missing_answer": missing_answer,
        "human_score": normalized_score,
        "note": note,
        "citation_ids": [str(item).strip() for item in citation_ids if str(item).strip()],
    }


def normalize_bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise InvalidFeedbackError("boolean feedback fields must be true, false, or null")
