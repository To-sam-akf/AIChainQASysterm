"""JSONL persistence for evaluation reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_RUN_DIR = ROOT_DIR / "data" / "eval_runs"
EVAL_RUNS_FILE = "eval_runs.jsonl"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class EvalRunNotFoundError(KeyError):
    """Raised when an eval run id cannot be found."""


class InvalidEvalRunError(ValueError):
    """Raised when a run id or payload is invalid."""


class EvalRunStore:
    def __init__(self, directory: Path | str = DEFAULT_EVAL_RUN_DIR) -> None:
        self.directory = Path(directory)
        self.path = self.directory / EVAL_RUNS_FILE

    def save(self, run: dict[str, Any]) -> dict[str, Any]:
        payload = dict(run)
        self._validate_id(str(payload.get("run_id") or ""))
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return payload

    def list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        runs = list(self._latest_runs().values())
        runs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        if limit is not None:
            runs = runs[:limit]
        return [self._summary(run) for run in runs]

    def get(self, run_id: str) -> dict[str, Any]:
        self._validate_id(run_id)
        run = self._latest_runs().get(run_id)
        if run is None:
            raise EvalRunNotFoundError(run_id)
        return run

    def _latest_runs(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
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
            run_id = str(payload.get("run_id") or "")
            if not run_id:
                continue
            latest[run_id] = payload
        return latest

    @staticmethod
    def _summary(run: dict[str, Any]) -> dict[str, Any]:
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
        return {
            "run_id": str(run.get("run_id") or ""),
            "created_at": str(run.get("created_at") or ""),
            "dataset_name": str(run.get("dataset", {}).get("name") if isinstance(run.get("dataset"), dict) else ""),
            "dataset_hash": str(run.get("dataset", {}).get("hash") if isinstance(run.get("dataset"), dict) else ""),
            "cases": int(summary.get("cases") or 0),
            "passed": int(summary.get("passed") or 0),
            "failed": int(summary.get("failed") or 0),
            "overall_score": float(summary.get("overall_score") or 0.0),
            "metrics": metrics,
        }

    @staticmethod
    def _validate_id(run_id: str) -> None:
        if not run_id or not RUN_ID_RE.match(run_id):
            raise InvalidEvalRunError("Invalid eval run id")
