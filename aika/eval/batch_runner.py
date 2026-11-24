"""Crash-safe journals, exclusive run locks and immutable resume checks."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from aika.eval.experiment_artifacts import canonical, digest


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def replace_jsonl(path: Path, rows) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            for row in rows:
                handle.write(canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_journal(path: Path, *, repair_tail: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    rows = []
    offset = 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Journal rows must be objects")
        except (ValueError, UnicodeDecodeError):
            if repair_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                # Preserve the damaged bytes for diagnosis before repairing a crash tail.
                path.with_suffix(path.suffix + ".partial").write_bytes(line)
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                break
            raise ValueError(f"Malformed journal row: {path}:{index + 1}")
        rows.append(row)
        offset += len(line)
    if repair_tail and rows and data and not data.endswith(b"\n") and offset == len(data):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return rows


@contextmanager
def run_lock(directory: Path) -> Iterator[None]:
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process is already using this run") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def source_identity(archive: Path) -> str:
    with zipfile.ZipFile(archive) as bundle:
        return digest({name: hashlib.sha256(bundle.read(name)).hexdigest() for name in sorted(bundle.namelist())})


def latest_results(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest = {}
    for row in rows:
        key = (row["case_id"], row["variant"])
        if key in latest and latest[key].get("terminal", True) and not row.get("terminal", True):
            raise ValueError(f"Nonterminal entry cannot overwrite a completed result: {key}")
        latest[key] = row
    return latest
