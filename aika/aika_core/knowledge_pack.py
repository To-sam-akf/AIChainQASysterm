"""Knowledge-pack validation helpers for bundled public data."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aika.aika_core.data_paths import (
    CLAIMS_FILE,
    ENTITIES_FILE,
    EVIDENCE_SPANS_FILE,
    EXAMPLES_FILE,
    MANIFEST_FILE,
    RELATIONS_FILE,
    SEGMENT_DOSSIERS_FILE,
)


REQUIRED_KNOWLEDGE_PACK_FILES = [
    ENTITIES_FILE,
    RELATIONS_FILE,
    CLAIMS_FILE,
    EVIDENCE_SPANS_FILE,
    SEGMENT_DOSSIERS_FILE,
    MANIFEST_FILE,
    EXAMPLES_FILE,
]
REQUIRED_MANIFEST_FIELDS = [
    "source_report_id",
    "source_title",
    "source_url",
    "published_at",
    "source_type",
    "license_or_usage_note",
    "included_fields",
]
MAX_TRACEABLE_TEXT_CHARS = 700
COPYRIGHT_RISK_PATTERNS = (
    "all rights reserved",
    "proprietary and confidential",
    "no part of this publication",
    "without prior written permission",
)


@dataclass(frozen=True)
class KnowledgePackIssue:
    severity: str
    file: str
    message: str
    row: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "severity": self.severity,
            "file": self.file,
            "message": self.message,
        }
        if self.row is not None:
            result["row"] = self.row
        return result


@dataclass(frozen=True)
class KnowledgePackValidationResult:
    path: str
    counts: dict[str, int] = field(default_factory=dict)
    issues: list[KnowledgePackIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "counts": self.counts,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_knowledge_pack(
    path: str | Path,
    *,
    sample_size: int = 20,
    max_text_chars: int = MAX_TRACEABLE_TEXT_CHARS,
) -> KnowledgePackValidationResult:
    """Validate a redistributable AIKA knowledge pack."""
    base = Path(path).expanduser().resolve()
    issues: list[KnowledgePackIssue] = []
    counts: dict[str, int] = {}

    if not base.exists():
        issues.append(KnowledgePackIssue("error", ".", f"Knowledge pack path does not exist: {base}"))
        return KnowledgePackValidationResult(str(base), counts, issues)
    if not base.is_dir():
        issues.append(KnowledgePackIssue("error", ".", f"Knowledge pack path is not a directory: {base}"))
        return KnowledgePackValidationResult(str(base), counts, issues)

    for name in REQUIRED_KNOWLEDGE_PACK_FILES:
        if not (base / name).is_file():
            issues.append(KnowledgePackIssue("error", name, "Required file is missing."))

    manifest_rows, manifest_fields = _read_csv(base / MANIFEST_FILE, MANIFEST_FILE, issues)
    claims, claim_fields = _read_csv(base / CLAIMS_FILE, CLAIMS_FILE, issues)
    evidence_rows, evidence_fields = _read_csv(base / EVIDENCE_SPANS_FILE, EVIDENCE_SPANS_FILE, issues)
    relations, relation_fields = _read_csv(base / RELATIONS_FILE, RELATIONS_FILE, issues)
    entities, _ = _read_csv(base / ENTITIES_FILE, ENTITIES_FILE, issues)
    dossiers = _read_jsonl(base / SEGMENT_DOSSIERS_FILE, SEGMENT_DOSSIERS_FILE, issues)
    examples = _read_jsonl(base / EXAMPLES_FILE, EXAMPLES_FILE, issues)

    counts.update(
        {
            "manifest": len(manifest_rows),
            "claims": len(claims),
            "evidence_spans": len(evidence_rows),
            "relations": len(relations),
            "entities": len(entities),
            "dossiers": len(dossiers),
            "examples": len(examples),
            "sampled_evidence": min(max(0, sample_size), len(evidence_rows)),
        }
    )

    _validate_manifest(manifest_rows, manifest_fields, issues)
    manifest_ids = {str(row.get("source_report_id") or "").strip() for row in manifest_rows}
    manifest_ids.discard("")

    claim_ids = {str(row.get("claim_id") or "").strip() for row in claims}
    evidence_ids = {str(row.get("evidence_id") or "").strip() for row in evidence_rows}

    _require_fields(CLAIMS_FILE, claim_fields, ["claim_id", "source_report_id", "source_title", "evidence_span"], issues)
    _require_fields(EVIDENCE_SPANS_FILE, evidence_fields, ["evidence_id", "claim_id", "source_report_id", "source_title", "page", "section", "text"], issues)
    _require_fields(RELATIONS_FILE, relation_fields, ["relation_id", "source_report_id", "source_title", "page", "section", "evidence"], issues)

    for index, row in enumerate(claims, start=2):
        _validate_source_ref(CLAIMS_FILE, row, index, manifest_ids, issues)
        _validate_text_cell(CLAIMS_FILE, "evidence_span", row, index, max_text_chars, issues)
        if not str(row.get("claim_id") or "").strip():
            issues.append(KnowledgePackIssue("error", CLAIMS_FILE, "claim_id is required.", index))

    for index, row in enumerate(evidence_rows, start=2):
        _validate_source_ref(EVIDENCE_SPANS_FILE, row, index, manifest_ids, issues)
        if not str(row.get("source_title") or "").strip():
            issues.append(KnowledgePackIssue("error", EVIDENCE_SPANS_FILE, "source_title is required.", index))
        if not str(row.get("page") or "").strip() and not str(row.get("section") or "").strip():
            issues.append(KnowledgePackIssue("error", EVIDENCE_SPANS_FILE, "page or section is required.", index))
        claim_id = str(row.get("claim_id") or "").strip()
        if not claim_id or claim_id not in claim_ids:
            issues.append(KnowledgePackIssue("error", EVIDENCE_SPANS_FILE, f"Unknown claim_id: {claim_id or '-'}", index))
        _validate_text_cell(EVIDENCE_SPANS_FILE, "text", row, index, max_text_chars, issues)

    for index, row in enumerate(relations, start=2):
        _validate_source_ref(RELATIONS_FILE, row, index, manifest_ids, issues)
        _validate_text_cell(RELATIONS_FILE, "evidence", row, index, max_text_chars, issues)

    known_support_ids = claim_ids | evidence_ids
    for index, dossier in enumerate(dossiers, start=1):
        if not isinstance(dossier, dict):
            issues.append(KnowledgePackIssue("error", SEGMENT_DOSSIERS_FILE, "JSONL row must be an object.", index))
            continue
        for value in dossier.get("evidence_ids") or []:
            support_id = str(value or "").strip()
            if support_id and support_id not in known_support_ids:
                issues.append(KnowledgePackIssue("error", SEGMENT_DOSSIERS_FILE, f"Unknown evidence_ids item: {support_id}", index))

    return KnowledgePackValidationResult(str(base), counts, issues)


def format_validation_result(result: KnowledgePackValidationResult) -> str:
    status = "PASS" if result.ok else "FAIL"
    lines = [f"Knowledge pack validation: {status}", f"Path: {result.path}"]
    if result.counts:
        counts = ", ".join(f"{key}={value}" for key, value in sorted(result.counts.items()))
        lines.append(f"Counts: {counts}")
    if result.issues:
        lines.append("Issues:")
        for issue in result.issues:
            row = f" row {issue.row}" if issue.row is not None else ""
            lines.append(f"- [{issue.severity}] {issue.file}{row}: {issue.message}")
    return "\n".join(lines)


def _read_csv(path: Path, label: str, issues: list[KnowledgePackIssue]) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    try:
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            fields = list(reader.fieldnames or [])
            if not fields:
                issues.append(KnowledgePackIssue("error", label, "CSV header is missing."))
                return [], []
            return [dict(row) for row in reader], fields
    except csv.Error as exc:
        issues.append(KnowledgePackIssue("error", label, f"CSV parse error: {exc}"))
    except OSError as exc:
        issues.append(KnowledgePackIssue("error", label, f"Could not read file: {exc}"))
    return [], []


def _read_jsonl(path: Path, label: str, issues: list[KnowledgePackIssue]) -> list[Any]:
    rows: list[Any] = []
    if not path.exists():
        return rows
    try:
        with path.open(encoding="utf-8") as file:
            for index, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    issues.append(KnowledgePackIssue("error", label, f"JSONL parse error: {exc.msg}", index))
    except OSError as exc:
        issues.append(KnowledgePackIssue("error", label, f"Could not read file: {exc}"))
    return rows


def _validate_manifest(rows: list[dict[str, str]], fields: list[str], issues: list[KnowledgePackIssue]) -> None:
    _require_fields(MANIFEST_FILE, fields, REQUIRED_MANIFEST_FIELDS, issues)
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        source_id = str(row.get("source_report_id") or "").strip()
        if not source_id:
            issues.append(KnowledgePackIssue("error", MANIFEST_FILE, "source_report_id is required.", index))
            continue
        if source_id in seen:
            issues.append(KnowledgePackIssue("error", MANIFEST_FILE, f"Duplicate source_report_id: {source_id}", index))
        seen.add(source_id)
        for field_name in REQUIRED_MANIFEST_FIELDS:
            if not str(row.get(field_name) or "").strip():
                issues.append(KnowledgePackIssue("error", MANIFEST_FILE, f"{field_name} is required.", index))


def _require_fields(
    label: str,
    fields: list[str],
    required: list[str],
    issues: list[KnowledgePackIssue],
) -> None:
    missing = [name for name in required if name not in fields]
    if missing:
        issues.append(KnowledgePackIssue("error", label, f"Missing required columns: {', '.join(missing)}"))


def _validate_source_ref(
    label: str,
    row: dict[str, str],
    index: int,
    manifest_ids: set[str],
    issues: list[KnowledgePackIssue],
) -> None:
    source_id = str(row.get("source_report_id") or "").strip()
    if not source_id:
        issues.append(KnowledgePackIssue("error", label, "source_report_id is required.", index))
    elif source_id not in manifest_ids:
        issues.append(KnowledgePackIssue("error", label, f"source_report_id is not in manifest: {source_id}", index))


def _validate_text_cell(
    label: str,
    field_name: str,
    row: dict[str, str],
    index: int,
    max_text_chars: int,
    issues: list[KnowledgePackIssue],
) -> None:
    text = str(row.get(field_name) or "")
    if len(text) > max_text_chars:
        issues.append(
            KnowledgePackIssue(
                "error",
                label,
                f"{field_name} is too long: {len(text)} chars > {max_text_chars}",
                index,
            )
        )
    normalized = re.sub(r"\s+", " ", text).casefold()
    if any(pattern in normalized for pattern in COPYRIGHT_RISK_PATTERNS):
        issues.append(KnowledgePackIssue("error", label, f"{field_name} contains copyright boilerplate.", index))
