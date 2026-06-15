"""LLM-powered automatic judgment for unjudged retrieval review rows.

Populates grade (0/1/2) and matched_unit_id for chunks the evaluation
dataset has not yet labelled, using batch LLM calls grouped by case.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.eval.rag_dataset import (
    DEFAULT_RAG_RETRIEVAL_BENCHMARK,
    EvidenceAlternative,
    EvidenceUnit,
    RagRetrievalCase,
    load_rag_retrieval_cases,
)
from src.llm_client import OpenAICompatibleClient

ROOT_DIR = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AutoJudgeConfig:
    model: str = ""
    temperature: float = 0.0
    max_chunks_per_batch: int = 8
    batch_delay_seconds: float = 0.5
    low_confidence_threshold: float = 0.7
    max_retries: int = 2


@dataclass
class ReviewRow:
    case_id: str
    category: str
    question: str
    chunk_id: str
    source_title: str
    page: str
    section: str
    company: str
    snippet: str
    rank: int = 0
    retriever: str = ""


@dataclass(frozen=True)
class AutoJudgment:
    case_id: str
    chunk_id: str
    grade: int                     # 0, 1, or 2
    matched_unit_id: str           # empty when grade == 0
    confidence: float              # 0.0 – 1.0
    reasoning: str
    needs_review: bool
    model: str
    judged_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)] + "..."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Evidence unit formatting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceUnitSummary:
    unit_id: str
    required: bool
    description: str

    @classmethod
    def from_evidence_unit(cls, unit: EvidenceUnit) -> "EvidenceUnitSummary":
        return cls(unit_id=unit.unit_id, required=unit.required, description=unit.description)


def format_evidence_units_text(units: list[EvidenceUnitSummary]) -> str:
    """Render evidence units for the LLM prompt."""
    lines: list[str] = []
    for unit in units:
        tag = "[required]" if unit.required else "[optional]"
        lines.append(f"- unit_id: \"{unit.unit_id}\" {tag}")
        if unit.description:
            lines.append(f"  描述: {unit.description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是评估数据集标注专家。你的任务是判断一个检索返回的文档片段（chunk）"
    "是否与给定问题的证据单元（evidence unit）相关，并给出相关度等级。\n\n"
    "判定标准：\n"
    "- grade=2：片段直接包含证据单元描述中的事实，可独立支撑回答\n"
    "- grade=1：片段与证据单元主题相关，但不直接包含所要求的具体事实\n"
    "- grade=0：片段与证据单元无关或完全不包含有用信息\n\n"
    "重要规则：\n"
    "1. 严格以片段实际内容为准，不要推断或假设\n"
    "2. 如果片段只包含财务报表附注、公司注册信息、审计意见等模板性文字，"
    "而证据单元要求的是业务描述或风险披露，应判定为 grade=0\n"
    "3. 如果片段包含证据单元描述中的关键事实，即使措辞不同，也应判定为 grade=2\n"
    "4. 如果片段部分涉及主题但缺少关键具体信息，判定为 grade=1\n"
    "5. 每个片段必须且只能匹配到一个 unit_id；如果完全不相关，unit_id 为空字符串\n"
    "6. confidence 反映你对判定结果的确信程度（0.0=完全不确定, 1.0=非常确信）"
)


def build_judge_user_prompt(
    question: str,
    evidence_units_text: str,
    chunks: list[dict[str, str]],
) -> str:
    """Build user prompt for a single case with multiple chunks."""
    chunks_text = "\n".join(
        f"[{idx}] chunk_id: {chunk['chunk_id']}\n"
        f"    来源: {chunk['source_title']}, 第{chunk['page']}页\n"
        f"    章节: {chunk['section']}\n"
        f"    公司: {chunk['company']}\n"
        f"    内容: {chunk['snippet']}"
        for idx, chunk in enumerate(chunks, start=1)
    )
    return (
        f"问题：{question}\n\n"
        f"证据单元列表：\n{evidence_units_text}\n\n"
        f"---\n待判定片段列表：\n{chunks_text}\n\n"
        "请输出 JSON：\n"
        '{"judgments": ['
        '{"chunk_id": "...", "grade": 0, "matched_unit_id": "", "confidence": 0.95, "reasoning": "判定理由"},'
        ' ...'
        "]}"
    )


# ---------------------------------------------------------------------------
#  LLM judgment extraction
# ---------------------------------------------------------------------------

def _extract_judgments(
    response: dict[str, Any],
    expected_chunk_ids: set[str],
) -> list[dict[str, Any]]:
    """Extract and validate the judgments array from an LLM JSON response."""
    raw = response.get("judgments")
    if not isinstance(raw, list) or not raw:
        raise ValueError("LLM response missing 'judgments' array")

    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not chunk_id or chunk_id not in expected_chunk_ids:
            continue
        if chunk_id in seen:
            continue
        seen.add(chunk_id)

        grade = _safe_int(item.get("grade"))
        if grade not in {0, 1, 2}:
            grade = 0
        matched_unit_id = str(item.get("matched_unit_id") or "").strip()
        if grade == 0:
            matched_unit_id = ""
        confidence = max(0.0, min(1.0, _safe_float(item.get("confidence"), 0.7)))
        reasoning = _short_text(str(item.get("reasoning") or ""), 200)

        valid.append({
            "chunk_id": chunk_id,
            "grade": grade,
            "matched_unit_id": matched_unit_id,
            "confidence": confidence,
            "reasoning": reasoning,
        })

    # Fill missing chunks as failed-to-judge
    for chunk_id in sorted(expected_chunk_ids - seen):
        valid.append({
            "chunk_id": chunk_id,
            "grade": 0,
            "matched_unit_id": "",
            "confidence": 0.0,
            "reasoning": "LLM未返回该chunk的判定结果",
        })

    return valid


# ---------------------------------------------------------------------------
#  Core auto-judge logic
# ---------------------------------------------------------------------------

def auto_judge_batch(
    llm_client: OpenAICompatibleClient,
    case: RagRetrievalCase,
    chunks: list[dict[str, str]],
    *,
    config: AutoJudgeConfig | None = None,
) -> list[AutoJudgment]:
    """Judge a batch of unjudged chunks for a single case in one LLM call.

    Args:
        llm_client: Configured OpenAI-compatible client.
        case: The evaluation case the chunks belong to.
        chunks: List of dicts with keys chunk_id, source_title, page,
                section, company, snippet.
        config: Optional configuration overrides.

    Returns:
        One AutoJudgment per input chunk.
    """
    if not chunks:
        return []

    cfg = config or AutoJudgeConfig()
    units = [EvidenceUnitSummary.from_evidence_unit(unit) for unit in case.evidence_units]
    evidence_text = format_evidence_units_text(units)
    user_prompt = build_judge_user_prompt(case.question, evidence_text, chunks)

    last_error: Exception | None = None
    for attempt in range(max(1, cfg.max_retries)):
        try:
            response = llm_client.chat_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=cfg.temperature,
            )
            expected_ids = {chunk["chunk_id"] for chunk in chunks}
            items = _extract_judgments(response, expected_ids)
            now = datetime.now().isoformat(timespec="seconds")
            model = cfg.model or llm_client.model
            return [
                AutoJudgment(
                    case_id=case.case_id,
                    chunk_id=item["chunk_id"],
                    grade=item["grade"],
                    matched_unit_id=item["matched_unit_id"],
                    confidence=item["confidence"],
                    reasoning=item["reasoning"],
                    needs_review=item["confidence"] < cfg.low_confidence_threshold,
                    model=model,
                    judged_at=now,
                )
                for item in items
            ]
        except Exception as exc:
            last_error = exc
            if attempt + 1 < cfg.max_retries:
                time.sleep(1.5 * (attempt + 1))

    # All retries exhausted — return failed judgments
    now = datetime.now().isoformat(timespec="seconds")
    model = cfg.model or llm_client.model
    return [
        AutoJudgment(
            case_id=case.case_id,
            chunk_id=chunk["chunk_id"],
            grade=0,
            matched_unit_id="",
            confidence=0.0,
            reasoning=f"LLM判定失败: {last_error}",
            needs_review=True,
            model=model,
            judged_at=now,
        )
        for chunk in chunks
    ]


# ---------------------------------------------------------------------------
#  Review queue processing
# ---------------------------------------------------------------------------

def load_review_rows(path: Path | str) -> list[ReviewRow]:
    """Load unjudged review rows from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"review queue file not found: {path}")

    rows: list[ReviewRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        rows.append(ReviewRow(
            case_id=str(payload.get("case_id") or ""),
            category=str(payload.get("category") or ""),
            question=str(payload.get("question") or ""),
            chunk_id=str(payload.get("chunk_id") or ""),
            source_title=str(payload.get("source_title") or ""),
            page=str(payload.get("page") or ""),
            section=str(payload.get("section") or ""),
            company=str(payload.get("company") or ""),
            snippet=str(payload.get("snippet") or ""),
            rank=_safe_int(payload.get("rank")),
            retriever=str(payload.get("retriever") or ""),
        ))
    return rows


def dedupe_review_rows(
    rows: list[ReviewRow],
    known_chunk_ids: set[str] | None = None,
) -> list[ReviewRow]:
    """Deduplicate review rows: keep one row per (case_id, chunk_id).

    Prefer the row with the longest snippet (lowest rank).

    Also excludes chunks already present in *known_chunk_ids* (e.g. already
    judged in the dataset).
    """
    if known_chunk_ids is None:
        known_chunk_ids = set()

    best: dict[tuple[str, str], ReviewRow] = {}
    for row in rows:
        key = (row.case_id, row.chunk_id)
        if not key[0] or not key[1]:
            continue
        if row.chunk_id in known_chunk_ids:
            continue
        if key in best:
            if len(row.snippet) > len(best[key].snippet):
                best[key] = row
        else:
            best[key] = row
    return list(best.values())


def group_by_case(rows: list[ReviewRow]) -> dict[str, list[ReviewRow]]:
    """Group review rows by case_id."""
    groups: dict[str, list[ReviewRow]] = {}
    for row in rows:
        groups.setdefault(row.case_id, []).append(row)
    return groups


def referenced_chunk_ids_from_cases(cases: list[RagRetrievalCase]) -> set[str]:
    """Collect all already-judged chunk IDs from a list of cases."""
    ids: set[str] = set()
    for case in cases:
        for unit in case.evidence_units:
            for alt in unit.alternatives:
                ids.add(alt.chunk_id)
        ids.update(case.judged_irrelevant_chunk_ids)
        ids.update(case.hard_negatives)
    return ids


def auto_judge_review_queue(
    llm_client: OpenAICompatibleClient,
    review_rows: list[ReviewRow],
    cases: list[RagRetrievalCase],
    *,
    config: AutoJudgeConfig | None = None,
    progress_callback: Any = None,
) -> list[AutoJudgment]:
    """Run auto-judgment over an entire review queue.

    Groups rows by case, batches chunks within each case, and calls the
    LLM once per batch.
    """
    cfg = config or AutoJudgeConfig()
    known_ids = referenced_chunk_ids_from_cases(cases)
    deduped = dedupe_review_rows(review_rows, known_chunk_ids=known_ids)
    grouped = group_by_case(deduped)

    case_map: dict[str, RagRetrievalCase] = {case.case_id: case for case in cases}

    all_judgments: list[AutoJudgment] = []
    total_batches = sum(
        (len(rows) + cfg.max_chunks_per_batch - 1) // cfg.max_chunks_per_batch
        for rows in grouped.values()
    )
    batch_idx = 0

    for case_id, rows in sorted(grouped.items()):
        case = case_map.get(case_id)
        if case is None:
            continue

        for i in range(0, len(rows), cfg.max_chunks_per_batch):
            batch = rows[i : i + cfg.max_chunks_per_batch]
            chunk_dicts = [
                {
                    "chunk_id": row.chunk_id,
                    "source_title": row.source_title,
                    "page": row.page,
                    "section": row.section,
                    "company": row.company,
                    "snippet": row.snippet,
                }
                for row in batch
            ]
            judgments = auto_judge_batch(llm_client, case, chunk_dicts, config=cfg)
            all_judgments.extend(judgments)

            batch_idx += 1
            if progress_callback is not None:
                progress_callback(batch_idx, total_batches, case_id)

            if i + cfg.max_chunks_per_batch < len(rows):
                time.sleep(cfg.batch_delay_seconds)

    return all_judgments


# ---------------------------------------------------------------------------
#  Save helpers
# ---------------------------------------------------------------------------

def save_judgments(judgments: list[AutoJudgment], path: Path | str) -> None:
    """Write auto judgments to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for judgment in judgments:
            fh.write(json.dumps(judgment.to_dict(), ensure_ascii=False) + "\n")


def load_judgments(path: Path | str) -> list[AutoJudgment]:
    """Read auto judgments from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"judgments file not found: {path}")

    judgments: list[AutoJudgment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        judgments.append(AutoJudgment(
            case_id=str(payload.get("case_id") or ""),
            chunk_id=str(payload.get("chunk_id") or ""),
            grade=_safe_int(payload.get("grade")),
            matched_unit_id=str(payload.get("matched_unit_id") or ""),
            confidence=_safe_float(payload.get("confidence"), 0.0),
            reasoning=str(payload.get("reasoning") or ""),
            needs_review=bool(payload.get("needs_review", True)),
            model=str(payload.get("model") or ""),
            judged_at=str(payload.get("judged_at") or ""),
        ))
    return judgments