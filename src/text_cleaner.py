"""Text cleaning and chunking for KG extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.extraction_schema import load_jsonl, stable_id, write_jsonl


ROOT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_DIR = ROOT_DIR / "data" / "chunks"

RELEVANT_TERMS = (
    "业务概要",
    "主营业务",
    "核心竞争力",
    "管理层讨论",
    "经营情况",
    "研发投入",
    "风险因素",
    "财务指标",
    "产业链",
    "AI",
    "人工智能",
    "算力",
    "服务器",
    "光模块",
    "液冷",
    "芯片",
    "数据中心",
    "产品",
    "技术",
)

SECTION_PATTERNS = (
    re.compile(r"第[一二三四五六七八九十]+节\s*([^\n]{2,40})"),
    re.compile(r"^\s*[一二三四五六七八九十]+[、.．]\s*([^\n]{2,40})", re.M),
    re.compile(r"^\s*\d+(?:\.\d+)*[、.．]\s*([^\n]{2,40})", re.M),
    re.compile(r"^\s*[(（]?[一二三四五六七八九十\d]+[)）]\s*([^\n]{2,40})", re.M),
)

DISCLAIMER_LINES = (
    "请务必阅读正文之后的免责声明",
    "请务必仔细阅读正文后的",
    "法律声明及风险提示",
    "评级说明及声明",
    "LEGAL NOTICE",
    "ALL RIGHTS RESERVED",
    "MERCHANTABILITY",
    "FITNESS FOR A PARTICULAR PURPOSE",
    "NO LICENSE",
    "GOVERNING DOCUMENTS",
)

TABLE_FRAGMENT_PATTERNS = (
    re.compile(r"^[\d,.%％+\-—/ ]+$"),
    re.compile(r"^(?:图|表)\s*\d+[:：]?\s*$"),
    re.compile(r"^\d+(?:\.\d+)?%$"),
)


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    if re.search(r"\.{5,}\s*\d{1,4}$", stripped):
        return True
    if stripped in {"目录", "释义", "重要提示"}:
        return True
    if any(term in stripped for term in DISCLAIMER_LINES):
        return True
    if any(term in stripped.upper() for term in DISCLAIMER_LINES):
        return True
    if len(stripped) < 4 and re.fullmatch(r"[-_=—]+", stripped):
        return True
    return False


def is_table_fragment(value: str) -> bool:
    stripped = str(value or "").strip()
    if not stripped:
        return True
    return any(pattern.fullmatch(stripped) for pattern in TABLE_FRAGMENT_PATTERNS)


def is_valid_section_title(value: str) -> bool:
    title = re.sub(r"\s+", "", str(value or "").strip())
    if len(title) < 2 or len(title) > 42:
        return False
    if is_table_fragment(title):
        return False
    if "%" in title or "％" in title:
        return False
    if re.fullmatch(r"[\d一二三四五六七八九十]+", title):
        return False
    digit_count = sum(char.isdigit() for char in title)
    if digit_count and digit_count / max(len(title), 1) > 0.45:
        return False
    if any(token in title.casefold() for token in ("www.", "http", ".com", ".cn")):
        return False
    return True


def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if not is_noise_line(line)]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def detect_section(text: str) -> str:
    for pattern in SECTION_PATTERNS:
        match = pattern.search(text)
        if match and is_valid_section_title(match.group(1)):
            return match.group(1).strip()
    for line in text.splitlines()[:12]:
        candidate = line.strip()
        if is_valid_section_title(candidate) and any(term in candidate for term in RELEVANT_TERMS):
            return candidate
    for term in RELEVANT_TERMS:
        if term in text:
            return term
    return ""


def relevance_score(text: str) -> int:
    return sum(2 if term in {"算力", "产业链", "AI", "人工智能"} else 1 for term in RELEVANT_TERMS if term in text)


def split_text(text: str, max_chars: int = 2800, overlap: int = 200) -> list[str]:
    overlap = max(0, min(overlap, max_chars // 3))
    step = max(1, max_chars - overlap)
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + max_chars].strip())
                start += step
            continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap and len(current) > overlap else ""
            current = f"{tail}\n\n{paragraph}" if tail else paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_chunks_from_pages(
    pages: list[dict[str, Any]],
    *,
    max_chars: int = 2800,
    overlap: int = 200,
    include_all_if_no_relevant: bool = True,
) -> list[dict[str, Any]]:
    page_units = []
    inherited_section = ""
    for page in pages:
        cleaned = clean_text(page.get("text", ""))
        if not cleaned:
            continue
        score = relevance_score(cleaned)
        detected_section = detect_section(cleaned)
        if detected_section:
            inherited_section = detected_section
        page_units.append((score, page, cleaned, inherited_section))
    selected = [item for item in page_units if item[0] > 0]
    if not selected and include_all_if_no_relevant:
        selected = page_units
    chunks: list[dict[str, Any]] = []
    for _, page, cleaned, inherited in selected:
        for index, text in enumerate(split_text(cleaned, max_chars=max_chars, overlap=overlap), start=1):
            section = detect_section(text) or inherited
            chunk_id = stable_id("chunk", page["report_id"], page["page"], index, text[:80])
            context = build_chunk_context(page, section)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "report_id": page["report_id"],
                    "kind": page.get("kind", ""),
                    "company": page.get("company", ""),
                    "stock_code": page.get("stock_code", ""),
                    "year": page.get("year", ""),
                    "source_title": page.get("source_title", ""),
                    "source_url": page.get("source_url", ""),
                    "source_tier": page.get("source_tier", ""),
                    "source_type": page.get("source_type", ""),
                    "page": page.get("page", ""),
                    "section": section,
                    "context": context,
                    "text": text,
                }
            )
    return chunks


def build_chunk_context(page: dict[str, Any], section: str) -> str:
    parts = []
    if page.get("source_title"):
        parts.append(f"报告：{page.get('source_title')}")
    if page.get("company"):
        parts.append(f"公司：{page.get('company')}")
    if page.get("year"):
        parts.append(f"年份：{page.get('year')}")
    if section:
        parts.append(f"章节：{section}")
    if page.get("page"):
        parts.append(f"页码：{page.get('page')}")
    return "；".join(str(part) for part in parts if part)


def build_chunks_file(parsed_jsonl: Path, output_dir: Path = CHUNKS_DIR, *, max_chars: int = 2800) -> Path:
    pages = load_jsonl(parsed_jsonl)
    chunks = build_chunks_from_pages(pages, max_chars=max_chars)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / parsed_jsonl.name
    write_jsonl(output_path, chunks)
    return output_path
