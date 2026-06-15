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
    "营业收入",
    "净利润",
    "毛利率",
    "研发费用",
    "市场份额",
    "出货量",
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
    if re.search(r"(?:\.|·|…){5,}\s*\d{1,4}$", stripped):
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

# 去页码、目录、免责声明、空行等噪声
def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\x00", "")
    text = "".join(char for char in text if char in "\n\t" or ord(char) >= 32)
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
    page_units: list[dict[str, Any]] = []
    inherited_section = ""
    for page in pages:
        cleaned = clean_text(page.get("text", ""))
        detected_section = detect_section(cleaned) if cleaned else ""
        if detected_section:
            inherited_section = detected_section
        page_score = relevance_score(cleaned)
        if cleaned:
            page_units.append(
                {
                    "score": page_score,
                    "content_type": "text",
                    "page": page,
                    "text": cleaned,
                    "section": inherited_section,
                }
            )
        for table in page.get("tables", []) or []:
            table_title = str(table.get("title") or "").strip()
            table_text = " ".join(
                str(value or "")
                for value in (
                    table_title,
                    " ".join(str(cell or "") for cell in table.get("headers", [])),
                    table.get("markdown", ""),
                )
            )
            table_section = detect_section(table_title)
            if not table_section and is_valid_section_title(table_title):
                table_section = table_title
            page_units.append(
                {
                    "score": max(page_score, relevance_score(table_text)),
                    "content_type": "table",
                    "page": page,
                    "table": table,
                    "section": table_section or inherited_section,
                }
            )
    selected = [item for item in page_units if item["score"] > 0]
    if not selected and include_all_if_no_relevant:
        selected = page_units
    chunks: list[dict[str, Any]] = []
    for unit in selected:
        page = unit["page"]
        if unit["content_type"] == "table":
            chunks.extend(build_table_chunks(page, unit["table"], unit["section"], max_chars=max_chars))
            continue
        for index, text in enumerate(split_text(unit["text"], max_chars=max_chars, overlap=overlap), start=1):
            section = detect_section(text) or unit["section"]
            chunk_id = stable_id("chunk", page["report_id"], page["page"], index, text[:80])
            chunks.append(
                build_chunk_record(
                    page,
                    chunk_id=chunk_id,
                    section=section,
                    text=text,
                    content_type="text",
                )
            )
    return chunks


def build_table_chunks(
    page: dict[str, Any],
    table: dict[str, Any],
    section: str,
    *,
    max_chars: int,
) -> list[dict[str, Any]]:
    headers = [normalize_table_value(cell) for cell in table.get("headers", [])]
    rows = [
        [normalize_table_value(cell) for cell in row]
        for row in table.get("rows", [])
        if isinstance(row, list) and any(normalize_table_value(cell) for cell in row)
    ]
    title = normalize_table_value(table.get("title", ""))
    table_id = str(table.get("table_id") or stable_id("table", page["report_id"], page.get("page", ""), title))
    row_groups = split_table_rows(headers, rows, title=title, max_chars=max_chars)
    if not row_groups:
        markdown = str(table.get("markdown") or "").strip()
        if not markdown:
            return []
        row_groups = [(0, 0, markdown)]

    chunks = []
    for index, (row_start, row_end, text) in enumerate(row_groups, start=1):
        chunk_id = stable_id("chunk", table_id, row_start, row_end, index, text[:80])
        chunks.append(
            build_chunk_record(
                page,
                chunk_id=chunk_id,
                section=section,
                text=text,
                content_type="table",
                table_id=table_id,
                table_title=title,
                table_row_start=row_start,
                table_row_end=row_end,
            )
        )
    return chunks


def split_table_rows(
    headers: list[str],
    rows: list[list[str]],
    *,
    title: str = "",
    max_chars: int = 2800,
) -> list[tuple[int, int, str]]:
    if not headers and not rows:
        return []
    if not rows:
        return [(0, 0, render_table_markdown(headers, [], title=title))]

    groups: list[tuple[int, int, str]] = []
    current_rows: list[list[str]] = []
    current_start = 1
    for row_number, row in enumerate(rows, start=1):
        candidate_rows = [*current_rows, row]
        candidate = render_table_markdown(headers, candidate_rows, title=title)
        if current_rows and len(candidate) > max_chars:
            groups.append(
                (
                    current_start,
                    row_number - 1,
                    render_table_markdown(headers, current_rows, title=title),
                )
            )
            current_rows = [row]
            current_start = row_number
        else:
            current_rows = candidate_rows
    if current_rows:
        groups.append(
            (
                current_start,
                current_start + len(current_rows) - 1,
                render_table_markdown(headers, current_rows, title=title),
            )
        )
    return groups


def render_table_markdown(headers: list[str], rows: list[list[str]], *, title: str = "") -> str:
    column_count = max([len(headers), *(len(row) for row in rows)], default=0)
    if column_count == 0:
        return ""
    padded_headers = [headers[index] if index < len(headers) else "" for index in range(column_count)]
    lines = []
    if title:
        lines.extend([f"### {title}", ""])
    lines.extend(
        [
            "| " + " | ".join(markdown_escape(cell) for cell in padded_headers) + " |",
            "| " + " | ".join("---" for _ in range(column_count)) + " |",
        ]
    )
    for row in rows:
        padded = [row[index] if index < len(row) else "" for index in range(column_count)]
        lines.append("| " + " | ".join(markdown_escape(cell) for cell in padded) + " |")
    return "\n".join(lines)


def normalize_table_value(value: Any) -> str:
    value = str(value or "").replace("\u3000", " ").replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return " ".join(line for line in lines if line)


def markdown_escape(value: str) -> str:
    return str(value or "").replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def build_chunk_record(
    page: dict[str, Any],
    *,
    chunk_id: str,
    section: str,
    text: str,
    content_type: str,
    table_id: str = "",
    table_title: str = "",
    table_row_start: int = 0,
    table_row_end: int = 0,
) -> dict[str, Any]:
    return {
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
        "context": build_chunk_context(
            page,
            section,
            content_type=content_type,
            table_title=table_title,
        ),
        "content_type": content_type,
        "table_id": table_id,
        "table_title": table_title,
        "table_row_start": table_row_start,
        "table_row_end": table_row_end,
        "text": text,
    }


def build_chunk_context(
    page: dict[str, Any],
    section: str,
    *,
    content_type: str = "text",
    table_title: str = "",
) -> str:
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
    if content_type == "table":
        parts.append("内容：表格")
    if table_title:
        parts.append(f"表格：{table_title}")
    return "；".join(str(part) for part in parts if part)


def build_chunks_file(parsed_jsonl: Path, output_dir: Path = CHUNKS_DIR, *, max_chars: int = 2800) -> Path:
    pages = load_jsonl(parsed_jsonl)
    chunks = build_chunks_from_pages(pages, max_chars=max_chars)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / parsed_jsonl.name
    write_jsonl(output_path, chunks)
    return output_path
