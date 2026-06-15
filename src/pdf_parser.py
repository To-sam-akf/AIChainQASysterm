"""Layout-aware PDF parsing utilities for stage 2."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import fitz

from src.extraction_schema import stable_id, write_jsonl


ROOT_DIR = Path(__file__).resolve().parents[1]
PARSED_TEXT_DIR = ROOT_DIR / "data" / "parsed_text"
OCR_MODES = {"auto", "off", "force"}
HEADER_FOOTER_ZONE_RATIO = 0.12


def read_downloaded_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [row for row in rows if row.get("status") == "downloaded" and row.get("local_path")]


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def clean_page_text(text: str) -> str:
    """Normalize control characters and soft line wraps without losing paragraphs."""
    text = str(text or "").replace("\u3000", " ").replace("\x00", "")
    text = "".join(char for char in text if char in "\n\t" or ord(char) >= 32)
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in paragraph.splitlines()]
        lines = [line for line in lines if line]
        if lines:
            paragraphs.append(join_soft_lines(lines))
    return "\n\n".join(paragraphs).strip()


def join_soft_lines(lines: list[str]) -> str:
    result = ""
    for line in lines:
        if not result:
            result = line
            continue
        if re.search(r"[A-Za-z]-$", result) and re.match(r"^[A-Za-z]", line):
            result = result[:-1] + line
        elif is_cjk_boundary(result, line):
            result += line
        else:
            result += " " + line
    return result


def is_cjk_boundary(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return bool(re.match(r"[\u3400-\u9fff]", right[0]) or re.search(r"[\u3400-\u9fff]$", left))


def normalize_cell(value: Any) -> str:
    text = str(value or "").replace("\u3000", " ").replace("\x00", "")
    text = "".join(char for char in text if char in "\n\t" or ord(char) >= 32)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return clean_page_text("\n".join(line for line in lines if line))


def markdown_escape(value: str) -> str:
    return str(value or "").replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def table_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    column_count = max([len(headers), *(len(row) for row in rows)], default=0)
    if column_count == 0:
        return ""
    normalized_headers = [headers[index] if index < len(headers) else "" for index in range(column_count)]
    lines = [
        "| " + " | ".join(markdown_escape(cell) for cell in normalized_headers) + " |",
        "| " + " | ".join("---" for _ in range(column_count)) + " |",
    ]
    for row in rows:
        padded = [row[index] if index < len(row) else "" for index in range(column_count)]
        lines.append("| " + " | ".join(markdown_escape(cell) for cell in padded) + " |")
    return "\n".join(lines)


def normalize_table_data(
    raw_rows: Iterable[Iterable[Any]],
    raw_headers: Iterable[Any] | None = None,
) -> tuple[list[str], list[list[str]]]:
    rows = [[normalize_cell(cell) for cell in row] for row in raw_rows]
    headers = [normalize_cell(cell) for cell in (raw_headers or [])]
    column_count = max([len(headers), *(len(row) for row in rows)], default=0)
    if column_count == 0:
        return [], []
    headers.extend([""] * (column_count - len(headers)))
    for row in rows:
        row.extend([""] * (column_count - len(row)))

    keep_columns = [
        index
        for index in range(column_count)
        if headers[index] or any(row[index] for row in rows)
    ]
    headers = [headers[index] for index in keep_columns]
    rows = [[row[index] for index in keep_columns] for row in rows]
    rows = [row for row in rows if any(row)]
    if not headers and not rows:
        return [], []

    meaningful_headers = sum(bool(cell) for cell in headers)
    if meaningful_headers == 0 and rows:
        headers = rows.pop(0)
    elif rows and canonical_row(rows[0]) == canonical_row(headers):
        rows.pop(0)
    return headers, rows


def canonical_row(row: Iterable[str]) -> tuple[str, ...]:
    return tuple(re.sub(r"\s+", "", str(cell or "")).casefold() for cell in row)


def parse_pdf_pages(
    report: dict[str, str],
    *,
    max_pages: int | None = None,
    ocr_mode: str = "auto",
    ocr_language: str = "chi_sim+eng",
    min_text_chars: int = 80,
) -> list[dict[str, Any]]:
    if ocr_mode not in OCR_MODES:
        raise ValueError(f"Unsupported OCR mode: {ocr_mode}")
    if min_text_chars < 0:
        raise ValueError("min_text_chars must be non-negative")

    pdf_path = resolve_project_path(report["local_path"])
    working_pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        limit = min(total_pages, max_pages) if max_pages else total_pages
        for index in range(limit):
            page = doc.load_page(index)
            working_pages.append(
                extract_page(
                    page,
                    report,
                    page_number=index + 1,
                    total_pages=total_pages,
                    ocr_mode=ocr_mode,
                    ocr_language=ocr_language,
                    min_text_chars=min_text_chars,
                )
            )

    repeated_edge_text = find_repeated_edge_text(working_pages)
    pages = []
    for working in working_pages:
        blocks = [
            block
            for block in working.pop("_blocks")
            if not should_remove_edge_block(block, working, repeated_edge_text)
        ]
        working["text"] = clean_page_text("\n\n".join(block["text"] for block in blocks))
        extracted_content = working["text"] + "".join(
            str(table.get("markdown") or "") for table in working.get("tables", [])
        )
        if len(re.sub(r"\s+", "", extracted_content)) < min_text_chars:
            working["warnings"].append("low_text_after_extraction")
        working["warnings"] = list(dict.fromkeys(working["warnings"]))
        pages.append(working)
    return pages


def extract_page(
    page: fitz.Page,
    report: dict[str, str],
    *,
    page_number: int,
    total_pages: int,
    ocr_mode: str,
    ocr_language: str,
    min_text_chars: int,
) -> dict[str, Any]:
    warnings: list[str] = []
    native_blocks = get_text_blocks(page)
    native_text_chars = len(re.sub(r"\s+", "", " ".join(block["text"] for block in native_blocks)))
    tables = extract_tables(page, report["report_id"], page_number, native_blocks, warnings)
    blocks = native_blocks
    extraction_method = "text"

    should_ocr = ocr_mode == "force" or (ocr_mode == "auto" and native_text_chars < min_text_chars)
    if should_ocr:
        try:
            textpage = page.get_textpage_ocr(language=ocr_language, dpi=300, full=True)
            blocks = get_text_blocks(page, textpage=textpage)
            extraction_method = "ocr"
        except Exception as exc:
            if ocr_mode == "force":
                raise RuntimeError(f"OCR failed on page {page_number}: {exc}") from exc
            warnings.append(f"ocr_unavailable: {exc}")

    table_rects = [fitz.Rect(table["bbox"]) for table in tables]
    body_blocks = [block for block in blocks if not overlaps_table(block["bbox"], table_rects)]
    return {
        "report_id": report["report_id"],
        "kind": report.get("kind", ""),
        "company": report.get("company", ""),
        "stock_code": report.get("stock_code", ""),
        "year": report.get("year", ""),
        "source_title": report.get("title", ""),
        "source_url": report.get("source_url", ""),
        "source_tier": report.get("source_tier", ""),
        "source_type": report.get("source_type", ""),
        "pdf_path": report.get("local_path", ""),
        "page": page_number,
        "total_pages": total_pages,
        "page_width": round(float(page.rect.width), 2),
        "page_height": round(float(page.rect.height), 2),
        "text": "",
        "tables": tables,
        "extraction_method": extraction_method,
        "warnings": warnings,
        "table_count": len(tables),
        "_blocks": body_blocks,
    }


def get_text_blocks(page: fitz.Page, *, textpage: fitz.TextPage | None = None) -> list[dict[str, Any]]:
    raw_blocks = page.get_text("blocks", sort=True, textpage=textpage)
    blocks = []
    for raw in raw_blocks:
        if len(raw) > 6 and raw[6] != 0:
            continue
        text = clean_page_text(raw[4])
        if not text:
            continue
        blocks.append(
            {
                "bbox": [round(float(value), 2) for value in raw[:4]],
                "text": text,
            }
        )
    return blocks


def extract_tables(
    page: fitz.Page,
    report_id: str,
    page_number: int,
    blocks: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    try:
        found = page.find_tables()
    except Exception as exc:
        warnings.append(f"table_detection_failed: {exc}")
        return []

    tables = []
    for index, table in enumerate(found.tables, start=1):
        try:
            raw_headers = getattr(getattr(table, "header", None), "names", None)
            headers, rows = normalize_table_data(table.extract(), raw_headers)
            if not headers and not rows:
                continue
            bbox = [round(float(value), 2) for value in table.bbox]
            title = infer_table_title(blocks, fitz.Rect(bbox), page.rect.height)
            table_id = stable_id("table", report_id, page_number, index, *bbox)
            tables.append(
                {
                    "table_id": table_id,
                    "bbox": bbox,
                    "title": title,
                    "headers": headers,
                    "rows": rows,
                    "markdown": table_to_markdown(headers, rows),
                    "row_count": len(rows),
                    "column_count": max([len(headers), *(len(row) for row in rows)], default=0),
                }
            )
        except Exception as exc:
            warnings.append(f"table_{index}_extraction_failed: {exc}")
    return tables


def infer_table_title(blocks: list[dict[str, Any]], table_rect: fitz.Rect, page_height: float) -> str:
    candidates: list[tuple[float, str]] = []
    for block in blocks:
        rect = fitz.Rect(block["bbox"])
        text = re.sub(r"\s+", " ", block["text"]).strip()
        if rect.y1 > table_rect.y0 or table_rect.y0 - rect.y1 > min(90, page_height * 0.12):
            continue
        if not text or len(text) > 120 or is_page_number(text, 10000):
            continue
        candidates.append((table_rect.y0 - rect.y1, text))
    return min(candidates, default=(0.0, ""), key=lambda item: item[0])[1]


def overlaps_table(bbox: list[float], table_rects: list[fitz.Rect]) -> bool:
    rect = fitz.Rect(bbox)
    if rect.is_empty:
        return False
    center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
    for table_rect in table_rects:
        intersection = rect & table_rect
        if table_rect.contains(center) or (
            not intersection.is_empty and intersection.get_area() / max(rect.get_area(), 1.0) >= 0.35
        ):
            return True
    return False


def edge_fingerprint(text: str) -> str:
    value = re.sub(r"\s+", "", str(text or "")).casefold()
    value = re.sub(r"\d+", "#", value)
    return value


def find_repeated_edge_text(pages: list[dict[str, Any]]) -> set[str]:
    occurrences: dict[str, set[int]] = defaultdict(set)
    for page in pages:
        height = float(page.get("page_height") or 0)
        for block in page.get("_blocks", []):
            rect = fitz.Rect(block["bbox"])
            if rect.y1 <= height * HEADER_FOOTER_ZONE_RATIO or rect.y0 >= height * (1 - HEADER_FOOTER_ZONE_RATIO):
                fingerprint = edge_fingerprint(block["text"])
                if fingerprint:
                    occurrences[fingerprint].add(int(page["page"]))
    page_count = len(pages)
    threshold = 2 if page_count <= 4 else max(3, math.ceil(page_count * 0.3))
    return {fingerprint for fingerprint, page_numbers in occurrences.items() if len(page_numbers) >= threshold}


def should_remove_edge_block(
    block: dict[str, Any],
    page: dict[str, Any],
    repeated_edge_text: set[str],
) -> bool:
    text = block["text"]
    if is_page_number(text, int(page.get("total_pages") or len(repeated_edge_text) or 1)):
        return True
    rect = fitz.Rect(block["bbox"])
    height = float(page.get("page_height") or 0)
    is_edge = rect.y1 <= height * HEADER_FOOTER_ZONE_RATIO or rect.y0 >= height * (1 - HEADER_FOOTER_ZONE_RATIO)
    return is_edge and edge_fingerprint(text) in repeated_edge_text


def is_page_number(text: str, total_pages: int) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    match = re.fullmatch(r"(?:第)?(\d{1,4})(?:页|/\d{1,4})?", compact, re.I)
    if match:
        return int(match.group(1)) <= max(total_pages + 5, 20)
    return bool(re.fullmatch(r"page\d{1,4}(?:of\d{1,4})?", compact, re.I))


def parsed_quality_summary(pages: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "pages": len(pages),
        "text_pages": sum(bool(str(page.get("text") or "").strip()) for page in pages),
        "table_count": sum(int(page.get("table_count") or 0) for page in pages),
        "ocr_pages": sum(page.get("extraction_method") == "ocr" for page in pages),
        "low_text_pages": sum("low_text_after_extraction" in page.get("warnings", []) for page in pages),
        "warning_count": sum(len(page.get("warnings", [])) for page in pages),
    }


def write_parsed_report(
    report: dict[str, str],
    pages: list[dict[str, Any]],
    output_dir: Path = PARSED_TEXT_DIR,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_id = report["report_id"]
    jsonl_path = output_dir / f"{report_id}.jsonl"
    txt_path = output_dir / f"{report_id}.txt"
    write_jsonl(jsonl_path, pages)
    with txt_path.open("w", encoding="utf-8") as file:
        for page in pages:
            if page["text"] or page.get("tables"):
                file.write(f"\n\n=== page {page['page']} ===\n")
            if page["text"]:
                file.write(page["text"])
            for table in page.get("tables", []):
                file.write(f"\n\n--- table {table['table_id']} ---\n")
                if table.get("title"):
                    file.write(f"{table['title']}\n\n")
                file.write(table.get("markdown", ""))
    return jsonl_path, txt_path
