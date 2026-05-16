"""PDF parsing utilities for stage 2."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import fitz

from src.extraction_schema import write_jsonl


ROOT_DIR = Path(__file__).resolve().parents[1]
PARSED_TEXT_DIR = ROOT_DIR / "data" / "parsed_text"


def read_downloaded_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [row for row in rows if row.get("status") == "downloaded" and row.get("local_path")]


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def clean_page_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf_pages(report: dict[str, str], *, max_pages: int | None = None) -> list[dict[str, Any]]:
    pdf_path = resolve_project_path(report["local_path"])
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        limit = min(total_pages, max_pages) if max_pages else total_pages
        for index in range(limit):
            page = doc.load_page(index)
            text = clean_page_text(page.get_text("text"))
            pages.append(
                {
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
                    "page": index + 1,
                    "total_pages": total_pages,
                    "text": text,
                }
            )
    return pages


def write_parsed_report(report: dict[str, str], pages: list[dict[str, Any]], output_dir: Path = PARSED_TEXT_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_id = report["report_id"]
    jsonl_path = output_dir / f"{report_id}.jsonl"
    txt_path = output_dir / f"{report_id}.txt"
    write_jsonl(jsonl_path, pages)
    with txt_path.open("w", encoding="utf-8") as file:
        for page in pages:
            if page["text"]:
                file.write(f"\n\n=== page {page['page']} ===\n")
                file.write(page["text"])
    return jsonl_path, txt_path
