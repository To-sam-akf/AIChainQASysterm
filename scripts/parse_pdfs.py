#!/usr/bin/env python3
"""Parse downloaded PDFs and create extraction chunks."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from aika.pdf_parser import (
    OCR_MODES,
    PARSED_TEXT_DIR,
    parse_pdf_pages,
    parsed_quality_summary,
    read_downloaded_manifest,
    write_parsed_report,
)
from aika.extraction_schema import load_jsonl
from aika.text_cleaner import CHUNKS_DIR, build_chunks_file


DEFAULT_MANIFEST = ROOT_DIR / "data" / "metadata" / "reports_manifest.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse stage-1 PDFs and build text chunks.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parsed-dir", type=Path, default=PARSED_TEXT_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--max-reports", type=int, default=0, help="Limit reports for smoke tests; 0 means all.")
    parser.add_argument("--max-pages", type=int, default=0, help="Limit pages per report; 0 means all.")
    parser.add_argument("--max-chars", type=int, default=2800)
    parser.add_argument("--ocr-mode", choices=sorted(OCR_MODES), default="auto")
    parser.add_argument("--ocr-language", default="chi_sim+eng")
    parser.add_argument("--min-text-chars", type=int, default=80)
    parser.add_argument("--force", action="store_true", help="Overwrite existing parsed/chunk files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reports = read_downloaded_manifest(args.manifest)
    if args.max_reports:
        reports = reports[: args.max_reports]
    args.parsed_dir.mkdir(parents=True, exist_ok=True)
    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    error_path = args.parsed_dir / "parse_errors.csv"
    quality_path = args.parsed_dir / "parse_quality.csv"
    errors: list[dict[str, str]] = []
    quality_rows: list[dict[str, str | int]] = []
    parsed_count = 0
    chunk_count = 0
    for report in reports:
        parsed_jsonl = args.parsed_dir / f"{report['report_id']}.jsonl"
        chunk_jsonl = args.chunks_dir / f"{report['report_id']}.jsonl"
        if parsed_jsonl.exists() and chunk_jsonl.exists() and not args.force:
            summary = build_quality_row(report, load_jsonl(parsed_jsonl), chunk_jsonl)
            quality_rows.append(summary)
            parsed_count += 1
            chunk_count += int(summary["chunks"])
            print(
                f"SKIP {report['report_id']} pages={summary['pages']} "
                f"tables={summary['table_count']} chunks={summary['chunks']}"
            )
            continue
        try:
            pages = parse_pdf_pages(
                report,
                max_pages=args.max_pages or None,
                ocr_mode=args.ocr_mode,
                ocr_language=args.ocr_language,
                min_text_chars=args.min_text_chars,
            )
            write_parsed_report(report, pages, args.parsed_dir)
            build_chunks_file(parsed_jsonl, args.chunks_dir, max_chars=args.max_chars)
            summary = build_quality_row(report, pages, chunk_jsonl)
            quality_rows.append(summary)
            parsed_count += 1
            chunk_count += int(summary["chunks"])
            print(
                f"OK   {report['report_id']} pages={summary['pages']} "
                f"tables={summary['table_count']} ocr={summary['ocr_pages']} "
                f"low_text={summary['low_text_pages']} chunks={summary['chunks']}"
            )
        except Exception as exc:
            errors.append({"report_id": report.get("report_id", ""), "path": report.get("local_path", ""), "error": str(exc)})
            print(f"FAIL {report.get('report_id', '')}: {exc}")
    with error_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["report_id", "path", "error"])
        writer.writeheader()
        writer.writerows(errors)
    quality_fields = [
        "report_id",
        "path",
        "pages",
        "text_pages",
        "table_count",
        "ocr_pages",
        "low_text_pages",
        "warning_count",
        "chunks",
        "warnings",
    ]
    with quality_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=quality_fields)
        writer.writeheader()
        writer.writerows(quality_rows)
    totals = Counter()
    for row in quality_rows:
        for key in ("pages", "text_pages", "table_count", "ocr_pages", "low_text_pages", "warning_count"):
            totals[key] += int(row[key])
    print(
        f"Parsed reports: {parsed_count}; pages: {totals['pages']}; tables: {totals['table_count']}; "
        f"OCR pages: {totals['ocr_pages']}; low-text pages: {totals['low_text_pages']}; "
        f"chunks: {chunk_count}; errors: {len(errors)}"
    )
    print(f"Quality report: {quality_path}")
    return 1 if errors else 0


def build_quality_row(
    report: dict[str, str],
    pages: list[dict],
    chunk_jsonl: Path,
) -> dict[str, str | int]:
    summary: dict[str, str | int] = parsed_quality_summary(pages)
    summary.update(
        {
            "report_id": report["report_id"],
            "path": report.get("local_path", ""),
            "chunks": sum(1 for _ in chunk_jsonl.open(encoding="utf-8")),
            "warnings": " | ".join(
                f"p{page['page']}: {warning}"
                for page in pages
                for warning in page.get("warnings", [])
            ),
        }
    )
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
