#!/usr/bin/env python3
"""Parse downloaded PDFs and create extraction chunks."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.pdf_parser import PARSED_TEXT_DIR, parse_pdf_pages, read_downloaded_manifest, write_parsed_report
from src.text_cleaner import CHUNKS_DIR, build_chunks_file


DEFAULT_MANIFEST = ROOT_DIR / "data" / "metadata" / "reports_manifest.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse stage-1 PDFs and build text chunks.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parsed-dir", type=Path, default=PARSED_TEXT_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--max-reports", type=int, default=0, help="Limit reports for smoke tests; 0 means all.")
    parser.add_argument("--max-pages", type=int, default=0, help="Limit pages per report; 0 means all.")
    parser.add_argument("--max-chars", type=int, default=2800)
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
    errors: list[dict[str, str]] = []
    parsed_count = 0
    chunk_count = 0
    for report in reports:
        parsed_jsonl = args.parsed_dir / f"{report['report_id']}.jsonl"
        chunk_jsonl = args.chunks_dir / f"{report['report_id']}.jsonl"
        if parsed_jsonl.exists() and chunk_jsonl.exists() and not args.force:
            print(f"SKIP {report['report_id']}")
            parsed_count += 1
            continue
        try:
            pages = parse_pdf_pages(report, max_pages=args.max_pages or None)
            write_parsed_report(report, pages, args.parsed_dir)
            build_chunks_file(parsed_jsonl, args.chunks_dir, max_chars=args.max_chars)
            parsed_count += 1
            chunk_count += sum(1 for _ in chunk_jsonl.open(encoding="utf-8"))
            print(f"OK   {report['report_id']} pages={len(pages)} chunks={chunk_jsonl}")
        except Exception as exc:
            errors.append({"report_id": report.get("report_id", ""), "path": report.get("local_path", ""), "error": str(exc)})
            print(f"FAIL {report.get('report_id', '')}: {exc}")
    if errors:
        with error_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["report_id", "path", "error"])
            writer.writeheader()
            writer.writerows(errors)
    print(f"Parsed reports: {parsed_count}; new chunks: {chunk_count}; errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
