"""PDF export helpers for AIKA HTML reports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aika.report.render_html import render_html
from aika.report.spec import ReportSpec


PLAYWRIGHT_INSTALL_FIX = (
    "Install PDF rendering support with `pip install 'aika-research-mcp[pdf]'` "
    "and `python -m playwright install chromium`."
)


class PdfRenderError(RuntimeError):
    """Raised when HTML-to-PDF export cannot run."""

    def __init__(self, message: str, *, fix: str = PLAYWRIGHT_INSTALL_FIX) -> None:
        super().__init__(message)
        self.fix = fix


def render_report_pdf(
    spec: ReportSpec | dict[str, Any],
    *,
    output_dir: str | Path,
    html: str | None = None,
) -> dict[str, str]:
    """Write report HTML and export it to PDF with Playwright."""

    active = spec if isinstance(spec, ReportSpec) else ReportSpec.model_validate(spec)
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename(active.title or active.topic or "aika-report")
    html_path = target_dir / f"{stem}.html"
    pdf_path = target_dir / f"{stem}.pdf"

    html_path.write_text(html if html is not None else render_html(active), encoding="utf-8")
    _export_pdf_with_playwright(html_path, pdf_path)
    return {"html_path": str(html_path), "pdf_path": str(pdf_path)}


def _export_pdf_with_playwright(html_path: Path, pdf_path: Path) -> None:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - covered through MCP boundary tests.
        raise PdfRenderError("Playwright is not installed.") from exc

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    margin={"top": "16mm", "right": "14mm", "bottom": "16mm", "left": "14mm"},
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        _remove_partial_pdf(pdf_path)
        raise PdfRenderError(f"Playwright could not render the PDF: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive boundary for browser/runtime failures.
        _remove_partial_pdf(pdf_path)
        raise PdfRenderError(f"PDF rendering failed: {exc}") from exc


def _remove_partial_pdf(pdf_path: Path) -> None:
    try:
        if pdf_path.exists():
            pdf_path.unlink()
    except OSError:
        pass


def _safe_filename(value: str) -> str:
    text = re.sub(r"\s+", "_", str(value or "").strip())
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text[:80] or "aika-report"
