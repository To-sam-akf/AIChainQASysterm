"""Structured report specs and renderers for AIKA research outputs."""

from aika.report.builder import build_report_spec
from aika.report.render_html import render_html
from aika.report.render_markdown import render_markdown, render_markdown_sections
from aika.report.spec import (
    AppendixSpec,
    ChartsSpec,
    CoverageSpec,
    ExecutiveSummarySpec,
    ReportSpec,
)

__all__ = [
    "AppendixSpec",
    "ChartsSpec",
    "CoverageSpec",
    "ExecutiveSummarySpec",
    "ReportSpec",
    "build_report_spec",
    "render_html",
    "render_markdown",
    "render_markdown_sections",
]
