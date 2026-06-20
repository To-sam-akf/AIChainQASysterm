"""HTML renderer for ReportSpec."""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from aika.report.spec import ReportSpec


def render_html(spec: ReportSpec | dict[str, Any]) -> str:
    active = spec if isinstance(spec, ReportSpec) else ReportSpec.model_validate(spec)
    template_name = "coverage_audit.html.j2" if active.report_type == "evidence_coverage_audit" else "research_brief.html.j2"
    environment = Environment(
        loader=PackageLoader("aika.report", "templates"),
        autoescape=select_autoescape(["html", "xml", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return environment.get_template(template_name).render(report=active.model_dump())
