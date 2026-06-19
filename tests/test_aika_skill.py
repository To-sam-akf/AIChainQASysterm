from __future__ import annotations

import re
from pathlib import Path

from aika.aika_mcp.tools import tool_names


SKILL_PATH = Path("skills/aika-research/SKILL.md")


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_aika_skill_exists_and_has_valid_frontmatter() -> None:
    text = _skill_text()
    match = re.match(r"^---\n(?P<yaml>.*?)\n---\n", text, re.DOTALL)

    assert match is not None
    frontmatter = match.group("yaml")
    assert "name: aika-research" in frontmatter
    assert "description:" in frontmatter
    assert "AI 算力产业链" in frontmatter
    assert "中文投研" in frontmatter
    assert "公司对比" in frontmatter
    assert "技术路线" in frontmatter
    assert "风险审查" in frontmatter
    assert "证据缺口" in frontmatter
    assert "液冷" in frontmatter
    assert "光模块" in frontmatter


def test_aika_skill_lists_all_mcp_tools() -> None:
    text = _skill_text()

    for name in tool_names():
        assert f"`{name}`" in text


def test_aika_skill_preserves_citations_and_handles_missing_evidence() -> None:
    text = _skill_text()

    assert "citation_id" in text
    assert "证据卡片" in text
    assert "conclusions" in text
    assert "evidence_links" in text
    assert "freshness_status" in text or "时效" in text
    assert "Preserve" in text or "保留" in text
    assert "当前证据不足" in text
    assert "do not turn absent evidence into a confirmed fact" in text
    assert "uncited" in text


def test_aika_skill_contains_compliance_boundaries() -> None:
    text = _skill_text()

    assert "买卖建议" in text
    assert "目标价" in text
    assert "收益预测" in text
    assert "buy/sell advice" in text


def test_aika_skill_stays_concise() -> None:
    lines = _skill_text().splitlines()

    assert len(lines) <= 90
