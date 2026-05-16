from pathlib import Path

import pytest
from pypdf import PdfWriter

from scripts.prepare_stage1_data import (
    INDUSTRY_SOURCES_CSV,
    MANIFEST_FIELDS,
    ManifestStore,
    industry_candidate,
    load_companies,
    safe_filename,
    validate_companies,
    validate_pdf,
)
from src.data_config import load_industry_sources, load_research_keywords, validate_industry_sources


def test_company_list_has_expected_thirty_targets() -> None:
    companies = load_companies()

    validate_companies(companies)

    assert len(companies) == 30
    assert {company.company for company in companies} >= {"浪潮信息", "海光信息", "澜起科技", "紫光股份"}
    assert all(company.stock_code.isdigit() and len(company.stock_code) == 6 for company in companies)
    assert all(company.aliases for company in companies)


def test_research_keywords_are_config_driven() -> None:
    keywords = load_research_keywords()

    assert "AI算力产业链" in keywords
    assert any("液冷" in keyword for keyword in keywords)
    assert any("PCB" in keyword for keyword in keywords)


def test_industry_sources_have_downloadable_pdf_metadata() -> None:
    sources = load_industry_sources(INDUSTRY_SOURCES_CSV)

    validate_industry_sources(sources)

    assert sources
    assert all(source.pdf_url.endswith(".pdf") for source in sources)
    assert all(source.source_tier in {"1", "2", "3"} for source in sources)
    assert {source.source_type for source in sources} == {"authority_whitepaper"}

    candidate = industry_candidate(sources[0])
    assert candidate.kind == "industry"
    assert candidate.local_path.parts[-2:] == ("industry", candidate.local_path.name)
    assert candidate.source_tier == "1"


def test_safe_filename_preserves_chinese_company_name() -> None:
    filename = safe_filename('浪潮信息: 2025 年度报告/正式版?.pdf')

    assert filename.startswith("浪潮信息")
    assert "/" not in filename
    assert ":" not in filename
    assert "?" not in filename


def test_validate_pdf_rejects_html_error_page(tmp_path: Path) -> None:
    html_file = tmp_path / "error.pdf"
    html_file.write_bytes(b"<html>not a pdf</html>" * 200)

    with pytest.raises(ValueError, match="not a PDF"):
        validate_pdf(html_file)


def test_validate_pdf_accepts_real_pdf(tmp_path: Path) -> None:
    pdf_file = tmp_path / "ok.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_file.open("wb") as file:
        writer.write(file)

    assert validate_pdf(pdf_file) == 1


def test_manifest_upsert_deduplicates_report_id(tmp_path: Path) -> None:
    manifest_path = tmp_path / "reports_manifest.csv"
    store = ManifestStore(manifest_path)
    base_row = {field: "" for field in MANIFEST_FIELDS}
    base_row.update(
        {
            "report_id": "annual_000977_2025",
            "kind": "annual",
            "company": "浪潮信息",
            "status": "downloaded",
            "local_path": "data/raw_pdfs/annual/浪潮信息_2025年报.pdf",
        }
    )

    store.upsert(base_row)
    updated_row = dict(base_row)
    updated_row["pages"] = "300"
    store.upsert(updated_row)

    reloaded = ManifestStore(manifest_path)
    assert len(reloaded.rows) == 1
    assert reloaded.rows[0]["pages"] == "300"
