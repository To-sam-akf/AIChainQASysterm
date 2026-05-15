from pathlib import Path

import pytest
from pypdf import PdfWriter

from scripts.prepare_stage1_data import (
    MANIFEST_FIELDS,
    ManifestStore,
    load_companies,
    safe_filename,
    validate_companies,
    validate_pdf,
)


def test_company_list_has_expected_ten_targets() -> None:
    companies = load_companies()

    validate_companies(companies)

    assert len(companies) == 10
    assert {company.company for company in companies} >= {"浪潮信息", "海光信息"}
    assert all(company.stock_code.isdigit() and len(company.stock_code) == 6 for company in companies)


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
