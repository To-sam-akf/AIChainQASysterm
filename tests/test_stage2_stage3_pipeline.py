import csv
import json
from pathlib import Path

import fitz
import pytest

from aika.extraction_schema import SchemaError, sanitize_extraction_payload, validate_extraction_payload
from aika.graph_builder import build_verified_graph
from aika.kg_loader import CONSTRAINT_QUERIES, assert_label, assert_relation_type, validate_graph_csvs
from aika.llm_extractor import build_user_prompt
from aika.pdf_parser import clean_page_text, normalize_table_data, parse_pdf_pages
from aika.text_cleaner import build_chunks_from_pages, split_table_rows


def make_text_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def make_table_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "AI compute financial summary")
    page.insert_text((72, 110), "Liquid cooling supports dense AI servers.")
    xs = [72, 230, 365, 500]
    ys = [150, 180, 210, 240]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    cells = [
        ["Metric", "2024", "2025"],
        ["Revenue", "100", "120"],
        ["Gross margin", "20%", "25%"],
    ]
    for row_index, row in enumerate(cells):
        for column_index, value in enumerate(row):
            page.insert_text((xs[column_index] + 5, ys[row_index] + 20), value)
    doc.save(path)
    doc.close()


def test_pdf_parser_outputs_non_empty_page_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    make_text_pdf(pdf_path, "Inspur AI server compute business")
    report = {
        "report_id": "annual_000977_2025",
        "kind": "annual",
        "company": "浪潮信息",
        "stock_code": "000977",
        "year": "2025",
        "title": "2025年年度报告",
        "source_url": "https://example.com",
        "local_path": str(pdf_path),
    }

    pages = parse_pdf_pages(report, ocr_mode="off")

    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert "AI server" in pages[0]["text"]


def test_pdf_parser_separates_layout_text_and_structured_table(tmp_path: Path) -> None:
    pdf_path = tmp_path / "table.pdf"
    make_table_pdf(pdf_path)
    report = {
        "report_id": "annual_table",
        "kind": "annual",
        "company": "测试公司",
        "title": "测试年报",
        "local_path": str(pdf_path),
    }

    pages = parse_pdf_pages(report, ocr_mode="off")

    assert len(pages) == 1
    assert "Liquid cooling" in pages[0]["text"]
    assert "Revenue" not in pages[0]["text"]
    assert pages[0]["table_count"] == 1
    table = pages[0]["tables"][0]
    assert table["headers"] == ["Metric", "2024", "2025"]
    assert table["rows"][0] == ["Revenue", "100", "120"]
    assert "| Gross margin | 20% | 25% |" in table["markdown"]


def test_table_normalization_preserves_values_and_removes_empty_axes() -> None:
    headers, rows = normalize_table_data(
        [
            ["指标", "2024", "2025", None],
            ["毛利率", "20%", "25%", None],
            ["净利润", "(10)", "-12", None],
            [None, None, None, None],
        ],
        ["指标", "2024", "2025", None],
    )

    assert headers == ["指标", "2024", "2025"]
    assert rows == [["毛利率", "20%", "25%"], ["净利润", "(10)", "-12"]]


def test_parser_removes_repeated_headers_footers_and_page_numbers(tmp_path: Path) -> None:
    pdf_path = tmp_path / "repeated.pdf"
    doc = fitz.open()
    for page_number in range(1, 4):
        page = doc.new_page()
        page.insert_text((72, 35), "Repeated annual report header")
        page.insert_text((72, 150), f"Unique body content {page_number} about AI servers")
        page.insert_text((290, 810), str(page_number))
    doc.save(pdf_path)
    doc.close()

    pages = parse_pdf_pages(
        {"report_id": "repeated", "title": "报告", "local_path": str(pdf_path)},
        ocr_mode="off",
    )

    assert all("Repeated annual report header" not in page["text"] for page in pages)
    assert all("Unique body content" in page["text"] for page in pages)
    assert all(page["text"].strip()[-1] != str(page["page"]) for page in pages)


def test_soft_line_cleaning_joins_english_hyphen_and_chinese_lines() -> None:
    cleaned = clean_page_text("intercon-\nnect bandwidth\n\n算力基础\n设施")

    assert cleaned == "interconnect bandwidth\n\n算力基础设施"


def test_ocr_auto_warns_but_force_fails_when_ocr_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "blank.pdf"
    make_text_pdf(pdf_path, "")
    report = {"report_id": "blank", "title": "空白", "local_path": str(pdf_path)}

    def fail_ocr(*args, **kwargs):
        raise RuntimeError("tesseract missing")

    monkeypatch.setattr(fitz.Page, "get_textpage_ocr", fail_ocr)
    pages = parse_pdf_pages(report, ocr_mode="auto")
    assert any("ocr_unavailable" in warning for warning in pages[0]["warnings"])
    with pytest.raises(RuntimeError, match="OCR failed"):
        parse_pdf_pages(report, ocr_mode="force")


def test_text_chunking_preserves_source_fields_and_size() -> None:
    pages = [
        {
            "report_id": "annual_000977_2025",
            "kind": "annual",
            "company": "浪潮信息",
            "source_title": "2025年年度报告",
            "source_url": "https://example.com",
            "page": 12,
            "text": "核心竞争力\n浪潮信息持续布局AI服务器和算力基础设施。" * 30,
        }
    ]

    chunks = build_chunks_from_pages(pages, max_chars=180)

    assert chunks
    assert all(len(chunk["text"]) <= 180 for chunk in chunks)
    assert chunks[0]["report_id"] == "annual_000977_2025"
    assert chunks[0]["page"] == 12
    assert chunks[0]["source_title"] == "2025年年度报告"
    assert chunks[0]["context"]
    assert chunks[0]["content_type"] == "text"


def test_text_chunking_rejects_numeric_table_fragment_as_section() -> None:
    pages = [
        {
            "report_id": "industry_1",
            "kind": "industry",
            "company": "",
            "source_title": "绿色算力发展研究报告",
            "source_url": "https://example.com",
            "page": 19,
            "text": "2. 50%\n在未来智算中心持续扩张情境下，局部地区为解决算力对电量的需求，电网扩容压力将加剧。",
        }
    ]

    chunks = build_chunks_from_pages(pages, max_chars=300)

    assert chunks
    assert chunks[0]["section"] != "50%"
    assert "算力" in chunks[0]["section"] or "AI" in chunks[0]["section"]


def test_table_chunks_repeat_headers_and_preserve_complete_rows() -> None:
    pages = [
        {
            "report_id": "annual_table",
            "kind": "annual",
            "company": "测试公司",
            "source_title": "测试年报",
            "page": 12,
            "text": "财务指标",
            "tables": [
                {
                    "table_id": "table_1",
                    "title": "主要财务指标",
                    "headers": ["指标", "2024", "2025"],
                    "rows": [[f"指标{i}", str(i), str(i + 1)] for i in range(1, 8)],
                    "markdown": "",
                }
            ],
        }
    ]

    chunks = [chunk for chunk in build_chunks_from_pages(pages, max_chars=100) if chunk["content_type"] == "table"]

    assert len(chunks) > 1
    assert all("| 指标 | 2024 | 2025 |" in chunk["text"] for chunk in chunks)
    assert all(chunk["table_id"] == "table_1" for chunk in chunks)
    assert all(chunk["section"] == "主要财务指标" for chunk in chunks)
    assert chunks[0]["table_row_start"] == 1
    assert chunks[-1]["table_row_end"] == 7
    assert sum(chunk["table_row_end"] - chunk["table_row_start"] + 1 for chunk in chunks) == 7


def test_single_oversized_table_row_is_not_split_mid_row() -> None:
    groups = split_table_rows(["指标", "说明"], [["算力", "长" * 120]], max_chars=60)

    assert len(groups) == 1
    assert groups[0][0:2] == (1, 1)
    assert "长" * 120 in groups[0][2]


def test_table_chunk_prompt_requires_strict_header_value_alignment() -> None:
    prompt = build_user_prompt(
        {
            "report_id": "annual_table",
            "kind": "annual",
            "content_type": "table",
            "table_id": "table_1",
            "table_title": "主要财务指标",
            "text": "| 指标 | 2025 |\n|---|---|\n| 毛利率 | 25% |",
        }
    )

    assert "不得沿用上一行" in prompt
    assert "table_1" in prompt


def test_extraction_schema_accepts_valid_and_rejects_missing_evidence() -> None:
    valid = {
        "entities": [
            {"type": "Company", "name": "浪潮信息"},
            {"type": "Technology", "name": "AI服务器"},
        ],
        "relations": [
            {
                "head_type": "Company",
                "head": "浪潮信息",
                "relation": "USES_TECHNOLOGY",
                "tail_type": "Technology",
                "tail": "AI服务器",
                "evidence": "公司持续布局AI服务器。",
            }
        ],
    }
    cleaned = validate_extraction_payload(valid)
    assert cleaned["relations"][0]["relation"] == "USES_TECHNOLOGY"

    invalid = json.loads(json.dumps(valid, ensure_ascii=False))
    invalid["relations"][0]["evidence"] = ""
    with pytest.raises(SchemaError, match="evidence"):
        validate_extraction_payload(invalid)


def test_extraction_schema_supports_industry_ontology_relations() -> None:
    valid = {
        "entities": [
            {"type": "IndustryConcept", "name": "智能算力"},
            {"type": "ValueChainSegment", "name": "智算中心"},
            {"type": "Policy", "name": "东数西算政策"},
        ],
        "relations": [
            {
                "head_type": "IndustryConcept",
                "head": "智能算力",
                "relation": "DEFINES",
                "tail_type": "ValueChainSegment",
                "tail": "智算中心",
                "evidence": "智能算力基础设施包括智算中心等形态。",
            },
            {
                "head_type": "ValueChainSegment",
                "head": "智算中心",
                "relation": "SUPPORTED_BY_POLICY",
                "tail_type": "Policy",
                "tail": "东数西算政策",
                "evidence": "东数西算政策支撑算力基础设施建设。",
            },
        ],
    }

    cleaned = validate_extraction_payload(valid)

    assert {relation["relation"] for relation in cleaned["relations"]} == {"DEFINES", "SUPPORTED_BY_POLICY"}


def test_metric_entity_requires_structured_fields() -> None:
    valid = {"entities": [{"type": "Metric", "name": "2025年智能算力规模160 EFLOPS"}], "relations": []}
    cleaned = validate_extraction_payload(valid)

    assert cleaned["entities"][0]["properties"]["year"] == "2025"
    assert cleaned["entities"][0]["properties"]["value"]

    invalid = {"entities": [{"type": "Metric", "name": "营业收入"}], "relations": []}
    with pytest.raises(SchemaError, match="Metric must include"):
        validate_extraction_payload(invalid)


def test_sanitize_extraction_payload_drops_invalid_relations() -> None:
    payload = {
        "entities": [{"type": "Report", "name": "报告"}, {"type": "Company", "name": "浪潮信息"}],
        "relations": [
            {
                "head_type": "Report",
                "head": "报告",
                "relation": "MENTIONED_IN",
                "tail_type": "Company",
                "tail": "浪潮信息",
                "evidence": "错误方向关系。",
            }
        ],
    }

    cleaned, rejected = sanitize_extraction_payload(payload)

    assert len(cleaned["entities"]) == 2
    assert cleaned["relations"] == []
    assert rejected


def test_sanitize_extraction_payload_drops_unstructured_metric_relations() -> None:
    payload = {
        "entities": [{"type": "Company", "name": "浪潮信息"}],
        "relations": [
            {
                "head_type": "Company",
                "head": "浪潮信息",
                "relation": "HAS_METRIC",
                "tail_type": "Metric",
                "tail": "收入表现",
                "evidence": "公司披露收入表现良好。",
            }
        ],
    }

    cleaned, rejected = sanitize_extraction_payload(payload)

    assert cleaned["relations"] == []
    assert any("Metric relation tail" in item for item in rejected)


def test_build_verified_graph_deduplicates_entities_and_relations(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "report_id",
                "kind",
                "company",
                "stock_code",
                "year",
                "title",
                "source_site",
                "source_url",
                "pdf_url",
                "local_path",
                "published_at",
                "downloaded_at",
                "sha256",
                "file_size",
                "pages",
                "status",
                "error",
                "source_tier",
                "source_type",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "report_id": "annual_000977_2025",
                "kind": "annual",
                "company": "浪潮信息",
                "stock_code": "000977",
                "year": "2025",
                "title": "2025年年度报告",
                "status": "downloaded",
                "source_tier": "1",
                "source_type": "company_annual_report",
            }
        )
    extraction = tmp_path / "extractions.jsonl"
    record = {
        "chunk_id": "chunk_1",
        "report_id": "annual_000977_2025",
        "source_title": "2025年年度报告",
        "page": "8",
        "entities": [
            {"type": "Company", "name": "浪潮信息"},
            {"type": "Company", "name": "浪潮信息"},
            {"type": "Technology", "name": "AI服务器"},
        ],
        "relations": [
            {
                "head_type": "Company",
                "head_name": "浪潮信息",
                "relation": "USES_TECHNOLOGY",
                "tail_type": "Technology",
                "tail_name": "AI服务器",
                "evidence": "浪潮信息布局AI服务器。",
            },
            {
                "head_type": "Company",
                "head_name": "浪潮信息",
                "relation": "USES_TECHNOLOGY",
                "tail_type": "Technology",
                "tail_name": "AI服务器",
                "evidence": "浪潮信息布局AI服务器。",
            },
        ],
    }
    extraction.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    entities_csv = tmp_path / "entities.csv"
    relations_csv = tmp_path / "relations.csv"

    entities, relations = build_verified_graph(
        extraction_paths=[extraction],
        manifest_path=manifest,
        entities_csv=entities_csv,
        relations_csv=relations_csv,
    )

    assert sum(row["type"] == "Company" and row["name"] == "浪潮信息" for row in entities) == 1
    assert sum(row["relation"] == "USES_TECHNOLOGY" for row in relations) == 1
    assert any(row["type"] == "Report" for row in entities)
    assert any(row["relation"] == "MENTIONED_IN" for row in relations)
    assert any(row["is_core_company"] == "true" for row in entities if row["type"] == "Company")
    assert any(row["source_tier"] == "1" for row in relations)


def test_neo4j_loader_rejects_unknown_labels_and_relation_types() -> None:
    assert_label("Company")
    assert_relation_type("USES_TECHNOLOGY")
    with pytest.raises(ValueError):
        assert_label("BadLabel")
    with pytest.raises(ValueError):
        assert_relation_type("BAD_REL")
    assert all("CREATE CONSTRAINT" in query for query in CONSTRAINT_QUERIES)


def test_validate_graph_csvs_counts_rows(tmp_path: Path) -> None:
    entities_csv = tmp_path / "entities.csv"
    relations_csv = tmp_path / "relations.csv"
    entities_csv.write_text(
        "entity_id,type,name,normalized_name,properties,source_report_ids,review_status\n"
        "e1,Company,浪潮信息,浪潮信息,{},[],auto\n"
        "e2,Technology,AI服务器,ai服务器,{},[],auto\n",
        encoding="utf-8",
    )
    relations_csv.write_text(
        "relation_id,head_type,head_name,relation,tail_type,tail_name,evidence,source_report_id,source_title,page,section,confidence,review_status\n"
        "r1,Company,浪潮信息,USES_TECHNOLOGY,Technology,AI服务器,涉及AI服务器,r,报告,1,,0.9,auto\n",
        encoding="utf-8",
    )

    assert validate_graph_csvs(entities_csv, relations_csv) == (2, 1)
