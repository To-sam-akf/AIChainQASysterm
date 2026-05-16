import csv
import json
from pathlib import Path

import fitz
import pytest

from src.extraction_schema import SchemaError, sanitize_extraction_payload, validate_extraction_payload
from src.graph_builder import build_verified_graph
from src.kg_loader import CONSTRAINT_QUERIES, assert_label, assert_relation_type, validate_graph_csvs
from src.pdf_parser import parse_pdf_pages
from src.text_cleaner import build_chunks_from_pages


def make_text_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
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

    pages = parse_pdf_pages(report)

    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert "AI server" in pages[0]["text"]


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
