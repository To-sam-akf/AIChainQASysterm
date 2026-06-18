from __future__ import annotations

import csv
import json
from pathlib import Path

from aika.aika_core.knowledge_pack import validate_knowledge_pack
from scripts.build_sample_pack import build_sample_pack


CLAIM_HEADER = (
    "claim_id,claim_type,topic,claim_text,companies,mechanism,direction,horizon,metric,value,unit,"
    "source_report_id,source_title,page,section,source_tier,evidence_span,confidence,as_of_date,"
    "exposure_level,review_status,reviewer_note,quality_flags,conflict_group_id"
)
EVIDENCE_HEADER = "evidence_id,claim_id,source_report_id,source_title,page,section,source_tier,text,as_of_date,quality"
RELATION_HEADER = (
    "relation_id,head_type,head_name,relation,tail_type,tail_name,evidence,source_report_id,"
    "source_title,page,section,source_tier,confidence,review_status"
)
ENTITY_HEADER = "entity_id,type,name,normalized_name,properties,source_report_ids,review_status,is_core_company"
MANIFEST_HEADER = (
    "source_report_id,source_title,source_url,published_at,source_type,license_or_usage_note,included_fields"
)


def write_valid_pack(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "claims.csv").write_text(
        "\n".join(
            [
                CLAIM_HEADER,
                'claim_1,mechanism,液冷,液冷支撑高功率机柜,"[""英维克""]",高功率机柜散热,positive,,,,,report_1,液冷白皮书,3,产业链,1,液冷产业链包括冷板和CDU,0.90,2026,direct,auto,,,',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "evidence_spans.csv").write_text(
        "\n".join(
            [
                EVIDENCE_HEADER,
                "evidence_1,claim_1,report_1,液冷白皮书,3,产业链,1,液冷产业链包括冷板和CDU,2026,high",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "relations.csv").write_text(
        "\n".join(
            [
                RELATION_HEADER,
                "relation_1,Company,英维克,HAS_PRODUCT,Product,液冷温控,英维克布局液冷温控产品,report_1,液冷白皮书,3,产业链,1,0.90,auto",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "entities.csv").write_text(
        "\n".join(
            [
                ENTITY_HEADER,
                'entity_1,Company,英维克,英维克,{},"[""report_1""]",auto,true',
                'entity_2,Product,液冷温控,液冷温控,{},"[""report_1""]",auto,',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "manifest.csv").write_text(
        "\n".join(
            [
                MANIFEST_HEADER,
                "report_1,液冷白皮书,https://example.com/report-1,2026,authority_whitepaper,Short snippets only,claims;evidence_spans;relations",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "segment_dossiers.jsonl").write_text(
        json.dumps({"topic": "液冷", "summary": "液冷样例", "evidence_ids": ["claim_1"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (path / "examples.jsonl").write_text(
        json.dumps({"query": "液冷产业链", "topics": ["液冷"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_validate_knowledge_pack_success(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    write_valid_pack(pack)

    result = validate_knowledge_pack(pack)

    assert result.ok is True
    assert result.exit_code == 0
    assert result.counts["claims"] == 1
    assert result.counts["sampled_evidence"] == 1


def test_validate_knowledge_pack_rejects_missing_manifest_field(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    write_valid_pack(pack)
    (pack / "manifest.csv").write_text(
        "source_report_id,source_title,published_at,source_type,license_or_usage_note,included_fields\n"
        "report_1,液冷白皮书,2026,authority_whitepaper,Short snippets only,claims\n",
        encoding="utf-8",
    )

    result = validate_knowledge_pack(pack)

    assert result.ok is False
    assert any("source_url" in issue.message for issue in result.issues)


def test_validate_knowledge_pack_rejects_unknown_source(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    write_valid_pack(pack)
    text = (pack / "claims.csv").read_text(encoding="utf-8").replace("report_1", "missing_report", 1)
    (pack / "claims.csv").write_text(text, encoding="utf-8")

    result = validate_knowledge_pack(pack)

    assert result.ok is False
    assert any("not in manifest" in issue.message for issue in result.issues)


def test_validate_knowledge_pack_rejects_long_evidence(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    write_valid_pack(pack)
    long_text = "x" * 701
    (pack / "evidence_spans.csv").write_text(
        "\n".join(
            [
                EVIDENCE_HEADER,
                f"evidence_1,claim_1,report_1,液冷白皮书,3,产业链,1,{long_text},2026,high",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_knowledge_pack(pack)

    assert result.ok is False
    assert any("too long" in issue.message for issue in result.issues)


def test_validate_knowledge_pack_rejects_bad_jsonl(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    write_valid_pack(pack)
    (pack / "examples.jsonl").write_text("{bad json\n", encoding="utf-8")

    result = validate_knowledge_pack(pack)

    assert result.ok is False
    assert any("JSONL parse error" in issue.message for issue in result.issues)


def test_build_sample_pack_filters_broker_research_and_is_deterministic(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    curated.mkdir()
    source_manifest = tmp_path / "reports_manifest.csv"
    write_curated_inputs(curated, source_manifest)

    output_a = tmp_path / "sample_a"
    output_b = tmp_path / "sample_b"
    build_sample_pack(curated_dir=curated, source_manifest=source_manifest, output_dir=output_a, claim_limit=10, relation_limit=10)
    build_sample_pack(curated_dir=curated, source_manifest=source_manifest, output_dir=output_b, claim_limit=10, relation_limit=10)

    result = validate_knowledge_pack(output_a)
    claims = read_csv(output_a / "claims.csv")
    manifest = read_csv(output_a / "manifest.csv")

    assert result.ok is True
    assert [row["source_report_id"] for row in claims] == ["official_1"]
    assert {row["source_report_id"] for row in manifest} == {"official_1"}
    assert (output_a / "claims.csv").read_text(encoding="utf-8") == (output_b / "claims.csv").read_text(encoding="utf-8")
    assert (output_a / "manifest.csv").read_text(encoding="utf-8") == (output_b / "manifest.csv").read_text(encoding="utf-8")


def write_curated_inputs(curated: Path, source_manifest: Path) -> None:
    (curated / "claims.csv").write_text(
        "\n".join(
            [
                CLAIM_HEADER,
                'claim_official,mechanism,液冷,液冷支撑高功率机柜,"[""英维克""]",高功率机柜散热,positive,,,,,official_1,官方白皮书,3,产业链,1,液冷产业链包括冷板和CDU,0.90,2026,direct,auto,,,',
                'claim_broker,mechanism,液冷,券商研报样例,"[""英维克""]",券商文本,positive,,,,,broker_1,券商研报,3,产业链,1,券商研报里的液冷证据,0.90,2026,direct,auto,,,',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (curated / "evidence_spans.csv").write_text(
        "\n".join(
            [
                EVIDENCE_HEADER,
                "evidence_official,claim_official,official_1,官方白皮书,3,产业链,1,液冷产业链包括冷板和CDU,2026,high",
                "evidence_broker,claim_broker,broker_1,券商研报,3,产业链,1,券商研报里的液冷证据,2026,high",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (curated / "relations.csv").write_text(
        "\n".join(
            [
                RELATION_HEADER,
                "relation_official,Company,英维克,HAS_PRODUCT,Product,液冷温控,英维克布局液冷温控产品,official_1,官方白皮书,3,产业链,1,0.90,auto",
                "relation_broker,Company,英维克,HAS_PRODUCT,Product,液冷温控,券商研报液冷产品证据,broker_1,券商研报,3,产业链,1,0.90,auto",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (curated / "entities.csv").write_text(
        "\n".join(
            [
                ENTITY_HEADER,
                'entity_1,Company,英维克,英维克,{},"[""official_1""]",auto,true',
                'entity_2,Product,液冷温控,液冷温控,{},"[""official_1""]",auto,',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_manifest.write_text(
        "\n".join(
            [
                "report_id,kind,company,stock_code,year,title,source_site,source_url,pdf_url,local_path,published_at,downloaded_at,sha256,file_size,pages,status,error,source_tier,source_type",
                "official_1,industry,,,,官方白皮书,官方,https://example.com/official.pdf,https://example.com/official.pdf,,2026,,,,,downloaded,,1,authority_whitepaper",
                "broker_1,research,,,,券商研报,券商,https://example.com/broker.pdf,https://example.com/broker.pdf,,2026,,,,,downloaded,,2,broker_research",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return [dict(row) for row in csv.DictReader(file)]
