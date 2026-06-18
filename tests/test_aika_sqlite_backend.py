from __future__ import annotations

import json
from pathlib import Path

from aika import aika_cli
from aika.aika_core.backends.sqlite_backend import SQLiteResearchBackend, build_sqlite_index
from aika.aika_core.models import ClaimRecord, EvidenceCard


def write_sample_knowledge(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "claims.csv").write_text(
        "\n".join(
            [
                "claim_id,claim_type,topic,claim_text,companies,mechanism,direction,horizon,metric,value,unit,source_report_id,source_title,page,section,source_tier,evidence_span,confidence,as_of_date,exposure_level,review_status,reviewer_note,quality_flags,conflict_group_id",
                'claim_1,company_exposure,光模块,中际旭创在光模块产业链具备直接敞口,"[""中际旭创""]",光模块需求增长,positive,,,,,report_1,测试报告,12,业务,1,中际旭创800G光模块需求增长,0.90,2026,core,auto,,,',
                'claim_2,mechanism,液冷,液冷产业链包括冷板 CDU 和温控设备,"[""英维克""]",散热支撑高功率机柜,positive,,,,,report_2,液冷报告,3,产业链,1,液冷产业链包括冷板和CDU,0.85,2026,direct,auto,,,',
                'claim_3,risk,光模块,光模块需求可能受资本开支节奏影响,"[""中际旭创""]",资本开支波动,negative,,,,,report_1,测试报告,15,风险,1,资本开支节奏会影响光模块订单,0.70,2026,core,auto,,,',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "evidence_spans.csv").write_text(
        "\n".join(
            [
                "evidence_id,claim_id,source_report_id,source_title,page,section,source_tier,text,as_of_date,quality",
                "evidence_1,claim_1,report_1,测试报告,12,业务,1,中际旭创800G光模块需求增长,2026,high",
                "evidence_2,claim_2,report_2,液冷报告,3,产业链,1,液冷产业链包括冷板和CDU,2026,high",
                "evidence_3,claim_3,report_1,测试报告,15,风险,1,资本开支节奏会影响光模块订单,2026,medium",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "segment_dossiers.jsonl").write_text(
        json.dumps(
            {
                "topic": "液冷",
                "summary": "液冷产业链包括冷板、CDU、温控和机柜配套。",
                "technology_mechanism": ["高功率机柜推动液冷渗透率提升。"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "relations.csv").write_text(
        "\n".join(
            [
                "relation_id,head_type,head_name,relation,tail_type,tail_name,evidence,source_report_id,source_title,page,section,source_tier,confidence,review_status",
                "relation_1,Company,中际旭创,HAS_PRODUCT,Product,光模块,中际旭创布局高速光模块,report_1,测试报告,12,业务,1,0.90,auto",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "entities.csv").write_text(
        "\n".join(
            [
                "type,name,normalized_name,is_core_company",
                "Company,中际旭创,中际旭创,true",
                "Company,英维克,英维克,true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "manifest.csv").write_text(
        "\n".join(
            [
                "source_report_id,source_title,source_url,published_at,source_type,license_or_usage_note,included_fields",
                "report_1,测试报告,https://example.com/report-1,2026,company_annual_report,Public disclosure short snippets only,claims;evidence_spans;relations",
                "report_2,液冷报告,https://example.com/report-2,2026,authority_whitepaper,Public source short snippets only,claims;evidence_spans",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "examples.jsonl").write_text(
        json.dumps(
            {
                "query": "液冷产业链",
                "topics": ["液冷"],
                "suggested_command": 'aika search-evidence "液冷产业链" --top-k 5',
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_index_and_search_evidence_returns_citation_ids(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge" / "sample"
    index_path = tmp_path / "indexes" / "sample.sqlite"
    write_sample_knowledge(knowledge_dir)

    result = build_sqlite_index(knowledge_dir, index_path)
    backend = SQLiteResearchBackend(index_path)
    cards = backend.search_evidence("液冷产业链", top_k=5)

    assert index_path.exists()
    assert result["counts"]["evidence_spans"] == 3
    assert 0 < len(cards) <= 5
    assert all(isinstance(card, EvidenceCard) for card in cards)
    assert all(card.citation_id for card in cards)
    assert cards[0].citation_id == "E1"


def test_search_claims_returns_structured_records(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge" / "sample"
    index_path = tmp_path / "indexes" / "sample.sqlite"
    write_sample_knowledge(knowledge_dir)
    build_sqlite_index(knowledge_dir, index_path)

    claims = SQLiteResearchBackend(index_path).search_claims("中际旭创 光模块", top_k=5)

    assert 0 < len(claims) <= 5
    assert all(isinstance(claim, ClaimRecord) for claim in claims)
    assert claims[0].claim_id
    assert any(claim.claim_id == "claim_1" for claim in claims)


def test_filters_apply_to_claims_and_evidence(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge" / "sample"
    index_path = tmp_path / "indexes" / "sample.sqlite"
    write_sample_knowledge(knowledge_dir)
    build_sqlite_index(knowledge_dir, index_path)
    backend = SQLiteResearchBackend(index_path)

    claims = backend.search_claims("光模块", top_k=5, company="中际旭创", topic="光模块", claim_type="risk")
    evidence = backend.search_evidence("液冷", top_k=5, company="英维克", topic="液冷")
    no_match = backend.search_claims("光模块", top_k=5, company="英维克", claim_type="risk")

    assert claims
    assert all(claim.claim_type == "risk" for claim in claims)
    assert all("中际旭创" in claim.companies for claim in claims)
    assert evidence
    assert all(card.company == "英维克" for card in evidence)
    assert no_match == []


def test_missing_data_returns_empty_lists(tmp_path: Path) -> None:
    empty_knowledge_dir = tmp_path / "empty"
    index_path = tmp_path / "indexes" / "empty.sqlite"
    empty_knowledge_dir.mkdir()

    build_sqlite_index(empty_knowledge_dir, index_path)
    backend = SQLiteResearchBackend(index_path)
    missing_backend = SQLiteResearchBackend(tmp_path / "missing.sqlite")

    assert backend.search_evidence("液冷", top_k=5) == []
    assert backend.search_claims("液冷", top_k=5) == []
    assert missing_backend.search_evidence("液冷", top_k=5) == []
    assert missing_backend.search_claims("液冷", top_k=5) == []


def test_aika_cli_init_build_index_and_doctor(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    sample_source = tmp_path / "source"
    home = tmp_path / "home"
    write_sample_knowledge(sample_source)
    monkeypatch.setattr(aika_cli, "SAMPLE_SOURCE_DIR", sample_source)
    monkeypatch.setattr(
        aika_cli,
        "_check_mcp_server_and_tools",
        lambda timeout_seconds: [
            aika_cli.CliDoctorCheck("mcp_server", aika_cli.STATUS_PASS, "started"),
            aika_cli.CliDoctorCheck("mcp_tools", aika_cli.STATUS_PASS, "registered"),
        ],
    )
    monkeypatch.setattr(
        aika_cli,
        "_check_postgres_not_required",
        lambda: aika_cli.CliDoctorCheck("postgres_not_required", aika_cli.STATUS_PASS, "psycopg was not imported."),
    )

    assert aika_cli.main(["init", "--sample", "--home", str(home), "--force"]) == 0
    assert aika_cli.main(["build-index", "--home", str(home)]) == 0
    assert aika_cli.main(["validate-data", "--path", str(home / "knowledge" / "sample")]) == 0
    assert aika_cli.main(["doctor", "--home", str(home)]) == 0
    assert (home / "indexes" / "sample.sqlite").exists()
    assert (home / "logs").is_dir()
    for name in aika_cli.SAMPLE_FILES:
        assert (home / "knowledge" / "sample" / name).is_file()


def test_aika_cli_demo_uses_sqlite_sample_path(tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    sample_source = tmp_path / "source"
    home = tmp_path / "home"
    write_sample_knowledge(sample_source)
    monkeypatch.setattr(aika_cli, "SAMPLE_SOURCE_DIR", sample_source)

    assert aika_cli.main(["init", "--sample", "--home", str(home), "--force"]) == 0
    assert aika_cli.main(["build-index", "--home", str(home)]) == 0
    assert aika_cli.main(["demo", "--home", str(home)]) == 0

    output = capsys.readouterr().out
    assert "AIKA local demo" in output
    assert "Evidence cards:" in output
    assert "Claims:" in output
    assert "Brief title:" in output


def test_aika_cli_doctor_reports_missing_home_as_failure(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    sample_source = tmp_path / "source"
    write_sample_knowledge(sample_source)
    monkeypatch.setattr(aika_cli, "SAMPLE_SOURCE_DIR", sample_source)
    monkeypatch.setattr(
        aika_cli,
        "_check_mcp_server_and_tools",
        lambda timeout_seconds: [
            aika_cli.CliDoctorCheck("mcp_server", aika_cli.STATUS_PASS, "started"),
            aika_cli.CliDoctorCheck("mcp_tools", aika_cli.STATUS_PASS, "registered"),
        ],
    )

    report = aika_cli.run_cli_doctor(home=tmp_path / "missing")

    assert report.exit_code == 2
    assert any(check.name == "home" and check.status == aika_cli.STATUS_FAIL for check in report.checks)
    assert any(check.name == "sample_query" and check.status == aika_cli.STATUS_FAIL for check in report.checks)


def test_sample_source_status_falls_back_to_repo_data() -> None:
    status = aika_cli.sample_source_status(aika_cli.SAMPLE_SOURCE_DIR)

    assert status.available is True
    assert status.missing == []
