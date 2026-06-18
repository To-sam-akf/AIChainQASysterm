import csv
import json
from pathlib import Path

from aika.frontend_data import LocalKnowledgeGraph
from aika.qa_engine import QAEngine
from aika.question_planner import heuristic_plan_question
from aika.research_claims import (
    CLAIM_CSV_FIELDS,
    ResearchMemory,
    build_research_artifacts,
    claims_from_technical_chunks,
)


RELATION_FIELDS = [
    "relation_id",
    "head_type",
    "head_name",
    "relation",
    "tail_type",
    "tail_name",
    "evidence",
    "source_report_id",
    "source_title",
    "page",
    "section",
    "source_tier",
    "confidence",
    "review_status",
]


def write_relations(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RELATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_chunks(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_research_artifacts_ranks_direct_liquid_cooling_exposure(tmp_path: Path) -> None:
    relations = tmp_path / "relations.csv"
    output = tmp_path / "curated"
    write_relations(
        relations,
        [
            {
                "relation_id": "r1",
                "head_type": "Company",
                "head_name": "英维克",
                "relation": "HAS_PRODUCT",
                "tail_type": "Product",
                "tail_name": "液冷CDU",
                "evidence": "英维克提供从冷板、Manifold、CDU 到管路冷源的端到端液冷产品。",
                "source_report_id": "annual_002837_2025",
                "source_title": "英维克2025年年度报告",
                "page": "9",
                "section": "液冷",
                "source_tier": "1",
                "confidence": "0.95",
                "review_status": "auto",
            },
            {
                "relation_id": "r2",
                "head_type": "Company",
                "head_name": "欧陆通",
                "relation": "USES_TECHNOLOGY",
                "tail_type": "Technology",
                "tail_name": "高压直流输入",
                "evidence": "公司围绕高功率、高效率、高效液冷兼容设计等服务器电源技术方向深化布局。",
                "source_report_id": "annual_300870_2025",
                "source_title": "欧陆通2025年年度报告",
                "page": "15",
                "section": "服务器电源",
                "source_tier": "1",
                "confidence": "0.80",
                "review_status": "auto",
            },
        ],
    )

    claims, evidence_spans, dossiers = build_research_artifacts(relations_csv=relations, output_dir=output)

    assert (output / "claims.csv").exists()
    assert (output / "evidence_spans.csv").exists()
    assert (output / "segment_dossiers.jsonl").exists()
    assert claims
    assert evidence_spans
    liquid = next(dossier for dossier in dossiers if dossier["topic"] == "液冷")
    assert "英维克" in liquid["company_exposure"]["core"]
    assert "欧陆通" in liquid["company_exposure"]["indirect"]


def test_technical_chunks_generate_direct_claims_and_dossiers(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    write_chunks(
        chunks_dir / "industry_tech_sample.jsonl",
        [
            {
                "chunk_id": "tech_legal",
                "report_id": "industry_tech_ualink_200g_2025",
                "kind": "industry",
                "source_title": "UALink 200G 1.0 Specification",
                "source_tier": "1",
                "source_type": "open_specification",
                "page": "2",
                "section": "LEGAL NOTICE",
                "year": "2025",
                "text": "LEGAL NOTICE ALL RIGHTS RESERVED TRADEMARK NO LICENSE GOVERNING DOCUMENTS.",
            },
            {
                "chunk_id": "tech_ualink",
                "report_id": "industry_tech_ualink_200g_2025",
                "kind": "industry",
                "source_title": "UALink 200G 1.0 Specification",
                "source_tier": "1",
                "source_type": "open_specification",
                "page": "20",
                "section": "Scale Up",
                "year": "2025",
                "text": "UALink is a scale-up interconnect for AI accelerator pods. The 200G interface enables a switch fabric for low-latency accelerator-to-accelerator communication.",
            },
            {
                "chunk_id": "tech_deepseek",
                "report_id": "industry_tech_deepseek_v3_2025",
                "kind": "industry",
                "source_title": "DeepSeek-V3 Technical Report",
                "source_tier": "2",
                "source_type": "model_technical_report",
                "page": "5",
                "section": "FP8 Training",
                "year": "2025",
                "text": "DeepSeek-V3 validates FP8 mixed precision training on an extremely large-scale model and overcomes the communication bottleneck with compute-communication overlap, requiring 2.788M H800 GPU hours.",
            },
            {
                "chunk_id": "tech_ucie",
                "report_id": "industry_tech_ucie_2_whitepaper_2024",
                "kind": "industry",
                "source_title": "UCIe 2.0 Specification Continuing Innovation to Drive an Open Chiplet Ecosystem",
                "source_tier": "1",
                "source_type": "open_specification",
                "page": "4",
                "section": "Chiplet",
                "year": "2024",
                "text": "UCIe supports an open chiplet ecosystem and advanced packaging, enabling heterogeneous integration for AI accelerators.",
            },
            {
                "chunk_id": "tech_thermal",
                "report_id": "industry_tech_hir_thermal_2023",
                "kind": "industry",
                "source_title": "HIR 2023 Thermal",
                "source_tier": "1",
                "source_type": "technical_roadmap",
                "page": "9",
                "section": "Thermal Management",
                "year": "2023",
                "text": "Liquid cooling with cold plate and CDU architecture relieves thermal management constraints as AI accelerator power density rises.",
            },
        ],
    )
    relations = tmp_path / "relations.csv"
    output = tmp_path / "curated"
    write_relations(relations, [])

    direct_claims = claims_from_technical_chunks(chunks_dir)
    claims, _, dossiers = build_research_artifacts(relations_csv=relations, output_dir=output, chunks_dir=chunks_dir)

    assert direct_claims
    assert all("LEGAL NOTICE" not in claim["evidence_span"] for claim in direct_claims)
    assert {claim["topic"] for claim in direct_claims} >= {"算力网络", "AI芯片", "液冷"}
    assert not any(claim["companies"] for claim in direct_claims)
    assert any(claim["claim_type"] == "indicator" and "GPU hours" in claim["evidence_span"] for claim in claims)
    network = next(dossier for dossier in dossiers if dossier["topic"] == "算力网络")
    chip = next(dossier for dossier in dossiers if dossier["topic"] == "AI芯片")
    thermal = next(dossier for dossier in dossiers if dossier["topic"] == "液冷")
    assert "UALink" in network["summary"]
    assert "FP8" in chip["summary"] or "UCIe" in chip["summary"]
    assert "Liquid cooling" in thermal["summary"]


def test_research_memory_improves_topic_company_answer(tmp_path: Path) -> None:
    claims_path = tmp_path / "claims.csv"
    dossiers_path = tmp_path / "segment_dossiers.jsonl"
    with claims_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CLAIM_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "claim_id": "c1",
                "claim_type": "company_exposure",
                "topic": "液冷",
                "claim_text": "英维克 对 液冷 的公司敞口为 core：拥有液冷CDU。",
                "companies": "[\"英维克\"]",
                "mechanism": "英维克 在 液冷 的关系为拥有产品液冷CDU",
                "direction": "neutral",
                "horizon": "near_term",
                "metric": "",
                "value": "",
                "unit": "",
                "source_report_id": "annual_002837_2025",
                "source_title": "英维克2025年年度报告",
                "page": "9",
                "section": "液冷",
                "source_tier": "1",
                "evidence_span": "英维克提供端到端液冷产品。",
                "confidence": "0.95",
                "as_of_date": "2025",
                "exposure_level": "core",
            }
        )
    dossiers_path.write_text(
        '{"topic":"液冷","summary":"液冷直接敞口公司包括英维克","company_exposure":{"core":["英维克"]},"technology_mechanism":["液冷降低高功率机柜散热瓶颈"],"leading_indicators":[],"bottlenecks":[],"risks":[],"gaps":["缺少订单指标"],"evidence_ids":["c1"]}\n',
        encoding="utf-8",
    )
    memory = ResearchMemory.load(tmp_path)
    graph = LocalKnowledgeGraph(
        entities=[],
        relations=[
            {
                "head_type": "Company",
                "head_name": "英维克",
                "relation": "HAS_PRODUCT",
                "tail_type": "Product",
                "tail_name": "液冷CDU",
                "evidence": "英维克提供端到端液冷产品。",
                "source_title": "英维克2025年年度报告",
                "page": "9",
                "source_tier": "1",
                "section": "液冷",
            }
        ],
    )
    engine = QAEngine(csv_graph=graph, rag_index=None, research_memory=memory, llm_client=None)

    result = engine.answer_question("液冷产业链有哪些上市公司？")

    assert result["diagnostics"]["research_hits"] >= 1
    assert "核心敞口" in result["answer"]
    assert "英维克" in result["answer"]


def test_research_memory_returns_dossier_for_bottleneck_question(tmp_path: Path) -> None:
    claims_path = tmp_path / "claims.csv"
    dossiers_path = tmp_path / "segment_dossiers.jsonl"
    with claims_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CLAIM_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "claim_id": "c1",
                "claim_type": "bottleneck",
                "topic": "国产算力",
                "claim_text": "国产算力 的约束或瓶颈：国产高性能芯片受限。",
                "companies": "[]",
                "mechanism": "芯力掣肘约束智算一体化服务",
                "direction": "negative",
                "horizon": "near_term",
                "metric": "",
                "value": "",
                "unit": "",
                "source_report_id": "research_1",
                "source_title": "国产算力研报",
                "page": "19",
                "section": "算力",
                "source_tier": "2",
                "evidence_span": "国产高性能芯片受限、异构资源效率损失。",
                "confidence": "0.9",
                "as_of_date": "2025",
                "exposure_level": "",
            }
        )
    dossiers_path.write_text(
        '{"topic":"国产算力","summary":"国产算力核心瓶颈在高性能芯片和异构资源效率","company_exposure":{},"technology_mechanism":["软硬件协同带动一体化发展"],"leading_indicators":["国产芯片供给和上架率"],"bottlenecks":["国产高性能芯片受限"],"risks":[],"gaps":[],"evidence_ids":["c1"]}\n',
        encoding="utf-8",
    )
    memory = ResearchMemory.load(tmp_path)
    plan = heuristic_plan_question("国产算力产业链的核心瓶颈和跟踪指标是什么？")

    hits = memory.search("国产算力产业链的核心瓶颈和跟踪指标是什么？", plan)

    assert hits[0].kind == "dossier"
    assert any(hit.claim_type == "bottleneck" for hit in hits)
