#!/usr/bin/env python3
"""Build the redistributable public AIKA sample knowledge pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from aika.aika_core.data_paths import (  # noqa: E402
    CLAIMS_FILE,
    ENTITIES_FILE,
    EVIDENCE_SPANS_FILE,
    EXAMPLES_FILE,
    MANIFEST_FILE,
    RELATIONS_FILE,
    SEGMENT_DOSSIERS_FILE,
)
from aika.aika_core.knowledge_pack import REQUIRED_MANIFEST_FIELDS, validate_knowledge_pack  # noqa: E402
from aika.research_claims import build_segment_dossiers  # noqa: E402


DEFAULT_CURATED_DIR = ROOT_DIR / "data" / "curated"
DEFAULT_SOURCE_MANIFEST = ROOT_DIR / "data" / "metadata" / "reports_manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "knowledge_packs" / "sample"

TARGET_TOPICS = ["液冷", "AI芯片", "算力网络", "光模块", "数据中心"]
TARGET_TERMS = {
    "液冷": ("液冷", "冷板", "浸没", "CDU", "温控", "散热", "thermal", "cooling"),
    "AI芯片": ("AI芯片", "GPU", "chiplet", "HBM", "UCIe", "加速", "异构", "封装", "DeepSeek", "MoE"),
    "算力网络": ("算力网络", "交换机", "以太网", "Ultra Ethernet", "UALink", "RDMA", "网络", "互联"),
    "光模块": ("光模块", "光通信", "CPO", "LPO", "硅光", "optical", "photonics"),
    "数据中心": ("数据中心", "智算中心", "机柜", "PUE", "IDC", "AIDC", "power density"),
}
ALLOWED_SOURCE_TYPES = {
    "company_annual_report",
    "authority_whitepaper",
    "technical_roadmap",
    "open_specification",
    "manual_open_specification",
    "benchmark_methodology",
    "technical_paper",
    "model_technical_report",
}
EXCLUDED_SOURCE_TYPES = {"broker_research", "broker_research_seed"}
CLAIM_TYPE_PRIORITY = {
    "company_exposure": 0,
    "mechanism": 1,
    "bottleneck": 2,
    "indicator": 3,
    "supply_chain": 4,
    "risk": 5,
    "trend": 6,
    "policy": 7,
}
LICENSE_NOTES = {
    "company_annual_report": "Public company disclosure; sample redistributes structured metadata and short evidence snippets only.",
    "authority_whitepaper": "Public authority publication; sample redistributes structured metadata and short evidence snippets only.",
    "technical_roadmap": "Public roadmap/reference material; sample redistributes structured metadata and short evidence snippets only.",
    "open_specification": "Open specification/reference material; sample redistributes structured metadata and short evidence snippets only.",
    "manual_open_specification": "Open specification reference; sample redistributes structured metadata and short evidence snippets only.",
    "benchmark_methodology": "Public benchmark methodology; sample redistributes structured metadata and short evidence snippets only.",
    "technical_paper": "Public technical paper; sample redistributes structured metadata and short evidence snippets only.",
    "model_technical_report": "Public technical report; sample redistributes structured metadata and short evidence snippets only.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the public AIKA sample knowledge pack.")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--claim-limit", type=int, default=160)
    parser.add_argument("--relation-limit", type=int, default=120)
    parser.add_argument("--max-text-chars", type=int, default=500)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_sample_pack(
        curated_dir=args.curated_dir,
        source_manifest=args.manifest,
        output_dir=args.output,
        claim_limit=args.claim_limit,
        relation_limit=args.relation_limit,
        max_text_chars=args.max_text_chars,
    )
    validation = validate_knowledge_pack(args.output)
    counts = ", ".join(f"{key}={value}" for key, value in sorted(result.items()))
    print(f"Built sample pack: {args.output}")
    print(f"Counts: {counts}")
    if not validation.ok:
        for issue in validation.issues:
            row = f" row {issue.row}" if issue.row is not None else ""
            print(f"[{issue.severity}] {issue.file}{row}: {issue.message}", file=sys.stderr)
        return validation.exit_code
    return 0


def build_sample_pack(
    *,
    curated_dir: Path,
    source_manifest: Path,
    output_dir: Path,
    claim_limit: int = 160,
    relation_limit: int = 120,
    max_text_chars: int = 500,
) -> dict[str, int]:
    source_rows = {row["report_id"]: row for row in read_csv(source_manifest) if row.get("report_id")}
    allowed_source_ids = {
        report_id
        for report_id, row in source_rows.items()
        if is_allowed_source(row.get("source_type", ""))
    }

    claims_fields, claims = read_csv_with_fields(curated_dir / CLAIMS_FILE)
    evidence_fields, evidence_rows = read_csv_with_fields(curated_dir / EVIDENCE_SPANS_FILE)
    relation_fields, relations = read_csv_with_fields(curated_dir / RELATIONS_FILE)
    entity_fields, entities = read_csv_with_fields(curated_dir / ENTITIES_FILE)

    selected_claims = select_claims(claims, allowed_source_ids, claim_limit)
    selected_claim_ids = {row.get("claim_id", "") for row in selected_claims}
    selected_evidence = [row for row in evidence_rows if row.get("claim_id", "") in selected_claim_ids]
    selected_relations = select_relations(relations, allowed_source_ids, relation_limit)
    selected_sources = sorted(
        {
            row.get("source_report_id", "")
            for row in [*selected_claims, *selected_evidence, *selected_relations]
            if row.get("source_report_id", "")
        }
    )

    selected_claims = [truncate_row(row, ("evidence_span",), max_text_chars) for row in selected_claims]
    selected_evidence = [truncate_row(row, ("text",), max_text_chars) for row in selected_evidence]
    selected_relations = [truncate_row(row, ("evidence",), max_text_chars) for row in selected_relations]
    selected_entities = select_entities(entities, entity_fields, selected_relations, selected_claims, selected_sources)
    dossiers = build_segment_dossiers(selected_claims)
    manifest_rows = build_pack_manifest(source_rows, selected_sources, selected_claims, selected_evidence, selected_relations)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / CLAIMS_FILE, claims_fields, selected_claims)
    write_csv(output_dir / EVIDENCE_SPANS_FILE, evidence_fields, selected_evidence)
    write_csv(output_dir / RELATIONS_FILE, relation_fields, selected_relations)
    write_csv(output_dir / ENTITIES_FILE, entity_fields, selected_entities)
    write_csv(output_dir / MANIFEST_FILE, REQUIRED_MANIFEST_FIELDS, manifest_rows)
    write_jsonl(output_dir / SEGMENT_DOSSIERS_FILE, dossiers)
    write_jsonl(output_dir / EXAMPLES_FILE, sample_examples())

    return {
        "claims": len(selected_claims),
        "evidence_spans": len(selected_evidence),
        "relations": len(selected_relations),
        "entities": len(selected_entities),
        "dossiers": len(dossiers),
        "manifest": len(manifest_rows),
        "examples": len(sample_examples()),
    }


def is_allowed_source(source_type: str) -> bool:
    value = str(source_type or "").strip()
    return value in ALLOWED_SOURCE_TYPES and value not in EXCLUDED_SOURCE_TYPES


def select_claims(rows: list[dict[str, str]], allowed_source_ids: set[str], limit: int) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row.get("source_report_id", "") in allowed_source_ids and row.get("topic", "") in TARGET_TOPICS
    ]
    candidates.sort(key=claim_sort_key)
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    per_topic_limit = max(1, limit // max(1, len(TARGET_TOPICS)))
    for topic in TARGET_TOPICS:
        for row in [item for item in candidates if item.get("topic", "") == topic][:per_topic_limit]:
            add_unique_row(selected, seen, row, "claim_id")
    for row in candidates:
        if len(selected) >= limit:
            break
        add_unique_row(selected, seen, row, "claim_id")
    selected.sort(key=claim_sort_key)
    return selected


def select_relations(rows: list[dict[str, str]], allowed_source_ids: set[str], limit: int) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row.get("source_report_id", "") in allowed_source_ids and relation_matches_target(row)
    ]
    candidates.sort(key=relation_sort_key)
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in candidates:
        if len(selected) >= limit:
            break
        add_unique_row(selected, seen, row, "relation_id")
    return selected


def claim_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    topic = row.get("topic", "")
    try:
        confidence = float(row.get("confidence", "") or 0)
    except ValueError:
        confidence = 0.0
    return (
        TARGET_TOPICS.index(topic) if topic in TARGET_TOPICS else len(TARGET_TOPICS),
        CLAIM_TYPE_PRIORITY.get(row.get("claim_type", ""), 99),
        -confidence,
        row.get("source_report_id", ""),
        natural_int(row.get("page", "")),
        row.get("claim_id", ""),
    )


def relation_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    score = max((topic_match_score(relation_text(row), topic) for topic in TARGET_TOPICS), default=0)
    return (-score, row.get("source_report_id", ""), natural_int(row.get("page", "")), row.get("relation_id", ""))


def relation_matches_target(row: dict[str, str]) -> bool:
    text = relation_text(row)
    return any(topic_match_score(text, topic) > 0 for topic in TARGET_TOPICS)


def topic_match_score(text: str, topic: str) -> int:
    normalized = normalize(text)
    score = 0
    for term in TARGET_TERMS.get(topic, (topic,)):
        if normalize(term) in normalized:
            score += 1
    return score


def relation_text(row: dict[str, str]) -> str:
    return " ".join(
        str(row.get(key, "") or "")
        for key in ("head_type", "head_name", "relation", "tail_type", "tail_name", "evidence", "section")
    )


def add_unique_row(rows: list[dict[str, str]], seen: set[str], row: dict[str, str], key: str) -> None:
    value = row.get(key, "")
    if not value or value in seen:
        return
    rows.append(dict(row))
    seen.add(value)


def select_entities(
    rows: list[dict[str, str]],
    fields: list[str],
    relations: list[dict[str, str]],
    claims: list[dict[str, str]],
    source_ids: list[str],
) -> list[dict[str, str]]:
    needed = {
        (row.get("head_type", ""), row.get("head_name", ""))
        for row in relations
        if row.get("head_type") and row.get("head_name")
    }
    needed.update(
        {
            (row.get("tail_type", ""), row.get("tail_name", ""))
            for row in relations
            if row.get("tail_type") and row.get("tail_name")
        }
    )
    for claim in claims:
        for company in parse_companies(claim.get("companies", "")):
            needed.add(("Company", company))

    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    selected_sources = set(source_ids)
    for row in rows:
        key = (row.get("type", ""), row.get("name", ""))
        source_report_ids = set(parse_json_list(row.get("source_report_ids", "")))
        if key not in needed and not (source_report_ids & selected_sources):
            continue
        if key in seen:
            continue
        selected.append(dict(row))
        seen.add(key)

    for entity_type, name in sorted(needed):
        key = (entity_type, name)
        if key in seen or not name:
            continue
        selected.append(synthetic_entity(fields, entity_type, name, source_ids))
        seen.add(key)

    selected.sort(key=lambda row: (row.get("type", ""), row.get("name", "")))
    return selected


def synthetic_entity(fields: list[str], entity_type: str, name: str, source_ids: list[str]) -> dict[str, str]:
    row = {field: "" for field in fields}
    row.update(
        {
            "entity_id": stable_id("entity", entity_type, name),
            "type": entity_type,
            "name": name,
            "normalized_name": normalize(name),
            "properties": "{}",
            "source_report_ids": json.dumps(source_ids, ensure_ascii=False),
            "review_status": "auto",
            "is_core_company": "true" if entity_type == "Company" else "",
        }
    )
    return row


def build_pack_manifest(
    source_rows: dict[str, dict[str, str]],
    source_ids: list[str],
    claims: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    relations: list[dict[str, str]],
) -> list[dict[str, str]]:
    included_by_source: dict[str, set[str]] = defaultdict(set)
    for row in claims:
        included_by_source[row.get("source_report_id", "")].add("claims")
    for row in evidence_rows:
        included_by_source[row.get("source_report_id", "")].add("evidence_spans")
    for row in relations:
        included_by_source[row.get("source_report_id", "")].add("relations")

    rows: list[dict[str, str]] = []
    for source_id in source_ids:
        source = source_rows.get(source_id, {})
        source_type = source.get("source_type", "")
        rows.append(
            {
                "source_report_id": source_id,
                "source_title": source.get("title", "") or source_id,
                "source_url": source.get("source_url", "") or source.get("pdf_url", ""),
                "published_at": source.get("published_at", "") or source.get("year", ""),
                "source_type": source_type,
                "license_or_usage_note": LICENSE_NOTES.get(
                    source_type,
                    "Sample redistributes structured metadata and short evidence snippets only.",
                ),
                "included_fields": ";".join(sorted(included_by_source.get(source_id, set()))),
            }
        )
    return rows


def sample_examples() -> list[dict[str, Any]]:
    return [
        {"query": "液冷产业链有哪些关键环节？", "topics": ["液冷"], "suggested_command": "aika search-evidence \"液冷产业链\" --top-k 5"},
        {"query": "AI芯片的主要技术瓶颈是什么？", "topics": ["AI芯片"], "suggested_command": "aika search-claims \"AI芯片 瓶颈\" --top-k 5"},
        {"query": "算力网络和高速互联有什么关系？", "topics": ["算力网络"], "suggested_command": "aika search-evidence \"算力网络 高速互联\" --top-k 5"},
        {"query": "数据中心能耗和PUE可以看哪些证据？", "topics": ["数据中心"], "suggested_command": "aika search-claims \"数据中心 PUE\" --top-k 5"},
        {"query": "光模块在AI算力链条中的作用是什么？", "topics": ["光模块"], "suggested_command": "aika search-evidence \"光模块 AI算力\" --top-k 5"},
    ]


def truncate_row(row: dict[str, str], fields: Iterable[str], limit: int) -> dict[str, str]:
    updated = dict(row)
    for field in fields:
        updated[field] = compact_text(updated.get(field, ""), limit)
    return updated


def compact_text(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def read_csv(path: Path) -> list[dict[str, str]]:
    _, rows = read_csv_with_fields(path)
    return rows


def read_csv_with_fields(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize_cell(row.get(field, "")) for field in fields})


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def serialize_cell(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "")


def parse_companies(value: str) -> list[str]:
    return [str(item).strip() for item in parse_json_list(value) if str(item).strip()]


def parse_json_list(value: str) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [item.strip() for item in str(value).split(";") if item.strip()]
    if isinstance(parsed, list):
        return parsed
    return []


def natural_int(value: str) -> int:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else 999999


def normalize(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


if __name__ == "__main__":
    raise SystemExit(main())
