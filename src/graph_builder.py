"""Build reviewable entity and relation CSV files from extraction JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.extraction_schema import (
    ENTITY_CSV_FIELDS,
    RELATION_CSV_FIELDS,
    coerce_entity,
    coerce_relation,
    json_dumps,
    normalize_name,
    read_csv,
    sanitize_extraction_payload,
    stable_id,
    write_csv,
)


def report_entity_from_manifest(row: dict[str, str]) -> dict[str, Any]:
    properties = {
        "report_id": row["report_id"],
        "kind": row.get("kind", ""),
        "year": row.get("year", ""),
        "source_site": row.get("source_site", ""),
        "source_url": row.get("source_url", ""),
        "pdf_url": row.get("pdf_url", ""),
        "local_path": row.get("local_path", ""),
        "published_at": row.get("published_at", ""),
        "sha256": row.get("sha256", ""),
        "pages": row.get("pages", ""),
    }
    return {
        "type": "Report",
        "name": row.get("title") or row["report_id"],
        "normalized_name": row["report_id"],
        "properties": properties,
        "source_report_id": row["report_id"],
    }


def entity_csv_row(entity: dict[str, Any], source_report_ids: set[str] | None = None) -> dict[str, str]:
    entity_type = entity["type"]
    name = entity["name"]
    normalized = entity.get("normalized_name") or normalize_name(name, entity_type)
    report_ids = sorted(source_report_ids or {entity.get("source_report_id", "")} - {""})
    return {
        "entity_id": stable_id("entity", entity_type, normalized),
        "type": entity_type,
        "name": name,
        "normalized_name": normalized,
        "properties": json_dumps(entity.get("properties", {})),
        "source_report_ids": json_dumps(report_ids),
        "review_status": "auto",
    }


def relation_csv_row(relation: dict[str, Any]) -> dict[str, str]:
    relation_id = stable_id(
        "relation",
        relation["head_type"],
        normalize_name(relation["head_name"], relation["head_type"]),
        relation["relation"],
        relation["tail_type"],
        normalize_name(relation["tail_name"], relation["tail_type"]),
        relation.get("source_report_id", ""),
        relation.get("evidence", ""),
    )
    return {
        "relation_id": relation_id,
        "head_type": relation["head_type"],
        "head_name": relation["head_name"],
        "relation": relation["relation"],
        "tail_type": relation["tail_type"],
        "tail_name": relation["tail_name"],
        "evidence": relation.get("evidence", ""),
        "source_report_id": relation.get("source_report_id", ""),
        "source_title": relation.get("source_title", ""),
        "page": relation.get("page", ""),
        "section": relation.get("section", ""),
        "confidence": relation.get("confidence", "0.70"),
        "review_status": "auto",
    }


def add_entity(
    entities: dict[tuple[str, str], dict[str, Any]],
    source_index: dict[tuple[str, str], set[str]],
    entity: dict[str, Any],
    source_report_id: str,
) -> None:
    coerced = coerce_entity(entity)
    key = (coerced["type"], coerced["normalized_name"])
    if key not in entities:
        entities[key] = coerced
    if source_report_id:
        source_index.setdefault(key, set()).add(source_report_id)


def add_relation(relations: dict[str, dict[str, Any]], relation: dict[str, Any]) -> None:
    coerced = coerce_relation(relation)
    row = relation_csv_row(coerced)
    relations.setdefault(row["relation_id"], coerced)


def load_extraction_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def build_verified_graph(
    *,
    extraction_paths: list[Path],
    manifest_path: Path,
    entities_csv: Path,
    relations_csv: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    manifest_rows = [row for row in read_csv(manifest_path) if row.get("status") == "downloaded"]
    entities: dict[tuple[str, str], dict[str, Any]] = {}
    source_index: dict[tuple[str, str], set[str]] = {}
    relations: dict[str, dict[str, Any]] = {}

    for report in manifest_rows:
        report_entity = report_entity_from_manifest(report)
        add_entity(entities, source_index, report_entity, report["report_id"])

    for record in load_extraction_records(extraction_paths):
        cleaned, _ = sanitize_extraction_payload({"entities": record.get("entities", []), "relations": record.get("relations", [])})
        source_report_id = record.get("report_id", "")
        source_title = record.get("source_title", "")
        page = str(record.get("page", ""))
        section = record.get("section", "")
        touched_entities: list[dict[str, Any]] = []
        for entity in cleaned["entities"]:
            if entity["type"] == "Report":
                continue
            entity["source_report_id"] = source_report_id
            add_entity(entities, source_index, entity, source_report_id)
            touched_entities.append(entity)
        for relation in cleaned["relations"]:
            if relation["relation"] == "MENTIONED_IN":
                continue
            relation["source_report_id"] = relation.get("source_report_id") or source_report_id
            relation["source_title"] = relation.get("source_title") or source_title
            relation["page"] = relation.get("page") or page
            relation["section"] = relation.get("section") or section
            add_entity(
                entities,
                source_index,
                {"type": relation["head_type"], "name": relation["head_name"]},
                relation["source_report_id"],
            )
            add_entity(
                entities,
                source_index,
                {"type": relation["tail_type"], "name": relation["tail_name"]},
                relation["source_report_id"],
            )
            add_relation(relations, relation)
            touched_entities.extend(
                [
                    {"type": relation["head_type"], "name": relation["head_name"]},
                    {"type": relation["tail_type"], "name": relation["tail_name"]},
                ]
            )
        if source_report_id:
            for entity in touched_entities:
                if entity["type"] == "Report":
                    continue
                mention_relation = {
                    "head_type": entity["type"],
                    "head_name": entity["name"],
                    "relation": "MENTIONED_IN",
                    "tail_type": "Report",
                    "tail_name": source_report_id,
                    "evidence": f"实体在报告《{source_title or source_report_id}》中被抽取。",
                    "source_report_id": source_report_id,
                    "source_title": source_title,
                    "page": page,
                    "section": section,
                    "confidence": "1.00",
                }
                add_relation(relations, mention_relation)

    entity_rows = [entity_csv_row(entity, source_index.get(key, set())) for key, entity in sorted(entities.items())]
    relation_rows = [relation_csv_row(relation) for _, relation in sorted(relations.items())]
    write_csv(entities_csv, ENTITY_CSV_FIELDS, entity_rows)
    write_csv(relations_csv, RELATION_CSV_FIELDS, relation_rows)
    return entity_rows, relation_rows
