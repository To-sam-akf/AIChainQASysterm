"""Build a cleaner professional graph view from automatically verified CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.domain_lexicon import (
    BUSINESS_RELATIONS,
    canonical_company_name,
    company_lookup,
    is_core_company,
    is_core_metric,
    is_disclaimer_text,
    is_noise_section,
    looks_like_definition_noise,
)
from src.extraction_schema import ENTITY_CSV_FIELDS, RELATION_CSV_FIELDS, normalize_name, read_csv, write_csv
from src.frontend_data import DEFAULT_ENTITIES_CSV, DEFAULT_RELATIONS_CSV


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CURATED_DIR = ROOT_DIR / "data" / "curated"
CURATED_ENTITIES_CSV = DEFAULT_CURATED_DIR / "entities.csv"
CURATED_RELATIONS_CSV = DEFAULT_CURATED_DIR / "relations.csv"


def relation_quality_reasons(row: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    relation = row.get("relation", "")
    head_type = row.get("head_type", "")
    evidence = row.get("evidence", "")
    section = row.get("section", "")

    if relation == "MENTIONED_IN":
        return reasons
    if is_disclaimer_text(evidence) and relation in BUSINESS_RELATIONS:
        reasons.append("disclaimer_or_research_boilerplate")
    if looks_like_definition_noise(row):
        reasons.append("definition_or_abbreviation_page")
    if relation in {"USES_TECHNOLOGY", "HAS_PRODUCT", "BELONGS_TO_CHAIN"} and is_noise_section(section):
        reasons.append("low_value_section")
    if relation == "HAS_METRIC" and not is_core_metric(row):
        reasons.append("non_core_metric")
    if head_type == "Company" and not is_core_company(row.get("head_name", "")):
        reasons.append("non_core_company_head")
    return reasons


def should_keep_relation(row: dict[str, str]) -> bool:
    relation = row.get("relation", "")
    if relation == "MENTIONED_IN":
        return False
    return not relation_quality_reasons(row)


def canonicalize_relation(row: dict[str, str]) -> dict[str, str]:
    updated = dict(row)
    if updated.get("head_type") == "Company":
        updated["head_name"] = canonical_company_name(updated.get("head_name", ""))
    if updated.get("tail_type") == "Company":
        updated["tail_name"] = canonical_company_name(updated.get("tail_name", ""))
    return updated


def entity_key(entity_type: str, name: str) -> tuple[str, str]:
    return entity_type, normalize_name(name, entity_type)


def relation_endpoint_keys(relations: Iterable[dict[str, str]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in relations:
        keys.add(entity_key(row["head_type"], row["head_name"]))
        keys.add(entity_key(row["tail_type"], row["tail_name"]))
    return keys


def build_curated_graph(
    *,
    entities_csv: Path = DEFAULT_ENTITIES_CSV,
    relations_csv: Path = DEFAULT_RELATIONS_CSV,
    output_dir: Path = DEFAULT_CURATED_DIR,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    entities = read_csv(entities_csv)
    relations = read_csv(relations_csv)

    kept_relations: list[dict[str, str]] = []
    seen_relation_ids: set[str] = set()
    for row in relations:
        updated = canonicalize_relation(row)
        if not should_keep_relation(updated):
            continue
        relation_id = updated.get("relation_id", "")
        if relation_id and relation_id in seen_relation_ids:
            continue
        seen_relation_ids.add(relation_id)
        kept_relations.append(updated)

    used_entities = relation_endpoint_keys(kept_relations)
    lookup = company_lookup()
    for company in lookup.companies:
        if company.is_core_company:
            used_entities.add(entity_key("Company", company.company))
    for entity in entities:
        if entity.get("type") == "Report":
            used_entities.add((entity["type"], entity["normalized_name"]))

    kept_entities: list[dict[str, str]] = []
    seen_entities: set[tuple[str, str]] = set()
    for entity in entities:
        row = dict(entity)
        if row.get("type") == "Company":
            row["name"] = canonical_company_name(row.get("name", ""))
            row["normalized_name"] = normalize_name(row["name"], "Company")
            row["is_core_company"] = "true" if is_core_company(row["name"]) else "false"
            if row["is_core_company"] != "true":
                continue
        key = (row["type"], row["normalized_name"])
        if key not in used_entities or key in seen_entities:
            continue
        seen_entities.add(key)
        kept_entities.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "entities.csv", ENTITY_CSV_FIELDS, kept_entities)
    write_csv(output_dir / "relations.csv", RELATION_CSV_FIELDS, kept_relations)
    return kept_entities, kept_relations

