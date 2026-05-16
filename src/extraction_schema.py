"""Schema and validation helpers for KG extraction outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ENTITY_TYPES = (
    "Company",
    "Technology",
    "Product",
    "IndustryChain",
    "IndustryConcept",
    "Policy",
    "Standard",
    "ValueChainSegment",
    "Metric",
    "Risk",
    "Report",
)

RELATION_TYPES = (
    "USES_TECHNOLOGY",
    "HAS_PRODUCT",
    "BELONGS_TO_CHAIN",
    "HAS_METRIC",
    "DISCLOSES_RISK",
    "MENTIONED_IN",
    "UPSTREAM_OF",
    "DOWNSTREAM_OF",
    "ENABLES",
    "CONSTRAINS",
    "DEFINES",
    "SUPPORTED_BY_POLICY",
)

CHAIN_ENTITY_TYPES = (
    "IndustryChain",
    "IndustryConcept",
    "Technology",
    "Product",
    "ValueChainSegment",
)

RELATION_SIGNATURES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "USES_TECHNOLOGY": (("Company",), ("Technology",)),
    "HAS_PRODUCT": (("Company",), ("Product",)),
    "BELONGS_TO_CHAIN": (("Company",), ("IndustryChain", "ValueChainSegment")),
    "HAS_METRIC": (("Company",), ("Metric",)),
    "DISCLOSES_RISK": (("Company",), ("Risk",)),
    "MENTIONED_IN": (tuple(t for t in ENTITY_TYPES if t != "Report"), ("Report",)),
    "UPSTREAM_OF": (CHAIN_ENTITY_TYPES, CHAIN_ENTITY_TYPES),
    "DOWNSTREAM_OF": (CHAIN_ENTITY_TYPES, CHAIN_ENTITY_TYPES),
    "ENABLES": (("Technology", "Product", "IndustryConcept", "ValueChainSegment"), CHAIN_ENTITY_TYPES),
    "CONSTRAINS": (("Risk", "Policy", "Standard", "IndustryConcept"), tuple(t for t in ENTITY_TYPES if t != "Report")),
    "DEFINES": (("IndustryConcept", "Policy", "Standard"), ("IndustryConcept", "Technology", "ValueChainSegment")),
    "SUPPORTED_BY_POLICY": (("Company", *CHAIN_ENTITY_TYPES), ("Policy",)),
}

ENTITY_CSV_FIELDS = [
    "entity_id",
    "type",
    "name",
    "normalized_name",
    "properties",
    "source_report_ids",
    "review_status",
    "is_core_company",
]

RELATION_CSV_FIELDS = [
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


class SchemaError(ValueError):
    """Raised when extracted KG data does not match the project schema."""


def normalize_name(value: str, entity_type: str | None = None) -> str:
    value = str(value or "").strip()
    value = re.sub(r"\s+", "", value)
    value = value.replace("（", "(").replace("）", ")")
    if entity_type == "Company":
        value = re.sub(r"(股份有限公司|有限公司|股份)$", "", value)
    return value.casefold()


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "||".join(str(part or "") for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence_match:
        text = fence_match.group(1)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaError("LLM response must be a JSON object")
    return payload


def _clean_properties(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def infer_metric_properties(name: str, evidence: str = "", properties: dict[str, Any] | None = None) -> dict[str, Any]:
    """Infer minimal structured metric fields from metric text when the LLM omits them."""
    cleaned = dict(properties or {})
    text = f"{name} {evidence}"
    if not cleaned.get("year"):
        year_match = re.search(r"(20\d{2})\s*年?", text)
        if year_match:
            cleaned["year"] = year_match.group(1)
    if not cleaned.get("value"):
        value_match = re.search(r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|亿元|万元|千元|亿美元|万台|台|个|EFLOPS|PFLOPS|TFLOPS|EB|GB|bps|G|T)?)", text, flags=re.I)
        if value_match:
            cleaned["value"] = value_match.group(1).strip()
    if not cleaned.get("unit") and cleaned.get("value"):
        unit_match = re.search(r"(%|亿元|万元|千元|亿美元|万台|台|个|EFLOPS|PFLOPS|TFLOPS|EB|GB|bps|G|T)$", str(cleaned["value"]), flags=re.I)
        if unit_match:
            cleaned["unit"] = unit_match.group(1)
    if not cleaned.get("metric_name"):
        cleaned["metric_name"] = name
    return cleaned


def metric_has_structured_fields(properties: dict[str, Any]) -> bool:
    return any(str(properties.get(key, "")).strip() for key in ("year", "value", "unit"))


def coerce_entity(entity: dict[str, Any]) -> dict[str, Any]:
    entity_type = str(entity.get("type") or "").strip()
    name = str(entity.get("name") or "").strip()
    if entity_type not in ENTITY_TYPES:
        raise SchemaError(f"Invalid entity type: {entity_type}")
    if not name:
        raise SchemaError("Entity name is required")
    normalized = str(entity.get("normalized_name") or "").strip() or normalize_name(name, entity_type)
    properties = _clean_properties(entity.get("properties"))
    if entity_type == "Metric":
        properties = infer_metric_properties(name, properties=properties)
        if not metric_has_structured_fields(properties):
            raise SchemaError("Metric must include at least one of year, value, or unit")
    return {
        "type": entity_type,
        "name": name,
        "normalized_name": normalized,
        "properties": properties,
    }


def coerce_relation(relation: dict[str, Any]) -> dict[str, Any]:
    relation_type = str(relation.get("relation") or "").strip()
    head_type = str(relation.get("head_type") or "").strip()
    tail_type = str(relation.get("tail_type") or "").strip()
    head_name = str(relation.get("head") or relation.get("head_name") or "").strip()
    tail_name = str(relation.get("tail") or relation.get("tail_name") or "").strip()
    evidence = str(relation.get("evidence") or "").strip()
    if relation_type not in RELATION_TYPES:
        raise SchemaError(f"Invalid relation type: {relation_type}")
    if head_type not in ENTITY_TYPES or tail_type not in ENTITY_TYPES:
        raise SchemaError(f"Invalid relation endpoint types: {head_type}->{tail_type}")
    allowed_heads, allowed_tails = RELATION_SIGNATURES[relation_type]
    if head_type not in allowed_heads or tail_type not in allowed_tails:
        raise SchemaError(f"Relation {relation_type} does not allow {head_type}->{tail_type}")
    if not head_name or not tail_name:
        raise SchemaError("Relation endpoints are required")
    if not evidence:
        raise SchemaError("Relation evidence is required")
    if tail_type == "Metric":
        metric_properties = infer_metric_properties(tail_name, evidence)
        if not metric_has_structured_fields(metric_properties):
            raise SchemaError("Metric relation tail must include at least one of year, value, or unit")
    confidence = relation.get("confidence", 0.7)
    try:
        confidence_float = float(confidence)
    except (TypeError, ValueError):
        confidence_float = 0.7
    confidence_float = max(0.0, min(1.0, confidence_float))
    return {
        "head_type": head_type,
        "head_name": head_name,
        "relation": relation_type,
        "tail_type": tail_type,
        "tail_name": tail_name,
        "evidence": evidence,
        "source_report_id": str(relation.get("source_report_id") or relation.get("source") or "").strip(),
        "source_title": str(relation.get("source_title") or "").strip(),
        "page": str(relation.get("page") or "").strip(),
        "section": str(relation.get("section") or "").strip(),
        "source_tier": str(relation.get("source_tier") or "").strip(),
        "confidence": f"{confidence_float:.2f}",
    }


def validate_extraction_payload(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if "entities" not in payload or "relations" not in payload:
        raise SchemaError("Extraction JSON must contain entities and relations")
    entities = payload.get("entities")
    relations = payload.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise SchemaError("entities and relations must be lists")
    cleaned_entities = []
    cleaned_relations = []
    for entity in entities:
        if not isinstance(entity, dict):
            raise SchemaError("Entity item must be an object")
        cleaned_entities.append(coerce_entity(entity))
    for relation in relations:
        if not isinstance(relation, dict):
            raise SchemaError("Relation item must be an object")
        cleaned_relations.append(coerce_relation(relation))
    return {"entities": cleaned_entities, "relations": cleaned_relations}


def sanitize_extraction_payload(payload: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Coerce valid items and return rejected item errors without failing the batch."""
    rejected: list[str] = []
    entities = payload.get("entities", [])
    relations = payload.get("relations", [])
    cleaned_entities: list[dict[str, Any]] = []
    cleaned_relations: list[dict[str, Any]] = []
    if not isinstance(entities, list):
        rejected.append("entities must be a list")
        entities = []
    if not isinstance(relations, list):
        rejected.append("relations must be a list")
        relations = []
    for index, entity in enumerate(entities):
        try:
            if not isinstance(entity, dict):
                raise SchemaError("Entity item must be an object")
            cleaned_entities.append(coerce_entity(entity))
        except SchemaError as exc:
            rejected.append(f"entities[{index}]: {exc}")
    for index, relation in enumerate(relations):
        try:
            if not isinstance(relation, dict):
                raise SchemaError("Relation item must be an object")
            cleaned_relations.append(coerce_relation(relation))
        except SchemaError as exc:
            rejected.append(f"relations[{index}]: {exc}")
    return {"entities": cleaned_entities, "relations": cleaned_relations}, rejected
