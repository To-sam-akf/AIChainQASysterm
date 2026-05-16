"""Neo4j import helpers for verified KG CSV files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from src.extraction_schema import ENTITY_TYPES, RELATION_TYPES, normalize_name, read_csv
from src.llm_client import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]

CONSTRAINT_QUERIES = [
    f"CREATE CONSTRAINT {label.lower()}_normalized_name IF NOT EXISTS FOR (n:{label}) REQUIRE n.normalized_name IS UNIQUE"
    for label in ENTITY_TYPES
] + ["CREATE CONSTRAINT report_id IF NOT EXISTS FOR (n:Report) REQUIRE n.report_id IS UNIQUE"]


def assert_label(label: str) -> str:
    if label not in ENTITY_TYPES:
        raise ValueError(f"Unsupported entity label: {label}")
    return label


def assert_relation_type(relation_type: str) -> str:
    if relation_type not in RELATION_TYPES:
        raise ValueError(f"Unsupported relation type: {relation_type}")
    return relation_type


def parse_properties(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def validate_graph_csvs(entities_csv: Path, relations_csv: Path) -> tuple[int, int]:
    entity_rows = read_csv(entities_csv)
    relation_rows = read_csv(relations_csv)
    known_entities = set()
    for row in entity_rows:
        label = assert_label(row["type"])
        if not row.get("name") or not row.get("normalized_name"):
            raise ValueError(f"Invalid entity row: {row}")
        known_entities.add((label, row["normalized_name"]))
    for row in relation_rows:
        assert_label(row["head_type"])
        assert_label(row["tail_type"])
        assert_relation_type(row["relation"])
        if not row.get("head_name") or not row.get("tail_name") or not row.get("evidence"):
            raise ValueError(f"Invalid relation row: {row}")
    return len(entity_rows), len(relation_rows)


class Neo4jGraphLoader:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None) -> None:
        load_dotenv()
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password123")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        self.driver.close()

    def create_constraints(self) -> None:
        with self.driver.session() as session:
            for query in CONSTRAINT_QUERIES:
                session.run(query).consume()

    def clear_project_graph(self) -> None:
        with self.driver.session() as session:
            session.run(
                "MATCH (n) WHERE any(label IN labels(n) WHERE label IN $labels) DETACH DELETE n",
                labels=list(ENTITY_TYPES),
            ).consume()

    def load_entities(self, entities_csv: Path) -> int:
        rows = read_csv(entities_csv)
        with self.driver.session() as session:
            for row in rows:
                label = assert_label(row["type"])
                properties = parse_properties(row.get("properties", ""))
                params = {
                    "entity_id": row["entity_id"],
                    "name": row["name"],
                    "normalized_name": row["normalized_name"],
                    "properties": properties,
                    "source_report_ids": json.loads(row.get("source_report_ids") or "[]"),
                    "review_status": row.get("review_status", ""),
                    "report_id": properties.get("report_id", row["normalized_name"] if label == "Report" else ""),
                }
                query = (
                    f"MERGE (n:{label} {{normalized_name: $normalized_name}}) "
                    "SET n.entity_id = $entity_id, n.name = $name, "
                    "n.source_report_ids = $source_report_ids, n.review_status = $review_status, "
                    "n += $properties "
                    "FOREACH (_ IN CASE WHEN $report_id <> '' THEN [1] ELSE [] END | SET n.report_id = $report_id)"
                )
                session.run(query, **params).consume()
        return len(rows)

    def load_relations(self, relations_csv: Path) -> int:
        rows = read_csv(relations_csv)
        with self.driver.session() as session:
            for row in rows:
                head_label = assert_label(row["head_type"])
                tail_label = assert_label(row["tail_type"])
                relation_type = assert_relation_type(row["relation"])
                head_norm = normalize_name(row["head_name"], head_label)
                tail_norm = row["tail_name"] if tail_label == "Report" and row["tail_name"].startswith(("annual_", "research_", "industry_")) else normalize_name(row["tail_name"], tail_label)
                params = {
                    "head_norm": head_norm,
                    "tail_norm": tail_norm,
                    "head_name": row["head_name"],
                    "tail_name": row["tail_name"],
                    "relation_id": row["relation_id"],
                    "evidence": row.get("evidence", ""),
                    "source_report_id": row.get("source_report_id", ""),
                    "source_title": row.get("source_title", ""),
                    "page": row.get("page", ""),
                    "section": row.get("section", ""),
                    "source_tier": row.get("source_tier", ""),
                    "confidence": float(row.get("confidence") or 0.7),
                    "review_status": row.get("review_status", ""),
                }
                query = (
                    f"MERGE (h:{head_label} {{normalized_name: $head_norm}}) "
                    "ON CREATE SET h.name = $head_name "
                    f"MERGE (t:{tail_label} {{normalized_name: $tail_norm}}) "
                    "ON CREATE SET t.name = $tail_name "
                    f"MERGE (h)-[r:{relation_type} {{relation_id: $relation_id}}]->(t) "
                    "SET r.evidence = $evidence, r.source_report_id = $source_report_id, "
                    "r.source_title = $source_title, r.page = $page, r.section = $section, "
                    "r.source_tier = $source_tier, r.confidence = $confidence, "
                    "r.review_status = $review_status"
                )
                session.run(query, **params).consume()
        return len(rows)

    def load_graph(self, entities_csv: Path, relations_csv: Path, *, clear: bool = False) -> tuple[int, int]:
        if clear:
            self.clear_project_graph()
        self.create_constraints()
        entity_count = self.load_entities(entities_csv)
        relation_count = self.load_relations(relations_csv)
        return entity_count, relation_count
