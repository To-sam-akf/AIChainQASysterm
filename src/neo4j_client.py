"""Read-only Neo4j access for the QA engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.graph import Node, Path as Neo4jPath, Relationship

from src.cypher_guard import ensure_limit
from src.llm_client import ROOT_DIR, load_dotenv


DEFAULT_QUERY_LIMIT = 50


def to_plain_value(value: Any) -> Any:
    if isinstance(value, Node):
        payload = dict(value)
        payload["_labels"] = sorted(value.labels)
        return payload
    if isinstance(value, Relationship):
        payload = dict(value)
        payload["_type"] = value.type
        return payload
    if isinstance(value, Neo4jPath):
        return {
            "nodes": [to_plain_value(node) for node in value.nodes],
            "relationships": [to_plain_value(rel) for rel in value.relationships],
        }
    if isinstance(value, list):
        return [to_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain_value(item) for key, item in value.items()}
    return value


class Neo4jReadClient:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        *,
        database: str | None = None,
    ) -> None:
        load_dotenv(Path(ROOT_DIR) / ".env")
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password123")
        self.database = database or os.getenv("NEO4J_DATABASE", "")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        self.driver.close()

    def check_connection(self) -> tuple[bool, str]:
        try:
            with self._session() as session:
                value = session.run("RETURN 1 AS ok").single()
            return bool(value and value.get("ok") == 1), ""
        except Exception as exc:  # pragma: no cover - depends on local Neo4j service
            return False, str(exc)

    def run_read_query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> list[dict[str, Any]]:
        safe_cypher = ensure_limit(cypher, limit=limit)
        with self._session() as session:
            result = session.run(safe_cypher, **(params or {}))
            rows = []
            for record in result:
                rows.append({key: to_plain_value(record[key]) for key in record.keys()})
            return rows

    def _session(self) -> Any:
        if self.database:
            return self.driver.session(database=self.database)
        return self.driver.session()
