"""CSV knowledge-graph adapters for AIKA Core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.aika_core.data_paths import DEFAULT_CURATED_DIR
from src.aika_core.models import GraphEdge
from src.frontend_data import LocalKnowledgeGraph, RELATION_LABELS, subgraph_edges


def load_graph(data_dir: str | Path = DEFAULT_CURATED_DIR) -> LocalKnowledgeGraph:
    path = Path(data_dir)
    if not path.exists():
        return LocalKnowledgeGraph(entities=[], relations=[])
    return LocalKnowledgeGraph.from_dir(path)


def query_graph_edges(
    graph: LocalKnowledgeGraph,
    *,
    company: str = "",
    technology: str = "",
    relation_type: str = "",
    limit: int = 80,
) -> list[GraphEdge]:
    rows = graph.subgraph_relations(
        company=company,
        technology=technology,
        relation_type=relation_type,
        limit=max(0, int(limit)),
    )
    return [edge_from_relation_row(row) for row in rows]


def edge_from_relation_row(row: dict[str, Any]) -> GraphEdge:
    relation = str(row.get("relation") or "")
    return GraphEdge(
        source=str(row.get("head_name") or ""),
        target=str(row.get("tail_name") or ""),
        relation=relation,
        label=RELATION_LABELS.get(relation, relation),
        source_type=str(row.get("head_type") or ""),
        target_type=str(row.get("tail_type") or ""),
        evidence=str(row.get("evidence") or ""),
        source_title=str(row.get("source_title") or row.get("source") or ""),
        page=str(row.get("page") or ""),
        section=str(row.get("section") or ""),
        source_tier=str(row.get("source_tier") or ""),
        report_id=str(row.get("source_report_id") or row.get("report_id") or ""),
        raw=dict(row),
    )


def edge_from_graph_record(row: dict[str, Any]) -> GraphEdge:
    relation = str(row.get("relation") or "")
    return GraphEdge(
        source=str(row.get("company") or row.get("head_name") or ""),
        target=str(row.get("target") or row.get("tail_name") or ""),
        relation=relation,
        label=RELATION_LABELS.get(relation, relation),
        source_type=str((row.get("company_labels") or [row.get("head_type", "")])[0] or ""),
        target_type=str((row.get("target_labels") or [row.get("tail_type", "")])[0] or ""),
        evidence=str(row.get("evidence") or ""),
        source_title=str(row.get("source") or row.get("source_title") or ""),
        page=str(row.get("page") or ""),
        section=str(row.get("section") or ""),
        source_tier=str(row.get("source_tier") or ""),
        report_id=str(row.get("report_id") or row.get("source_report_id") or ""),
        raw=dict(row),
    )


__all__ = [
    "LocalKnowledgeGraph",
    "RELATION_LABELS",
    "edge_from_graph_record",
    "edge_from_relation_row",
    "load_graph",
    "query_graph_edges",
    "subgraph_edges",
]
