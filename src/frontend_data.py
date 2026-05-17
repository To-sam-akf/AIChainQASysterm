"""Data access and local QA helpers for the Streamlit frontend."""

from __future__ import annotations

import csv
import html
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.extraction_schema import normalize_name


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENTITIES_CSV = ROOT_DIR / "data" / "verified" / "entities.csv"
DEFAULT_RELATIONS_CSV = ROOT_DIR / "data" / "verified" / "relations.csv"

RELATION_LABELS = {
    "USES_TECHNOLOGY": "涉及技术",
    "HAS_PRODUCT": "拥有产品",
    "BELONGS_TO_CHAIN": "属于环节",
    "HAS_METRIC": "披露指标",
    "DISCLOSES_RISK": "披露风险",
    "MENTIONED_IN": "来源报告",
    "UPSTREAM_OF": "上游",
    "DOWNSTREAM_OF": "下游",
    "ENABLES": "使能",
    "CONSTRAINS": "约束",
    "DEFINES": "定义",
    "SUPPORTED_BY_POLICY": "政策支撑",
}

QUESTION_RELATIONS = {
    "技术": "USES_TECHNOLOGY",
    "产品": "HAS_PRODUCT",
    "产业链": "BELONGS_TO_CHAIN",
    "环节": "BELONGS_TO_CHAIN",
    "指标": "HAS_METRIC",
    "财务": "HAS_METRIC",
    "风险": "DISCLOSES_RISK",
    "政策": "SUPPORTED_BY_POLICY",
    "定义": "DEFINES",
    "上游": "UPSTREAM_OF",
    "下游": "DOWNSTREAM_OF",
}

ENTITY_ORDER = [
    "Company",
    "IndustryChain",
    "ValueChainSegment",
    "IndustryConcept",
    "Technology",
    "Product",
    "Metric",
    "Risk",
    "Policy",
    "Standard",
    "Report",
]

ENTITY_COLORS = {
    "Company": "#1f6f70",
    "Technology": "#2563eb",
    "Product": "#7a4cc2",
    "IndustryChain": "#b86519",
    "ValueChainSegment": "#c2410c",
    "IndustryConcept": "#0e7490",
    "Metric": "#477148",
    "Risk": "#c43c39",
    "Policy": "#4f5f9f",
    "Standard": "#8a4f9f",
    "Report": "#64748b",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def contains_name(value: str, query: str) -> bool:
    value_norm = normalize_name(value)
    query_norm = normalize_name(query)
    return bool(query_norm and (query_norm in value_norm or value_norm in query_norm))


def unique_sorted(values: list[str]) -> list[str]:
    return sorted({value for value in values if value}, key=lambda item: (len(item), item))


@dataclass
class LocalKnowledgeGraph:
    entities: list[dict[str, str]]
    relations: list[dict[str, str]]
    _relations_by_company: dict[str, list[dict[str, str]]] = field(init=False, repr=False)
    _relations_by_relation: dict[str, list[dict[str, str]]] = field(init=False, repr=False)
    _relations_by_relation_company: dict[tuple[str, str], list[dict[str, str]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        by_company: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_relation: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_relation_company: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in self.relations:
            company = row.get("head_name", "")
            relation = row.get("relation", "")
            if company:
                by_company[company].append(row)
            if relation:
                by_relation[relation].append(row)
            if company and relation:
                by_relation_company[(relation, company)].append(row)
        self._relations_by_company = dict(by_company)
        self._relations_by_relation = dict(by_relation)
        self._relations_by_relation_company = dict(by_relation_company)

    @classmethod
    def from_csvs(
        cls,
        entities_csv: Path = DEFAULT_ENTITIES_CSV,
        relations_csv: Path = DEFAULT_RELATIONS_CSV,
    ) -> "LocalKnowledgeGraph":
        return cls(read_csv_rows(entities_csv), read_csv_rows(relations_csv))

    @classmethod
    def from_dir(cls, data_dir: Path) -> "LocalKnowledgeGraph":
        return cls.from_csvs(data_dir / "entities.csv", data_dir / "relations.csv")

    def entity_counts(self) -> Counter:
        return Counter(row["type"] for row in self.entities)

    def relation_counts(self) -> Counter:
        return Counter(row["relation"] for row in self.relations)

    def names_by_type(self, entity_type: str) -> list[str]:
        return unique_sorted([row["name"] for row in self.entities if row["type"] == entity_type])

    def reports_count(self) -> int:
        return sum(1 for row in self.entities if row["type"] == "Report")

    def subgraph_relations(
        self,
        *,
        company: str = "",
        technology: str = "",
        relation_type: str = "",
        limit: int = 80,
    ) -> list[dict[str, str]]:
        rows = self.relations
        if relation_type:
            rows = [row for row in rows if row["relation"] == relation_type]
        if company:
            rows = [row for row in rows if row["head_type"] == "Company" and contains_name(row["head_name"], company)]
        if technology:
            rows = [
                row
                for row in rows
                if row["tail_type"] == "Technology" and contains_name(row["tail_name"], technology)
            ]
        return rows[:limit]

    def relation_candidates(
        self,
        *,
        companies: list[str] | set[str] | tuple[str, ...] | None = None,
        relations: list[str] | set[str] | tuple[str, ...] | None = None,
        head_type: str = "",
        exclude_relations: set[str] | None = None,
    ) -> list[dict[str, str]]:
        company_values = [company for company in (companies or []) if company]
        relation_values = [relation for relation in (relations or []) if relation]
        if company_values and relation_values:
            rows = [
                row
                for relation in relation_values
                for company in company_values
                for row in self._relations_by_relation_company.get((relation, company), [])
            ]
        elif company_values:
            rows = [row for company in company_values for row in self._relations_by_company.get(company, [])]
        elif relation_values:
            rows = [row for relation in relation_values for row in self._relations_by_relation.get(relation, [])]
        else:
            rows = self.relations
        if head_type:
            rows = [row for row in rows if row.get("head_type") == head_type]
        if exclude_relations:
            rows = [row for row in rows if row.get("relation") not in exclude_relations]
        return rows

    def companies_for_technology(self, technology: str) -> list[dict[str, str]]:
        return [
            row
            for row in self.relations
            if row["relation"] == "USES_TECHNOLOGY"
            and row["head_type"] == "Company"
            and row["tail_type"] == "Technology"
            and contains_name(row["tail_name"], technology)
        ]

    def companies_for_topic(self, topic: str) -> list[dict[str, str]]:
        topic = topic.strip()
        if not topic:
            return []
        return [
            row
            for row in self.relations
            if row["head_type"] == "Company"
            and row["relation"] != "MENTIONED_IN"
            and (
                topic in row.get("tail_name", "")
                or topic in row.get("evidence", "")
                or topic in row.get("section", "")
            )
        ]

    def company_relations(self, company: str, relation_type: str) -> list[dict[str, str]]:
        return [
            row
            for row in self.relations
            if row["relation"] == relation_type
            and row["head_type"] == "Company"
            and contains_name(row["head_name"], company)
        ]


def infer_question(graph: LocalKnowledgeGraph, question: str) -> dict[str, Any]:
    question = question.strip()
    companies = graph.names_by_type("Company")
    technologies = graph.names_by_type("Technology")

    matched_company = next((name for name in companies if name and name in question), "")
    matched_technology = next((name for name in technologies if name and name in question), "")

    topic = extract_company_topic(question)
    if ("哪些公司" in question or "公司" in question) and topic:
        cypher = (
            "MATCH (c:Company)-[r]->(x) "
            f'WHERE type(r) <> "MENTIONED_IN" AND (x.name CONTAINS "{topic}" OR r.evidence CONTAINS "{topic}") '
            "RETURN c.name AS company, type(r) AS relation, x.name AS target, "
            "r.evidence AS evidence, r.source_title AS source, r.page AS page"
        )
        records = graph.companies_for_topic(topic)
        return build_answer(question, "topic_to_company", cypher, records)

    if ("哪些公司" in question or "公司" in question) and matched_technology:
        cypher = (
            f'MATCH (c:Company)-[r:USES_TECHNOLOGY]->(t:Technology {{name: "{matched_technology}"}}) '
            "RETURN c.name AS company, t.name AS technology, r.evidence AS evidence, "
            "r.source_title AS source, r.page AS page"
        )
        records = graph.companies_for_technology(matched_technology)
        return build_answer(question, "topic_to_company", cypher, records)

    if matched_company:
        relation_type = "USES_TECHNOLOGY"
        for keyword, candidate in QUESTION_RELATIONS.items():
            if keyword in question:
                relation_type = candidate
                break
        tail_label = relation_type_tail_label(relation_type)
        cypher = (
            f'MATCH (c:Company {{name: "{matched_company}"}})-[r:{relation_type}]->(x:{tail_label}) '
            "RETURN c.name AS company, x.name AS target, r.evidence AS evidence, "
            "r.source_title AS source, r.page AS page"
        )
        records = graph.company_relations(matched_company, relation_type)
        return build_answer(question, "company_to_relation", cypher, records)

    cypher = (
        "MATCH (c:Company)-[r]->(x) "
        "RETURN c.name AS company, type(r) AS relation, x.name AS target, "
        "r.evidence AS evidence, r.source_title AS source, r.page AS page LIMIT 20"
    )
    records = [
        row
        for row in graph.relations
        if row["head_type"] == "Company" and question and any(token in row["evidence"] for token in question[:12])
    ][:20]
    return build_answer(question, "fallback_keyword", cypher, records)


def relation_type_tail_label(relation_type: str) -> str:
    return {
        "USES_TECHNOLOGY": "Technology",
        "HAS_PRODUCT": "Product",
        "BELONGS_TO_CHAIN": "IndustryChain",
        "HAS_METRIC": "Metric",
        "DISCLOSES_RISK": "Risk",
    }.get(relation_type, "Technology")


def extract_company_topic(question: str) -> str:
    if "哪些公司" not in question:
        return ""
    patterns = [
        r"哪些公司(?:.*?)(?:涉及|使用|布局|拥有|披露|属于)([^？?，,。；;]+)",
        r"([^？?，,。；;]+)有哪些公司",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            topic = match.group(1).strip()
            topic = re.sub(r"^(了|的|相关|以下|这些)", "", topic).strip()
            return topic
    return ""


def build_answer(question: str, intent: str, cypher: str, records: list[dict[str, str]]) -> dict[str, Any]:
    if not records:
        answer = "当前知识图谱中未找到相关证据。"
    elif intent in {"tech_to_company", "topic_to_company"}:
        companies = unique_sorted([row["head_name"] for row in records])
        answer = f"当前知识图谱中，涉及该技术的公司包括：{'、'.join(companies)}。"
    elif intent == "company_to_relation":
        relation = records[0]["relation"]
        targets = unique_sorted([row["tail_name"] for row in records])
        answer = f"当前知识图谱中，该公司{RELATION_LABELS.get(relation, relation)}包括：{'、'.join(targets)}。"
    else:
        targets = unique_sorted([f"{row['head_name']} - {RELATION_LABELS.get(row['relation'], row['relation'])} - {row['tail_name']}" for row in records])
        answer = f"当前知识图谱中找到 {len(records)} 条相关证据：" + "；".join(targets[:5]) + "。"
    return {
        "question": question,
        "intent": intent,
        "cypher": cypher,
        "records": records,
        "answer": answer,
        "evidence": evidence_rows(records),
        "subgraph": subgraph_edges(records),
    }


def evidence_rows(records: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for record in records:
        rows.append(
            {
                "head": record.get("head_name", ""),
                "relation": RELATION_LABELS.get(record.get("relation", ""), record.get("relation", "")),
                "tail": record.get("tail_name", ""),
                "evidence": record.get("evidence", ""),
                "source": record.get("source_title", ""),
                "page": record.get("page", ""),
            }
        )
    return rows


def subgraph_edges(records: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "source": row.get("head_name", ""),
            "target": row.get("tail_name", ""),
            "label": RELATION_LABELS.get(row.get("relation", ""), row.get("relation", "")),
            "source_type": row.get("head_type", ""),
            "target_type": row.get("tail_type", ""),
        }
        for row in records
    ]


def render_svg_graph(edges: list[dict[str, str]], *, width: int = 980, height: int = 560) -> str:
    edges = [edge for edge in edges if edge.get("source") and edge.get("target")][:140]
    if not edges:
        return '<div class="empty-graph">当前筛选条件下没有可展示的子图。</div>'

    nodes: dict[str, dict[str, Any]] = {}
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        source_node = nodes.setdefault(
            source,
            {"type": edge.get("source_type", ""), "degree": 0, "source_count": 0, "target_count": 0},
        )
        target_node = nodes.setdefault(
            target,
            {"type": edge.get("target_type", ""), "degree": 0, "source_count": 0, "target_count": 0},
        )
        source_node["degree"] += 1
        source_node["source_count"] += 1
        target_node["degree"] += 1
        target_node["target_count"] += 1

    def order_index(entity_type: str) -> int:
        try:
            return ENTITY_ORDER.index(entity_type)
        except ValueError:
            return len(ENTITY_ORDER)

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, int, str]:
        name, node = item
        return (order_index(str(node.get("type", ""))), -int(node.get("degree", 0)), normalize_name(name))

    source_names = [
        name
        for name, node in sorted(nodes.items(), key=sort_key)
        if node.get("type") == "Company" or (node.get("source_count", 0) and node.get("source_count", 0) >= node.get("target_count", 0))
    ]
    if not source_names:
        source_names = [name for name, node in sorted(nodes.items(), key=sort_key) if node.get("source_count", 0)]
    source_set = set(source_names)

    grouped_targets: dict[str, list[str]] = defaultdict(list)
    for name, node in sorted(nodes.items(), key=sort_key):
        if name not in source_set:
            grouped_targets[str(node.get("type", "")) or "Entity"].append(name)

    columns: list[tuple[str, str, list[str]]] = []
    max_rows_per_column = 24

    def append_columns(column_id: str, label: str, names: list[str]) -> None:
        for start in range(0, len(names), max_rows_per_column):
            chunk = names[start : start + max_rows_per_column]
            suffix = f" {start // max_rows_per_column + 1}" if len(names) > max_rows_per_column else ""
            columns.append((f"{column_id}-{start}", f"{label}{suffix}", chunk))

    if source_names:
        append_columns("source", "起点", source_names)
    for entity_type in sorted(grouped_targets, key=lambda value: (order_index(value), value)):
        append_columns(entity_type, entity_type, grouped_targets[entity_type])
    if not columns:
        columns.append(("entities", "实体", list(nodes)))

    node_gap = 48
    column_gap = 260
    left_padding = 86
    top_padding = 68
    right_padding = 220
    bottom_padding = 84
    max_rows = max((len(names) for _, _, names in columns), default=1)
    canvas_width = max(width, left_padding + (len(columns) - 1) * column_gap + right_padding)
    canvas_height = max(height, top_padding + max(max_rows - 1, 0) * node_gap + bottom_padding)
    positions: dict[str, tuple[float, float]] = {}
    for column_index, (_, _, names) in enumerate(columns):
        x = left_padding + column_index * column_gap
        for node_index, name in enumerate(names):
            positions[name] = (x, top_padding + node_index * node_gap)

    def color(entity_type: str) -> str:
        return ENTITY_COLORS.get(entity_type, "#334155")

    def radius(name: str) -> float:
        return min(18, max(9, 8 + math.sqrt(float(nodes[name].get("degree", 1))) * 2.2))

    def path_for(edge: dict[str, str], index: int) -> str:
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        direction = 1 if x2 >= x1 else -1
        start_x = x1 + radius(edge["source"]) * direction
        end_x = x2 - radius(edge["target"]) * direction
        offset = (index % 9 - 4) * 3.5
        if abs(end_x - start_x) < 28:
            loop = 44 + (index % 5) * 10
            return (
                f"M {start_x:.1f} {y1:.1f} C {start_x + loop:.1f} {y1 - 22:.1f}, "
                f"{end_x + loop:.1f} {y2 + 22:.1f}, {end_x:.1f} {y2:.1f}"
            )
        curve = max(72, abs(end_x - start_x) * 0.42)
        return (
            f"M {start_x:.1f} {y1:.1f} C {start_x + curve * direction:.1f} {y1 + offset:.1f}, "
            f"{end_x - curve * direction:.1f} {y2 - offset:.1f}, {end_x:.1f} {y2:.1f}"
        )

    show_edge_labels = len(edges) <= 18 and len(nodes) <= 34
    svg_parts = [
        f'<svg viewBox="0 0 {canvas_width} {canvas_height}" width="100%" height="{canvas_height}" role="img" data-layout="layered">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto"><path d="M0,0 L0,7 L7,3.5 z" fill="#94a3b8"/></marker></defs>',
    ]
    for column_index, (_, label, names) in enumerate(columns):
        x = left_padding + column_index * column_gap
        svg_parts.append(
            f'<text x="{x:.1f}" y="32" text-anchor="middle" font-size="12" font-weight="700" fill="#475569">'
            f'{html.escape(label)}</text>'
        )
        svg_parts.append(
            f'<text x="{x:.1f}" y="49" text-anchor="middle" font-size="11" fill="#94a3b8">{len(names)}</text>'
        )
    for index, edge in enumerate(edges):
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        svg_parts.append(
            f'<path d="{path_for(edge, index)}" fill="none" stroke="#94a3b8" stroke-width="1.15" '
            'stroke-linecap="round" opacity="0.52" marker-end="url(#arrow)">'
            f'<title>{html.escape(edge["source"])} - {html.escape(edge.get("label", ""))} - {html.escape(edge["target"])}</title>'
            "</path>"
        )
        if show_edge_labels:
            svg_parts.append(
                f'<text x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 - 7:.1f}" text-anchor="middle" '
                'font-size="10" fill="#475569" paint-order="stroke" stroke="#f8fafc" stroke-width="4">'
                f'{html.escape(edge.get("label", ""))}</text>'
            )
    for name, node in nodes.items():
        x, y = positions[name]
        entity_type = str(node.get("type", ""))
        escaped = html.escape(short_label(name, 10))
        svg_parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius(name):.1f}" fill="{color(entity_type)}" '
            'stroke="rgba(255,255,255,0.92)" stroke-width="2"/>'
        )
        svg_parts.append(
            f'<text x="{x + radius(name) + 8:.1f}" y="{y + 4:.1f}" font-size="12" font-weight="650" '
            'fill="#17202a" paint-order="stroke" stroke="#f8fafc" stroke-width="4">'
            f'{escaped}</text>'
        )
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def short_label(value: str, length: int = 12) -> str:
    value = value or ""
    return value if len(value) <= length else value[:length] + "..."
