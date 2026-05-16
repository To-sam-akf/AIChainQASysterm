"""Data access and local QA helpers for the Streamlit frontend."""

from __future__ import annotations

import csv
import html
import math
import re
from collections import Counter
from dataclasses import dataclass
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
    edges = edges[:80]
    nodes: dict[str, str] = {}
    for edge in edges:
        nodes.setdefault(edge["source"], edge.get("source_type", ""))
        nodes.setdefault(edge["target"], edge.get("target_type", ""))
    if not nodes:
        return '<div class="empty-graph">当前筛选条件下没有可展示的子图。</div>'

    names = list(nodes.keys())
    center_x, center_y = width / 2, height / 2
    radius = min(width, height) * 0.36
    positions = {}
    for index, name in enumerate(names):
        angle = 2 * math.pi * index / max(len(names), 1)
        positions[name] = (center_x + radius * math.cos(angle), center_y + radius * math.sin(angle))

    def color(entity_type: str) -> str:
        return {
            "Company": "#2563eb",
            "Technology": "#059669",
            "Product": "#7c3aed",
            "IndustryChain": "#d97706",
            "Metric": "#0f766e",
            "Risk": "#dc2626",
            "Report": "#475569",
            "IndustryConcept": "#0891b2",
            "Policy": "#4f46e5",
            "Standard": "#9333ea",
            "ValueChainSegment": "#ea580c",
        }.get(entity_type, "#334155")

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#64748b"/></marker></defs>',
    ]
    for edge in edges:
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        svg_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#64748b" stroke-width="1.4" marker-end="url(#arrow)" opacity="0.78"/>'
        )
        svg_parts.append(
            f'<text x="{mid_x:.1f}" y="{mid_y:.1f}" text-anchor="middle" font-size="11" fill="#334155">'
            f'{html.escape(edge["label"])}</text>'
        )
    for name, entity_type in nodes.items():
        x, y = positions[name]
        escaped = html.escape(short_label(name))
        svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="24" fill="{color(entity_type)}"/>')
        svg_parts.append(
            f'<text x="{x:.1f}" y="{y + 42:.1f}" text-anchor="middle" font-size="12" fill="#0f172a">{escaped}</text>'
        )
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def short_label(value: str, length: int = 12) -> str:
    value = value or ""
    return value if len(value) <= length else value[:length] + "..."
