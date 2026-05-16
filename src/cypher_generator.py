"""Generate safe read-only Cypher for graph QA."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from src.cypher_guard import CypherSafetyError, ensure_limit
from src.extraction_schema import normalize_name


GRAPH_SCHEMA = """节点:
- Company(name, normalized_name)
- Technology(name, normalized_name)
- Product(name, normalized_name)
- IndustryChain(name, normalized_name)
- IndustryConcept(name, normalized_name)
- Policy(name, normalized_name)
- Standard(name, normalized_name)
- ValueChainSegment(name, normalized_name)
- Metric(name, normalized_name)
- Risk(name, normalized_name)
- Report(name, report_id)

关系:
- (Company)-[:USES_TECHNOLOGY]->(Technology)
- (Company)-[:HAS_PRODUCT]->(Product)
- (Company)-[:BELONGS_TO_CHAIN]->(IndustryChain)
- (Company)-[:HAS_METRIC]->(Metric)
- (Company)-[:DISCLOSES_RISK]->(Risk)
- (任意非 Report 实体)-[:MENTIONED_IN]->(Report)
- (IndustryConcept|Technology|Product|ValueChainSegment)-[:UPSTREAM_OF|DOWNSTREAM_OF]->(IndustryConcept|Technology|Product|ValueChainSegment)
- (IndustryConcept|Technology|Product|ValueChainSegment)-[:ENABLES]->(IndustryConcept|Technology|Product|ValueChainSegment)
- (Risk|Policy|Standard|IndustryConcept)-[:CONSTRAINS]->(任意非 Report 实体)
- (IndustryConcept|Policy|Standard)-[:DEFINES]->(IndustryConcept|Technology|ValueChainSegment)
- (Company|IndustryConcept|Technology|Product|ValueChainSegment)-[:SUPPORTED_BY_POLICY]->(Policy)

关系属性:
evidence, source_report_id, source_title, page, section, confidence
"""

SYSTEM_PROMPT = """你是 Neo4j Cypher 助手。只生成只读 Cypher，不解释。
必须输出 JSON 对象：{"cypher": "..."}。
只能使用 MATCH、OPTIONAL MATCH、WITH、WHERE、RETURN、LIMIT。
禁止 CREATE、MERGE、DELETE、SET、REMOVE、DROP、CALL、LOAD、UNWIND、APOC、GDS。
查询必须 RETURN evidence、source、page 等证据字段，并 LIMIT 50。"""

RELATION_BY_KEYWORD = {
    "风险": "DISCLOSES_RISK",
    "政策": "SUPPORTED_BY_POLICY",
    "定义": "DEFINES",
    "上游": "UPSTREAM_OF",
    "下游": "DOWNSTREAM_OF",
    "财务": "HAS_METRIC",
    "指标": "HAS_METRIC",
    "产品": "HAS_PRODUCT",
    "业务": "HAS_PRODUCT",
    "产业链": "BELONGS_TO_CHAIN",
    "环节": "BELONGS_TO_CHAIN",
    "技术": "USES_TECHNOLOGY",
    "算力": "USES_TECHNOLOGY",
}

TAIL_LABEL_BY_RELATION = {
    "USES_TECHNOLOGY": "Technology",
    "HAS_PRODUCT": "Product",
    "BELONGS_TO_CHAIN": "IndustryChain",
    "HAS_METRIC": "Metric",
    "DISCLOSES_RISK": "Risk",
    "SUPPORTED_BY_POLICY": "Policy",
    "DEFINES": "IndustryConcept",
    "UPSTREAM_OF": "ValueChainSegment",
    "DOWNSTREAM_OF": "ValueChainSegment",
}


@dataclass(frozen=True)
class GeneratedCypher:
    cypher: str
    params: dict[str, Any]
    source: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_cypher_prompt(question: str) -> str:
    return f"""请根据下面 schema 为用户问题生成 Neo4j 只读 Cypher。

{GRAPH_SCHEMA}

用户问题：{question}
"""


def generate_cypher(
    question: str,
    *,
    client: Any | None = None,
    enable_llm: bool = True,
    limit: int = 50,
) -> GeneratedCypher:
    if client is not None and enable_llm:
        try:
            payload = client.chat_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_cypher_prompt(question),
                temperature=0.0,
            )
            cypher = ensure_limit(str(payload.get("cypher", "")), limit=limit)
            return GeneratedCypher(cypher=cypher, params={}, source="llm")
        except (CypherSafetyError, Exception) as exc:
            fallback = heuristic_cypher(question, limit=limit)
            return GeneratedCypher(
                cypher=fallback.cypher,
                params=fallback.params,
                source="heuristic",
                error=f"LLM Cypher unavailable or unsafe: {exc}",
            )
    return heuristic_cypher(question, limit=limit)


def heuristic_cypher(question: str, *, limit: int = 50) -> GeneratedCypher:
    relation_type = infer_relation_type(question)
    company = extract_company(question)
    topic = extract_topic(question)
    if company:
        tail_label = TAIL_LABEL_BY_RELATION[relation_type]
        cypher = (
            f"MATCH (c:Company)-[r:{relation_type}]->(x:{tail_label})\n"
            "WHERE c.name CONTAINS $company OR c.normalized_name CONTAINS $company_norm\n"
            "RETURN c.name AS company, labels(c) AS company_labels, type(r) AS relation, "
            "x.name AS target, labels(x) AS target_labels, r.evidence AS evidence, "
            "r.source_title AS source, r.source_tier AS source_tier, r.page AS page, "
            "r.source_report_id AS report_id\n"
            f"LIMIT {limit}"
        )
        return GeneratedCypher(
            cypher=cypher,
            params={"company": company, "company_norm": normalize_name(company, "Company")},
            source="heuristic",
        )
    if topic:
        cypher = (
            "MATCH (c:Company)-[r]->(x)\n"
            "WHERE type(r) <> 'MENTIONED_IN' "
            "AND (x.name CONTAINS $topic OR r.evidence CONTAINS $topic OR r.section CONTAINS $topic)\n"
            "RETURN c.name AS company, labels(c) AS company_labels, type(r) AS relation, "
            "x.name AS target, labels(x) AS target_labels, r.evidence AS evidence, "
            "r.source_title AS source, r.source_tier AS source_tier, r.page AS page, "
            "r.source_report_id AS report_id\n"
            f"LIMIT {limit}"
        )
        return GeneratedCypher(cypher=cypher, params={"topic": topic}, source="heuristic")
    cypher = (
        "MATCH (c:Company)-[r]->(x)\n"
        "WHERE type(r) <> 'MENTIONED_IN'\n"
        "RETURN c.name AS company, labels(c) AS company_labels, type(r) AS relation, "
        "x.name AS target, labels(x) AS target_labels, r.evidence AS evidence, "
        "r.source_title AS source, r.source_tier AS source_tier, r.page AS page, "
        "r.source_report_id AS report_id\n"
        f"LIMIT {limit}"
    )
    return GeneratedCypher(cypher=cypher, params={}, source="heuristic")


def infer_relation_type(question: str) -> str:
    for keyword, relation_type in RELATION_BY_KEYWORD.items():
        if keyword in question:
            return relation_type
    return "USES_TECHNOLOGY"


def extract_company(question: str) -> str:
    patterns = [
        r"([\u4e00-\u9fffA-Za-z0-9（）()·.\-]{2,40})(?:涉及|使用|布局|拥有|披露|属于|有哪些|有什么)",
        r"(?:公司|企业)\s*([\u4e00-\u9fffA-Za-z0-9（）()·.\-]{2,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if not match:
            continue
        candidate = cleanup_phrase(match.group(1))
        if candidate and not candidate.startswith(("哪些", "什么")):
            return candidate
    return ""


def extract_topic(question: str) -> str:
    patterns = [
        r"哪些公司(?:.*?)(?:涉及|使用|布局|拥有|披露|属于)([^？?，,。；;]+)",
        r"([^？?，,。；;]+)有哪些公司",
        r"(?:关于|围绕|查询|检索)([^？?，,。；;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            topic = cleanup_phrase(match.group(1))
            if topic:
                return topic
    terms = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", question)
    stopwords = {"哪些公司", "有哪些", "涉及", "使用", "布局", "披露", "属于", "什么", "请问"}
    for term in sorted(terms, key=len, reverse=True):
        if term not in stopwords:
            return term
    return ""


def cleanup_phrase(value: str) -> str:
    value = re.sub(r"^(了|的|相关|以下|这些|公司|企业)", "", str(value or "")).strip()
    value = re.sub(r"(有哪些|是什么|分别是什么)$", "", value).strip()
    return value
