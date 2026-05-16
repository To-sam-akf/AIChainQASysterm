"""LLM-based entity and relation extraction."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.extraction_schema import ENTITY_TYPES, RELATION_TYPES, sanitize_extraction_payload
from src.data_config import load_companies


SYSTEM_PROMPT = """你是知识图谱构建助手。你只从给定文本中抽取明确出现的信息，不编造。
输出必须是严格 JSON 对象，包含 entities 和 relations 两个数组。"""


@lru_cache(maxsize=1)
def core_company_names() -> tuple[str, ...]:
    return tuple(company.company for company in load_companies() if company.is_core_company)


def kind_instructions(kind: str) -> str:
    if kind == "annual":
        return """财报抽取重点：
- 只把本报告对应上市公司作为 Company；子公司、供应商、客户、券商、境外科技公司不要抽成 Company。
- 优先抽取公司基本面：主营业务、产品、技术、研发、财务指标、风险、所处产业链环节。
- Metric 必须在 properties 中尽量给出 metric_name、year、value、unit、basis。"""
    if kind == "research":
        return """研报抽取重点：
- 优先抽取产业链映射、技术路线、竞争格局、公司所处环节。
- Company 优先限制在核心上市公司白名单；非上市公司仅在研报明确作为产业主体分析时才抽取。
- 保留券商研报原文证据，不输出投资建议、评级、目标价。"""
    if kind == "industry":
        return """权威行业知识抽取重点：
- 优先抽取 IndustryConcept、ValueChainSegment、Technology、Policy、Standard。
- 优先抽取 DEFINES、UPSTREAM_OF、DOWNSTREAM_OF、ENABLES、CONSTRAINS、SUPPORTED_BY_POLICY。
- 除非文本明确关联核心上市公司，否则不要输出 Company；行业白皮书主要用于概念、政策、技术和产业链知识库。"""
    return "按报告文本抽取明确出现且有证据的 AI 算力产业链知识。"


def build_user_prompt(chunk: dict[str, Any]) -> str:
    kind = str(chunk.get("kind", ""))
    core_companies = "、".join(core_company_names())
    return f"""请从以下财报或研报文本中抽取中国 AI 算力产业链相关知识。

实体类型只能使用：{", ".join(ENTITY_TYPES)}
关系类型只能使用：{", ".join(RELATION_TYPES)}

核心上市公司白名单：
{core_companies}

{kind_instructions(kind)}

关系约束：
- Company USES_TECHNOLOGY Technology
- Company HAS_PRODUCT Product
- Company BELONGS_TO_CHAIN IndustryChain
- Company HAS_METRIC Metric
- Company DISCLOSES_RISK Risk
- IndustryConcept/Technology/Product/ValueChainSegment UPSTREAM_OF 或 DOWNSTREAM_OF IndustryConcept/Technology/Product/ValueChainSegment
- Technology/Product/IndustryConcept/ValueChainSegment ENABLES IndustryConcept/Technology/Product/ValueChainSegment
- Risk/Policy/Standard/IndustryConcept CONSTRAINS 任意非 Report 实体
- IndustryConcept/Policy/Standard DEFINES IndustryConcept/Technology/ValueChainSegment
- Company/IndustryConcept/Technology/Product/ValueChainSegment SUPPORTED_BY_POLICY Policy
- 不要输出 MENTIONED_IN；系统会自动根据 source_report_id 生成来源关系。
- 不要输出 Report 实体；系统会自动根据 manifest 生成报告节点。

要求：
1. 只抽取文本中明确出现的信息。
2. 不要编造公司、技术、产品、指标或风险。
3. 每条关系必须给出 evidence 原文证据。
4. 每条关系尽量给出 confidence，范围 0 到 1。
5. 每个文本块最多输出 30 个实体和 12 条关系，优先保留 Company 直接相关关系。
6. Metric 实体必须在 properties 中包含 year、value、unit 至少一个字段；没有数值或年份证据时不要抽 Metric。
6. 输出 JSON 格式：
{{
  "entities": [{{"type": "Company", "name": "公司名", "properties": {{}}}}],
  "relations": [
    {{
      "head_type": "Company",
      "head": "公司名",
      "relation": "USES_TECHNOLOGY",
      "tail_type": "Technology",
      "tail": "技术名",
      "evidence": "原文证据",
      "confidence": 0.8
    }}
  ]
}}

来源：
- report_id: {chunk.get("report_id", "")}
- kind: {chunk.get("kind", "")}
- company: {chunk.get("company", "")}
- title: {chunk.get("source_title", "")}
- source_tier: {chunk.get("source_tier", "")}
- source_type: {chunk.get("source_type", "")}
- page: {chunk.get("page", "")}
- section: {chunk.get("section", "")}

文本：
{chunk.get("text", "")}
"""


def extract_from_chunk(chunk: dict[str, Any], client: Any) -> dict[str, Any]:
    payload = client.chat_json(system_prompt=SYSTEM_PROMPT, user_prompt=build_user_prompt(chunk), temperature=0.0)
    cleaned, rejected = sanitize_extraction_payload(payload)
    for relation in cleaned["relations"]:
        relation["source_report_id"] = relation.get("source_report_id") or chunk.get("report_id", "")
        relation["source_title"] = relation.get("source_title") or chunk.get("source_title", "")
        relation["page"] = relation.get("page") or str(chunk.get("page", ""))
        relation["section"] = relation.get("section") or chunk.get("section", "")
        relation["source_tier"] = relation.get("source_tier") or chunk.get("source_tier", "")
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "report_id": chunk.get("report_id", ""),
        "source_title": chunk.get("source_title", ""),
        "page": chunk.get("page", ""),
        "section": chunk.get("section", ""),
        "kind": chunk.get("kind", ""),
        "entities": cleaned["entities"],
        "relations": cleaned["relations"],
        "rejected": rejected,
    }
