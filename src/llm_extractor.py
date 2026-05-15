"""LLM-based entity and relation extraction."""

from __future__ import annotations

from typing import Any

from src.extraction_schema import ENTITY_TYPES, RELATION_TYPES, sanitize_extraction_payload


SYSTEM_PROMPT = """你是知识图谱构建助手。你只从给定文本中抽取明确出现的信息，不编造。
输出必须是严格 JSON 对象，包含 entities 和 relations 两个数组。"""


def build_user_prompt(chunk: dict[str, Any]) -> str:
    return f"""请从以下财报或研报文本中抽取中国 AI 算力产业链相关知识。

实体类型只能使用：{", ".join(ENTITY_TYPES)}
关系类型只能使用：{", ".join(RELATION_TYPES)}

关系约束：
- Company USES_TECHNOLOGY Technology
- Company HAS_PRODUCT Product
- Company BELONGS_TO_CHAIN IndustryChain
- Company HAS_METRIC Metric
- Company DISCLOSES_RISK Risk
- 不要输出 MENTIONED_IN；系统会自动根据 source_report_id 生成来源关系。
- 不要输出 Report 实体；系统会自动根据 manifest 生成报告节点。

要求：
1. 只抽取文本中明确出现的信息。
2. 不要编造公司、技术、产品、指标或风险。
3. 每条关系必须给出 evidence 原文证据。
4. 每条关系尽量给出 confidence，范围 0 到 1。
5. 每个文本块最多输出 30 个实体和 12 条关系，优先保留 Company 直接相关关系。
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
- company: {chunk.get("company", "")}
- title: {chunk.get("source_title", "")}
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
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "report_id": chunk.get("report_id", ""),
        "source_title": chunk.get("source_title", ""),
        "page": chunk.get("page", ""),
        "section": chunk.get("section", ""),
        "entities": cleaned["entities"],
        "relations": cleaned["relations"],
        "rejected": rejected,
    }
