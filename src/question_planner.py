"""Question planning for professional KG + RAG QA."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.domain_lexicon import (
    BOTTLENECK_TERMS,
    THEME_SYNONYMS,
    canonical_company_name,
    company_lookup,
    expanded_terms,
    infer_themes,
    normalize_topic,
)


@dataclass(frozen=True)
class QuestionPlan:
    question: str
    answer_type: str
    companies: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    expanded_topics: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    core_companies_only: bool = True
    needs_comparison: bool = False
    needs_risk: bool = False
    needs_metrics: bool = False
    needs_chain: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_question(
    question: str,
    *,
    client: Any | None = None,
    core_companies_only: bool = True,
) -> QuestionPlan:
    deterministic = heuristic_plan_question(question, core_companies_only=core_companies_only)
    if client is None or not hasattr(client, "chat_json"):
        return deterministic
    try:
        payload = client.chat_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=build_planner_prompt(question, deterministic),
            temperature=0.0,
        )
        return merge_llm_plan(question, deterministic, payload, core_companies_only=core_companies_only)
    except Exception:
        return deterministic


PLANNER_SYSTEM_PROMPT = """你是中国 AI 算力产业链投研问答系统的问题规划器。
只输出严格 JSON，不回答问题本身。不要生成投资建议、目标价或买卖结论。"""


def build_planner_prompt(question: str, fallback: QuestionPlan) -> str:
    return f"""请解析用户问题，输出字段：
- answer_type: topic_to_company/company_compare/risk_analysis/industry_bottleneck/company_profile/thematic_research
- companies: 问题中的核心 A 股公司名称
- topics: 产业主题或产品技术关键词
- relations: 需要查询的关系，限 USES_TECHNOLOGY/HAS_PRODUCT/BELONGS_TO_CHAIN/HAS_METRIC/DISCLOSES_RISK/SUPPORTED_BY_POLICY/CONSTRAINS
- core_companies_only: “哪些公司/上市公司”默认 true

可参考的启发式结果：
{json.dumps(fallback.to_dict(), ensure_ascii=False)}

用户问题：{question}
"""


def merge_llm_plan(
    question: str,
    fallback: QuestionPlan,
    payload: dict[str, Any],
    *,
    core_companies_only: bool,
) -> QuestionPlan:
    answer_type = str(payload.get("answer_type") or fallback.answer_type)
    allowed_types = {
        "topic_to_company",
        "company_compare",
        "risk_analysis",
        "industry_bottleneck",
        "company_profile",
        "thematic_research",
    }
    if answer_type not in allowed_types:
        answer_type = fallback.answer_type
    companies = unique_companies(listify(payload.get("companies")) or fallback.companies)
    topics = unique_strings(listify(payload.get("topics")) or fallback.topics)
    if not topics:
        topics = fallback.topics
    relations = [rel for rel in unique_strings(listify(payload.get("relations")) or fallback.relations) if rel in RELATIONS]
    if not relations:
        relations = fallback.relations
    planner_core_only = payload.get("core_companies_only", fallback.core_companies_only)
    if not isinstance(planner_core_only, bool):
        planner_core_only = fallback.core_companies_only
    if core_companies_only:
        planner_core_only = True
    return QuestionPlan(
        question=question.strip(),
        answer_type=answer_type,
        companies=companies,
        topics=topics,
        expanded_topics=expanded_terms(topics),
        relations=relations,
        core_companies_only=planner_core_only,
        needs_comparison=answer_type == "company_compare" or fallback.needs_comparison,
        needs_risk=answer_type == "risk_analysis" or fallback.needs_risk,
        needs_metrics=fallback.needs_metrics or "HAS_METRIC" in relations,
        needs_chain=fallback.needs_chain or "BELONGS_TO_CHAIN" in relations,
    )


RELATIONS = {
    "USES_TECHNOLOGY",
    "HAS_PRODUCT",
    "BELONGS_TO_CHAIN",
    "HAS_METRIC",
    "DISCLOSES_RISK",
    "SUPPORTED_BY_POLICY",
    "CONSTRAINS",
}


def heuristic_plan_question(question: str, *, core_companies_only: bool = True) -> QuestionPlan:
    question = question.strip()
    companies = extract_companies(question)
    topics = extract_topics(question, companies)
    needs_comparison = len(companies) >= 2 or any(term in question for term in ("比较", "对比", "差异", "区别", "相较"))
    needs_risk = "风险" in question or "不确定" in question
    needs_metrics = any(term in question for term in ("指标", "财务", "营收", "利润", "毛利", "业绩", "收入"))
    needs_chain = any(term in question for term in ("产业链", "环节", "上游", "下游", "位置", "分布"))
    asks_company_list = any(term in question for term in ("哪些公司", "上市公司", "企业", "标的"))
    bottleneck = any(term in question for term in BOTTLENECK_TERMS) or "最大" in question and "问题" in question

    if needs_comparison and companies:
        answer_type = "company_compare"
    elif needs_risk and companies:
        answer_type = "risk_analysis"
    elif bottleneck:
        answer_type = "industry_bottleneck"
    elif asks_company_list:
        answer_type = "topic_to_company"
    elif companies:
        answer_type = "company_profile"
    else:
        answer_type = "thematic_research"

    relations = infer_relations(question, topics, needs_risk=needs_risk, needs_metrics=needs_metrics, needs_chain=needs_chain)
    if answer_type == "industry_bottleneck" and "CONSTRAINS" not in relations:
        relations.append("CONSTRAINS")
    return QuestionPlan(
        question=question,
        answer_type=answer_type,
        companies=companies,
        topics=topics,
        expanded_topics=expanded_terms(topics),
        relations=relations,
        core_companies_only=core_companies_only or asks_company_list,
        needs_comparison=needs_comparison,
        needs_risk=needs_risk,
        needs_metrics=needs_metrics,
        needs_chain=needs_chain,
    )


def extract_companies(question: str) -> list[str]:
    lookup = company_lookup()
    matches: list[tuple[int, str]] = []
    normalized_question = normalize_topic(question)
    for company, aliases in lookup.aliases_by_company.items():
        for alias in aliases:
            alias_norm = normalize_topic(alias)
            if alias_norm and alias_norm in normalized_question:
                matches.append((len(alias_norm), company))
                break
    ordered = [company for _, company in sorted(matches, key=lambda item: (-item[0], item[1]))]
    return unique_companies(ordered)


def extract_topics(question: str, companies: list[str]) -> list[str]:
    cleaned = question
    lookup = company_lookup()
    for company in companies:
        for alias in lookup.aliases_by_company.get(company, (company,)):
            cleaned = cleaned.replace(alias, " ")

    themes = infer_themes(cleaned)
    candidates = list(themes)
    patterns = [
        r"(?:涉及|布局|拥有|关于|围绕|看|关注|受益于|属于)([^？?，,。；;]+)",
        r"([^？?，,。；;]{2,20})(?:有哪些公司|上市公司|产业链)",
        r"在([^？?，,。；;]{2,20})(?:业务|领域|环节)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned):
            value = cleanup_topic(match.group(1))
            if value:
                candidates.extend(infer_themes(value) or [value])
    for theme, aliases in THEME_SYNONYMS.items():
        if any(alias in cleaned for alias in aliases):
            candidates.append(theme)
    return unique_strings(candidates)


def cleanup_topic(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"^(了|的|相关|以下|这些|公司|上市公司|企业)", "", value).strip()
    value = re.sub(r"(是什么|分别是什么|有哪些|如何|怎么样)$", "", value).strip()
    stop = {"主要风险", "进展和主要风险", "差异", "区别", "当前最大的瓶颈"}
    if value in stop or len(value) > 24:
        return ""
    return value


def infer_relations(
    question: str,
    topics: list[str],
    *,
    needs_risk: bool,
    needs_metrics: bool,
    needs_chain: bool,
) -> list[str]:
    relations: list[str] = []
    if any(term in question for term in ("技术", "算力", "芯片", "液冷", "光模块", "服务器", "网络", "PCB")) or topics:
        relations.extend(["USES_TECHNOLOGY", "HAS_PRODUCT"])
    if needs_chain:
        relations.append("BELONGS_TO_CHAIN")
    if needs_metrics:
        relations.append("HAS_METRIC")
    if needs_risk:
        relations.append("DISCLOSES_RISK")
    if "政策" in question:
        relations.append("SUPPORTED_BY_POLICY")
    if not relations:
        relations.extend(["USES_TECHNOLOGY", "HAS_PRODUCT", "BELONGS_TO_CHAIN"])
    return [rel for rel in unique_strings(relations) if rel in RELATIONS]


def unique_companies(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        company = canonical_company_name(value)
        if company and company not in seen:
            seen.add(company)
            result.append(company)
    return result


def unique_strings(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        value = str(value or "").strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def listify(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []

