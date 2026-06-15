"""Question planning for professional KG + RAG QA.

该模块负责将用户自然语言问题解析为结构化的查询计划（QuestionPlan），
是 QA 流水线的第一步。支持两级规划策略：
1. 启发式规则（heuristic_plan_question）—— 快速、零成本
2. LLM 增强规划（plan_question） — 当启发式结果质量不足时，由 LLM 优化
"""

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
    """结构化查询计划，描述如何回答用户的一个问题。

    Attributes:
        question: 用户原始问题文本
        answer_type: 答案类型，决定后续的查询和生成策略
            - topic_to_company: 主题→公司（哪些公司涉及某主题）
            - company_compare: 公司对比
            - risk_analysis: 风险分析
            - industry_bottleneck: 产业瓶颈分析
            - company_profile: 公司概况
            - thematic_research: 主题研究
        companies: 问题中识别出的公司列表（规范名称）
        topics: 识别出的产业主题/关键词
        expanded_topics: 主题的扩展词（用于语义检索，增加召回率）
        relations: 需要查询的知识图谱关系类型列表
        core_companies_only: 是否只查询核心上市公司（过滤非核心标的）
        needs_comparison: 是否需要公司对比分析
        needs_risk: 是否需要风险评估
        needs_metrics: 是否需要财务/经营指标
        needs_chain: 是否需要产业链分析
    """
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
        """将查询计划序列化为字典，用于日志/调试。"""
        return asdict(self)


def plan_question(
    question: str,
    *,
    client: Any | None = None,
    core_companies_only: bool = True,
    llm_options: dict[str, Any] | None = None,
) -> QuestionPlan:
    """使用 LLM 增强方式规划问题。

    先用启发式规则生成基础计划（heuristic_plan_question），
    然后让 LLM 基于该结果进行二次优化，最后通过 merge_llm_plan
    将 LLM 输出与启发性结果安全合并（LLM 可能输出非法值）。

    Args:
        question: 用户问题
        client: LLM 客户端（如果为 None 或没有 chat_json 方法则直接返回启发式结果）
        core_companies_only: 是否只查询核心上市公司
        llm_options: 额外的 LLM 调用参数

    Returns:
        合并后的 QuestionPlan
    """
    # 1. 先跑启发式规则作为 fallback
    deterministic = heuristic_plan_question(question, core_companies_only=core_companies_only)

    # 2. 没有 LLM 客户端时直接返回启发式结果
    if client is None or not hasattr(client, "chat_json"):
        return deterministic

    # 3. 调用 LLM 进行优化，失败时静默 fallback 到启发式结果
    try:
        kwargs = {
            "system_prompt": PLANNER_SYSTEM_PROMPT,
            "user_prompt": build_planner_prompt(question, deterministic),
            "temperature": 0.0,
        }
        if llm_options:
            kwargs.update(llm_options)
        try:
            payload = client.chat_json(**kwargs)
        except TypeError:
            # 兼容旧版 LLM 客户端接口
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
    """构建 LLM planner 的 user prompt。

    将启发式结果作为参考信息提供给 LLM，引导 LLM 在启发式基础上
    进行修正和优化，而不是从头生成。

    Args:
        question: 用户问题
        fallback: 启发式规则生成的 QuestionPlan（作为参考）

    Returns:
        prompt 字符串
    """
    return f"""请解析用户问题，输出字段：
- answer_type: topic_to_company/company_compare/risk_analysis/industry_bottleneck/company_profile/thematic_research
- companies: 问题中的核心 A 股公司名称
- topics: 产业主题或产品技术关键词
- relations: 需要查询的关系，限 USES_TECHNOLOGY/HAS_PRODUCT/BELONGS_TO_CHAIN/HAS_METRIC/DISCLOSES_RISK/SUPPORTED_BY_POLICY/CONSTRAINS
- 对"为什么/瓶颈/趋势/受益/跟踪指标"类问题，可加入 DRIVES/DEPENDS_ON/RELIEVES/HAS_EXPOSURE/HAS_INDICATOR/BENEFITS_FROM
- core_companies_only: "哪些公司/上市公司"默认 true

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
    """将 LLM 的 JSON 输出与启发式 fallback 计划安全合并。

    对每个字段做合法性校验：如果 LLM 返回的值非法（如 answer_type 不在允许列表中、
    relations 不在 RELATIONS 集合中、core_companies_only 不是 bool 等），
    则回退使用 fallback 的值，确保不会因为 LLM 输出异常导致系统崩溃。

    Args:
        question: 用户问题
        fallback: 启发式规则生成的 QuestionPlan
        payload: LLM 返回的 JSON 字典
        core_companies_only: 顶级调用方设置的 core_companies_only 标志

    Returns:
        合并后的 QuestionPlan
    """
    # answer_type 合法性校验
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

    # companies 合法性校验
    companies = unique_companies(listify(payload.get("companies")) or fallback.companies)

    # topics 合法性校验
    topics = unique_strings(listify(payload.get("topics")) or fallback.topics)
    if not topics:
        topics = fallback.topics

    # relations 合法性校验（必须在 RELATIONS 集合中）
    relations = [rel for rel in unique_strings(listify(payload.get("relations")) or fallback.relations) if rel in RELATIONS]
    if not relations:
        relations = fallback.relations

    # core_companies_only 合法性校验（必须是 bool）
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


# 知识图谱中支持的完整关系类型集合
# 用于 Cypher 查询生成和证据卡片分类
RELATIONS = {
    "USES_TECHNOLOGY",     # 使用某技术
    "HAS_PRODUCT",         # 拥有某产品
    "BELONGS_TO_CHAIN",    # 属于某产业链环节
    "HAS_METRIC",          # 有某财务/经营指标
    "DISCLOSES_RISK",      # 披露某风险
    "SUPPORTED_BY_POLICY", # 受某政策支持
    "CONSTRAINS",          # 构成瓶颈/约束
    "DRIVES",              # 驱动某因素
    "DEPENDS_ON",          # 依赖于某因素
    "RELIEVES",            # 缓解某瓶颈
    "HAS_EXPOSURE",        # 有某敞口
    "HAS_INDICATOR",       # 有某跟踪指标
    "BENEFITS_FROM",       # 受益于某因素
}


def heuristic_plan_question(question: str, *, core_companies_only: bool = True) -> QuestionPlan:
    """基于规则的启发式问题规划。

    通过关键词匹配和模式识别，快速解析问题中的：
    - 公司名（通过公司别名表匹配）
    - 主题词（通过主题词表和正则模式提取）
    - 答案类型（根据问题中是否包含风险、比较、瓶颈等关键词判断）
    - 关系类型（根据问题领域推断需要的知识图谱关系）
    - 其他标志位（needs_comparison, needs_risk 等）

    Args:
        question: 用户问题
        core_companies_only: 是否只查询核心上市公司

    Returns:
        基于规则解析的 QuestionPlan
    """
    question = question.strip()
    companies = extract_companies(question)
    topics = extract_topics(question, companies)

    # 检测问题中的各种需求标志
    needs_comparison = len(companies) >= 2 or any(term in question for term in ("比较", "对比", "差异", "区别", "相较"))
    needs_risk = "风险" in question or "不确定" in question
    needs_metrics = any(term in question for term in ("指标", "财务", "营收", "利润", "毛利", "业绩", "收入"))
    needs_chain = any(term in question for term in ("产业链", "环节", "上游", "下游", "位置", "分布"))
    asks_company_list = any(term in question for term in ("哪些公司", "上市公司", "企业", "标的"))
    bottleneck = any(term in question for term in BOTTLENECK_TERMS) or "最大" in question and "问题" in question

    # 根据检测到的标志确定答案类型（优先级从高到低）
    if needs_risk and companies:
        answer_type = "risk_analysis"
    elif needs_comparison and companies:
        answer_type = "company_compare"
    elif bottleneck:
        answer_type = "industry_bottleneck"
    elif asks_company_list:
        answer_type = "topic_to_company"
    elif companies:
        answer_type = "company_profile"
    else:
        answer_type = "thematic_research"

    # 推断需要的知识图谱关系
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
    """从问题中提取公司名称。

    使用 company_lookup() 获取公司别名表，遍历所有公司的所有别名，
    在问题文本中匹配（归一化后的模糊匹配）。按别名长度降序排列，
    优先匹配更长的、更精确的别名（避免短别名误匹配）。

    Args:
        question: 用户问题

    Returns:
        匹配到的公司规范名称列表（去重，按匹配长度降序）
    """
    lookup = company_lookup()
    matches: list[tuple[int, str]] = []
    normalized_question = normalize_topic(question)
    for company, aliases in lookup.aliases_by_company.items():
        for alias in aliases:
            alias_norm = normalize_topic(alias)
            if alias_norm and alias_norm in normalized_question:
                matches.append((len(alias_norm), company))
                break
    # 按别名长度降序排序（优先精确匹配长别名）
    ordered = [company for _, company in sorted(matches, key=lambda item: (-item[0], item[1]))]
    return unique_companies(ordered)


def extract_topics(question: str, companies: list[str]) -> list[str]:
    """从问题中提取产业主题/关键词。

    流程：
    1. 从问题中移除已识别的公司名（避免公司名被误认为主题）
    2. 使用 infer_themes() 识别隐含主题
    3. 使用正则模式从句子结构中提取主题短语
    4. 通过主题同义词表补充标准化主题名

    Args:
        question: 用户问题
        companies: 已识别的公司列表（从中提取别名用于清洗）

    Returns:
        主题词列表（去重）
    """
    # 1. 清洗问题：移除公司名，避免干扰主题提取
    cleaned = question
    lookup = company_lookup()
    for company in companies:
        for alias in lookup.aliases_by_company.get(company, (company,)):
            cleaned = cleaned.replace(alias, " ")

    # 2. 从清洗后的文本中推断主题
    themes = infer_themes(cleaned)
    candidates = list(themes)

    # 3. 用正则提取模式化的主题短语
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

    # 4. 补充主题同义词（如 "AI" → "人工智能"）
    for theme, aliases in THEME_SYNONYMS.items():
        if any(alias in cleaned for alias in aliases):
            candidates.append(theme)

    return unique_strings(candidates)


def cleanup_topic(value: str) -> str:
    """清洗主题提取结果，去除噪点前缀/后缀。

    移除常见的无用前缀（"了"、"的"、"相关"等）和后缀（"是什么"、"有哪些"等）。
    如果结果为停用词或过长，返回空字符串。

    Args:
        value: 原始提取的主题文本

    Returns:
        清洗后的主题文本，或空字符串（无效时）
    """
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
    """根据问题内容推断需要查询的知识图谱关系。

    通过关键词匹配判断问题涉及的技术领域、风险、指标、产业链等维度，
    返回对应的 RELATIONS 集合中的关系类型列表。

    Args:
        question: 用户问题
        topics: 已提取的主题列表
        needs_risk: 是否需要风险分析
        needs_metrics: 是否需要指标数据
        needs_chain: 是否需要产业链分析

    Returns:
        关系类型列表（在 RELATIONS 集合中的合法值）
    """
    relations: list[str] = []

    # 技术/产品相关
    if any(term in question for term in ("技术", "算力", "芯片", "液冷", "光模块", "服务器", "网络", "PCB")) or topics:
        relations.extend(["USES_TECHNOLOGY", "HAS_PRODUCT"])

    # 产业链相关
    if needs_chain:
        relations.append("BELONGS_TO_CHAIN")

    # 财务指标相关
    if needs_metrics:
        relations.append("HAS_METRIC")

    # 风险相关
    if needs_risk:
        relations.append("DISCLOSES_RISK")

    # 政策相关
    if "政策" in question:
        relations.append("SUPPORTED_BY_POLICY")

    # 因果/传导/受益关系
    if any(term in question for term in ("为什么", "驱动", "传导", "受益", "趋势")):
        relations.extend(["DRIVES", "BENEFITS_FROM", "DEPENDS_ON"])

    # 瓶颈/约束关系
    if any(term in question for term in ("瓶颈", "约束", "缓解")):
        relations.extend(["CONSTRAINS", "RELIEVES"])

    # 风险敞口关系
    if any(term in question for term in ("敞口", "受益公司", "谁受益")):
        relations.append("HAS_EXPOSURE")

    # 跟踪指标关系
    if any(term in question for term in ("跟踪指标", "领先指标", "验证指标")):
        relations.append("HAS_INDICATOR")

    # 默认关系：如果以上都没匹配到，使用最通用的技术/产品/产业链关系
    if not relations:
        relations.extend(["USES_TECHNOLOGY", "HAS_PRODUCT", "BELONGS_TO_CHAIN"])

    return [rel for rel in unique_strings(relations) if rel in RELATIONS]


def unique_companies(values: list[str]) -> list[str]:
    """公司名称去重并转换为规范名称。

    Args:
        values: 原始公司名称列表（可能包含别名或重复项）

    Returns:
        规范公司名称列表（去重、保留首次出现顺序）
    """
    result = []
    seen = set()
    for value in values:
        company = canonical_company_name(value)
        if company and company not in seen:
            seen.add(company)
            result.append(company)
    return result


def unique_strings(values: list[str]) -> list[str]:
    """字符串列表去重（大小写不敏感）。

    Args:
        values: 原始字符串列表

    Returns:
        去重后的字符串列表（保留首次出现顺序）
    """
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
    """将任意值规范化为字符串列表。

    - 如果 value 已经是 list，将其元素转为字符串
    - 如果 value 是 str，作为单元素列表返回
    - 否则返回空列表

    Args:
        value: 任意输入

    Returns:
        字符串列表
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []