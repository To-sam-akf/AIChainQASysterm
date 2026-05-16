"""Domain vocabulary and quality rules for professional AI compute QA."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from src.data_config import Company, load_companies
from src.extraction_schema import normalize_name


THEME_SYNONYMS: dict[str, tuple[str, ...]] = {
    "AI服务器": ("AI服务器", "智算服务器", "GPU服务器", "训练服务器", "推理服务器", "人工智能服务器"),
    "光模块": ("光模块", "高速光模块", "800G", "1.6T", "硅光", "CPO", "LPO", "光器件", "光引擎"),
    "液冷": ("液冷", "冷板", "冷板式液冷", "浸没式液冷", "液冷散热", "温控", "热管理", "CDU"),
    "AI芯片": ("AI芯片", "GPU", "DCU", "CPU", "算力芯片", "推理芯片", "训练芯片", "加速卡"),
    "国产算力": ("国产算力", "自主可控", "国产替代", "信创", "国产AI芯片", "国产服务器"),
    "算力网络": ("算力网络", "交换机", "以太网", "Scale Up", "Scale-Out", "高速互联", "网络设备"),
    "PCB": ("PCB", "印制电路板", "高多层板", "封装基板", "CCL", "覆铜板"),
    "数据中心": ("数据中心", "智算中心", "AIDC", "IDC", "算力中心", "绿色数据中心"),
    "电源": ("电源", "UPS", "服务器电源", "数据中心电源", "电力模块"),
}

PROFESSIONAL_THEMES = tuple(THEME_SYNONYMS.keys())

BUSINESS_RELATIONS = {
    "USES_TECHNOLOGY",
    "HAS_PRODUCT",
    "BELONGS_TO_CHAIN",
    "DISCLOSES_RISK",
    "SUPPORTED_BY_POLICY",
}

TOPIC_RELATIONS = {
    "USES_TECHNOLOGY",
    "HAS_PRODUCT",
    "BELONGS_TO_CHAIN",
    "SUPPORTED_BY_POLICY",
}

CORE_METRIC_KEYWORDS = (
    "营业收入",
    "主营业务收入",
    "归属于上市公司股东的净利润",
    "归母净利润",
    "扣除非经常性损益",
    "扣非净利润",
    "毛利率",
    "净利率",
    "研发投入",
    "研发费用",
    "经营活动产生的现金流量净额",
    "经营现金流",
    "分产品",
    "分行业",
    "销售收入",
    "海外收入",
    "订单",
    "产能",
    "市场份额",
)

LOW_VALUE_METRIC_KEYWORDS = (
    "交易性金融资产",
    "货币资金",
    "使用权资产",
    "长期股权投资",
    "固定资产",
    "无形资产",
    "递延所得税",
    "应付账款",
    "应收账款",
    "预付款项",
    "其他流动资产",
    "其他非流动资产",
    "存货跌价准备",
    "坏账准备",
    "期初余额",
    "期末余额",
    "资产减值",
    "租赁负债",
)

NOISE_SECTIONS = (
    "重要提示",
    "目录",
    "释义",
    "公司简介",
    "主要财务指标",
    "股份变动",
    "股东信息",
    "董事",
    "监事",
    "审计报告",
    "财务报表",
    "财务报告",
    "会计政策",
    "合并资产负债表",
    "母公司资产负债表",
    "合并利润表",
    "现金流量表",
    "附注",
)

DISCLAIMER_TERMS = (
    "请务必仔细阅读正文后的",
    "法律声明",
    "免责声明",
    "评级说明",
    "证券研究报告",
    "投资建议",
    "风险提示",
)

BOTTLENECK_TERMS = (
    "瓶颈",
    "制约",
    "短板",
    "供给不足",
    "紧缺",
    "受限",
    "功耗",
    "电力",
    "散热",
    "网络",
    "芯片",
    "产能",
    "良率",
)


@dataclass(frozen=True)
class CompanyLookup:
    companies: tuple[Company, ...]
    canonical_by_norm: dict[str, str]
    segment_by_company: dict[str, str]
    aliases_by_company: dict[str, tuple[str, ...]]

    @property
    def core_names(self) -> set[str]:
        return {company.company for company in self.companies if company.is_core_company}


@lru_cache(maxsize=1)
def company_lookup() -> CompanyLookup:
    companies = tuple(load_companies())
    canonical_by_norm: dict[str, str] = {}
    segment_by_company: dict[str, str] = {}
    aliases_by_company: dict[str, tuple[str, ...]] = {}
    for company in companies:
        names = (company.company, *company.aliases)
        aliases_by_company[company.company] = names
        segment_by_company[company.company] = company.chain_segment
        for name in names:
            canonical_by_norm[normalize_name(name, "Company")] = company.company
    return CompanyLookup(
        companies=companies,
        canonical_by_norm=canonical_by_norm,
        segment_by_company=segment_by_company,
        aliases_by_company=aliases_by_company,
    )


def canonical_company_name(value: str) -> str:
    return company_lookup().canonical_by_norm.get(normalize_name(value, "Company"), str(value or "").strip())


def is_core_company(value: str) -> bool:
    return canonical_company_name(value) in company_lookup().core_names


def company_segment(value: str) -> str:
    return company_lookup().segment_by_company.get(canonical_company_name(value), "")


def expanded_terms(terms: Iterable[str]) -> list[str]:
    values: list[str] = []
    for term in terms:
        term = str(term or "").strip()
        if not term:
            continue
        values.append(term)
        for canonical, aliases in THEME_SYNONYMS.items():
            if term == canonical or any(term.casefold() in alias.casefold() or alias.casefold() in term.casefold() for alias in aliases):
                values.extend(aliases)
                values.append(canonical)
    deduped = []
    seen = set()
    for value in values:
        key = normalize_topic(value)
        if key and key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def infer_themes(text: str) -> list[str]:
    normalized = normalize_topic(text)
    themes = []
    for theme, aliases in THEME_SYNONYMS.items():
        if any(normalize_topic(alias) in normalized for alias in aliases):
            themes.append(theme)
    return themes


def normalize_topic(value: str) -> str:
    value = str(value or "").casefold()
    value = value.replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", value)


def text_matches_terms(text: str, terms: Iterable[str]) -> bool:
    normalized = normalize_topic(text)
    return any(normalize_topic(term) in normalized for term in terms if str(term or "").strip())


def is_noise_section(section: str) -> bool:
    return any(term in str(section or "") for term in NOISE_SECTIONS)


def is_disclaimer_text(text: str) -> bool:
    return any(term in str(text or "") for term in DISCLAIMER_TERMS)


def looks_like_definition_noise(row: dict[str, str]) -> bool:
    section = row.get("section", "")
    evidence = row.get("evidence", "")
    page = str(row.get("page", ""))
    relation = row.get("relation", "")
    if relation not in {"USES_TECHNOLOGY", "HAS_PRODUCT", "BELONGS_TO_CHAIN"}:
        return False
    if "释义" in section and ("指" in evidence or len(evidence) < 140):
        return True
    if page.isdigit() and int(page) <= 8 and is_noise_section(section) and "指" in evidence:
        return True
    return False


def is_core_metric(row: dict[str, str]) -> bool:
    if row.get("relation") != "HAS_METRIC":
        return True
    text = f"{row.get('tail_name', '')} {row.get('evidence', '')} {row.get('section', '')}"
    if any(term in text for term in LOW_VALUE_METRIC_KEYWORDS):
        return False
    return any(term in text for term in CORE_METRIC_KEYWORDS)


def company_groups_by_segment(companies: Iterable[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for company in companies:
        segment = company_segment(company) or "其他"
        if company not in groups[segment]:
            groups[segment].append(company)
    return dict(groups)

