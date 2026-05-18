"""Build and search an investment-research claim layer over the curated KG.

The curated graph is good at fact lookup, but professional questions usually need
causal claims: why a technology matters, which companies have direct exposure,
which indicators confirm the thesis, and where evidence is missing. This module
derives that layer deterministically from the curated relation evidence so the QA
path can improve without requiring a new embedding service.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.domain_lexicon import (
    THEME_SYNONYMS,
    canonical_company_name,
    company_segment,
    expanded_terms,
    infer_themes,
    normalize_topic,
    text_matches_terms,
)
from src.extraction_schema import read_csv, stable_id
from src.frontend_data import RELATION_LABELS


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESEARCH_DIR = ROOT_DIR / "data" / "curated"
CLAIMS_FILE = "claims.csv"
EVIDENCE_SPANS_FILE = "evidence_spans.csv"
SEGMENT_DOSSIERS_FILE = "segment_dossiers.jsonl"

CLAIM_CSV_FIELDS = [
    "claim_id",
    "claim_type",
    "topic",
    "claim_text",
    "companies",
    "mechanism",
    "direction",
    "horizon",
    "metric",
    "value",
    "unit",
    "source_report_id",
    "source_title",
    "page",
    "section",
    "source_tier",
    "evidence_span",
    "confidence",
    "as_of_date",
    "exposure_level",
]

EVIDENCE_SPAN_FIELDS = [
    "evidence_id",
    "claim_id",
    "source_report_id",
    "source_title",
    "page",
    "section",
    "source_tier",
    "text",
    "as_of_date",
    "quality",
]

EXPOSURE_ORDER = {"core": 0, "direct": 1, "indirect": 2, "mentioned": 3, "": 4}
CLAIM_TYPE_BONUS = {
    "company_exposure": 3.0,
    "mechanism": 2.4,
    "bottleneck": 2.4,
    "indicator": 2.0,
    "risk": 1.8,
    "supply_chain": 1.7,
    "policy": 1.2,
    "trend": 1.2,
}

EXPLICIT_BOTTLENECK_TERMS = (
    "瓶颈",
    "制约",
    "短板",
    "供给不足",
    "紧缺",
    "受限",
    "掣肘",
    "卡点",
    "难以满足",
    "扩容压力",
    "不及预期",
)

DIRECT_TOPIC_TERMS = {
    "AI服务器": ("ai服务器", "智算服务器", "gpu服务器", "推理服务器", "训练服务器", "服务器整机"),
    "光模块": ("光模块", "高速光模块", "光器件", "光芯片", "硅光", "cpo", "lpo", "光引擎"),
    "液冷": ("液冷", "冷板", "浸没", "cdu", "温控", "热管理", "快接头", "分水器"),
    "AI芯片": ("ai芯片", "gpu", "dcu", "cpu", "加速卡", "推理芯片", "训练芯片"),
    "国产算力": ("国产算力", "自主可控", "国产替代", "信创", "国产ai芯片", "国产服务器"),
    "算力网络": ("交换机", "以太网", "scaleup", "scale-out", "高速互联", "算力网络"),
    "PCB": ("pcb", "印制电路板", "高多层板", "封装基板", "ccl", "覆铜板"),
    "数据中心": ("数据中心", "智算中心", "aidc", "idc", "算力中心", "机柜"),
    "电源": ("电源", "ups", "服务器电源", "数据中心电源", "电力模块"),
}


@dataclass(frozen=True)
class ResearchHit:
    kind: str
    title: str
    text: str
    topic: str
    source: str = ""
    page: str = ""
    section: str = ""
    source_tier: str = ""
    company: str = ""
    claim_type: str = ""
    exposure_level: str = ""
    confidence: str = ""
    as_of_date: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "text": self.text,
            "topic": self.topic,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "source_tier": self.source_tier,
            "company": self.company,
            "claim_type": self.claim_type,
            "exposure_level": self.exposure_level,
            "confidence": self.confidence,
            "as_of_date": self.as_of_date,
            "score": self.score,
        }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize_cell(row.get(field, "")) for field in fields})


def serialize_cell(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "")


def build_research_artifacts(
    *,
    relations_csv: Path,
    output_dir: Path = DEFAULT_RESEARCH_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build claims, evidence spans, and segment dossiers from curated relations."""
    relations = read_csv(relations_csv)
    claims: list[dict[str, Any]] = []
    evidence_spans: list[dict[str, Any]] = []
    seen_claims: set[str] = set()

    for row in relations:
        for claim in claims_from_relation(row):
            if claim["claim_id"] in seen_claims:
                continue
            seen_claims.add(claim["claim_id"])
            claims.append(claim)
            evidence_spans.append(evidence_span_from_claim(claim))

    dossiers = build_segment_dossiers(claims)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / CLAIMS_FILE, CLAIM_CSV_FIELDS, claims)
    write_csv(output_dir / EVIDENCE_SPANS_FILE, EVIDENCE_SPAN_FIELDS, evidence_spans)
    with (output_dir / SEGMENT_DOSSIERS_FILE).open("w", encoding="utf-8") as file:
        for dossier in dossiers:
            file.write(json.dumps(dossier, ensure_ascii=False, sort_keys=True) + "\n")
    return claims, evidence_spans, dossiers


def claims_from_relation(row: dict[str, str]) -> list[dict[str, Any]]:
    relation = row.get("relation", "")
    if relation == "MENTIONED_IN" or not row.get("evidence"):
        return []
    topics = infer_relation_topics(row)
    if not topics:
        return []
    return [claim_from_relation_topic(row, topic) for topic in topics]


def infer_relation_topics(row: dict[str, str]) -> list[str]:
    text = relation_text(row)
    topics = infer_themes(text)
    if row.get("head_type") == "Company":
        topics.extend(infer_themes(company_segment(row.get("head_name", ""))))
    if row.get("tail_type") == "Company":
        topics.extend(infer_themes(company_segment(row.get("tail_name", ""))))
    if "国产" in text and ("算力" in text or "芯片" in text or "服务器" in text):
        topics.append("国产算力")
    deduped = []
    seen = set()
    for topic in topics:
        if topic in THEME_SYNONYMS and topic not in seen:
            seen.add(topic)
            deduped.append(topic)
    return deduped[:3]


def claim_from_relation_topic(row: dict[str, str], topic: str) -> dict[str, Any]:
    relation = row.get("relation", "")
    claim_type = infer_claim_type(row)
    companies = relation_companies(row)
    exposure_level = infer_exposure_level(row, topic) if companies else ""
    metric, value, unit = infer_metric_fields(row)
    direction = infer_direction(row)
    horizon = infer_horizon(row)
    evidence = clean_evidence(row.get("evidence", ""))
    mechanism = infer_mechanism(row, topic)
    claim_text = build_claim_text(row, topic, claim_type, exposure_level, evidence)
    claim_id = stable_id(
        "claim",
        claim_type,
        topic,
        row.get("head_type", ""),
        row.get("head_name", ""),
        relation,
        row.get("tail_type", ""),
        row.get("tail_name", ""),
        row.get("source_report_id", ""),
        evidence[:120],
    )
    return {
        "claim_id": claim_id,
        "claim_type": claim_type,
        "topic": topic,
        "claim_text": claim_text,
        "companies": companies,
        "mechanism": mechanism,
        "direction": direction,
        "horizon": horizon,
        "metric": metric,
        "value": value,
        "unit": unit,
        "source_report_id": row.get("source_report_id", ""),
        "source_title": row.get("source_title", ""),
        "page": row.get("page", ""),
        "section": row.get("section", ""),
        "source_tier": row.get("source_tier", ""),
        "evidence_span": evidence,
        "confidence": row.get("confidence", "0.70"),
        "as_of_date": infer_as_of_date(row),
        "exposure_level": exposure_level,
    }


def evidence_span_from_claim(claim: dict[str, Any]) -> dict[str, Any]:
    text = str(claim.get("evidence_span", ""))
    quality = "high" if len(text) >= 30 and not looks_like_low_quality_text(text) else "medium"
    return {
        "evidence_id": stable_id("evidence", claim["claim_id"], claim.get("source_report_id", ""), claim.get("page", "")),
        "claim_id": claim["claim_id"],
        "source_report_id": claim.get("source_report_id", ""),
        "source_title": claim.get("source_title", ""),
        "page": claim.get("page", ""),
        "section": claim.get("section", ""),
        "source_tier": claim.get("source_tier", ""),
        "text": text,
        "as_of_date": claim.get("as_of_date", ""),
        "quality": quality,
    }


def build_segment_dossiers(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        grouped[str(claim.get("topic", ""))].append(claim)

    dossiers: list[dict[str, Any]] = []
    built_at = datetime.now(timezone.utc).isoformat()
    for topic, rows in sorted(grouped.items()):
        if not topic:
            continue
        exposure = best_company_exposure(rows)
        mechanisms = top_claim_texts(rows, {"mechanism", "trend", "supply_chain"}, 5)
        bottlenecks = top_claim_texts(rows, {"bottleneck"}, 5)
        indicators = top_claim_texts(rows, {"indicator"}, 5)
        risks = top_claim_texts(rows, {"risk"}, 5)
        policies = top_claim_texts(rows, {"policy"}, 3)
        gaps = infer_dossier_gaps(topic, exposure, indicators, risks)
        dossier = {
            "topic": topic,
            "built_at": built_at,
            "summary": build_dossier_summary(topic, exposure, mechanisms, bottlenecks, indicators, risks),
            "technology_mechanism": mechanisms,
            "industry_chain": sorted({claim.get("mechanism", "") for claim in rows if claim.get("claim_type") == "supply_chain" and claim.get("mechanism")})[:8],
            "company_exposure": exposure,
            "leading_indicators": indicators,
            "bottlenecks": bottlenecks,
            "risks": risks,
            "policies": policies,
            "evidence_ids": [claim["claim_id"] for claim in sorted_claims(rows)[:20]],
            "gaps": gaps,
            "conflicts": [],
        }
        dossiers.append(dossier)
    return dossiers


def best_company_exposure(claims: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_company: dict[str, tuple[int, str]] = {}
    for claim in claims:
        if claim.get("claim_type") != "company_exposure":
            continue
        level = str(claim.get("exposure_level", "mentioned"))
        for company in parse_companies(claim.get("companies", [])):
            current = by_company.get(company)
            rank = EXPOSURE_ORDER.get(level, 4)
            if current is None or rank < current[0]:
                by_company[company] = (rank, level)
    grouped: dict[str, list[str]] = {"core": [], "direct": [], "indirect": [], "mentioned": []}
    for company, (_, level) in sorted(by_company.items(), key=lambda item: (item[1][0], company_segment(item[0]), item[0])):
        grouped.setdefault(level, []).append(company)
    return {level: names for level, names in grouped.items() if names}


def top_claim_texts(claims: list[dict[str, Any]], claim_types: set[str], limit: int) -> list[str]:
    return [claim["claim_text"] for claim in sorted_claims(claims) if claim.get("claim_type") in claim_types][:limit]


def sorted_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(claim: dict[str, Any]) -> tuple[float, int, str]:
        confidence = safe_float(claim.get("confidence"), 0.7)
        tier_bonus = 0.2 if claim.get("source_tier") == "1" else 0.0
        exposure_rank = EXPOSURE_ORDER.get(str(claim.get("exposure_level", "")), 4)
        return (-(confidence + tier_bonus), exposure_rank, str(claim.get("claim_text", "")))

    return sorted(claims, key=key)


def infer_dossier_gaps(
    topic: str,
    exposure: dict[str, list[str]],
    indicators: list[str],
    risks: list[str],
) -> list[str]:
    gaps = []
    if not exposure.get("core") and not exposure.get("direct"):
        gaps.append(f"{topic} 缺少可稳定排序的直接公司敞口证据。")
    if not indicators:
        gaps.append(f"{topic} 缺少订单、收入、毛利率、产能或渗透率等领先指标证据。")
    if not risks:
        gaps.append(f"{topic} 缺少明确风险或反证证据。")
    return gaps


def build_dossier_summary(
    topic: str,
    exposure: dict[str, list[str]],
    mechanisms: list[str],
    bottlenecks: list[str],
    indicators: list[str],
    risks: list[str],
) -> str:
    direct = exposure.get("core") or exposure.get("direct") or []
    parts = [f"{topic}："]
    if mechanisms:
        parts.append(f"技术机理集中在{shorten(mechanisms[0], 80)}")
    if bottlenecks:
        parts.append(f"主要瓶颈包括{shorten(bottlenecks[0], 80)}")
    if direct:
        parts.append(f"直接敞口公司包括{'、'.join(direct[:8])}")
    if indicators:
        parts.append(f"可跟踪指标包括{shorten(indicators[0], 70)}")
    if risks:
        parts.append(f"主要风险包括{shorten(risks[0], 70)}")
    return "；".join(parts)


class ResearchMemory:
    def __init__(self, claims: list[dict[str, Any]], dossiers: list[dict[str, Any]]) -> None:
        self.claims = [normalize_claim_row(row) for row in claims]
        self.dossiers = dossiers

    @classmethod
    def load(cls, data_dir: Path = DEFAULT_RESEARCH_DIR) -> "ResearchMemory":
        claims_path = data_dir / CLAIMS_FILE
        dossiers_path = data_dir / SEGMENT_DOSSIERS_FILE
        if not claims_path.exists() or not dossiers_path.exists():
            raise FileNotFoundError(f"Research artifacts not found in {data_dir}")
        claims = read_csv(claims_path)
        dossiers: list[dict[str, Any]] = []
        with dossiers_path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    dossiers.append(json.loads(line))
        return cls(claims, dossiers)

    def search(self, question: str, plan: Any, *, limit: int = 8) -> list[ResearchHit]:
        topics = query_topics(question, getattr(plan, "topics", []), getattr(plan, "expanded_topics", []))
        hits: list[ResearchHit] = []
        if should_use_dossier(question, plan):
            hits.extend(self._search_dossiers(question, topics, plan))
        hits.extend(self._search_claims(question, topics, plan))
        hits.sort(key=lambda hit: (-hit.score, hit.kind, hit.topic, hit.company, hit.title))
        return diversify_research_hits(dedupe_research_hits(hits))[:limit]

    def _search_dossiers(self, question: str, topics: list[str], plan: Any) -> list[ResearchHit]:
        hits = []
        for dossier in self.dossiers:
            topic = str(dossier.get("topic", ""))
            if topics and topic not in topics and not text_matches_terms(topic, topics):
                continue
            score = 25.0 + topic_match_score(topic, question, topics)
            if topics and topic == topics[0]:
                score += 5.0
            if getattr(plan, "answer_type", "") in {"industry_bottleneck", "thematic_research"}:
                score += 2.0
            text = dossier_to_text(dossier)
            hits.append(
                ResearchHit(
                    kind="dossier",
                    title=f"{topic} 产业链投研摘要",
                    text=text,
                    topic=topic,
                    score=round(score, 4),
                )
            )
        return hits

    def _search_claims(self, question: str, topics: list[str], plan: Any) -> list[ResearchHit]:
        hits = []
        for claim in self.claims:
            topic = str(claim.get("topic", ""))
            if topics and topic not in topics and not text_matches_terms(topic, topics):
                continue
            score = score_claim(claim, question, topics, plan)
            if score <= 0:
                continue
            company = first_company(claim.get("companies", []))
            hits.append(
                ResearchHit(
                    kind="claim",
                    title=claim_title(claim),
                    text=str(claim.get("claim_text", "")),
                    topic=topic,
                    source=str(claim.get("source_title", "")),
                    page=str(claim.get("page", "")),
                    section=str(claim.get("section", "")),
                    source_tier=str(claim.get("source_tier", "")),
                    company=company,
                    claim_type=str(claim.get("claim_type", "")),
                    exposure_level=str(claim.get("exposure_level", "")),
                    confidence=str(claim.get("confidence", "")),
                    as_of_date=str(claim.get("as_of_date", "")),
                    score=round(score, 4),
                )
            )
        return hits


def score_claim(claim: dict[str, Any], question: str, topics: list[str], plan: Any) -> float:
    text = " ".join(str(claim.get(key, "")) for key in ("topic", "claim_type", "claim_text", "evidence_span", "section"))
    score = topic_match_score(text, question, topics)
    claim_type = str(claim.get("claim_type", ""))
    score += CLAIM_TYPE_BONUS.get(claim_type, 0.8)
    if str(claim.get("source_tier", "")) == "1":
        score += 0.5
    score += safe_float(claim.get("confidence"), 0.7)
    level = str(claim.get("exposure_level", ""))
    if level == "core":
        score += 2.0
    elif level == "direct":
        score += 1.4
    elif level == "indirect":
        score += 0.4
    companies = parse_companies(claim.get("companies", []))
    plan_companies = set(getattr(plan, "companies", []) or [])
    if companies and plan_companies and plan_companies & set(companies):
        score += 2.5
    answer_type = getattr(plan, "answer_type", "")
    if answer_type == "industry_bottleneck" and claim_type in {"bottleneck", "mechanism", "indicator"}:
        score += 2.0
    if answer_type == "topic_to_company" and claim_type == "company_exposure":
        score += 2.0
    if answer_type == "risk_analysis" and claim_type == "risk":
        score += 2.0
    if answer_type == "company_compare" and companies:
        score += 1.2
    if looks_like_low_quality_text(text):
        score -= 4.0
    return score


def topic_match_score(text: str, question: str, topics: list[str]) -> float:
    score = 0.0
    normalized = normalize_topic(text)
    q_norm = normalize_topic(question)
    for term in expanded_terms([*topics, question]):
        term_norm = normalize_topic(term)
        if not term_norm or len(term_norm) < 2:
            continue
        if term_norm in normalized:
            score += 1.4
        if term_norm in q_norm:
            score += 0.2
    return score


def query_topics(question: str, plan_topics: Iterable[str], expanded_plan_topics: Iterable[str]) -> list[str]:
    del expanded_plan_topics
    topics = [topic for topic in plan_topics if topic in THEME_SYNONYMS]
    topics.extend(infer_themes(question))
    if "国产" in question and "算力" in question:
        topics.append("国产算力")
    if not topics and "算力" in question:
        topics.extend(["AI服务器", "AI芯片", "数据中心", "光模块", "液冷", "算力网络", "电源"])
    result = []
    seen = set()
    for topic in topics:
        if topic and topic not in seen:
            seen.add(topic)
            result.append(topic)
    return result


def should_use_dossier(question: str, plan: Any) -> bool:
    answer_type = getattr(plan, "answer_type", "")
    if answer_type in {"industry_bottleneck", "thematic_research", "topic_to_company"}:
        return True
    return any(term in question for term in ("为什么", "趋势", "瓶颈", "传导", "受益", "跟踪指标", "怎么看"))


def dossier_to_text(dossier: dict[str, Any]) -> str:
    lines = [str(dossier.get("summary", ""))]
    exposure = dossier.get("company_exposure", {}) or {}
    if exposure:
        exposure_text = []
        for level in ("core", "direct", "indirect", "mentioned"):
            names = exposure.get(level) or []
            if names:
                exposure_text.append(f"{level}:{'、'.join(names[:10])}")
        lines.append("公司敞口：" + "；".join(exposure_text))
    for label, key in (
        ("技术机理", "technology_mechanism"),
        ("瓶颈", "bottlenecks"),
        ("领先指标", "leading_indicators"),
        ("风险与反证", "risks"),
        ("证据缺口", "gaps"),
    ):
        values = dossier.get(key) or []
        if values:
            lines.append(f"{label}：" + "；".join(shorten(str(item), 120) for item in values[:4]))
    return "\n".join(line for line in lines if line.strip())


def dedupe_research_hits(hits: list[ResearchHit]) -> list[ResearchHit]:
    output = []
    seen = set()
    for hit in hits:
        key = (hit.kind, hit.topic, hit.company, normalize_topic(hit.text)[:120])
        if key in seen:
            continue
        seen.add(key)
        output.append(hit)
    return output


def diversify_research_hits(hits: list[ResearchHit]) -> list[ResearchHit]:
    output = []
    exposure_seen: set[tuple[str, str, str]] = set()
    bucket_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for hit in hits:
        if hit.claim_type == "company_exposure" and hit.company:
            exposure_key = (hit.topic, hit.company, hit.exposure_level)
            if exposure_key in exposure_seen:
                continue
            exposure_seen.add(exposure_key)
        bucket_key = (hit.kind, hit.topic, hit.company, hit.claim_type)
        if bucket_counts[bucket_key] >= (1 if hit.claim_type == "company_exposure" else 2):
            continue
        bucket_counts[bucket_key] += 1
        output.append(hit)
    return output


def normalize_claim_row(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated["companies"] = parse_companies(updated.get("companies", []))
    return updated


def relation_text(row: dict[str, str]) -> str:
    return " ".join(str(row.get(key, "") or "") for key in ("head_name", "tail_name", "relation", "evidence", "section"))


def relation_companies(row: dict[str, str]) -> list[str]:
    companies = []
    if row.get("head_type") == "Company":
        companies.append(canonical_company_name(row.get("head_name", "")))
    if row.get("tail_type") == "Company":
        companies.append(canonical_company_name(row.get("tail_name", "")))
    return [company for index, company in enumerate(companies) if company and company not in companies[:index]]


def infer_claim_type(row: dict[str, str]) -> str:
    relation = row.get("relation", "")
    text = relation_text(row)
    if relation == "DISCLOSES_RISK":
        return "risk"
    if relation == "HAS_METRIC" or any(term in text for term in ("订单", "毛利率", "营收", "收入", "产能", "PUE", "市场份额", "渗透率")):
        return "indicator"
    if relation == "CONSTRAINS" or any(term in text for term in EXPLICIT_BOTTLENECK_TERMS):
        return "bottleneck"
    if relation in {"HAS_PRODUCT", "USES_TECHNOLOGY", "BELONGS_TO_CHAIN", "HAS_EXPOSURE"} and row.get("head_type") == "Company":
        return "company_exposure"
    if relation in {"UPSTREAM_OF", "DOWNSTREAM_OF", "DEPENDS_ON"}:
        return "supply_chain"
    if relation in {"ENABLES", "DRIVES", "RELIEVES", "BENEFITS_FROM"}:
        return "mechanism"
    if relation == "SUPPORTED_BY_POLICY":
        return "policy"
    return "trend"


def infer_exposure_level(row: dict[str, str], topic: str) -> str:
    if row.get("head_type") != "Company":
        return ""
    relation = row.get("relation", "")
    company = row.get("head_name", "")
    segment_norm = normalize_topic(company_segment(company))
    topic_norm = normalize_topic(topic)
    tail_norm = normalize_topic(row.get("tail_name", ""))
    evidence_norm = normalize_topic(row.get("evidence", ""))
    direct_terms = DIRECT_TOPIC_TERMS.get(topic, tuple(normalize_topic(term) for term in THEME_SYNONYMS.get(topic, (topic,))))
    direct_match = any(normalize_topic(term) in tail_norm for term in direct_terms)
    segment_match = topic_norm in segment_norm or any(normalize_topic(term) in segment_norm for term in direct_terms)
    if segment_match:
        return "core"
    if topic == "液冷":
        if any(term in segment_norm for term in ("散热", "热管理", "温控")):
            return "direct"
        if direct_match or topic_norm in evidence_norm:
            return "indirect"
    if relation in {"HAS_PRODUCT", "USES_TECHNOLOGY", "HAS_EXPOSURE"} and direct_match:
        return "direct"
    if topic_norm in evidence_norm or any(normalize_topic(term) in evidence_norm for term in direct_terms):
        return "indirect"
    return "mentioned"


def infer_metric_fields(row: dict[str, str]) -> tuple[str, str, str]:
    text = f"{row.get('tail_name', '')} {row.get('evidence', '')}"
    metric = row.get("tail_name", "") if row.get("tail_type") in {"Metric", "LeadingIndicator"} else ""
    value = ""
    unit = ""
    match = re.search(r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)(\s*(?:%|亿元|万元|元|亿美元|万台|台|个|EFLOPS|PFLOPS|TFLOPS|kW|KW|MW|GW|PUE)?)", text, flags=re.I)
    if match:
        value = match.group(1).replace(",", "")
        unit = match.group(2).strip()
    return metric, value, unit


def infer_direction(row: dict[str, str]) -> str:
    relation = row.get("relation", "")
    if relation in {"CONSTRAINS", "DISCLOSES_RISK"}:
        return "negative"
    if relation in {"ENABLES", "DRIVES", "RELIEVES", "BENEFITS_FROM", "SUPPORTED_BY_POLICY"}:
        return "positive"
    return "neutral"


def infer_horizon(row: dict[str, str]) -> str:
    text = relation_text(row)
    if any(term in text for term in ("长期", "2030", "未来五年", "远期")):
        return "long_term"
    if any(term in text for term in ("2027", "2028", "中期", "逐步")):
        return "mid_term"
    if any(term in text for term in ("当前", "报告期", "2025", "2026", "短期")):
        return "near_term"
    return ""


def infer_mechanism(row: dict[str, str], topic: str) -> str:
    relation = row.get("relation", "")
    label = RELATION_LABELS.get(relation, relation)
    head = row.get("head_name", "")
    tail = row.get("tail_name", "")
    if relation in {"ENABLES", "DRIVES", "RELIEVES", "CONSTRAINS", "DEPENDS_ON", "UPSTREAM_OF", "DOWNSTREAM_OF"}:
        return f"{head} {label} {tail}".strip()
    if row.get("head_type") == "Company":
        return f"{head} 在 {topic} 的关系为{label}{tail}".strip()
    return f"{topic}：{label}{tail}".strip()


def build_claim_text(
    row: dict[str, str],
    topic: str,
    claim_type: str,
    exposure_level: str,
    evidence: str,
) -> str:
    head = row.get("head_name", "")
    tail = row.get("tail_name", "")
    relation_label = RELATION_LABELS.get(row.get("relation", ""), row.get("relation", ""))
    if claim_type == "company_exposure":
        return f"{head} 对 {topic} 的公司敞口为 {exposure_level or 'mentioned'}：{relation_label}{tail}。证据：{evidence}"
    if claim_type == "risk":
        return f"{head} 在 {topic} 相关业务中的风险：{tail}。证据：{evidence}"
    if claim_type == "indicator":
        return f"{topic} 的可跟踪指标：{head}{relation_label}{tail}。证据：{evidence}"
    if claim_type == "bottleneck":
        return f"{topic} 的约束或瓶颈：{head}{relation_label}{tail}。证据：{evidence}"
    return f"{topic} 机理/传导：{head}{relation_label}{tail}。证据：{evidence}"


def infer_as_of_date(row: dict[str, str]) -> str:
    text = " ".join(str(row.get(key, "") or "") for key in ("source_title", "source_report_id", "evidence"))
    match = re.search(r"(20\d{2})", text)
    return match.group(1) if match else ""


def clean_evidence(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:500]


def looks_like_low_quality_text(text: str) -> bool:
    value = str(text or "")
    if "免责声明" in value or "请务必" in value:
        return True
    if re.fullmatch(r"[\d\s,.%％+\-—/]+", value.strip()):
        return True
    return False


def parse_companies(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value)
            items = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            items = []
    elif isinstance(value, str) and value.strip():
        items = [value]
    else:
        items = []
    result = []
    for item in items:
        company = canonical_company_name(str(item))
        if company and company not in result:
            result.append(company)
    return result


def first_company(value: Any) -> str:
    companies = parse_companies(value)
    return companies[0] if companies else ""


def claim_title(claim: dict[str, Any]) -> str:
    company = first_company(claim.get("companies", []))
    topic = str(claim.get("topic", ""))
    claim_type = str(claim.get("claim_type", ""))
    if company:
        return f"{company} {topic} {claim_type}".strip()
    return f"{topic} {claim_type}".strip()


def shorten(text: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)] + "..."


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
