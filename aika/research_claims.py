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

from aika.domain_lexicon import (
    TECHNICAL_SOURCE_TYPES,
    THEME_SYNONYMS,
    canonical_company_name,
    company_lookup,
    company_segment,
    expanded_terms,
    infer_themes,
    normalize_topic,
    text_matches_terms,
)
from aika.extraction_schema import load_jsonl, read_csv, stable_id
from aika.frontend_data import RELATION_LABELS


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESEARCH_DIR = ROOT_DIR / "data" / "curated"
CLAIMS_FILE = "claims.csv"
EVIDENCE_SPANS_FILE = "evidence_spans.csv"
SEGMENT_DOSSIERS_FILE = "segment_dossiers.jsonl"
CLAIM_REVIEWS_FILE = "claim_reviews.jsonl"

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
    "review_status",
    "reviewer_note",
    "quality_flags",
    "conflict_group_id",
]

REVIEWABLE_CLAIM_FIELDS = {
    "claim_text",
    "claim_type",
    "topic",
    "companies",
    "mechanism",
    "direction",
    "horizon",
    "metric",
    "value",
    "unit",
    "evidence_span",
    "confidence",
    "as_of_date",
    "exposure_level",
    "review_status",
    "reviewer_note",
    "quality_flags",
    "conflict_group_id",
}

CLAIM_REVIEW_STATUSES = {"auto", "approved", "revised", "rejected", "needs_review"}

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

TECHNICAL_REPORT_PREFIX = "industry_tech_"
MAX_DIRECT_CLAIMS_PER_REPORT = 80

TECHNICAL_HIGH_SIGNAL_TERMS = (
    "UCIe",
    "UALink",
    "Ultra Ethernet",
    "UET",
    "RDMA",
    "congestion control",
    "Scale Up",
    "Scale-Up",
    "Scale Out",
    "Scale-Out",
    "SerDes",
    "224G",
    "PAM4",
    "InfiniBand",
    "NVLink",
    "all-to-all",
    "switch fabric",
    "CPO",
    "LPO",
    "NPO",
    "silicon photonics",
    "integrated photonics",
    "optical engine",
    "HBM",
    "chiplet",
    "advanced packaging",
    "heterogeneous integration",
    "2.5D",
    "3D integration",
    "FP8",
    "MoE",
    "MLA",
    "FlashAttention",
    "PagedAttention",
    "MLPerf",
    "time-to-train",
    "throughput",
    "latency",
    "KV cache",
    "memory bandwidth",
    "communication overhead",
    "compute-communication overlap",
    "memory wall",
    "cold plate",
    "liquid cooling",
    "thermal management",
    "CDU",
    "rack manifold",
    "power density",
    "AI accelerator",
    "GPU hours",
)

TECHNICAL_LOW_VALUE_TERMS = (
    "LEGAL NOTICE",
    "ALL RIGHTS RESERVED",
    "MERCHANTABILITY",
    "FITNESS FOR A PARTICULAR PURPOSE",
    "TRADEMARK",
    "INTELLECTUAL PROPERTY",
    "NO LICENSE",
    "GOVERNING DOCUMENTS",
    "WARRANTIES",
    "COPYRIGHT",
    "REFERENCES",
    "BIBLIOGRAPHY",
    "ACM SIGCOMM",
    "IEEE TRANSACTIONS",
)

TECHNICAL_MECHANISM_TERMS = (
    "enable",
    "enables",
    "support",
    "supports",
    "achieve",
    "achieves",
    "improve",
    "improves",
    "reduce",
    "reduces",
    "accelerate",
    "accelerates",
    "efficient",
    "efficiency",
    "overlap",
    "scale",
    "scaling",
    "优化",
    "提升",
    "降低",
    "支撑",
    "推动",
    "提高",
)

TECHNICAL_BOTTLENECK_TERMS = (
    "bottleneck",
    "constraint",
    "constrained",
    "limited",
    "limitation",
    "overhead",
    "latency",
    "congestion",
    "memory wall",
    "bandwidth",
    "power density",
    "thermal",
    "功耗",
    "散热",
    "瓶颈",
    "约束",
    "受限",
    "带宽",
    "拥塞",
)

TECHNICAL_INDICATOR_TERMS = (
    "benchmark",
    "throughput",
    "latency",
    "time-to-train",
    "GPU hours",
    "tokens",
    "parameters",
    "GB/s",
    "Tb/s",
    "GT/s",
    "PAM4",
    "kW",
    "MW",
    "PUE",
    "指标",
    "吞吐",
    "时延",
    "带宽",
)

DIRECT_METRIC_PATTERN = re.compile(
    r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:[KMBT])?\s*(?:GPU hours|tokens|parameters|GB/s|Tb/s|GT/s|W|kW|MW|PUE|ms|us|ns|%))",
    flags=re.I,
)


@dataclass(frozen=True)
class ResearchHit:
    kind: str
    title: str
    text: str
    topic: str
    claim_id: str = ""
    source: str = ""
    page: str = ""
    section: str = ""
    source_tier: str = ""
    company: str = ""
    claim_type: str = ""
    exposure_level: str = ""
    confidence: str = ""
    as_of_date: str = ""
    evidence_span: str = ""
    review_status: str = ""
    reviewer_note: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "text": self.text,
            "topic": self.topic,
            "claim_id": self.claim_id,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "source_tier": self.source_tier,
            "company": self.company,
            "claim_type": self.claim_type,
            "exposure_level": self.exposure_level,
            "confidence": self.confidence,
            "as_of_date": self.as_of_date,
            "evidence_span": self.evidence_span,
            "review_status": self.review_status,
            "reviewer_note": self.reviewer_note,
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
    chunks_dir: Path | None = None,
    include_direct_claims: bool = True,
    write_outputs: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build claims, evidence spans, and segment dossiers.

    The base layer is derived from curated KG relations. For professional
    technical sources, we also derive direct original-text claims from chunks so
    roadmap/spec/paper evidence can inform mechanisms and bottlenecks even when
    it is not naturally represented as company-centric triples.
    """
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

    if include_direct_claims and chunks_dir is not None:
        for claim in claims_from_technical_chunks(chunks_dir):
            if claim["claim_id"] in seen_claims:
                continue
            seen_claims.add(claim["claim_id"])
            claims.append(claim)
            evidence_spans.append(evidence_span_from_claim(claim))

    dossiers = build_segment_dossiers(claims)
    if write_outputs:
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
        "review_status": row.get("review_status", "auto") or "auto",
        "reviewer_note": "",
        "quality_flags": "",
        "conflict_group_id": "",
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


def claims_from_technical_chunks(chunks_dir: Path) -> list[dict[str, Any]]:
    """Derive original-text claims from professional technical source chunks."""
    claims: list[dict[str, Any]] = []
    report_counts: dict[str, int] = defaultdict(int)
    if not chunks_dir.exists():
        return claims
    for path in sorted(chunks_dir.glob(f"{TECHNICAL_REPORT_PREFIX}*.jsonl")):
        for chunk in load_jsonl(path):
            report_id = str(chunk.get("report_id", ""))
            if report_counts[report_id] >= MAX_DIRECT_CLAIMS_PER_REPORT:
                continue
            if not is_professional_technical_chunk(chunk):
                continue
            if looks_like_low_value_technical_chunk(chunk):
                continue
            evidence = select_technical_evidence(chunk)
            if not evidence:
                continue
            topics = infer_technical_topics(chunk, evidence)
            if not topics:
                continue
            claim_types = infer_technical_claim_types(evidence)
            companies = companies_explicitly_in_text(evidence)
            for topic in topics[:2]:
                for claim_type in claim_types[:2]:
                    claims.append(technical_claim_from_chunk(chunk, topic, claim_type, evidence, companies))
                    report_counts[report_id] += 1
                    if report_counts[report_id] >= MAX_DIRECT_CLAIMS_PER_REPORT:
                        break
                if report_counts[report_id] >= MAX_DIRECT_CLAIMS_PER_REPORT:
                    break
    return claims


def is_professional_technical_chunk(chunk: dict[str, Any]) -> bool:
    report_id = str(chunk.get("report_id", ""))
    source_type = str(chunk.get("source_type", ""))
    return report_id.startswith(TECHNICAL_REPORT_PREFIX) or source_type in TECHNICAL_SOURCE_TYPES


def looks_like_low_value_technical_chunk(chunk: dict[str, Any]) -> bool:
    text = re.sub(r"\s+", " ", str(chunk.get("text", "") or "")).strip()
    if len(text) < 80:
        return True
    upper = text.upper()
    if any(term in upper for term in TECHNICAL_LOW_VALUE_TERMS):
        signal_terms = count_matching_terms(text, TECHNICAL_HIGH_SIGNAL_TERMS)
        if signal_terms <= 1:
            return True
    section = normalize_topic(chunk.get("section", ""))
    if section in {"contents", "目录"}:
        return True
    if text.count(". . .") >= 4 or text.count(" . . ") >= 8:
        return True
    if looks_like_low_quality_text(text):
        return True
    return count_matching_terms(text, TECHNICAL_HIGH_SIGNAL_TERMS) == 0


def select_technical_evidence(chunk: dict[str, Any]) -> str:
    text = re.sub(r"\s+", " ", str(chunk.get("text", "") or "")).strip()
    candidates = split_evidence_candidates(text)
    if not candidates:
        return ""
    scored: list[tuple[int, str]] = []
    for candidate in candidates:
        if len(candidate) < 35 or looks_like_low_quality_text(candidate):
            continue
        if any(term in candidate.upper() for term in TECHNICAL_LOW_VALUE_TERMS):
            continue
        if looks_like_reference_candidate(candidate):
            continue
        signal = count_matching_terms(candidate, TECHNICAL_HIGH_SIGNAL_TERMS)
        if not signal:
            continue
        score = signal * 4
        score += count_matching_terms(candidate, TECHNICAL_MECHANISM_TERMS)
        score += count_matching_terms(candidate, TECHNICAL_BOTTLENECK_TERMS)
        score += count_matching_terms(candidate, TECHNICAL_INDICATOR_TERMS)
        if re.search(r"\d", candidate):
            score += 2
        scored.append((score, candidate))
    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return clean_evidence(scored[0][1])


def split_evidence_candidates(text: str) -> list[str]:
    rough_parts = re.split(r"(?<=[。！？.!?])\s+|\n+", text)
    candidates: list[str] = []
    for part in rough_parts:
        cleaned = re.sub(r"\s+", " ", part).strip(" -•\t")
        if not cleaned:
            continue
        if len(cleaned) > 650:
            chunks = re.split(r";\s+|；|,\s+(?=[A-Z])", cleaned)
            candidates.extend(item.strip(" -•\t") for item in chunks if item.strip())
        else:
            candidates.append(cleaned)
    return candidates


def looks_like_reference_candidate(text: str) -> bool:
    value = str(text or "")
    if re.search(r"\bpp\.\s*\d", value, flags=re.I):
        return True
    if re.search(r"\b(?:Proceedings|Conference|SIGCOMM|arXiv preprint)\b", value, flags=re.I):
        return True
    if re.search(r"^[A-Z][A-Za-z\-]+,\s+\".+\"", value):
        return True
    if value.count(" et al.") >= 1 and re.search(r"\b20\d{2}\b", value):
        return True
    return False


def infer_technical_topics(chunk: dict[str, Any], evidence: str) -> list[str]:
    text = " ".join(
        str(chunk.get(key, "") or "") for key in ("source_title", "source_type", "section", "context")
    )
    raw_text = f"{text} {evidence}"
    topics = infer_themes(raw_text)
    normalized = normalize_topic(raw_text)
    if "ucie" in normalized or "chiplet" in normalized or "advancedpackaging" in normalized:
        topics.append("AI芯片")
    if "ualink" in normalized or "ultraethernet" in normalized or "uet" in normalized:
        topics.append("算力网络")
    if "flashattention" in normalized or "pagedattention" in normalized or "fp8" in normalized:
        topics.append("AI芯片")
    if "liquidcooling" in normalized or "coldplate" in normalized or "thermalmanagement" in normalized:
        topics.append("液冷")
    if "siliconphotonics" in normalized or "integratedphotonics" in normalized or "opticalengine" in normalized:
        topics.append("光模块")
    deduped: list[str] = []
    for topic in topics:
        if topic in THEME_SYNONYMS and topic not in deduped and technical_topic_supported(topic, raw_text, normalized):
            deduped.append(topic)
    return deduped


def technical_topic_supported(topic: str, raw_text: str, normalized_text: str) -> bool:
    if topic == "光模块":
        return any(
            term in normalized_text
            for term in (
                "光模块",
                "高速光模块",
                "硅光",
                "光器件",
                "光引擎",
                "siliconphotonics",
                "integratedphotonics",
                "opticalengine",
                "co-packagedoptics",
            )
        ) or bool(re.search(r"\b(?:CPO|LPO|NPO)\b", raw_text, flags=re.I))
    if topic == "液冷":
        return any(
            term in normalized_text
            for term in (
                "液冷",
                "冷板",
                "温控",
                "热管理",
                "coldplate",
                "liquidcooling",
                "thermalmanagement",
                "powerdensity",
                "heat",
                "thermal",
            )
        ) or bool(re.search(r"\bCDU\b", raw_text, flags=re.I))
    if topic == "电源":
        return any(term in normalized_text for term in ("电源", "ups", "powerdelivery", "powersupply"))
    return True


def infer_technical_claim_types(evidence: str) -> list[str]:
    claim_types: list[str] = []
    if count_matching_terms(evidence, TECHNICAL_BOTTLENECK_TERMS):
        claim_types.append("bottleneck")
    if has_direct_metric(evidence):
        claim_types.append("indicator")
    if any(term in normalize_topic(evidence) for term in ("upstream", "downstream", "supplychain", "packaging", "interconnect", "fabric", "cluster")):
        claim_types.append("supply_chain")
    if count_matching_terms(evidence, TECHNICAL_MECHANISM_TERMS):
        claim_types.append("mechanism")
    if not claim_types:
        claim_types.append("trend")
    return dedupe_strings(claim_types)


def technical_claim_from_chunk(
    chunk: dict[str, Any],
    topic: str,
    claim_type: str,
    evidence: str,
    companies: list[str],
) -> dict[str, Any]:
    metric, value, unit = infer_direct_metric_fields(evidence) if claim_type == "indicator" else ("", "", "")
    source_report_id = str(chunk.get("report_id", ""))
    claim_id = stable_id(
        "claim",
        "direct_technical",
        claim_type,
        topic,
        source_report_id,
        chunk.get("chunk_id", ""),
        evidence[:160],
    )
    mechanism = infer_direct_mechanism(topic, claim_type, evidence)
    return {
        "claim_id": claim_id,
        "claim_type": claim_type,
        "topic": topic,
        "claim_text": build_direct_claim_text(topic, claim_type, evidence, str(chunk.get("source_title", ""))),
        "companies": companies,
        "mechanism": mechanism,
        "direction": "negative" if claim_type in {"bottleneck", "risk"} else "positive" if claim_type == "mechanism" else "neutral",
        "horizon": infer_direct_horizon(evidence),
        "metric": metric,
        "value": value,
        "unit": unit,
        "source_report_id": source_report_id,
        "source_title": str(chunk.get("source_title", "")),
        "page": str(chunk.get("page", "")),
        "section": str(chunk.get("section", "")),
        "source_tier": str(chunk.get("source_tier", "")),
        "evidence_span": evidence,
        "confidence": "0.78" if chunk.get("source_tier") == "1" else "0.70",
        "as_of_date": infer_direct_as_of_date(chunk, evidence),
        "exposure_level": "mentioned" if companies else "",
        "review_status": "auto",
        "reviewer_note": "",
        "quality_flags": "",
        "conflict_group_id": "",
    }


def build_direct_claim_text(topic: str, claim_type: str, evidence: str, source_title: str = "") -> str:
    labels = {
        "mechanism": "技术机理",
        "bottleneck": "约束或瓶颈",
        "indicator": "可跟踪指标",
        "supply_chain": "产业传导",
        "risk": "风险或反证",
        "trend": "技术趋势",
    }
    label = labels.get(claim_type, "技术判断")
    prefix = f"{source_title}：" if source_title and source_title not in evidence else ""
    return f"{topic} 的{label}：{prefix}{evidence}"


def infer_direct_mechanism(topic: str, claim_type: str, evidence: str) -> str:
    if claim_type == "bottleneck":
        return f"{topic} 受 {shorten(evidence, 80)} 约束"
    if claim_type == "indicator":
        return f"{topic} 可通过 {shorten(evidence, 80)} 跟踪"
    if claim_type == "supply_chain":
        return f"{topic} 的产业链传导涉及 {shorten(evidence, 80)}"
    return f"{topic} 由 {shorten(evidence, 80)} 驱动"


def infer_direct_metric_fields(evidence: str) -> tuple[str, str, str]:
    metric = ""
    value = ""
    unit = ""
    metric_terms = (
        "GPU hours",
        "tokens",
        "parameters",
        "throughput",
        "latency",
        "bandwidth",
        "time-to-train",
        "power density",
        "GPU小时",
        "吞吐",
        "时延",
        "带宽",
    )
    for term in metric_terms:
        if term.lower() in evidence.lower():
            metric = term
            break
    match = DIRECT_METRIC_PATTERN.search(evidence)
    if match:
        value = match.group(1).replace(",", "").strip()
        unit_match = re.search(r"(GPU hours|tokens|parameters|GB/s|Tb/s|GT/s|W|kW|MW|PUE|ms|us|ns|%)$", value, flags=re.I)
        if unit_match:
            unit = unit_match.group(1)
    return metric, value, unit


def has_direct_metric(evidence: str) -> bool:
    if DIRECT_METRIC_PATTERN.search(evidence):
        return True
    return bool(re.search(r"\d", evidence) and count_matching_terms(evidence, TECHNICAL_INDICATOR_TERMS))


def infer_direct_horizon(evidence: str) -> str:
    text = evidence.lower()
    if any(term in text for term in ("2030", "long term", "roadmap", "future generation", "未来")):
        return "long_term"
    if any(term in text for term in ("2027", "2028", "next generation", "中期", "逐步")):
        return "mid_term"
    if any(term in text for term in ("2024", "2025", "2026", "current", "rev 1.0", "当前")):
        return "near_term"
    return ""


def infer_direct_as_of_date(chunk: dict[str, Any], evidence: str) -> str:
    text = " ".join(str(chunk.get(key, "") or "") for key in ("year", "source_title", "report_id"))
    match = re.search(r"(20\d{2})", f"{text} {evidence}")
    return match.group(1) if match else ""


def companies_explicitly_in_text(text: str) -> list[str]:
    lookup = company_lookup()
    normalized = normalize_topic(text)
    companies: list[str] = []
    for company, aliases in lookup.aliases_by_company.items():
        if any(normalize_topic(alias) in normalized for alias in aliases):
            companies.append(company)
    return companies


def count_matching_terms(text: str, terms: Iterable[str]) -> int:
    normalized = normalize_topic(text)
    return sum(1 for term in terms if normalize_topic(term) in normalized)


def dedupe_strings(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        key = str(value).casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(str(value))
    return result


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


def load_claim_reviews(data_dir: Path = DEFAULT_RESEARCH_DIR) -> dict[str, dict[str, Any]]:
    reviews_path = data_dir / CLAIM_REVIEWS_FILE
    reviews: dict[str, dict[str, Any]] = {}
    if not reviews_path.exists():
        return reviews
    with reviews_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue
            claim_id = str(review.get("claim_id") or "").strip()
            if claim_id:
                reviews[claim_id] = normalize_claim_review(claim_id, review)
    return reviews


def write_claim_review(
    data_dir: Path,
    claim_id: str,
    updates: dict[str, Any],
    *,
    reviewer: str = "frontend",
) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    review = normalize_claim_review(claim_id, updates, reviewer=reviewer)
    reviews_path = data_dir / CLAIM_REVIEWS_FILE
    with reviews_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(review, ensure_ascii=False, sort_keys=True) + "\n")
    return review


def normalize_claim_review(
    claim_id: str,
    updates: dict[str, Any],
    *,
    reviewer: str = "",
) -> dict[str, Any]:
    claim_id = str(claim_id or updates.get("claim_id") or "").strip()
    if not claim_id:
        raise ValueError("claim_id is required")
    review: dict[str, Any] = {
        "claim_id": claim_id,
        "updated_at": str(updates.get("updated_at") or datetime.now(timezone.utc).isoformat()),
        "reviewer": str(updates.get("reviewer") or reviewer or "frontend"),
    }
    for field in REVIEWABLE_CLAIM_FIELDS:
        if field not in updates:
            continue
        value = updates.get(field)
        if field == "companies":
            value = parse_companies(value)
        elif isinstance(value, list):
            value = [str(item).strip() for item in value if str(item).strip()]
        else:
            value = str(value or "").strip()
        if field == "review_status" and value and value not in CLAIM_REVIEW_STATUSES:
            value = "needs_review"
        review[field] = value
    if "review_status" not in review:
        review["review_status"] = "revised"
    return review


def apply_claim_reviews(claims: list[dict[str, Any]], reviews: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not reviews:
        return claims
    output: list[dict[str, Any]] = []
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        review = reviews.get(claim_id)
        if not review:
            output.append(claim)
            continue
        updated = dict(claim)
        for field in REVIEWABLE_CLAIM_FIELDS:
            if field in review:
                updated[field] = review[field]
        updated["reviewed_at"] = review.get("updated_at", "")
        updated["reviewer"] = review.get("reviewer", "")
        output.append(updated)
    return output


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
        claims = apply_claim_reviews(read_csv(claims_path), load_claim_reviews(data_dir))
        dossiers: list[dict[str, Any]] = []
        with dossiers_path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    dossiers.append(json.loads(line))
        return cls(claims, dossiers)

    def review_claim(self, claim_id: str, updates: dict[str, Any], *, reviewer: str = "frontend") -> dict[str, Any]:
        review = normalize_claim_review(claim_id, updates, reviewer=reviewer)
        for index, claim in enumerate(self.claims):
            if str(claim.get("claim_id") or "") != review["claim_id"]:
                continue
            updated = apply_claim_reviews([claim], {review["claim_id"]: review})[0]
            self.claims[index] = normalize_claim_row(updated)
            return self.claims[index]
        raise KeyError(f"Claim not found: {claim_id}")

    def get_claim(self, claim_id: str) -> dict[str, Any]:
        for claim in self.claims:
            if str(claim.get("claim_id") or "") == str(claim_id or ""):
                return claim
        raise KeyError(f"Claim not found: {claim_id}")

    def claim_stats(self) -> dict[str, Any]:
        reviewed = sum(1 for claim in self.claims if str(claim.get("review_status", "")) in {"approved", "revised", "rejected", "needs_review"})
        rejected = sum(1 for claim in self.claims if str(claim.get("review_status", "")) == "rejected")
        direct_companies = {
            company
            for claim in self.claims
            if claim.get("claim_type") == "company_exposure" and claim.get("exposure_level") in {"core", "direct"}
            for company in parse_companies(claim.get("companies", []))
        }
        by_type: dict[str, int] = defaultdict(int)
        for claim in self.claims:
            by_type[str(claim.get("claim_type", "") or "unknown")] += 1
        return {
            "claims": len(self.claims),
            "dossiers": len(self.dossiers),
            "reviewed_claims": reviewed,
            "rejected_claims": rejected,
            "direct_exposure_companies": len(direct_companies),
            "claim_type_counts": dict(by_type),
        }

    def search(self, question: str, plan: Any, *, limit: int = 8) -> list[ResearchHit]:
        topics = query_topics(question, getattr(plan, "topics", []), getattr(plan, "expanded_topics", []))
        hits: list[ResearchHit] = []
        if should_use_dossier(question, plan):
            hits.extend(self._search_dossiers(question, topics, plan))
        hits.extend(self._search_claims(question, topics, plan))
        hits.sort(key=lambda hit: (-hit.score, hit.kind, hit.topic, hit.company, hit.title))
        return diversify_research_hits(dedupe_research_hits(hits))[:limit]

    def search_global_dossiers(
        self,
        question: str,
        plan: Any,
        *,
        limit: int = 3,
        topics: list[str] | None = None,
    ) -> list[ResearchHit]:
        """Search topic-level dossiers for broad GraphRAG/global context."""
        selected_topics = topics or query_topics(question, getattr(plan, "topics", []), getattr(plan, "expanded_topics", []))
        hits = self._search_dossiers(question, selected_topics, plan)
        hits.sort(key=lambda hit: (-hit.score, hit.topic, hit.title))
        return dedupe_research_hits(hits)[: max(0, limit)]

    def search_local_claims(
        self,
        question: str,
        plan: Any,
        *,
        limit: int = 12,
        claim_types: list[str] | set[str] | tuple[str, ...] | None = None,
        companies: list[str] | set[str] | tuple[str, ...] | None = None,
        topics: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> list[ResearchHit]:
        """Search local claim bundles with optional company/type/topic filters."""
        selected_topics = list(topics or []) or query_topics(
            question,
            getattr(plan, "topics", []),
            getattr(plan, "expanded_topics", []),
        )
        claim_type_set = {str(item) for item in (claim_types or []) if str(item)}
        company_set = {str(item) for item in (companies or getattr(plan, "companies", []) or []) if str(item)}
        hits = []
        for hit in self._search_claims(question, selected_topics, plan):
            if claim_type_set and hit.claim_type not in claim_type_set:
                continue
            if company_set and hit.company and hit.company not in company_set:
                continue
            hits.append(hit)
        hits.sort(key=lambda hit: (-hit.score, hit.claim_type, hit.topic, hit.company, hit.title))
        return diversify_research_hits(dedupe_research_hits(hits))[: max(0, limit)]

    def _search_dossiers(self, question: str, topics: list[str], plan: Any) -> list[ResearchHit]:
        hits = []
        focus_terms = query_focus_terms(question)
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
            if focus_terms:
                if text_matches_terms(text, focus_terms):
                    score += 4.0
                else:
                    score -= 8.0
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
            if str(claim.get("review_status", "")) == "rejected":
                continue
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
                    claim_id=str(claim.get("claim_id", "")),
                    source=str(claim.get("source_title", "")),
                    page=str(claim.get("page", "")),
                    section=str(claim.get("section", "")),
                    source_tier=str(claim.get("source_tier", "")),
                    company=company,
                    claim_type=str(claim.get("claim_type", "")),
                    exposure_level=str(claim.get("exposure_level", "")),
                    confidence=str(claim.get("confidence", "")),
                    as_of_date=str(claim.get("as_of_date", "")),
                    evidence_span=str(claim.get("evidence_span", "")),
                    review_status=str(claim.get("review_status", "")),
                    reviewer_note=str(claim.get("reviewer_note", "")),
                    score=round(score, 4),
                )
            )
        return hits


def score_claim(claim: dict[str, Any], question: str, topics: list[str], plan: Any) -> float:
    text = " ".join(str(claim.get(key, "")) for key in ("topic", "claim_type", "claim_text", "evidence_span", "section"))
    score = topic_match_score(text, question, topics)
    focus_terms = query_focus_terms(question)
    if focus_terms:
        focus_matches = sum(1 for term in focus_terms if text_matches_terms(text, [term]))
        if focus_matches:
            score += min(24.0, 12.0 * focus_matches)
            if str(claim.get("source_report_id", "")).startswith(TECHNICAL_REPORT_PREFIX):
                score += 3.0
        else:
            score -= 3.0
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
    if companies and plan_companies and not (plan_companies & set(companies)) and getattr(plan, "answer_type", "") in {
        "company_compare",
        "risk_analysis",
        "company_profile",
    }:
        score -= 20.0
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


def query_focus_terms(question: str) -> list[str]:
    normalized_question = normalize_topic(question)
    terms = [
        term
        for term in TECHNICAL_HIGH_SIGNAL_TERMS
        if normalize_topic(term) and normalize_topic(term) in normalized_question
    ]
    terms.extend(
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+.\-_/]{2,}", question)
        if normalize_topic(token) not in {"what", "why", "how"}
    )
    return dedupe_strings(terms)


def query_topics(question: str, plan_topics: Iterable[str], expanded_plan_topics: Iterable[str]) -> list[str]:
    del expanded_plan_topics
    plan_topic_list = list(plan_topics)
    topics = [topic for topic in plan_topic_list if topic in THEME_SYNONYMS]
    topics.extend(infer_themes(question))
    if "国产" in question and "算力" in question:
        topics.append("国产算力")
    if not topics and "算力" in question:
        if any(term in question for term in ("瓶颈", "约束", "制约", "卡点", "最大")):
            topics.extend(["AI芯片", "算力网络", "液冷"])
        else:
            topics.extend(["AI服务器", "AI芯片", "算力网络"])
    result = []
    seen = set()
    for topic in topics:
        if topic and topic not in seen:
            seen.add(topic)
            result.append(topic)
        if len(result) >= 3 and not plan_topic_list:
            break
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
    upper = value.upper()
    if any(term in upper for term in ("LEGAL NOTICE", "ALL RIGHTS RESERVED", "MERCHANTABILITY", "TRADEMARK")):
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
