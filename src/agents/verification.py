"""Deterministic evidence verification and conflict grouping for QA answers."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Iterable

from src.question_planner import extract_companies


EXPOSURE_LABELS = {
    "core": "核心敞口",
    "direct": "直接敞口",
    "indirect": "间接敞口",
    "mentioned": "仅提及",
}
EXPOSURE_BY_LABEL = {label: value for value, label in EXPOSURE_LABELS.items()}
EXPOSURE_BY_LABEL.update({"core": "core", "direct": "direct", "indirect": "indirect", "mentioned": "mentioned"})

METRIC_TERMS = (
    "订单",
    "合同负债",
    "产能",
    "毛利率",
    "ASP",
    "客户结构",
    "资本开支",
    "PUE",
    "功率密度",
    "端口速率",
    "渗透率",
    "营收",
    "收入",
    "利润",
    "市场份额",
    "客户导入",
)
RISK_TERMS = ("风险", "不确定", "波动", "不及预期", "反证", "替代", "压力", "集中度", "受限", "约束")
POSITIVE_TERMS = ("增长", "受益", "提升", "扩张", "加速", "高增", "景气", "兑现", "突破", "需求")
TECH_ROUTE_PAIRS = (
    ("CPO", "LPO"),
    ("InfiniBand", "Ethernet"),
    ("风冷", "液冷"),
)

NUMERIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:19|20)\d{2}(?![A-Za-z0-9])"
    r"|[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|亿元|万元|元|万只|万台|台|套|个|GB/s|Tb/s|GT/s|G|T|kW|MW|PUE|ms|us|ns|GPU hours|tokens|parameters)",
    flags=re.I,
)
CITATION_PATTERN = re.compile(r"\[(E\d+)\]")


def verify_answer_support(
    answer: str,
    plan: Any,
    evidence_cards: list[Any],
    raw_cards: list[Any],
    *,
    question: str = "",
) -> dict[str, Any]:
    """Verify that answer claims are supported by the selected evidence pack."""
    cards = list(evidence_cards or [])
    raw = list(raw_cards or [])
    all_cards = dedupe_cards([*cards, *raw])
    evidence_text = support_text(all_cards)
    citation_ids = sorted({card_text(card, "citation_id") for card in cards if card_text(card, "citation_id")})
    cited_ids = sorted(set(CITATION_PATTERN.findall(str(answer or ""))))
    missing_citations = sorted(set(cited_ids) - set(citation_ids))

    company_check = company_coverage_checks(plan, cards, raw)
    numeric_check = numeric_support_check(answer, evidence_text)
    metric_check = metric_support_check(answer, question or card_text(plan, "question"), plan, all_cards)
    exposure_check = exposure_support_check(answer, all_cards)
    risk_check = risk_support_check(answer, question or card_text(plan, "question"), plan, all_cards)
    temporal_check = temporal_consistency_check(answer, all_cards)
    unsupported_terms = unsupported_answer_terms(answer, evidence_text, numeric_check)
    conflict_groups = detect_conflict_groups(all_cards)

    citation_check = {
        "citation_ids": citation_ids,
        "cited_ids": cited_ids,
        "missing_citations": missing_citations,
        "status": "fail" if missing_citations else "pass",
    }
    checks = {
        "evidence_count": len(cards),
        "raw_evidence_count": len(raw),
        "citation_ids": citation_ids,
        "missing_citations": missing_citations,
        "citation_validity": citation_check,
        "company_coverage": company_check,
        "numeric_support": numeric_check,
        "metric_support": metric_check,
        "exposure_support": exposure_check,
        "risk_support": risk_check,
        "temporal_consistency": temporal_check,
        "risk_evidence_count": count_risk_cards(cards),
        "unsupported_terms": unsupported_terms,
    }
    gaps = evidence_gap_rows(checks, conflict_groups)
    status = verification_status(checks, gaps, conflict_groups, bool(cards))
    return {
        "status": status,
        "checks": checks,
        "evidence_gaps": gaps,
        "conflict_groups": conflict_groups,
    }


def build_evidence_limited_answer(
    plan: Any,
    evidence_cards: list[Any],
    verification: dict[str, Any],
    coverage_report: Any | None = None,
) -> str:
    """Build a conservative answer that only states cited evidence and explicit gaps."""
    cards = list(evidence_cards or [])
    gaps = list(verification.get("evidence_gaps") or [])
    if coverage_report is not None:
        for gap in getattr(coverage_report, "gaps", []) or []:
            row = gap.to_dict() if hasattr(gap, "to_dict") else dict(gap)
            reason = str(row.get("reason") or row.get("gap") or "").strip()
            if reason:
                gaps.append({"gap": reason, "priority": str(row.get("priority") or "高"), "suggested_source": ""})
    subject = subject_label(plan)
    if not cards:
        return (
            f"核心判断：当前知识库无法给出关于{subject}的可验证结论。\n\n"
            f"证据缺口：{format_gap_lines(gaps) or '当前问题没有召回可用证据卡片。'}"
        )

    lines = [
        f"核心判断：以下只保留关于{subject}的已验证证据，未被证据包支撑的判断不展开。",
        "",
        "证据支持：",
    ]
    for card in cards[:6]:
        citation = card_text(card, "citation_id") or "E?"
        text = short_text(card_text(card, "evidence") or card_text(card, "evidence_span"), 180)
        source = format_card_source(card)
        lines.append(f"- [{citation}] {text}" + (f"（{source}）" if source else ""))
    conflict_groups = verification.get("conflict_groups") or []
    if conflict_groups:
        lines.extend(["", "冲突/边界："])
        for group in conflict_groups[:3]:
            resolution = str(group.get("resolution") or "").strip()
            lines.append(f"- {resolution or '存在方向不一致的证据，需要保守解释。'}")
    if gaps:
        lines.extend(["", "证据缺口：", format_gap_lines(gaps)])
    return "\n".join(line for line in lines if line is not None)


def detect_conflict_groups(cards: list[Any]) -> list[dict[str, Any]]:
    candidates = dedupe_cards(list(cards or []))
    groups: list[dict[str, Any]] = []
    groups.extend(exposure_conflicts(candidates))
    groups.extend(direction_conflicts(candidates))
    groups.extend(route_conflicts(candidates))
    return dedupe_conflict_groups(groups)[:12]


def exposure_conflicts(cards: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for card in cards:
        if card_text(card, "claim_type") != "company_exposure":
            continue
        company = card_text(card, "company")
        topic = card_text(card, "topic")
        level = card_text(card, "exposure_level")
        if company and topic and level:
            grouped[(topic, company)][level].append(card)
    output = []
    for (topic, company), by_level in grouped.items():
        levels = [level for level in ("core", "direct", "indirect", "mentioned") if by_level.get(level)]
        if len(levels) < 2:
            continue
        output.append(
            conflict_row(
                "exposure_level_conflict",
                topic,
                company,
                by_level[levels[0]][0],
                by_level[levels[1]][0],
                f"{company} 在 {topic} 的敞口等级存在 {EXPOSURE_LABELS.get(levels[0], levels[0])} 与 {EXPOSURE_LABELS.get(levels[1], levels[1])} 的不一致，需以更高质量或更新证据复核。",
                "0.86",
            )
        )
    return output


def direction_conflicts(cards: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(lambda: {"positive": [], "negative": []})
    for card in cards:
        topic = card_text(card, "topic")
        company = card_text(card, "company")
        if not topic or not company:
            continue
        direction = card_direction(card)
        if direction:
            grouped[(topic, company)][direction].append(card)
    output = []
    for (topic, company), rows in grouped.items():
        positives = rows["positive"]
        negatives = rows["negative"]
        if not positives or not negatives:
            continue
        conflict_type = "old_vs_new" if year_distance(positives[0], negatives[0]) >= 2 else "optimistic_vs_risk"
        resolution = (
            f"{company} 在 {topic} 同时存在增长/受益证据和风险/约束证据；"
            "结论应保守表达为有产业机会但需要跟踪风险兑现。"
        )
        if conflict_type == "old_vs_new":
            resolution = (
                f"{company} 在 {topic} 的正负证据来自不同年份，不能把旧证据直接外推为当前事实；"
                "应优先核验更新报告。"
            )
        output.append(conflict_row(conflict_type, topic, company, positives[0], negatives[0], resolution, "0.78"))
    return output


def route_conflicts(cards: list[Any]) -> list[dict[str, Any]]:
    output = []
    by_topic: dict[str, list[Any]] = defaultdict(list)
    for card in cards:
        topic = card_text(card, "topic") or "技术路线"
        by_topic[topic].append(card)
    for topic, rows in by_topic.items():
        for left, right in TECH_ROUTE_PAIRS:
            left_cards = [card for card in rows if term_in_card(left, card)]
            right_cards = [card for card in rows if term_in_card(right, card)]
            if not left_cards or not right_cards:
                continue
            output.append(
                conflict_row(
                    "technical_route_conflict",
                    topic,
                    "",
                    left_cards[0],
                    right_cards[0],
                    f"{topic} 同时出现 {left} 与 {right} 路线证据，不能单边断言路线胜出。",
                    "0.72",
                )
            )
    return output


def conflict_row(
    conflict_type: str,
    topic: str,
    company: str,
    claim_a: Any,
    claim_b: Any,
    resolution: str,
    confidence: str,
) -> dict[str, Any]:
    seed = "|".join(
        [
            conflict_type,
            topic,
            company,
            card_text(claim_a, "claim_id") or card_text(claim_a, "citation_id"),
            card_text(claim_b, "claim_id") or card_text(claim_b, "citation_id"),
        ]
    )
    return {
        "conflict_group_id": "conflict_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12],
        "conflict_type": conflict_type,
        "topic": topic,
        "company": company,
        "claim_a": conflict_claim(claim_a),
        "claim_b": conflict_claim(claim_b),
        "resolution": resolution,
        "confidence": confidence,
    }


def conflict_claim(card: Any) -> dict[str, str]:
    return {
        "citation_id": card_text(card, "citation_id"),
        "claim_id": card_text(card, "claim_id"),
        "title": card_text(card, "title"),
        "evidence": short_text(card_text(card, "evidence") or card_text(card, "evidence_span"), 260),
        "source": card_text(card, "source"),
        "page": card_text(card, "page"),
        "as_of_date": card_text(card, "as_of_date"),
        "claim_type": card_text(card, "claim_type"),
        "exposure_level": card_text(card, "exposure_level"),
    }


def numeric_support_check(answer: str, evidence_text: str) -> dict[str, Any]:
    found = unique_texts([match.group(0).strip() for match in NUMERIC_PATTERN.finditer(str(answer or ""))])
    unsupported = [value for value in found if not text_supports(value, evidence_text)]
    return {
        "found": found,
        "unsupported": unsupported,
        "status": "fail" if unsupported else "pass",
    }


def metric_support_check(answer: str, question: str, plan: Any, cards: list[Any]) -> dict[str, Any]:
    text = f"{question} {answer}"
    requested = bool(getattr(plan, "needs_metrics", False)) or any(term in str(question or "") for term in METRIC_TERMS)
    metric_terms = [term for term in METRIC_TERMS if term in text]
    supported_cards = [
        card
        for card in cards
        if card_text(card, "claim_type") == "indicator"
        or card_text(card, "relation") in {"HAS_METRIC", "HAS_INDICATOR"}
        or (any(term in card_full_text(card) for term in metric_terms) and NUMERIC_PATTERN.search(card_full_text(card)))
    ]
    status = "pass"
    if requested and not supported_cards:
        status = "fail"
    return {
        "requested": requested,
        "terms": metric_terms,
        "supported_citation_ids": [card_text(card, "citation_id") for card in supported_cards if card_text(card, "citation_id")],
        "status": status,
    }


def exposure_support_check(answer: str, cards: list[Any]) -> dict[str, Any]:
    text = str(answer or "")
    mentioned_levels = [level for label, level in EXPOSURE_BY_LABEL.items() if label and label in text]
    exposure_cards = [card for card in cards if card_text(card, "claim_type") == "company_exposure" or card_text(card, "relation") == "HAS_EXPOSURE"]
    dossier_levels = dossier_exposure_levels(cards)
    mismatches: list[dict[str, str]] = []
    for company in extract_companies(text):
        window = company_context(text, company)
        expected_levels = [level for label, level in EXPOSURE_BY_LABEL.items() if label and label in window]
        if not expected_levels:
            continue
        card_levels = {
            card_text(card, "exposure_level")
            for card in exposure_cards
            if card_text(card, "company") == company and card_text(card, "exposure_level")
        }
        card_levels.update(dossier_levels.get(company, set()))
        if card_levels and not any(level in card_levels for level in expected_levels):
            mismatches.append(
                {
                    "company": company,
                    "answer_level": EXPOSURE_LABELS.get(expected_levels[0], expected_levels[0]),
                    "evidence_levels": "、".join(EXPOSURE_LABELS.get(level, level) for level in sorted(card_levels)),
                }
            )
        elif not card_levels:
            mismatches.append(
                {
                    "company": company,
                    "answer_level": EXPOSURE_LABELS.get(expected_levels[0], expected_levels[0]),
                    "evidence_levels": "",
                }
            )
    if mentioned_levels and not exposure_cards and not dossier_levels:
        mismatches.append({"company": "", "answer_level": "、".join(unique_texts(mentioned_levels)), "evidence_levels": ""})
    return {
        "mentioned_levels": unique_texts(mentioned_levels),
        "supported_citation_ids": [
            card_text(card, "citation_id")
            for card in cards
            if card_text(card, "citation_id")
            and (card in exposure_cards or (card_text(card, "kind") == "dossier" and "公司敞口：" in card_text(card, "evidence")))
        ],
        "mismatches": mismatches,
        "status": "fail" if mismatches else "pass",
    }


def dossier_exposure_levels(cards: list[Any]) -> dict[str, set[str]]:
    levels_by_company: dict[str, set[str]] = defaultdict(set)
    for card in cards:
        if card_text(card, "kind") != "dossier":
            continue
        for line in card_text(card, "evidence").splitlines():
            if not line.startswith("公司敞口："):
                continue
            for part in line.split("：", 1)[-1].split("；"):
                if ":" not in part:
                    continue
                level, names = part.split(":", 1)
                level = level.strip()
                if level not in EXPOSURE_LABELS:
                    continue
                for name in names.split("、"):
                    name = name.strip()
                    if name:
                        levels_by_company[name].add(level)
    return levels_by_company


def risk_support_check(answer: str, question: str, plan: Any, cards: list[Any]) -> dict[str, Any]:
    text = f"{question} {answer}"
    requested = bool(getattr(plan, "needs_risk", False)) or getattr(plan, "answer_type", "") == "risk_analysis" or "风险" in text
    risk_cards = [card for card in cards if is_risk_card(card)]
    return {
        "requested": requested,
        "supported_citation_ids": [card_text(card, "citation_id") for card in risk_cards if card_text(card, "citation_id")],
        "status": "fail" if requested and not risk_cards else "pass",
    }


def temporal_consistency_check(answer: str, cards: list[Any]) -> dict[str, Any]:
    evidence_years = sorted({int(year) for card in cards for year in re.findall(r"(?:19|20)\d{2}", card_text(card, "as_of_date") or card_text(card, "source"))})
    answer_years = [int(year) for year in re.findall(r"(?:19|20)\d{2}", str(answer or ""))]
    mixed = len(evidence_years) >= 2 and (max(evidence_years) - min(evidence_years) >= 2)
    current_language = any(term in str(answer or "") for term in ("当前", "目前", "现在", "最新"))
    status = "warn" if mixed and current_language else "pass"
    return {
        "answer_years": answer_years,
        "evidence_years": evidence_years,
        "status": status,
    }


def company_coverage_checks(plan: Any, evidence_cards: list[Any], raw_cards: list[Any]) -> dict[str, Any]:
    required = list(getattr(plan, "companies", []) or [])
    if not required:
        return {"required": [], "covered": [], "missing": [], "status": "pass"}
    cards = evidence_cards or raw_cards
    covered = sorted({card_text(card, "company") for card in cards if card_text(card, "company") in set(required)})
    missing = [company for company in required if company not in covered]
    return {"required": required, "covered": covered, "missing": missing, "status": "fail" if missing else "pass"}


def unsupported_answer_terms(answer: str, evidence_text: str, numeric_check: dict[str, Any]) -> list[str]:
    unsupported: list[str] = []
    for company in extract_companies(str(answer or "")):
        if not text_supports(company, evidence_text):
            unsupported.append(company)
    unsupported.extend(str(item) for item in numeric_check.get("unsupported", []))
    return unique_texts(unsupported)[:20]


def evidence_gap_rows(checks: dict[str, Any], conflict_groups: list[dict[str, Any]]) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    if int(checks.get("evidence_count") or 0) == 0:
        gaps.append({"gap": "当前问题没有召回可用证据卡片。", "priority": "高", "suggested_source": "补充年报、研报原文或行业白皮书后重建 Claim/RAG 索引。"})
    if checks["citation_validity"]["missing_citations"]:
        gaps.append({"gap": "答案引用了不存在的证据编号：" + "、".join(checks["citation_validity"]["missing_citations"]), "priority": "高", "suggested_source": "重新生成答案或检查 evidence pack 编号。"})
    if checks["company_coverage"]["missing"]:
        gaps.append({"gap": "缺少公司证据：" + "、".join(checks["company_coverage"]["missing"]), "priority": "高", "suggested_source": "补充对应公司年报、公告、研报或投资者关系记录。"})
    if checks["numeric_support"]["unsupported"]:
        gaps.append({"gap": "答案中存在无来源年份或数值：" + "、".join(checks["numeric_support"]["unsupported"]), "priority": "高", "suggested_source": "回到原文页码、财务表或指标表核验后再引用。"})
    if checks["metric_support"]["status"] == "fail":
        gaps.append({"gap": "缺少订单、收入、毛利率、产能、客户导入或渗透率等指标证据。", "priority": "高", "suggested_source": "补充年报财务表、经营数据、公告和研报指标表。"})
    if checks["exposure_support"]["mismatches"]:
        gaps.append({"gap": "答案中的公司敞口等级缺少匹配证据或与证据不一致。", "priority": "高", "suggested_source": "补充或人工复核 company_exposure Claim。"})
    if checks["risk_support"]["status"] == "fail":
        gaps.append({"gap": "缺少明确风险、反证或不确定性证据。", "priority": "高", "suggested_source": "补充年报风险披露、行业竞争格局和技术路线替代证据。"})
    if conflict_groups:
        gaps.append({"gap": f"识别到 {len(conflict_groups)} 组冲突或边界证据，结论需保守解释。", "priority": "中", "suggested_source": "在任务详情中审阅冲突证据双方并人工确认。"})
    return dedupe_gap_rows(gaps)[:12]


def verification_status(checks: dict[str, Any], gaps: list[dict[str, str]], conflict_groups: list[dict[str, Any]], has_cards: bool) -> str:
    if not has_cards:
        return "fail"
    hard_checks = ("citation_validity", "company_coverage", "numeric_support", "metric_support", "exposure_support", "risk_support")
    if any(checks[name].get("status") == "fail" for name in hard_checks):
        return "fail"
    if checks["temporal_consistency"].get("status") == "warn" or conflict_groups or gaps:
        return "warn"
    return "pass"


def count_risk_cards(cards: list[Any]) -> int:
    return sum(1 for card in cards if is_risk_card(card))


def is_risk_card(card: Any) -> bool:
    return card_text(card, "claim_type") in {"risk", "bottleneck"} or card_text(card, "relation") == "DISCLOSES_RISK" or any(term in card_full_text(card) for term in RISK_TERMS)


def card_direction(card: Any) -> str:
    claim_type = card_text(card, "claim_type")
    text = card_full_text(card)
    if claim_type in {"risk", "bottleneck"} or any(term in text for term in RISK_TERMS):
        return "negative"
    if claim_type in {"company_exposure", "mechanism", "supply_chain", "trend", "indicator", "policy"} or any(term in text for term in POSITIVE_TERMS):
        return "positive"
    return ""


def year_distance(card_a: Any, card_b: Any) -> int:
    years_a = [int(year) for year in re.findall(r"(?:19|20)\d{2}", card_text(card_a, "as_of_date") or card_text(card_a, "source"))]
    years_b = [int(year) for year in re.findall(r"(?:19|20)\d{2}", card_text(card_b, "as_of_date") or card_text(card_b, "source"))]
    if not years_a or not years_b:
        return 0
    return abs(max(years_a) - max(years_b))


def term_in_card(term: str, card: Any) -> bool:
    return normalize_for_support(term) in normalize_for_support(card_full_text(card))


def text_supports(term: str, evidence_text: str) -> bool:
    normalized = normalize_for_support(term)
    if not normalized:
        return True
    return normalized in evidence_text


def support_text(cards: Iterable[Any]) -> str:
    return normalize_for_support(" ".join(card_full_text(card) for card in cards))


def card_full_text(card: Any) -> str:
    keys = ("title", "evidence", "evidence_span", "source", "section", "company", "target", "topic", "as_of_date", "claim_type", "exposure_level")
    return " ".join(card_text(card, key) for key in keys)


def card_text(card: Any, name: str) -> str:
    if isinstance(card, dict):
        value = card.get(name, "")
    else:
        value = getattr(card, name, "")
    if value is None:
        return ""
    return str(value)


def company_context(text: str, company: str, *, width: int = 36) -> str:
    index = text.find(company)
    if index < 0:
        return text
    return text[max(0, index - width) : index + len(company) + width]


def subject_label(plan: Any) -> str:
    companies = list(getattr(plan, "companies", []) or [])
    topics = list(getattr(plan, "topics", []) or [])
    if companies and topics:
        return "、".join(companies[:2]) + " " + "、".join(topics[:2])
    if companies:
        return "、".join(companies[:3])
    if topics:
        return "、".join(topics[:3])
    return "该问题"


def format_gap_lines(gaps: list[Any]) -> str:
    if not gaps:
        return ""
    lines = []
    for gap in gaps[:8]:
        if isinstance(gap, dict):
            text = str(gap.get("gap") or gap.get("reason") or "").strip()
            if text.startswith("答案中存在无来源年份或数值："):
                text = "答案中存在无来源年份或数值，已移除该判断。"
            elif text.startswith("答案中存在证据包未直接支撑"):
                text = "答案中存在证据包未直接支撑的表述，已移除该判断。"
            source = str(gap.get("suggested_source") or "").strip()
            lines.append(f"- {text}" + (f" 建议：{source}" if source else ""))
    return "\n".join(line for line in lines if line.strip())


def format_card_source(card: Any) -> str:
    source = card_text(card, "source")
    page = card_text(card, "page")
    return f"{source} p.{page}" if source and page else source


def normalize_for_support(value: str) -> str:
    text = str(value or "").casefold()
    text = text.replace("，", ",")
    return re.sub(r"\s+", "", text)


def short_text(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def dedupe_cards(cards: list[Any]) -> list[Any]:
    output = []
    seen = set()
    for card in cards:
        key = (
            card_text(card, "kind"),
            card_text(card, "citation_id"),
            card_text(card, "claim_id"),
            normalize_for_support(card_text(card, "evidence") or card_text(card, "evidence_span"))[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(card)
    return output


def dedupe_conflict_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for group in groups:
        key = str(group.get("conflict_group_id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(group)
    return output


def dedupe_gap_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    seen = set()
    for row in rows:
        key = normalize_for_support(row.get("gap", ""))[:160]
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def unique_texts(values: Iterable[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output
