"""Recognize numeric disclosures in raw text, without requiring extracted graph labels."""

from __future__ import annotations

import re


METRIC_GROUPS = (
    ("研发费用",), ("研发投入", "研发支出"), ("研发投入资本化", "资本化研发"),
    ("营业收入", "营收", "收入"), ("毛利率",), ("净利润",),
    ("订单",), ("产能",), ("市场份额",), ("渗透率",),
    ("出货量",), ("PUE",), ("功耗",), ("带宽",),
)
NUMBER = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?(?![\d.])")
UNIT = re.compile(r"%|％|亿元|万元|元|万台|台|万套|套|万吨|吨|MW|GW|kW|Gbps|Tbps", re.I)


def has_numeric_disclosure(text: str, question: str = "") -> bool:
    """Require a relevant metric label and an actual value, not headings or report years.

    This is a retrieval coverage signal, not a judgment that all requested values
    are answered or that their accounting scope is correct.
    """
    groups = [group for group in METRIC_GROUPS if any(term.casefold() in question.casefold() for term in group)]
    terms = [term for group in groups or METRIC_GROUPS for term in group]
    for line in text.splitlines():
        compact = re.sub(r"\s+", "", line)
        if not any(term.casefold() in compact.casefold() for term in terms):
            continue
        for number in NUMBER.finditer(compact):
            value = number.group().replace(",", "")
            # A year in a heading is not a disclosed metric value.
            if value.isdigit() and 1900 <= int(value) <= 2100 and "," not in number.group():
                continue
            if "|" in line or UNIT.search(compact[number.end():number.end() + 8]):
                return True
    return False
