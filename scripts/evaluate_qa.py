#!/usr/bin/env python3
"""Run a small professional QA regression set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from aika.qa_engine import QAEngine


EVAL_CASES = [
    {
        "question": "哪些公司涉及AI服务器？",
        "must_include": ["浪潮信息", "中科曙光"],
        "must_exclude": ["Amazon", "Meta", "AMD"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "液冷产业链有哪些上市公司，各自处于什么环节？",
        "must_include": ["英维克", "申菱环境", "高澜股份"],
        "must_exclude": ["阿里巴巴", "中国移动"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "中际旭创和新易盛在光模块业务上的差异是什么？",
        "must_include": ["中际旭创", "新易盛", "光模块"],
        "must_exclude": ["华工科技"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "英维克液冷业务进展和主要风险是什么？",
        "must_include": ["英维克", "风险"],
        "must_exclude": ["长期股权投资"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "AI算力产业链当前最大的瓶颈是什么？",
        "must_include": ["算力", "瓶颈"],
        "must_exclude": [],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "Ultra Ethernet 对算力网络有什么意义？",
        "must_include": ["Ultra Ethernet", "算力网络"],
        "must_exclude": ["商誉减值"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "DeepSeek-V3 对训练算力瓶颈有什么启示？",
        "must_include": ["DeepSeek", "瓶颈"],
        "must_exclude": ["买入", "目标价"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "UCIe/Chiplet 对国产算力产业链的传导是什么？",
        "must_include": ["UCIe", "Chiplet"],
        "must_exclude": ["目标价"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "CPO/LPO/硅光对光模块产业链有什么影响？",
        "must_include": ["光模块"],
        "must_exclude": ["长期股权投资"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "寒武纪在AI芯片业务上的主要风险是什么？",
        "must_include": ["寒武纪", "风险"],
        "must_exclude": ["Amazon"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "光模块产业链有哪些核心上市公司？",
        "must_include": ["中际旭创", "新易盛"],
        "must_exclude": ["Meta"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "国产算力产业链的主要传导路径是什么？",
        "must_include": ["国产算力"],
        "must_exclude": ["目标价"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "AI服务器需求如何传导到液冷？",
        "must_include": ["AI服务器", "液冷"],
        "must_exclude": ["买卖建议"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "PCB/CCL 在AI算力产业链中处于什么位置？",
        "must_include": ["PCB"],
        "must_exclude": ["目标价"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
    {
        "question": "哪些指标可以跟踪液冷产业链兑现？",
        "must_include": ["液冷"],
        "must_exclude": ["当前知识库中未找到"],
        "evidence_min": 1,
        "subgraph_min": 1,
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate professional QA quality with smoke assertions.")
    parser.add_argument("--json", action="store_true", help="Print full JSON results.")
    parser.add_argument("--use-llm", action="store_true", help="Use configured LLM instead of deterministic fallback answers.")
    return parser


def score_answer(result: dict, case: dict) -> tuple[int, list[str]]:
    answer = result.get("answer", "")
    must_include = case.get("must_include", [])
    must_exclude = case.get("must_exclude", [])
    failures = []
    for term in must_include:
        if term not in answer:
            failures.append(f"missing:{term}")
    for term in must_exclude:
        if term and term in answer:
            failures.append(f"unexpected:{term}")
    evidence_cards = result.get("evidence_cards", [])
    subgraph = result.get("subgraph", [])
    if len(evidence_cards) < case.get("evidence_min", 0):
        failures.append(f"evidence_lt:{case.get('evidence_min', 0)}")
    if len(subgraph) < case.get("subgraph_min", 0):
        failures.append(f"subgraph_lt:{case.get('subgraph_min', 0)}")
    if evidence_cards and not all(card.get("citation_id") for card in evidence_cards):
        failures.append("missing_citation_id")
    if not failures:
        return 2, failures
    if len(failures) < len(must_include) + len(must_exclude):
        return 1, failures
    return 0, failures


def main() -> int:
    args = build_parser().parse_args()
    engine = QAEngine.from_env()
    if not args.use_llm:
        engine.llm_client = None
        engine.status.llm_enabled = False
    results = []
    try:
        for case in EVAL_CASES:
            result = engine.answer_question(case["question"])
            answer = result["answer"]
            score, failures = score_answer(result, case)
            results.append(
                {
                    "question": case["question"],
                    "score": score,
                    "failures": failures,
                    "answer_type": result.get("answer_type", ""),
                    "graph_records": len(result.get("graph_records", [])),
                    "rag_hits": len(result.get("rag_hits", [])),
                    "evidence_cards": len(result.get("evidence_cards", [])),
                    "subgraph": len(result.get("subgraph", [])),
                    "unsupported_terms": result.get("diagnostics", {}).get("unsupported_terms", []),
                    "answer": answer,
                }
            )
    finally:
        engine.close()

    total = sum(item["score"] for item in results)
    max_score = 2 * len(results)
    accuracy = total / max_score if max_score else 0
    summary = {"cases": len(results), "score": total, "max_score": max_score, "accuracy": round(accuracy, 4)}
    if args.json:
        print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
    else:
        print(f"QA evaluation: {total}/{max_score} accuracy={accuracy:.1%}")
        for item in results:
            status = "PASS" if item["score"] == 2 else "PARTIAL" if item["score"] == 1 else "FAIL"
            print(f"[{status}] {item['question']} | score={item['score']} | failures={','.join(item['failures']) or '-'}")
    return 0 if total >= max_score * 0.6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
