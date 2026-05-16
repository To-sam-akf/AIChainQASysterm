#!/usr/bin/env python3
"""Run a small professional QA regression set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.qa_engine import QAEngine


EVAL_CASES = [
    {
        "question": "哪些公司涉及AI服务器？",
        "must_include": ["浪潮信息", "中科曙光"],
        "must_exclude": ["Amazon", "Meta", "AMD"],
    },
    {
        "question": "液冷产业链有哪些上市公司，各自处于什么环节？",
        "must_include": ["英维克", "申菱环境", "高澜股份"],
        "must_exclude": ["阿里巴巴", "中国移动"],
    },
    {
        "question": "中际旭创和新易盛在光模块业务上的差异是什么？",
        "must_include": ["中际旭创", "新易盛", "光模块"],
        "must_exclude": [],
    },
    {
        "question": "英维克液冷业务进展和主要风险是什么？",
        "must_include": ["英维克", "风险"],
        "must_exclude": ["长期股权投资"],
    },
    {
        "question": "AI算力产业链当前最大的瓶颈是什么？",
        "must_include": ["算力", "瓶颈"],
        "must_exclude": [],
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate professional QA quality with smoke assertions.")
    parser.add_argument("--json", action="store_true", help="Print full JSON results.")
    parser.add_argument("--use-llm", action="store_true", help="Use configured LLM instead of deterministic fallback answers.")
    return parser


def score_answer(answer: str, must_include: list[str], must_exclude: list[str]) -> tuple[int, list[str]]:
    failures = []
    for term in must_include:
        if term not in answer:
            failures.append(f"missing:{term}")
    for term in must_exclude:
        if term and term in answer:
            failures.append(f"unexpected:{term}")
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
            score, failures = score_answer(answer, case["must_include"], case["must_exclude"])
            results.append(
                {
                    "question": case["question"],
                    "score": score,
                    "failures": failures,
                    "answer_type": result.get("answer_type", ""),
                    "graph_records": len(result.get("graph_records", [])),
                    "rag_hits": len(result.get("rag_hits", [])),
                    "evidence_cards": len(result.get("evidence_cards", [])),
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
