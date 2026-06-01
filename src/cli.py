"""Command line interface for local demos, services, agent tasks, and evals."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.agents.research_agent import ResearchAgent, SUPPORTED_TASK_TYPES
from src.agents.store import AgentTaskStore
from src.qa_engine import QAEngine


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_TASK_DIR = ROOT_DIR / "data" / "agent_tasks"
DEFAULT_DEMO_QUESTION = "液冷产业链有哪些上市公司，各自处于什么环节？"
DEFAULT_DEMO_GOAL = "液冷产业链"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiqasys",
        description="AIQASYS engineering delivery CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    api_parser = subparsers.add_parser("api", help="Start the FastAPI backend.")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)
    api_parser.add_argument("--reload", action="store_true")
    api_parser.set_defaults(func=run_api)

    web_parser = subparsers.add_parser("web", help="Start the Vite frontend.")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=5173)
    web_parser.add_argument("--api", default="http://127.0.0.1:8000")
    web_parser.set_defaults(func=run_web)

    agent_parser = subparsers.add_parser("agent", help="Run one ResearchAgent task.")
    agent_parser.add_argument("goal", help="Research goal, for example: 液冷产业链")
    agent_parser.add_argument("--type", default="research_brief", choices=sorted(SUPPORTED_TASK_TYPES))
    agent_parser.add_argument("--task-dir", default=str(DEFAULT_AGENT_TASK_DIR))
    agent_parser.add_argument("--offline", action="store_true", help="Force local CSV/RAG mode and disable LLM/embedding.")
    agent_parser.add_argument("--use-llm", action="store_true", help="Use the configured LLM client.")
    agent_parser.add_argument("--use-embedding", action="store_true", help="Use the configured embedding index.")
    agent_parser.add_argument("--json", action="store_true", help="Print the full task JSON.")
    agent_parser.set_defaults(func=run_agent)

    eval_parser = subparsers.add_parser("eval", help="Run deterministic QA or Agent evaluations.")
    eval_parser.add_argument("--suite", choices=["qa", "agent"], default="qa")
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--task-dir", default=str(DEFAULT_AGENT_TASK_DIR))
    eval_parser.add_argument("--offline", action="store_true", help="Force local CSV/RAG mode and disable LLM/embedding.")
    eval_parser.add_argument("--use-llm", action="store_true", help="Use the configured LLM client.")
    eval_parser.add_argument("--use-embedding", action="store_true", help="Use the configured embedding index.")
    eval_parser.add_argument("--json", action="store_true", help="Print JSON results.")
    eval_parser.set_defaults(func=run_eval)

    demo_parser = subparsers.add_parser("demo", help="Run a minimal local QA + Agent demo.")
    demo_parser.add_argument("--offline", action="store_true", help="Force local CSV/RAG mode and disable LLM/embedding.")
    demo_parser.add_argument("--task-dir", default=str(DEFAULT_AGENT_TASK_DIR))
    demo_parser.add_argument("--use-llm", action="store_true", help="Use the configured LLM client.")
    demo_parser.add_argument("--use-embedding", action="store_true", help="Use the configured embedding index.")
    demo_parser.set_defaults(func=run_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def run_api(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def run_web(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env["VITE_API_PROXY_TARGET"] = args.api
    command = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    completed = subprocess.run(command, cwd=ROOT_DIR / "web", env=env, check=False)
    return completed.returncode


def run_agent(args: argparse.Namespace) -> int:
    engine = build_engine(args)
    try:
        store = AgentTaskStore(Path(args.task_dir))
        task = ResearchAgent(engine, store).run(
            task_type=args.type,
            goal=args.goal,
            thinking_enabled=False if effective_offline(args) else None,
        )
    finally:
        engine.close()

    if args.json:
        print(json.dumps(task, ensure_ascii=False, indent=2))
    else:
        final_outputs = task.get("final_outputs", {})
        print(f"Agent task completed: {task.get('task_id', '')}")
        print(f"Type: {task.get('task_type', '')}")
        print(f"Status: {task.get('status', '')}")
        print(f"Title: {final_outputs.get('report_title', '')}")
        print(f"Evidence cards: {final_outputs.get('evidence_card_count', 0)}")
        print(f"Evidence gaps: {final_outputs.get('evidence_gap_count', 0)}")
    return 0 if task.get("status") == "completed" else 1


def run_eval(args: argparse.Namespace) -> int:
    if args.suite == "agent":
        return run_agent_eval(args)
    return run_qa_eval(args)


def run_qa_eval(args: argparse.Namespace) -> int:
    from scripts.evaluate_qa import EVAL_CASES, score_answer

    engine = build_engine(args)
    results: list[dict[str, Any]] = []
    try:
        cases = EVAL_CASES[: args.limit] if args.limit and args.limit > 0 else EVAL_CASES
        for case in cases:
            result = engine.answer_question(case["question"])
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
                }
            )
    finally:
        engine.close()
    return print_eval_results("QA evaluation", results, args.json)


def run_agent_eval(args: argparse.Namespace) -> int:
    from scripts.evaluate_agent_tasks import AGENT_EVAL_CASES, score_task

    engine = build_engine(args)
    store = AgentTaskStore(Path(args.task_dir))
    agent = ResearchAgent(engine, store)
    results: list[dict[str, Any]] = []
    try:
        cases = AGENT_EVAL_CASES[: args.limit] if args.limit and args.limit > 0 else AGENT_EVAL_CASES
        for task_type, goal in cases:
            task = agent.run(task_type=task_type, goal=goal, thinking_enabled=False if effective_offline(args) else None)
            score, failures = score_task(task)
            results.append(
                {
                    "task_type": task_type,
                    "goal": goal,
                    "task_id": task.get("task_id", ""),
                    "status": task.get("status", ""),
                    "score": score,
                    "failures": failures,
                    "evidence_cards": len(task.get("evidence_cards", [])),
                    "evidence_gaps": task.get("final_outputs", {}).get("evidence_gap_count", 0),
                    "report_title": task.get("final_outputs", {}).get("report_title", ""),
                }
            )
    finally:
        engine.close()
    return print_eval_results("Agent task evaluation", results, args.json)


def run_demo(args: argparse.Namespace) -> int:
    engine = build_engine(args)
    task: dict[str, Any]
    try:
        status = engine.status
        qa_result = engine.answer_question(DEFAULT_DEMO_QUESTION, thinking_enabled=False if effective_offline(args) else None)
        store = AgentTaskStore(Path(args.task_dir))
        task = ResearchAgent(engine, store).run(
            task_type="research_brief",
            goal=DEFAULT_DEMO_GOAL,
            thinking_enabled=False if effective_offline(args) else None,
        )
    finally:
        engine.close()

    print("AIQASYS minimal demo")
    print(f"Graph backend: {status.graph_backend}")
    print(f"Research enabled: {status.research_enabled}")
    print(f"RAG enabled: {status.rag_enabled}")
    print(f"LLM enabled: {status.llm_enabled}")
    print("")
    print(f"Question: {DEFAULT_DEMO_QUESTION}")
    print(f"Answer type: {qa_result.get('answer_type', '')}")
    print(f"Evidence cards: {len(qa_result.get('evidence_cards', []))}")
    print(first_lines(str(qa_result.get("answer", "")), limit=8))
    print("")
    print(f"Agent task id: {task.get('task_id', '')}")
    print(f"Agent status: {task.get('status', '')}")
    print(f"Agent title: {task.get('final_outputs', {}).get('report_title', '')}")
    return 0 if task.get("status") == "completed" and qa_result.get("answer") else 1


def build_engine(args: argparse.Namespace) -> QAEngine:
    offline = effective_offline(args)
    if offline:
        os.environ["QA_GRAPH_BACKEND"] = "csv"
    if offline or not getattr(args, "use_embedding", False):
        os.environ["EMBEDDING_MODEL"] = ""

    engine = QAEngine.from_env()
    if offline or not getattr(args, "use_llm", False):
        engine.llm_client = None
        engine.status.llm_enabled = False
    if offline or not getattr(args, "use_embedding", False):
        engine.semantic_index = None
        engine.status.embedding_enabled = False
    if offline:
        engine.graph_client = None
        engine.status.neo4j_enabled = False
        engine.status.graph_backend = "csv" if engine.csv_graph is not None else "none"
    return engine


def effective_offline(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "offline", False)
        or (not getattr(args, "use_llm", False) and not getattr(args, "use_embedding", False))
    )


def print_eval_results(title: str, results: list[dict[str, Any]], as_json: bool) -> int:
    total = sum(int(item.get("score", 0)) for item in results)
    max_score = 2 * len(results)
    accuracy = total / max_score if max_score else 0.0
    summary = {"cases": len(results), "score": total, "max_score": max_score, "accuracy": round(accuracy, 4)}
    if as_json:
        print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
    else:
        print(f"{title}: {total}/{max_score} accuracy={accuracy:.1%}")
        for item in results:
            status = "PASS" if item.get("score") == 2 else "PARTIAL" if item.get("score") == 1 else "FAIL"
            label = item.get("question") or f"{item.get('task_type', '')} | {item.get('goal', '')}"
            failures = ",".join(item.get("failures", [])) or "-"
            print(f"[{status}] {label} | failures={failures}")
    return 0 if total >= max_score * 0.6 else 1


def first_lines(text: str, *, limit: int) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[:limit])
