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
DEFAULT_EVAL_RUN_DIR = ROOT_DIR / "data" / "eval_runs"
DEFAULT_QA_BENCHMARK = ROOT_DIR / "data" / "eval" / "qa_benchmark_v1.jsonl"
DEFAULT_RAG_BENCHMARK = ROOT_DIR / "data" / "eval" / "rag_retrieval_v1.jsonl"
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
    agent_parser.add_argument("--offline", action="store_true", help="Use CSV graph and disable LLM/semantic retrieval.")
    agent_parser.add_argument("--use-llm", action="store_true", help="Use the configured LLM client.")
    agent_parser.add_argument("--use-embedding", action="store_true", help="Use the configured embedding index.")
    agent_parser.add_argument("--json", action="store_true", help="Print the full task JSON.")
    agent_parser.set_defaults(func=run_agent)

    eval_parser = subparsers.add_parser("eval", help="Run QA, RAG retrieval, or Agent evaluations.")
    eval_parser.add_argument("--suite", choices=["qa", "rag", "agent"], default="qa")
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--task-dir", default=str(DEFAULT_AGENT_TASK_DIR))
    eval_parser.add_argument("--benchmark", default="", help="Benchmark JSONL file; defaults depend on --suite.")
    eval_parser.add_argument("--report-dir", default=str(DEFAULT_EVAL_RUN_DIR), help="Directory for eval run JSONL reports.")
    eval_parser.add_argument("--k", type=int, default=6, help="Top-k evidence cards used by retrieval metrics.")
    eval_parser.add_argument("--ks", default="1,3,6,12", help="Comma-separated K values for RAG retrieval metrics.")
    eval_parser.add_argument("--candidate-k", type=int, default=30, help="RAG candidate pool and review depth.")
    eval_parser.add_argument(
        "--retrievers",
        default="auto",
        help="RAG retrievers: auto, bm25, semantic, rrf, or a comma-separated combination.",
    )
    eval_parser.add_argument("--no-save", action="store_true", help="Do not append the eval report to the local report store.")
    eval_parser.add_argument("--offline", action="store_true", help="Use CSV graph and disable LLM/semantic retrieval.")
    eval_parser.add_argument("--use-llm", action="store_true", help="Use the configured LLM client.")
    eval_parser.add_argument("--use-embedding", action="store_true", help="Use the configured embedding index.")
    eval_parser.add_argument("--json", action="store_true", help="Print JSON results.")
    eval_parser.set_defaults(func=run_eval)

    demo_parser = subparsers.add_parser("demo", help="Run a minimal local QA + Agent demo.")
    demo_parser.add_argument("--offline", action="store_true", help="Use CSV graph and disable LLM/semantic retrieval.")
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
    if args.suite == "rag":
        return run_rag_eval(args)
    return run_qa_eval(args)


def run_qa_eval(args: argparse.Namespace) -> int:
    from src.eval.runner import run_qa_benchmark
    from src.eval.store import EvalRunStore

    engine = build_engine(args)
    try:
        report = run_qa_benchmark(
            engine,
            benchmark_path=Path(args.benchmark or DEFAULT_QA_BENCHMARK),
            limit=args.limit,
            k=args.k,
            store=EvalRunStore(Path(args.report_dir)),
            save=not args.no_save,
        )
    finally:
        engine.close()
    return print_qa_eval_report(report, args.json)


def run_rag_eval(args: argparse.Namespace) -> int:
    from src.embedding_client import OpenAICompatibleEmbeddingClient
    from src.eval.rag_runner import RagRetrievalEvalError, run_rag_retrieval_benchmark
    from src.eval.store import EvalRunStore
    from src.postgres_retrieval import (
        EMBEDDING_DIMENSIONS,
        PostgresRagIndex,
        PostgresRetrievalStore,
        PostgresSemanticIndex,
    )

    store = None
    try:
        retrievers = parse_rag_retrievers(args.retrievers, use_embedding=bool(args.use_embedding))
        ks = parse_positive_ints(args.ks)
        semantic_requested = bool({"semantic", "rrf"} & set(retrievers))
        if semantic_requested and not args.use_embedding:
            print("semantic/rrf retrieval requires --use-embedding", file=sys.stderr)
            return 2
        store = PostgresRetrievalStore.from_env()
        store.ensure_ready()
        rag_index = PostgresRagIndex(store)
        semantic_index = None
        if semantic_requested:
            embedding_client = OpenAICompatibleEmbeddingClient(dimensions=EMBEDDING_DIMENSIONS)
            semantic_index = PostgresSemanticIndex(
                store,
                embedding_client=embedding_client,
            )
        report_dir = Path(args.report_dir)
        report = run_rag_retrieval_benchmark(
            rag_index,
            benchmark_path=Path(args.benchmark or DEFAULT_RAG_BENCHMARK),
            semantic_index=semantic_index,
            retrievers=retrievers,
            ks=ks,
            candidate_k=args.candidate_k,
            limit=args.limit,
            store=EvalRunStore(report_dir),
            save=not args.no_save,
            review_dir=report_dir if not args.no_save else None,
        )
    except (OSError, RuntimeError, ValueError, RagRetrievalEvalError) as exc:
        print(f"RAG retrieval evaluation failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if store is not None:
            store.close()
    return print_rag_eval_report(report, args.json)


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
        os.environ["QA_DISABLE_POSTGRES"] = "true"
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


def print_qa_eval_report(report: dict[str, Any], as_json: bool) -> int:
    summary = report.get("summary", {})
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
        print(
            "QA benchmark evaluation: "
            f"run_id={report.get('run_id', '')} "
            f"cases={summary.get('cases', 0)} "
            f"score={float(summary.get('overall_score') or 0.0):.1%} "
            f"pass_rate={float(summary.get('pass_rate') or 0.0):.1%}"
        )
        print(
            "Metrics: "
            f"claim_recall@k={float(metrics.get('claim_recall@k') or 0.0):.1%}, "
            f"evidence_precision@k={float(metrics.get('evidence_precision@k') or 0.0):.1%}, "
            f"citation_validity={float(metrics.get('citation_validity') or 0.0):.1%}, "
            f"groundedness={float(metrics.get('answer_groundedness') or 0.0):.1%}, "
            f"unsupported_rate={float(metrics.get('unsupported_claim_rate') or 0.0):.1%}"
        )
        print(f"Dataset: {report.get('dataset', {}).get('name', '')} hash={report.get('dataset', {}).get('hash', '')}")
        failures = report.get("failed_examples") if isinstance(report.get("failed_examples"), list) else []
        for item in failures[:8]:
            failure_text = ",".join(str(failure) for failure in item.get("failures", [])) or "-"
            print(f"[{item.get('category', '')}] {item.get('case_id', '')} score={item.get('score', 0)} failures={failure_text}")
    return 0 if float(summary.get("overall_score") or 0.0) >= 0.35 else 1


def print_rag_eval_report(report: dict[str, Any], as_json: bool) -> int:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    summary = report.get("summary", {})
    primary_k = int(summary.get("primary_k") or 6)
    print(
        "RAG retrieval evaluation: "
        f"run_id={report.get('run_id', '')} "
        f"cases={summary.get('cases', 0)} "
        f"primary_k={primary_k}"
    )
    for retriever, row in summary.get("by_retriever", {}).items():
        metrics = row.get("primary", {})
        print(
            f"[{retriever}] "
            f"recall@{primary_k}={float(metrics.get('recall') or 0.0):.1%} "
            f"precision@{primary_k}={float(metrics.get('precision') or 0.0):.1%} "
            f"hit_rate@{primary_k}={float(metrics.get('hit_rate') or 0.0):.1%} "
            f"mrr@{primary_k}={float(metrics.get('mrr') or 0.0):.3f} "
            f"ndcg@{primary_k}={float(metrics.get('ndcg') or 0.0):.3f} "
            f"duplicate@{primary_k}={float(metrics.get('duplicate_rate') or 0.0):.1%} "
            f"unjudged@{primary_k}={float(metrics.get('unjudged_rate') or 0.0):.1%}"
        )
    dataset = report.get("dataset", {})
    print(f"Dataset: {dataset.get('name', '')} hash={dataset.get('hash', '')}")
    review_queue = report.get("review_queue", {})
    if review_queue.get("path"):
        print(f"Unjudged review queue: {review_queue.get('unjudged', 0)} -> {review_queue['path']}")
    return 0


def parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError("--ks must be a comma-separated list of positive integers") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError("--ks must contain at least one positive integer")
    return tuple(values)


def parse_rag_retrievers(value: str, *, use_embedding: bool) -> tuple[str, ...]:
    normalized = str(value or "auto").strip().casefold()
    if normalized == "auto":
        return ("bm25", "semantic", "rrf") if use_embedding else ("bm25",)
    values = tuple(item.strip() for item in normalized.split(",") if item.strip())
    allowed = {"bm25", "semantic", "rrf"}
    if not values or any(item not in allowed for item in values):
        raise ValueError("--retrievers supports bm25, semantic, rrf, or auto")
    return tuple(dict.fromkeys(values))


def first_lines(text: str, *, limit: int) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[:limit])
