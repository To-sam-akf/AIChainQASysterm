"""Generate comparable A/B/E/F outputs from a human-approved v2 benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import sys
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit, urlunsplit

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from aika.eval.benchmark_v2 import DEFAULT_BENCHMARK_V2, file_sha256, load_benchmark_v2, publication_end, validate_sources
from aika.eval.experiment_artifacts import canonical, raw_corpus_digest, runtime_snapshot, source_snapshot, write_json
from aika.eval.batch_runner import append_jsonl, latest_results, read_journal, replace_jsonl, run_lock, source_identity
from aika.eval.experiments import (
    EXPERIMENT_VERSION, PROMPT_VERSION, ConfigurationMismatch, ExperimentConfig, ExperimentRunner,
    Variant, normalize_variants,
)


def is_complete(rows, key):
    return key in rows and rows[key].get("terminal", True)


def create_runner(variants: list[Variant], config: ExperimentConfig) -> ExperimentRunner:
    """A-only never initializes retrieval; B-only never initializes the graph/Agent."""
    from aika.llm_client import OpenAICompatibleClient

    if any(variant in variants for variant in (Variant.E, Variant.F)):
        from aika.qa_engine import QAEngine

        engine = QAEngine.from_env()
        try:
            missing = [name for name in ("llm_client", "rag_index", "semantic_index", "research_memory")
                       if getattr(engine, name) is None]
            errors = [getattr(engine.status, name) for name in ("graph_error", "rag_error", "research_error", "embedding_error", "llm_error")
                      if getattr(engine.status, name)]
            if missing or errors or (engine.csv_graph is None and engine.graph_client is None):
                raise ConfigurationMismatch(f"Full Agent unavailable: missing={missing}, errors={errors}")
            engine.agent_runner = "langgraph"
            return ExperimentRunner(llm_client=engine.llm_client, semantic_index=engine.semantic_index, engine=engine, config=config)
        except Exception:
            engine.close()
            raise
    llm = OpenAICompatibleClient()
    semantic = None
    if Variant.B in variants:
        from aika.embedding_client import OpenAICompatibleEmbeddingClient
        from aika.postgres_retrieval import EMBEDDING_DIMENSIONS, PostgresRetrievalStore, PostgresSemanticIndex

        store = PostgresRetrievalStore.from_env()
        try:
            store.ensure_ready()
            semantic = PostgresSemanticIndex(store, embedding_client=OpenAICompatibleEmbeddingClient(dimensions=EMBEDDING_DIMENSIONS))
        except Exception:
            store.close()
            raise
    return ExperimentRunner(llm_client=llm, semantic_index=semantic, config=config)


def close_runner(runner: ExperimentRunner) -> None:
    if runner.engine is not None:
        runner.engine.close()
    elif runner.semantic_index is not None:
        runner.semantic_index.store.close()


def snapshot_runner(runner: ExperimentRunner) -> dict[str, Any]:
    if runner.semantic_index is None:
        return {"retrieval": "not_initialized"}
    engine = runner.engine or SimpleNamespace(
        retrieval_store=runner.semantic_index.store, semantic_index=runner.semantic_index,
        csv_graph=None, graph_client=None,
    )
    return runtime_snapshot(engine)


def validate_runtime_corpus(snapshot: dict[str, Any], cases: list[Any]) -> None:
    if "raw_corpus_sha256" not in snapshot:
        return
    expected = {case.corpus_snapshot.rag_corpus.path for case in cases}
    if len(expected) != 1:
        raise ConfigurationMismatch("This run requires one shared corpus snapshot")
    path = ROOT_DIR / next(iter(expected))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if raw_corpus_digest(rows) != snapshot["raw_corpus_sha256"]:
        raise ConfigurationMismatch("PostgreSQL raw corpus differs from the approved benchmark corpus")
    # Check the whole corpus's source dates, not the answer key's selected evidence.
    manifests = {case.corpus_snapshot.reports_manifest.path for case in cases}
    reports = {}
    for manifest in manifests:
        with (ROOT_DIR / manifest).open(encoding="utf-8-sig", newline="") as handle:
            reports.update({row["report_id"]: row for row in csv.DictReader(handle)})
    cutoff = min(case.as_of_date for case in cases)
    for row in rows:
        report = reports.get(row.get("report_id"))
        if report is None or not report.get("published_at") or publication_end(report["published_at"]) > cutoff:
            raise ConfigurationMismatch(f"Corpus contains an undated or post-cutoff report: {row.get('report_id')}")


def model_config(runner: ExperimentRunner) -> dict[str, Any]:
    client = runner.llm_client
    result = {key: getattr(client, key, None) for key in ("model", "max_tokens", "thinking_enabled", "reasoning_effort")}
    result.update(asdict(runner.config))
    url = urlsplit(getattr(client, "base_url", ""))
    result["provider"] = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
    embedding = getattr(runner.semantic_index, "embedding_client", None)
    if embedding is not None:
        result["embedding"] = {key: getattr(embedding, key, None) for key in ("model", "dimensions", "batch_size")}
    if runner.engine is not None:
        result["agent"] = {
            key: getattr(runner.engine, key) for key in (
                "agent_runner", "agent_max_steps", "enable_llm_cypher", "enable_llm_planner", "enable_hyde",
                "hyde_query_mode", "hyde_max_chars", "rerank_mode", "rag_top_k", "semantic_top_k", "graph_limit",
                "rerank_top_n", "evidence_top_n", "core_companies_only", "drift_max_subquestions",
                "global_dossier_top_k", "local_claim_top_k", "graph_path_top_k", "multi_agent_max_workers",
                "multi_agent_max_llm_calls", "multi_agent_task_timeout_seconds",
            )
        }
    return result


def run_benchmark(
    *, benchmark: Path | None = None, variants: Any = None, limit: int = 0,
    case_ids: list[str] | None = None, config: ExperimentConfig | None = None,
    output_dir: Path = ROOT_DIR / "data/eval_runs/experiments", dry_run: bool = False,
    runner_factory: Any = create_runner, snapshot_reader: Any = snapshot_runner,
    resume: Path | None = None,
) -> dict[str, Any]:
    previous = json.loads((resume / "manifest.json").read_text()) if resume is not None else None
    if previous is not None:
        if previous["prompt_version"] != PROMPT_VERSION or previous["schema_version"] != EXPERIMENT_VERSION:
            raise ConfigurationMismatch("Resume requires the original generation schema and prompt")
        config = config or ExperimentConfig(**previous["config"])
        selected = normalize_variants(variants if variants is not None else previous["variants"])
        if asdict(config) != previous["config"] or [v.value for v in selected] != previous["variants"]:
            raise ConfigurationMismatch("Cannot change variants or budgets while resuming")
        benchmark = benchmark or resume / "benchmark.jsonl"
        if file_sha256(benchmark) != previous["dataset"]["sha256"]:
            raise ConfigurationMismatch("Cannot change the benchmark while resuming")
        if not case_ids and not limit:
            case_ids = previous["dataset"]["case_ids"]
    else:
        config = config or ExperimentConfig()
        selected = normalize_variants(variants if variants is not None else tuple(Variant))
        benchmark = benchmark or DEFAULT_BENCHMARK_V2
    cases = load_benchmark_v2(benchmark, require_human_review=True)
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    if case_ids:
        unknown = set(case_ids) - {case.case_id for case in cases}
        if unknown:
            raise ValueError(f"Unknown case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case.case_id in case_ids]
    if limit:
        cases = cases[:limit]
    if previous is not None and [case.case_id for case in cases] != previous["dataset"]["case_ids"]:
        raise ConfigurationMismatch("Cannot change the selected cases while resuming")
    manifest: dict[str, Any] = {
        "schema_version": EXPERIMENT_VERSION, "protocol_version": "experiment_protocol_v1", "prompt_version": PROMPT_VERSION,
        "dataset": {"path": str(benchmark), "sha256": file_sha256(benchmark),
                    "versions": sorted({case.dataset_version for case in cases}), "case_ids": [case.case_id for case in cases],
                    "splits": sorted({case.split for case in cases})},
        "variants": [variant.value for variant in selected], "config": asdict(config),
        "planned_outputs": len(cases) * len(selected), "dry_run": dry_run,
    }
    if dry_run:
        if resume is not None:
            rows = latest_results(read_journal(resume / "results.jsonl"))
            manifest["remaining_outputs"] = sum(not is_complete(rows, (case.case_id, variant.value))
                for case in cases for variant in selected)
        return manifest
    manifest["source_checks"] = validate_sources(cases)
    run_id = "experiment_" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    directory = resume if resume is not None else output_dir / run_id
    directory.mkdir(parents=True, exist_ok=resume is not None)
    with run_lock(directory):
        return _execute_batch(directory, manifest, previous, cases, selected, config, benchmark,
                              runner_factory, snapshot_reader, run_id)


def _execute_batch(directory, manifest, previous, cases, selected, config, benchmark, runner_factory, snapshot_reader, run_id):
    if previous is not None:
        rows = read_journal(directory / "results.jsonl", repair_tail=True)
        latest = latest_results(rows)
        expected = {(case.case_id, variant.value) for case in cases for variant in selected}
        questions = {case.case_id: case.question for case in cases}
        if not set(latest) <= expected or any(row.get("question") != questions.get(row["case_id"]) for row in latest.values()):
            raise ConfigurationMismatch("Saved journal does not match the frozen case/variant selection")
        if (all(is_complete(latest, (case.case_id, variant.value)) for case in cases for variant in selected)
                and previous["status"] in ("success", "failed") and not previous.get("error")):
            return previous
        if file_sha256(directory / "source.zip") != previous["code"]["source_sha256"]:
            raise ConfigurationMismatch("Saved source archive was modified")
        with tempfile.TemporaryDirectory() as temporary:
            source_snapshot(Path(temporary))
            if source_identity(Path(temporary) / "source.zip") != source_identity(directory / "source.zip"):
                raise ConfigurationMismatch("Source changed; start a new run instead of mixing implementations")
        manifest = previous
        run_id = manifest["run_id"]
        manifest.setdefault("resumed_at", []).append(datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        manifest["status"] = "running"
        manifest.pop("error", None)
    else:
        rows, latest = [], {}
    manifest.update(run_id=run_id, status="running", output_dir=str(directory))
    manifest.setdefault("started_at", datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    if previous is None:
        manifest["code"] = source_snapshot(directory)
        shutil.copyfile(benchmark, directory / "benchmark.jsonl")
        if file_sha256(directory / "benchmark.jsonl") != manifest["dataset"]["sha256"]:
            raise ConfigurationMismatch("Benchmark changed during initialization")
        manifest["dataset"]["artifact"] = "benchmark.jsonl"
    (directory / "candidates").mkdir(exist_ok=True)
    manifest_path = directory / "manifest.json"
    write_json(manifest_path, manifest)
    counts: Counter[str] = Counter()
    billing = {call["call_id"]: call for row in rows for call in row.get("model_calls", [])}
    billing.update({call["call_id"]: call for call in read_journal(directory / "model_calls.jsonl", repair_tail=True)})
    runner = None
    fatal = None
    completed = {key for key, row in latest.items() if row.get("terminal", True)}

    def append(case: Any, payload: dict[str, Any]) -> None:
        payload = {**payload, "case_id": case.case_id, "category": case.category, "split": case.split, "run_id": run_id}
        append_jsonl(directory / "results.jsonl", payload)
        latest[(case.case_id, payload["variant"])] = payload
        if payload.get("terminal", True):
            completed.add((case.case_id, payload["variant"]))
        for call in payload.get("model_calls", []):
            billing[call["call_id"]] = call

    try:
        runner = runner_factory(selected, config)
        resolved = {call.get("response_model") for call in billing.values() if call.get("kind") == "llm" and call.get("response_model")}
        if len(resolved) > 1:
            raise ConfigurationMismatch("Previous outputs contain inconsistent resolved model identifiers")
        if resolved:
            runner._resolved_model = next(iter(resolved))
        if previous is not None and manifest.get("model_config") is not None and manifest["model_config"] != model_config(runner):
            raise ConfigurationMismatch("Cannot resume with a different model or feature configuration")
        manifest["model_config"] = model_config(runner)
        before = snapshot_reader(runner)
        if previous is not None and manifest.get("runtime_before") is not None and canonical(manifest["runtime_before"]) != canonical(before):
            raise ConfigurationMismatch("Cannot resume after a corpus, vector or graph change")
        manifest["runtime_before"] = before
        validate_runtime_corpus(before, cases)
        write_json(manifest_path, manifest)
        for case in cases:
            pending = [v for v in selected if (case.case_id, v.value) not in completed]
            if not pending:
                continue
            def record_call(call):
                billing[call["call_id"]] = call
                append_jsonl(directory / "model_calls.jsonl", call)
            runner.request_sink = record_call
            def save_candidate(payload: dict[str, Any]) -> None:
                from aika.eval.experiment_artifacts import digest
                filename = f"{digest(case.case_id)[:24]}.json"
                write_json(directory / "candidates" / filename, {"case_id": case.case_id, **payload})
            from aika.eval.experiment_artifacts import digest
            candidate_file = directory / "candidates" / f"{digest(case.case_id)[:24]}.json"
            saved_candidate = json.loads(candidate_file.read_text()) if candidate_file.exists() else None
            if saved_candidate is not None and (case.case_id, "E") in completed:
                saved_e = latest[(case.case_id, "E")]
                if any(saved_e.get(key) != saved_candidate["result"].get(key) for key in ("raw_answer", "pair_id", "candidate_sha256", "evidence")):
                    raise ConfigurationMismatch("Saved E output differs from the persisted candidate")
            if (case.case_id, "E") in completed and Variant.F in pending and saved_candidate is None:
                original = latest[(case.case_id, "E")]
                if original["status"] == "success":
                    raise ConfigurationMismatch("Cannot resume F without the original E candidate")
                append(case, {**original, "variant": "F", "errors": [*original.get("errors", []), "Shared generation failed; F did not regenerate"]})
                pending.remove(Variant.F)
                if not pending:
                    continue
            options = {"candidate_payload": saved_candidate} if saved_candidate is not None else {}
            runner.run(
                **case.public_input(), variants=pending, save_candidate=save_candidate, **options,
                save_result=lambda payload: append(case, payload),
            )
        after = snapshot_reader(runner)
        manifest["runtime_after"] = after
        if canonical(before) != canonical(after):
            raise ConfigurationMismatch("Runtime corpus, vectors, graph or index changed during this run")
        if manifest["model_config"] != model_config(runner):
            raise ConfigurationMismatch("Model or feature configuration changed during this run")
        validate_sources(cases)
    except (Exception, KeyboardInterrupt) as exc:
        fatal = str(exc) or type(exc).__name__
        manifest["error"] = fatal
        # Initialization failures still occupy every planned case/variant slot.
        for case in cases:
            for variant in selected:
                if (case.case_id, variant.value) not in completed:
                    append(case, {
                        "schema_version": EXPERIMENT_VERSION, "variant": variant.value, "status": "configuration_mismatch", "terminal": False,
                        "question": case.question, "raw_answer": "", "final_answer": "", "retrieved_materials": {},
                        "evidence": [], "generation_context": {}, "tool_calls": [], "model_calls": [],
                        "timings_ms": {}, "errors": [fatal], "cost": {"amount": None, "currency": None},
                    })
    finally:
        if runner is not None:
            try:
                close_runner(runner)
            except Exception as exc:
                fatal = f"Resource cleanup failed: {exc}"
                manifest["error"] = fatal
        counts.update(row["status"] for row in latest.values())
        manifest["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        manifest["counts"] = {status: counts[status] for status in ("success", "failed", "configuration_mismatch")}
        manifest["status"] = "configuration_mismatch" if fatal else "failed" if counts["failed"] or counts["configuration_mismatch"] else "success"
        manifest["valid_for_comparison"] = not fatal and not counts["configuration_mismatch"]
        manifest["billing"] = {
            "unique_attempts": len(billing), "unknown_usage_attempts": sum(call.get("usage") is None for call in billing.values()),
            "cost": None if billing else 0, "currency": None, "pricing": "unknown",
        }
        replace_jsonl(directory / "model_calls.jsonl", billing.values())
        write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--variants", nargs="+", choices=[variant.value for variant in Variant])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--context-max-chars", type=int)
    parser.add_argument("--resume", type=Path, help="resume only unfinished outputs under identical frozen conditions")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "data/eval_runs/experiments")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = run_benchmark(
            benchmark=args.benchmark, variants=args.variants, limit=args.limit, case_ids=args.case_id,
            config=ExperimentConfig(context_max_chars=args.context_max_chars) if args.context_max_chars is not None else None,
            output_dir=args.output_dir, dry_run=args.dry_run, resume=args.resume,
        )
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({key: manifest[key] for key in ("run_id", "output_dir", "status", "counts", "planned_outputs", "dry_run", "error") if key in manifest}, ensure_ascii=False, indent=2))
    return 0 if manifest.get("status", "success") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
