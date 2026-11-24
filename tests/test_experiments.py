import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from aika.eval.benchmark_v2 import load_benchmark_v2
from aika.eval.experiment_artifacts import canonical, digest
from aika.eval.experiments import ExperimentConfig, ExperimentRunner, RequestRecorder, Variant, budget_context
from aika.llm_client import OpenAICompatibleClient
from aika.professional_qa import EvidenceCard
from aika.qa_engine import QAEngine
from aika.rag_index import RagHit
from aika.semantic_index import SemanticHit
from scripts import run_experiments as cli


class Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class Provider:
    def __init__(self):
        self.requests = []
        self.answer = "浪潮信息2026年AI服务器收入为10亿元 [E1]。"

    def post(self, url, *, headers, json: dict, timeout):
        self.requests.append(copy.deepcopy(json))
        system, user = [message["content"] for message in json["messages"]]
        content = self.answer
        if "response_format" in json:
            payload = globals()["json"].loads(user)
            if "中心调度 Agent" in system:
                value = {"assignments": [{"agent_type": "rag", "query": payload["question"], "filters": {},
                                           "coverage_goals": payload["required_coverage"]}]}
            elif "查询 Agent" in system:
                value = {"query": payload["question"], "filters": {}}
            elif "证据审核 Agent" in system:
                value = {"accepted_ids": [item["candidate_id"] for item in payload["candidates"]],
                         "rejected_ids": [], "missing_coverage": [], "conflicts": []}
            elif "证据归纳 Agent" in system:
                value = {"cards": [{"candidate_ids": [item["candidate_id"]]} for item in payload["candidates"][:2]]}
            else:
                value = {}
            content = globals()["json"].dumps(value)
        return Response({"model": "resolved-test-model", "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                         "choices": [{"message": {"content": content}}]})


class SemanticIndex:
    def __init__(self):
        self.queries = []

    def search(self, question, *, top_k=8, filters=None):
        self.queries.append((question, top_k, filters))
        return [SemanticHit(doc_id="rag:chunk_1", kind="rag", title="测试报告", text="浪潮信息布局AI服务器。",
                            source="测试报告", page="1", company="浪潮信息", ref_id="chunk_1", score=0.9)]


class RagIndex:
    def __init__(self):
        self.queries = []

    def search(self, question, *, top_k=6, filters=None):
        self.queries.append(question)
        return [RagHit(chunk_id="chunk_1", report_id="report_1", source_title="测试报告", source_tier="1",
                       source_type="annual_report", page="1", section="主营业务", content_type="text", table_id="",
                       company="浪潮信息", text="浪潮信息布局AI服务器。", snippet="浪潮信息布局AI服务器。", score=1)]


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr("aika.llm_client.load_dotenv", lambda *args: None)
    monkeypatch.setenv("LLM_THINKING_ENABLED", "false")
    monkeypatch.setattr("aika.llm_client.time.sleep", lambda *_: None)
    provider = Provider()
    monkeypatch.setattr("aika.llm_client.requests.post", provider.post)
    llm = OpenAICompatibleClient(api_key="SECRET_TEST_CREDENTIAL", base_url="https://provider.invalid/v1", model="test-model")
    semantic, rag = SemanticIndex(), RagIndex()
    engine = QAEngine(llm_client=llm, semantic_index=semantic, rag_index=rag, enable_hyde=False, multi_agent_max_llm_calls=64)
    runner = ExperimentRunner(llm_client=llm, semantic_index=semantic, engine=engine)
    return SimpleNamespace(provider=provider, llm=llm, semantic=semantic, rag=rag, engine=engine, runner=runner)


def forbid(*args, **kwargs):
    raise AssertionError("Forbidden experiment call")


def test_a_and_b_have_explicit_isolation_and_same_prompt(setup, monkeypatch):
    monkeypatch.setattr("aika.agents.verification.verify_answer_support", forbid)
    monkeypatch.setattr(setup.engine, "answer_question", forbid)
    monkeypatch.setattr(setup.engine, "_query_graph", forbid)
    monkeypatch.setattr(setup.engine, "_search_research", forbid)
    question = "浪潮信息有哪些AI服务器产品？"
    a = setup.runner.run(question, variants=["A"])[0]
    assert a["status"] == "success"
    assert not setup.semantic.queries and not setup.rag.queries
    assert a["generation_context"]["chars_after"] == 0
    b = setup.runner.run(question, variants=["B"])[0]
    assert b["status"] == "success"
    assert setup.semantic.queries == [(question, 6, {"kinds": ["rag"]})]
    assert not setup.rag.queries
    assert a["generation_context"]["system_prompt"] == b["generation_context"]["system_prompt"]
    assert [call["tool"] for call in b["tool_calls"]] == ["raw_vector_search", "generate_answer"]
    assert a["model_usage"]["attempts"] == b["model_usage"]["attempts"] == 1
    assert b["model_calls"][0]["usage"]["prompt_tokens"] == 20
    assert "SECRET_TEST_CREDENTIAL" not in canonical([a, b])


def test_ef_share_one_generation_and_persist_before_check(setup, monkeypatch):
    from aika.agents.verification import verify_answer_support
    events = []
    snapshots = []
    def check(*args, **kwargs):
        events.append("check")
        return verify_answer_support(*args, **kwargs)
    monkeypatch.setattr("aika.agents.verification.verify_answer_support", check)
    def save_candidate(payload):
        events.append("save_candidate")
        snapshots.append(copy.deepcopy(payload))
    def save_result(payload):
        events.append("save_" + payload["variant"])
    e, f = setup.runner.run("浪潮信息有哪些AI服务器产品？", variants=["E", "F"],
                            save_candidate=save_candidate, save_result=save_result)
    assert e["status"] == f["status"] == "success", [e["errors"], f["errors"]]
    assert events[:3] == ["save_candidate", "save_E", "check"]
    assert events[-1] == "save_F"
    assert e["pair_id"] == f["pair_id"]
    assert e["candidate_sha256"] == f["candidate_sha256"] == digest(snapshots[0]["candidate"])
    assert e["raw_answer"] == e["final_answer"] == f["raw_answer"]
    assert "10亿元" in e["final_answer"] and "10亿元" not in f["final_answer"]
    assert e["verification"]["status"] == "skipped"
    assert not any(call["tool"] == "verify_answer_support" for call in e["tool_calls"])
    assert sum(call["tool"] == "verify_answer_support" for call in f["tool_calls"]) == 2
    assert e["model_calls"] == f["model_calls"]
    assert sum("response_format" not in call for call in setup.provider.requests) == 1
    assert f["timings_ms"]["total"] >= e["timings_ms"]["total"]
    assert snapshots[0]["candidate"]["raw_answer"] == e["final_answer"]
    assert e["retrieved_materials"]["retrieval_rounds"][0]["output"][0]["text"] == "浪潮信息布局AI服务器。"


def test_e_never_calls_verifier_and_f_only_still_saves_candidate(setup, monkeypatch):
    monkeypatch.setattr("aika.agents.verification.verify_answer_support", forbid)
    e = setup.runner.run("浪潮信息有哪些AI服务器产品？", variants=["E"])[0]
    assert e["status"] == "success"
    saved = []
    failed = setup.runner.run("浪潮信息有哪些AI服务器产品？", variants=["F"], save_candidate=saved.append)[0]
    assert len(saved) == 1 and failed["status"] == "failed"
    assert failed["raw_answer"] == saved[0]["candidate"]["raw_answer"]
    assert failed["model_calls"] == saved[0]["result"]["model_calls"]


def test_shared_generation_failure_does_not_regenerate_for_f(setup, monkeypatch):
    attempts = []
    def fail(*args):
        attempts.append(1)
        raise RuntimeError("generation failed")
    monkeypatch.setattr(setup.runner, "_candidate", fail)
    rows = setup.runner.run("问题", variants=["E", "F"])
    assert len(attempts) == 1
    assert [row["status"] for row in rows] == ["failed", "failed"]


def test_supported_candidate_is_unchanged_by_f(setup):
    setup.provider.answer = "浪潮信息布局AI服务器 [E1]。"
    e, f = setup.runner.run("浪潮信息有哪些AI服务器产品？", variants=["E", "F"])
    assert f["final_answer"] == e["final_answer"]
    assert not any(call["tool"] == "build_evidence_limited_answer" for call in f["tool_calls"])


def test_context_budget_counts_metadata_and_keeps_identifiers():
    card = EvidenceCard("E1", "rag", "报告", "正文" * 5000, evidence_span="重复全文" * 3000, source="原文报告", page="12")
    selected, context = budget_context([card], 1200)
    assert len(context["text"]) == context["chars_after"] <= 1200
    assert context["chars_before"] > 1200 and context["truncated"]
    actual = json.loads(context["text"])
    assert actual[0]["citation_id"] == "E1" and actual[0]["page"] == "12"
    assert selected[0].evidence == actual[0]["evidence"]
    assert selected[0].evidence_span == ""
    assert budget_context([card], 2)[0] == []


def test_context_sent_to_model_is_the_recorded_budget(setup):
    setup.runner.config = ExperimentConfig(context_max_chars=900)
    row = setup.runner.run("问题", variants=["B"])[0]
    request = row["model_calls"][0]["request"]
    assert request["messages"][1]["content"] == row["generation_context"]["user_prompt"]
    assert row["generation_context"]["text"] in request["messages"][1]["content"]
    assert row["generation_context"]["chars_after"] <= 900


def test_empty_retrieval_still_generates_without_offline_fallback(setup, monkeypatch):
    monkeypatch.setattr(setup.semantic, "search", lambda *args, **kwargs: [])
    row = setup.runner.run("没有资料的问题", variants=["B"])[0]
    assert row["status"] == "success" and row["evidence"] == []
    assert row["model_usage"]["attempts"] == 1


def test_transport_has_three_attempts_not_nested_retries(setup, monkeypatch):
    attempts = []
    def fail(*args, **kwargs):
        attempts.append(1)
        raise requests.ConnectionError("unavailable")
    monkeypatch.setattr("aika.llm_client.requests.post", fail)
    row = setup.runner.run("问题", variants=["A"])[0]
    assert len(attempts) == len(row["model_calls"]) == 3
    assert row["status"] == "failed" and row["model_usage"]["unknown_usage_attempts"] == 3
    assert row["cost"]["amount"] is None


def test_parameter_downgrade_is_recorded_and_invalidates_result(setup, monkeypatch):
    setup.llm.thinking_enabled = True
    attempts = []
    def downgraded(*args, **kwargs):
        attempts.append(copy.deepcopy(kwargs["json"]))
        if "thinking" in kwargs["json"]:
            return Response({}, 400)
        return setup.provider.post(*args, **kwargs)
    monkeypatch.setattr("aika.llm_client.requests.post", downgraded)
    row = setup.runner.run("问题", variants=["A"])[0]
    assert len(attempts) == 2
    assert row["status"] == "configuration_mismatch"
    assert row["model_calls"][0]["parameter_downgrade"] is True
    assert "thinking" not in row["model_calls"][1]["request"]
    assert setup.llm._thinking_runtime_disabled is False


def test_agent_fallback_is_not_a_successful_experiment(setup):
    setup.engine.multi_agent_max_llm_calls = 0
    rows = setup.runner.run("浪潮信息有哪些AI服务器产品？", variants=["E", "F"])
    assert all(row["status"] != "success" for row in rows)


def test_dry_run_validates_existing_approved_set_without_services_or_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "validate_sources", forbid)
    result = cli.run_benchmark(output_dir=tmp_path, dry_run=True, runner_factory=forbid, snapshot_reader=forbid)
    assert result["planned_outputs"] == 80 and len(result["dataset"]["case_ids"]) == 20
    assert list(tmp_path.iterdir()) == []
    assert cli.main(["--dry-run", "--variants", "A", "--limit", "1"]) == 0


def test_pending_and_stale_reviews_are_rejected_before_generation(tmp_path):
    case = load_benchmark_v2()[0].model_dump(mode="json")
    case["review"].update(human_status="pending", human_reviewer=None, human_reviewed_at=None, reviewed_content_sha256=None)
    path = tmp_path / "pending.jsonl"
    path.write_text(json.dumps(case) + "\n")
    with pytest.raises(ValueError, match="review"):
        cli.run_benchmark(benchmark=path, dry_run=True)
    case = load_benchmark_v2()[0].model_dump(mode="json")
    case["question"] += " changed"
    path.write_text(json.dumps(case) + "\n")
    with pytest.raises(ValueError, match="stale"):
        cli.run_benchmark(benchmark=path, dry_run=True)


@pytest.fixture
def local_batch(setup, monkeypatch):
    monkeypatch.setattr(cli, "validate_sources", lambda cases: {"cases": len(cases)})
    monkeypatch.setattr(cli, "close_runner", lambda runner: None)
    return {"runner_factory": lambda variants, config: setup.runner, "snapshot_reader": lambda runner: {"test_snapshot": "frozen"}}


def test_twenty_cases_produce_eighty_records_without_gold_in_prompts(setup, local_batch, tmp_path):
    manifest = cli.run_benchmark(output_dir=tmp_path, **local_batch)
    directory = Path(manifest["output_dir"])
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 80
    assert [(row["case_id"], row["variant"]) for row in rows[:4]] == [("pilot_001", value) for value in ("A", "B", "E", "F")]
    assert len(list((directory / "candidates").glob("*.json"))) == 20
    unique_calls = {call["call_id"] for row in rows for call in row["model_calls"]}
    assert manifest["billing"]["unique_attempts"] == len(unique_calls)
    assert manifest["billing"]["cost"] is None
    assert (directory / "source.zip").exists()
    for request in setup.provider.requests:
        text = canonical(request)
        assert "forbidden_inferences" not in text and "answer_points" not in text and "human_reviewer" not in text
    for i in range(0, 80, 4):
        e, f = rows[i + 2:i + 4]
        assert e["pair_id"] == f["pair_id"] and e["candidate_sha256"] == f["candidate_sha256"]


def test_snapshot_drift_marks_entire_run_invalid(setup, local_batch, tmp_path):
    snapshots = iter([{"version": 1}, {"version": 2}])
    local_batch["snapshot_reader"] = lambda runner: next(snapshots)
    manifest = cli.run_benchmark(output_dir=tmp_path, variants=["A"], limit=1, **local_batch)
    assert manifest["status"] == "configuration_mismatch"
    assert "changed" in manifest["error"]


def test_initialization_failure_is_recorded_for_each_planned_output(local_batch, tmp_path):
    def unavailable(*args):
        raise RuntimeError("database unavailable")
    local_batch["runner_factory"] = unavailable
    manifest = cli.run_benchmark(output_dir=tmp_path, limit=1, **local_batch)
    assert manifest["counts"]["configuration_mismatch"] == 4
    rows = (Path(manifest["output_dir"]) / "results.jsonl").read_text().splitlines()
    assert len(rows) == 4 and all("database unavailable" in row for row in rows)


def test_recorder_is_safe_for_concurrent_calls_and_redacts_errors():
    from concurrent.futures import ThreadPoolExecutor
    recorder = RequestRecorder(["secret-value"])
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda i: recorder({"request_id": str(i), "attempt": 1, "error": "secret-value"}), range(50)))
    calls = recorder.calls()
    assert len(calls) == len({call["call_id"] for call in calls}) == 50
    assert all(call["error"] == "[REDACTED]" for call in calls)


def test_embedding_observer_keeps_provider_usage_and_retry_attempts(monkeypatch):
    from aika.embedding_client import OpenAICompatibleEmbeddingClient
    recorder = RequestRecorder()
    requests_seen = []
    def post(*args, **kwargs):
        requests_seen.append(kwargs["json"])
        if len(requests_seen) == 1:
            return Response({}, 503)
        return Response({"model": "embedding-test", "usage": {"prompt_tokens": 7, "total_tokens": 7},
                         "data": [{"index": 0, "embedding": [1.0, 0.0]}]})
    monkeypatch.setattr("aika.embedding_client.requests.post", post)
    monkeypatch.setattr("aika.embedding_client.time.sleep", lambda *_: None)
    client = OpenAICompatibleEmbeddingClient(
        api_key="private-key", base_url="https://provider.invalid/v1", model="embedding-test",
        dimensions=2, request_observer=recorder,
    )
    assert client.embed_texts(["问题"]) == [[1.0, 0.0]]
    calls = recorder.calls()
    assert len(calls) == 2 and calls[0]["request_id"] == calls[1]["request_id"]
    assert calls[0]["usage"] is None and calls[0]["error"]
    assert calls[1]["usage"]["prompt_tokens"] == 7
    assert "private-key" not in canonical(calls) and "embedding\": [" not in canonical(calls)


def test_runtime_snapshot_detects_vector_change_without_content_change():
    from contextlib import nullcontext
    from aika.eval.experiment_artifacts import runtime_snapshot
    records = {
        "rag_chunks": [{"record": {"chunk_id": "a", "report_id": "r", "page": "1", "text": "原文"}, "vector_hash": "v1"}],
        "research_claims": [], "segment_dossiers": [],
    }
    class Cursor:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def execute(self, sql):
            self.table = next(table for table in records if f"FROM {table} " in sql)
        def __iter__(self):
            return iter(copy.deepcopy(records[self.table]))
    class Connection:
        def transaction(self):
            return nullcontext()
        def cursor(self, **kwargs):
            return Cursor()
        def execute(self, sql, params=None):
            assert sql.startswith("SET TRANSACTION") or sql.startswith("SELECT")
            return []
    connection = Connection()
    engine = SimpleNamespace(
        retrieval_store=SimpleNamespace(pool=SimpleNamespace(connection=lambda: nullcontext(connection)), hnsw_ef_search=100),
        csv_graph=None, graph_client=None, semantic_index=SimpleNamespace(metadata_dict=lambda: {"version": "test"}),
    )
    before = runtime_snapshot(engine)
    records["rag_chunks"][0]["vector_hash"] = "v2"
    after = runtime_snapshot(engine)
    assert before["tables"]["rag_chunks"]["sha256"] != after["tables"]["rag_chunks"]["sha256"]
    assert before["raw_corpus_sha256"] == after["raw_corpus_sha256"]


def test_runtime_corpus_must_match_whole_benchmark_snapshot(tmp_path, monkeypatch):
    from datetime import date
    from aika.eval.experiment_artifacts import raw_corpus_digest
    monkeypatch.setattr(cli, "ROOT_DIR", tmp_path)
    rows = [{"chunk_id": "a", "report_id": "r", "text": "原文", "page": "1"}]
    (tmp_path / "corpus.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (tmp_path / "reports.csv").write_text("report_id,published_at\nr,2026-05-01\n")
    cases = [SimpleNamespace(as_of_date=date(2026, 5, 15), corpus_snapshot=SimpleNamespace(
        rag_corpus=SimpleNamespace(path="corpus.jsonl"), reports_manifest=SimpleNamespace(path="reports.csv"),
    ))]
    cli.validate_runtime_corpus({"raw_corpus_sha256": raw_corpus_digest(rows)}, cases)
    with pytest.raises(ValueError, match="differs"):
        cli.validate_runtime_corpus({"raw_corpus_sha256": "changed"}, cases)
    (tmp_path / "reports.csv").write_text("report_id,published_at\nr,2026-06-01\n")
    with pytest.raises(ValueError, match="post-cutoff"):
        cli.validate_runtime_corpus({"raw_corpus_sha256": raw_corpus_digest(rows)}, cases)


def test_persistence_failure_stops_f_before_check(setup, monkeypatch):
    from aika.eval.experiments import ArtifactWriteError
    monkeypatch.setattr("aika.agents.verification.verify_answer_support", forbid)
    def disk_full(payload):
        raise OSError("disk full")
    with pytest.raises(ArtifactWriteError, match="disk full"):
        setup.runner.run("浪潮信息有哪些AI服务器产品？", variants=["E", "F"], save_candidate=disk_full)


def test_generation_failure_keeps_retrieval_and_shares_failed_calls(setup, monkeypatch):
    original = setup.provider.post
    def fail_answer(*args, **kwargs):
        if "response_format" not in kwargs["json"]:
            raise requests.ConnectionError("generation unavailable")
        return original(*args, **kwargs)
    monkeypatch.setattr("aika.llm_client.requests.post", fail_answer)
    b, e, f = setup.runner.run("浪潮信息有哪些AI服务器产品？", variants=["B", "E", "F"])
    assert b["retrieved_materials"]["semantic_hits"]
    assert e["retrieved_materials"]["rag_hits"]
    assert e["pair_id"] == f["pair_id"] and e["model_calls"] == f["model_calls"]
    assert e["retrieved_materials"] == f["retrieved_materials"]
    assert len({call["call_id"] for call in e["model_calls"]}) == len(e["model_calls"])


def test_metric_refusal_is_preserved_without_final_answer_call(setup):
    e, f = setup.runner.run("浪潮信息2025年AI服务器收入是多少？", variants=["E", "F"])
    assert e["raw_answer"] and e["raw_answer"] == f["raw_answer"] == f["final_answer"]
    assert not any(call["role"] == "answer" for call in e["model_calls"])
    assert any(call.get("args", {}).get("refused") for call in e["tool_calls"])


def test_interruption_cannot_leave_a_successful_manifest(local_batch, tmp_path):
    def interrupt(*args):
        raise KeyboardInterrupt()
    local_batch["runner_factory"] = interrupt
    manifest = cli.run_benchmark(output_dir=tmp_path, limit=1, **local_batch)
    assert manifest["status"] == "configuration_mismatch" and not manifest["valid_for_comparison"]
    assert manifest["error"] == "KeyboardInterrupt"


def test_resume_restores_f_from_saved_e_without_more_model_calls(setup, local_batch, tmp_path, monkeypatch):
    original_run = setup.runner.run
    def interrupted(question, *, save_result, **kwargs):
        def save(payload):
            save_result(payload)
            if payload["variant"] == "E":
                raise KeyboardInterrupt()
        return original_run(question, save_result=save, **kwargs)
    monkeypatch.setattr(setup.runner, "run", interrupted)
    first = cli.run_benchmark(output_dir=tmp_path, variants=["E", "F"], limit=1, **local_batch)
    assert first["counts"]["failed"] + first["counts"]["success"] + first["counts"]["configuration_mismatch"] == 2
    directory = Path(first["output_dir"])
    previous_calls = len(setup.provider.requests)
    monkeypatch.setattr(setup.runner, "run", original_run)
    resumed = cli.run_benchmark(resume=directory, **local_batch)
    assert len(setup.provider.requests) == previous_calls
    from aika.eval.batch_runner import latest_results, read_journal
    records = latest_results(read_journal(directory / "results.jsonl"))
    e, f = records[("pilot_001", "E")], records[("pilot_001", "F")]
    assert e["pair_id"] == f["pair_id"] and e["candidate_sha256"] == f["candidate_sha256"]
    assert e["raw_answer"] == f["raw_answer"]
    assert resumed["started_at"] == first["started_at"]
    assert resumed["billing"]["unique_attempts"] == first["billing"]["unique_attempts"]
    assert f.get("terminal", True) and "KeyboardInterrupt" not in f["errors"]


def test_resume_rejects_changed_budgets_before_model_initialization(setup, local_batch, tmp_path, monkeypatch):
    monkeypatch.setattr(setup.runner, "run", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    first = cli.run_benchmark(output_dir=tmp_path, variants=["A"], limit=1, **local_batch)
    with pytest.raises(ValueError, match="budgets"):
        cli.run_benchmark(resume=Path(first["output_dir"]), config=ExperimentConfig(top_k=7), **local_batch)


def test_resume_rejects_runtime_drift_before_generating(setup, local_batch, tmp_path, monkeypatch):
    original_run = setup.runner.run
    monkeypatch.setattr(setup.runner, "run", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    first = cli.run_benchmark(output_dir=tmp_path, variants=["A"], limit=1, **local_batch)
    monkeypatch.setattr(setup.runner, "run", original_run)
    local_batch["snapshot_reader"] = lambda runner: {"changed": True}
    second = cli.run_benchmark(resume=Path(first["output_dir"]), **local_batch)
    assert second["status"] == "configuration_mismatch"
    assert "corpus, vector or graph change" in second["error"]
    assert not setup.provider.requests


def test_completed_resume_is_noop_even_without_services(setup, local_batch, tmp_path):
    first = cli.run_benchmark(output_dir=tmp_path, variants=["A"], limit=1, **local_batch)
    local_batch["runner_factory"] = forbid
    second = cli.run_benchmark(resume=Path(first["output_dir"]), **local_batch)
    assert second == first


def test_saved_candidate_rejects_tampered_result_fields(setup):
    candidates = []
    setup.runner.run("浪潮信息有哪些AI服务器产品？", variants=["E"], save_candidate=candidates.append)
    payload = candidates[0]
    payload["result"]["raw_answer"] = "tampered"
    with pytest.raises(ValueError, match="hashed candidate"):
        setup.runner.run("浪潮信息有哪些AI服务器产品？", variants=["F"], candidate_payload=payload)


def test_resume_after_failed_e_does_not_regenerate_for_f(setup, local_batch, tmp_path, monkeypatch):
    original_run = setup.runner.run
    def fail_and_interrupt(question, *, save_result, **kwargs):
        save_result({"variant": "E", "status": "failed", "question": question, "raw_answer": "", "final_answer": "",
                     "errors": ["Shared generation failed"], "model_calls": [], "pair_id": "failed-pair"})
        raise KeyboardInterrupt()
    monkeypatch.setattr(setup.runner, "run", fail_and_interrupt)
    first = cli.run_benchmark(output_dir=tmp_path, variants=["E", "F"], limit=1, **local_batch)
    monkeypatch.setattr(setup.runner, "run", original_run)
    second = cli.run_benchmark(resume=Path(first["output_dir"]), **local_batch)
    assert second["counts"]["failed"] == 2 and not setup.provider.requests
