import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aika.eval import score_runner
from aika.eval.answer_judge import OriginalSources, evidence_coverage, judge_input, numeric_audit, pending_judgment, validate_judgment
from aika.eval.batch_runner import append_jsonl, read_journal, run_lock
from aika.eval.benchmark_v2 import DEFAULT_BENCHMARK_V2, file_sha256, load_benchmark_v2
from aika.eval.experiment_artifacts import canonical, digest
from aika.eval.faithfulness import ratio, score_metrics
from aika.eval.judge_schema import Judgment
from aika.eval.pricing import billing_summary
from aika.eval.task_judge import calibration_sample, calibration_status, import_reviews, review_queue
from aika.llm_client import OpenAICompatibleClient


def response_for(payload):
    source = payload["sources"][0]
    refs = {c["citation_id"]: c["original_source_ids"] for c in payload["citations"]}
    by_id = {s["id"]: s for s in payload["sources"]}
    support = [{"source_id": source["id"], "quote": source["text"][:30]}]
    points = [{"point_id": p["point_id"], "status": "correct", "answer_quote": payload["answer"][:10],
               "support": support, "reason": "Fixture original supports the requested point"} for p in payload["answer_points"]]
    segments, claims, citations = [], [], []
    for index, segment in enumerate(payload["segments"]):
        claim_id = f"c{index}"
        segments.append({"segment_id": segment["segment_id"], "kind": "factual", "reason": "Factual statement"})
        claims.append({"claim_id": claim_id, "segment_id": segment["segment_id"], "statement": segment["text"],
                       "status": "supported", "support": support, "citation_ids": segment["citation_ids"], "reason": "Original support"})
        for citation in segment["citation_ids"]:
            originals = refs.get(citation, [])
            cited_support = [{"source_id": originals[0], "quote": by_id[originals[0]]["text"][:30]}] if originals else []
            citations.append({"claim_id": claim_id, "citation_id": citation, "status": "correct", "reason": "Located original", "support": cited_support})
    return {"points": points, "segments": segments, "claims": claims, "citations": citations,
            "task": {"usability": "usable", "reason": "All required points supported", "unsupported_answer": False,
                     "over_refusal": False, "serious_errors": [], "serious_errors_pending": False}}


@pytest.fixture
def fixture(tmp_path):
    case = load_benchmark_v2()[0]
    source = case.sources[0]
    chunk_id = next(iter(source.chunks))
    doc = {"chunk_id": chunk_id, "report_id": source.report_id, "company": "紫光股份", "source_title": source.title,
           "page": str(source.pdf_page), "text": source.quotes[0]}
    corpus = tmp_path / case.corpus_snapshot.rag_corpus.path
    corpus.parent.mkdir(parents=True)
    corpus.write_text(json.dumps(doc, ensure_ascii=False) + "\n")
    manifest = tmp_path / case.corpus_snapshot.reports_manifest.path
    manifest.parent.mkdir(parents=True)
    with manifest.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["report_id", "published_at"])
        writer.writeheader()
        writer.writerow({"report_id": source.report_id, "published_at": source.published_at})
    originals = OriginalSources([case], root=tmp_path)
    answer = "网络设备、服务器、存储产品、云与智能、主动安全、智能终端[E1]。服务器包括通用计算、人工智能计算、关键业务计算、边缘计算服务器[E1]。"
    row = {"case_id": case.case_id, "variant": "E", "status": "success", "question": case.question,
           "final_answer": answer, "raw_answer": answer, "model_calls": [], "timings_ms": {"total": 1},
           "retrieved_materials": {"rag_hits": [doc]}, "evidence": [{"citation_id": "E1", "semantic_ref_id": chunk_id,
               "source": source.title, "page": str(source.pdf_page), "company": "紫光股份", "evidence": source.quotes[0]}],
           "verification": {"status": "fail", "sentinel": "DO_NOT_TRUST_PRODUCT_CHECK"}, "pair_id": "SECRET_PAIR_METADATA"}
    return SimpleNamespace(case=case, row=row, originals=originals, doc=doc, root=tmp_path)


def test_blind_input_excludes_variant_trace_and_product_checks(fixture):
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    text = canonical(payload)
    assert "DO_NOT_TRUST_PRODUCT_CHECK" not in text and "SECRET_PAIR_METADATA" not in text
    assert "variant" not in payload and "case_id" not in payload and "verification" not in payload
    assert payload["answer"] == fixture.row["final_answer"]
    assert any(source["id"].startswith("doc:") for source in payload["sources"])


def test_correct_claim_and_citation_require_their_own_original_quotes(fixture):
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    decision = response_for(payload)
    assert validate_judgment(Judgment.model_validate(decision), payload).task.usability == "usable"
    bad = copy.deepcopy(decision)
    bad["claims"][0]["support"][0]["quote"] = "The source never says this"
    checked = validate_judgment(Judgment.model_validate(bad), payload)
    assert checked.claims[0].status == "pending" and checked.task.usability == "pending"
    bad = copy.deepcopy(decision)
    bad["citations"][0]["support"] = bad["claims"][0]["support"]  # Gold source is not this citation's mapped original ID.
    checked = validate_judgment(Judgment.model_validate(bad), payload)
    assert checked.citations[0].status == "pending" and checked.task.usability == "pending"


@pytest.mark.parametrize('assertion', [
    '以上仅为截至2026-05-15已入库的年度报告资料',
    '资料截至2026-05-15入库，仅含2025年年度报告片段',
])
def test_report_quote_cannot_approve_a_historical_ingestion_claim(fixture, assertion):
    fixture.row['final_answer'] += '\n' + assertion + '[E1]。'
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    decision = response_for(payload)
    checked = validate_judgment(Judgment.model_validate(decision), payload)
    claim = next(c for c in checked.claims if '入库' in c.statement)
    assert claim.status == 'pending' and not claim.support
    assert all(c.status == 'pending' for c in checked.citations if c.claim_id == claim.claim_id)
    assert checked.task.usability == 'pending'
    # The raw model response remains untouched, including its false support.
    assert next(c for c in decision['claims'] if '入库' in c['statement'])['status'] == 'supported'


@pytest.mark.parametrize("context, status, text, has_materials", [
    ({"text": ""}, "recorded", "", False),
    ({"text": "仅提供编号1、2、4、6。", "user_prompt": "SECRET_PROMPT"}, "recorded", "仅提供编号1、2、4、6。", True),
    ({}, "unknown", None, None),
    ({"text": "准备过但未发送的资料", "sent_to_model": False}, "not_sent", None, None),
])
def test_actual_input_is_separate_from_gold_and_retrieved_evidence(fixture, context, status, text, has_materials):
    fixture.row["generation_context"] = context
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    assert payload["generation_materials"] == {"status": status, "text": text, "has_materials": has_materials}
    assert payload["sources"]  # Gold remains available even when generator input was empty/unknown.
    assert "SECRET_PROMPT" not in canonical(payload)


def test_no_input_refusal_keeps_task_incomplete_without_inventing_wrong_refusal(fixture):
    fixture.row.update(final_answer="本次未提供资料，无法比较。", generation_context={"text": ""})
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    decision = pending_judgment(payload, "Awaiting review")
    for point in decision.points:
        point.status = "missing"
    decision.task.usability = "unusable"
    decision.task.over_refusal = True  # An unsupported label must not become a confirmed error.
    checked = validate_judgment(decision, payload)
    assert checked.task.usability == "unusable"
    assert all(point.status == "missing" for point in checked.points)
    assert checked.task.over_refusal is False
    assert "不能仅凭标准原文" in checked.task.reason


def test_empty_input_limit_is_not_a_serious_enterprise_fact_error(fixture):
    fixture.row.update(final_answer='无法给出具体金额。',generation_context={'text':''})
    payload=judge_input(fixture.row,fixture.case,fixture.originals)
    j=pending_judgment(payload,'review')
    for p in j.points: p.status='missing'
    proof={'source_id':payload['sources'][0]['id'],'quote':payload['sources'][0]['text'][:30]}
    j.segments[0].kind='factual'
    data=j.model_dump()
    data['claims']=[{'claim_id':'c1','segment_id':'s1','statement':'无法给出具体金额','status':'contradicted','reason':'Gold contains data','support':[proof],'citation_ids':[]}]
    data['task'].update(usability='unusable',over_refusal=True,serious_errors_pending=False,
        serious_errors=[{'kind':'numeric_or_period','answer_quote':'无法给出具体金额','reason':'Gold contains data','support':[proof]}])
    checked=validate_judgment(Judgment.model_validate(data),payload)
    assert checked.task.usability=='unusable' and checked.task.over_refusal is False
    assert not checked.task.serious_errors and not checked.claims
    assert checked.segments[0].kind=='nonfactual'


def test_quote_recovery_only_changes_presentation_not_meaning():
    from aika.eval.answer_judge import recover_answer_quote
    text='研发费用为 **100.00 万元**，不是收入。'
    quote=recover_answer_quote('研发费用为100.00万元',text)
    assert quote is not None and quote in text
    assert recover_answer_quote('研发费用为1000.00万元',text) is None
    assert recover_answer_quote('公司收入为100.00万元',text) is None
    assert recover_answer_quote('金额100万元','金额100万元；金额100万元') is None
    assert recover_answer_quote('23','2**3') is None


def test_missing_citation_is_wrong_and_unlocated_citation_stays_pending(fixture):
    fixture.row["final_answer"] = fixture.row["final_answer"].replace("[E1]", "[E99]")
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    result = validate_judgment(Judgment.model_validate(response_for(payload)), payload)
    assert all(c.status == "incorrect" for c in result.citations)
    fixture.row["evidence"] = [{"citation_id": "E99", "source": "Unknown", "page": "1", "evidence": "Unverifiable"}]
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    result = validate_judgment(Judgment.model_validate(response_for(payload)), payload)
    assert all(c.status == "pending" for c in result.citations)
    assert result.task.usability == "pending"


def test_point_and_segment_omissions_are_rejected(fixture):
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    for field in ("points", "segments"):
        decision = response_for(payload)
        decision[field] = []
        with pytest.raises(ValueError, match="IDs"):
            validate_judgment(Judgment.model_validate(decision), payload)


def test_judge_invented_citation_holds_only_affected_claim(fixture):
    fixture.row['final_answer'] = fixture.row['final_answer'].replace('[E1]', '', 1)
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    decision = response_for(payload)
    claim = decision['claims'][0]
    assert not claim['citation_ids']
    claim['citation_ids'] = ['E1']  # The evidence exists, but this answer segment does not cite it.
    decision['citations'].append({**decision['citations'][0], 'claim_id': claim['claim_id']})
    original_answer = payload['answer']
    checked = validate_judgment(Judgment.model_validate(decision), payload)
    assert checked.claims[0].status == 'pending'
    assert checked.claims[0].citation_ids == []
    assert '不在该原答案分段中' in checked.claims[0].reason
    assert checked.claims[1].status == 'supported'
    assert all(c.claim_id != claim['claim_id'] for c in checked.citations)
    assert len(checked.citations) == 1 and checked.citations[0].status == 'correct'
    assert payload['answer'] == original_answer
    assert checked.task.usability == 'pending'
    assert validate_judgment(checked, payload).claims[0].status == 'pending'


def test_missing_citation_assessment_becomes_pending_not_correct(fixture):
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    decision = response_for(payload)
    decision['citations'].pop()
    checked = validate_judgment(Judgment.model_validate(decision), payload)
    assert len(checked.citations) == 2
    assert checked.citations[0].status == 'correct'
    assert checked.citations[1].status == 'pending'
    assert '评审遗漏' in checked.citations[1].reason
    assert checked.task.usability == 'pending'


def test_unrelated_citation_assessment_is_still_rejected(fixture):
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    decision = response_for(payload)
    decision['citations'].append({**decision['citations'][0], 'claim_id': 'invented_claim'})
    with pytest.raises(ValueError, match='unrelated'):
        validate_judgment(Judgment.model_validate(decision), payload)


def test_pending_is_not_zero_or_one_and_no_citations_are_na():
    metric = ratio(["correct", "incorrect", "pending"], positive={"correct"})
    assert metric == {"numerator": 1, "denominator": 2, "pending": 1, "total": 3, "value": 0.5,
                      "applicable": True, "provisional": True, "review_coverage": 2 / 3}
    assert ratio([], positive={"correct"})["value"] is None
    assert ratio(["pending"], positive={"correct"})["value"] is None


def test_failure_counts_as_uncompleted_task_and_zero_points(fixture):
    fixture.row["status"] = "failed"
    assessment, _ = score_runner.assess(fixture.row, fixture.case, fixture.originals, answer_id="anonymous", rules_only=True)
    metrics = assessment["metrics"]
    assert metrics["task_usability"] == "unusable"
    assert metrics["answer_point_coverage"]["value"] == 0
    assert metrics["evidence_coverage"]["retrieved"]["value"] == 0
    assert metrics["fact_support"]["value"] is None and metrics["citation_correctness"]["value"] is None


def test_a_retrieval_is_na_even_if_factually_correct(fixture):
    fixture.row["variant"] = "A"
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    judgment = validate_judgment(Judgment.model_validate(response_for(payload)), payload)
    metrics = score_metrics(fixture.case, fixture.row, judgment, {"retrieved": [{"status": "matched"}], "selected": [], "segments": payload["segments"]})
    assert metrics["answer_point_coverage"]["value"] == 1
    assert metrics["fact_support"]["value"] == 1
    assert metrics["evidence_coverage"]["retrieved"]["value"] is None


def test_evidence_pointer_alone_is_not_coverage(fixture):
    exact = evidence_coverage(fixture.case, [fixture.doc], fixture.originals)
    assert all(unit["status"] == "matched" for unit in exact)
    abbreviated = {**fixture.doc, "text": "相关业务"}
    assert all(unit["status"] == "pending" for unit in evidence_coverage(fixture.case, [abbreviated], fixture.originals))
    assert all(unit["status"] == "missing" for unit in evidence_coverage(fixture.case, [], fixture.originals))


def test_numeric_audit_normalizes_units_without_claiming_semantic_correctness():
    case = load_benchmark_v2()[10]
    number = case.facts[0].numeric_values[0]
    from decimal import Decimal
    equivalent = Decimal(number.value) / Decimal(10000)
    audit = numeric_audit(f"{number.subject} {number.period} {equivalent}亿元", case)
    assert audit[0]["equivalent_values"] and audit[0]["period_present"]
    assert audit[0]["decision"] == "semantic_review_required"


def test_calibration_has_eight_answers_balanced_by_category_and_version():
    rows = [{"answer_id": f"{category}_{variant}", "case_id": category, "category": category, "variant": variant}
            for category in ("simple_lookup", "complex_analysis", "numeric_conflict", "insufficient_evidence")
            for variant in ("A", "B", "E", "F")]
    sample = calibration_sample(rows, 42)
    selected = [row for row in rows if row["answer_id"] in sample["answer_ids"]]
    from collections import Counter
    assert len(selected) == 8 and set(Counter(r["category"] for r in selected).values()) == {2}
    assert Counter(r["variant"] for r in selected) == {"A": 2, "B": 2, "E": 2, "F": 2}
    assert any({r["variant"] for r in selected if r["case_id"] == category} == {"E", "F"} for category in {r["case_id"] for r in selected})
    assert calibration_sample(rows, 42) == sample
    assert calibration_sample(rows[:4], 42)["status"] == "insufficient_coverage"


def test_regrading_keeps_same_calibration_answers_but_not_old_ids():
    categories = ('simple_lookup', 'complex_analysis', 'numeric_conflict', 'insufficient_evidence')
    rows = [{'answer_id': f'{cat}_{variant}', 'answer_sha256': f'hash_{cat}_{variant}',
             'case_id': cat, 'category': cat, 'variant': variant}
            for cat in categories for variant in ('A', 'B', 'E', 'F')]
    original = calibration_sample(rows, 42)
    hashes = [next(r['answer_sha256'] for r in rows if r['answer_id'] == aid) for aid in original['answer_ids']]
    updated = [{**r, 'answer_id': 'new_' + r['answer_id']} for r in rows]
    sample = calibration_sample(updated, 42, hashes)
    assert sample['answer_ids'] == ['new_' + aid for aid in original['answer_ids']]
    assert sample['status'] == 'awaiting_human_review'
    with pytest.raises(ValueError, match='unchanged'):
        calibration_sample(updated, 42, hashes[:-1] + ['changed_answer'])
    invalid = [r['answer_sha256'] for r in rows[:8]]
    with pytest.raises(ValueError, match='requirements'):
        calibration_sample(updated, 42, invalid)


def test_partial_json_tail_is_repaired_but_internal_corruption_is_rejected(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b'{"id":1}\n{"id":')
    assert read_journal(path, repair_tail=True) == [{"id": 1}]
    assert path.with_suffix(".jsonl.partial").read_bytes() == b'{"id":'
    append_jsonl(path, {"id": 2})
    assert len(read_journal(path)) == 2
    path.write_text('{"id":1}\ninvalid\n{"id":2}\n')
    with pytest.raises(ValueError, match="Malformed"):
        read_journal(path, repair_tail=True)


def test_exclusive_lock_rejects_parallel_writers(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(ValueError, match="Another process"):
            with run_lock(tmp_path):
                pass


def test_pricing_deduplicates_shared_calls_and_counts_cache_once():
    call = {"call_id": "x", "kind": "llm", "response_model": "test", "usage": {
        "prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 40}}
    rates = {"currency": "TEST", "source": "fixture", "effective_date": "2026-11-22", "models": {
        "test": {"input_per_million": "2", "input_cached_per_million": "1", "output_per_million": "3"}}}
    result = billing_summary([call, call], rates)
    assert result["unique_attempts"] == 1 and result["amount"] == "0.00022"
    assert billing_summary([call])["amount"] is None
    assert billing_summary([{**call, "usage": None}], rates)["unknown_usage_attempts"] == 1


@pytest.fixture
def saved_run(fixture, tmp_path, monkeypatch):
    directory = tmp_path / "generation"
    directory.mkdir()
    (directory / "benchmark.jsonl").write_bytes(DEFAULT_BENCHMARK_V2.read_bytes())
    manifest = {"run_id": "fixture_run", "dataset": {"artifact": "benchmark.jsonl", "sha256": file_sha256(DEFAULT_BENCHMARK_V2),
                "case_ids": [fixture.case.case_id]}, "variants": ["A", "B", "E", "F"],
                "valid_for_comparison": True, "model_config": {"model": "test"}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    for variant in manifest["variants"]:
        append_jsonl(directory / "results.jsonl", {**fixture.row, "variant": variant})
    monkeypatch.setattr(score_runner, "validate_sources", lambda *args: None)
    monkeypatch.setattr(score_runner, "OriginalSources", lambda *args: fixture.originals)
    return directory


@pytest.fixture
def judge_model(monkeypatch):
    seen = []
    monkeypatch.setattr("aika.llm_client.load_dotenv", lambda *args: None)
    monkeypatch.setattr("aika.llm_client.time.sleep", lambda *args: None)
    def post(url, *, json, **kwargs):
        payload = globals()["json"].loads(json["messages"][1]["content"])
        seen.append(payload)
        response = {"model": "test", "usage": {"prompt_tokens": 20, "completion_tokens": 30},
                    "choices": [{"message": {"content": globals()["json"].dumps(response_for(payload))}}]}
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: response)
    monkeypatch.setattr("aika.llm_client.requests.post", post)
    client = OpenAICompatibleClient(api_key="judge-secret-key", base_url="https://judge.invalid/v1", model="test")
    client.thinking_enabled = False
    return seen, lambda model: client


def test_saved_answers_can_be_scored_and_resumed_without_regeneration(saved_run, judge_model, tmp_path):
    seen, factory = judge_model
    result = score_runner.run_scoring(saved_run, output_dir=tmp_path / "scores", client_factory=factory)
    assert result["assessed"] == 4 and result["judge_errors"] == 0
    assert len(seen) == 1  # Identical blind inputs reuse judgment, independent of variant labels.
    assert result["billing"]["judging"]["unique_attempts"] == 1
    assert result["ready_for_product_conclusions"] is False  # Only one category, no human calibration.
    scores = read_journal(Path(result["output_dir"]) / "scores.jsonl")
    assert all(score["metrics"]["answer_point_coverage"]["value"] == 1 for score in scores)
    assert all(score["metrics"]["evidence_coverage"]["retrieved"]["value"] is None for score in scores if score["variant"] == "A")
    def forbidden(*args):
        raise AssertionError("Completed scoring must not call the judge")
    resumed = score_runner.run_scoring(resume=Path(result["output_dir"]), client_factory=forbidden)
    assert resumed["assessed"] == 4 and len(seen) == 1
    packet = read_journal(Path(result["output_dir"]) / "review_packet.jsonl")
    assert packet and all("variant" not in item and "case_id" not in item for item in packet)


def test_rules_only_never_invents_scores_or_human_approval(saved_run, tmp_path):
    result = score_runner.run_scoring(saved_run, output_dir=tmp_path / "scores", rules_only=True)
    assert result["pending_answers"] == 4 and result["human_reviewed"] == 0
    rows = read_journal(Path(result["output_dir"]) / "scores.jsonl")
    assert all(row["metrics"]["answer_point_coverage"]["value"] is None for row in rows)
    assert result["billing"]["judging"]["unique_attempts"] == 0


def test_invalid_model_judgment_becomes_pending_not_success(saved_run, judge_model, tmp_path, monkeypatch):
    monkeypatch.setattr("aika.llm_client.requests.post", lambda *args, **kwargs: SimpleNamespace(
        status_code=200, raise_for_status=lambda: None, json=lambda: {"model": "test", "usage": {"prompt_tokens": 2},
        "choices": [{"message": {"content": '{"points":[]}'}}]}))
    result = score_runner.run_scoring(saved_run, output_dir=tmp_path / "scores", client_factory=judge_model[1])
    assert result["judge_errors"] == result["pending_answers"] == 4
    rows = read_journal(Path(result["output_dir"]) / "scores.jsonl")
    assert all(row["metrics"]["fact_support"]["value"] is None for row in rows)


def test_review_import_requires_attribution_and_current_hashes(saved_run, judge_model, tmp_path):
    result = score_runner.run_scoring(saved_run, output_dir=tmp_path / "scores", client_factory=judge_model[1])
    directory = Path(result["output_dir"])
    rows = read_journal(directory / "scores.jsonl")
    row = rows[0]
    review = {"answer_id": row["answer_id"], "answer_sha256": row["answer_sha256"], "judgment_sha256": row["judgment_sha256"],
              "reviewer": "fixture-reviewer", "reviewed_at": "2026-11-22T21:00:00+08:00", "decision": "approved",
              "notes": "Reviewed original and answer", "minutes": 2, "judgment": None}
    path = tmp_path / "human.jsonl"
    append_jsonl(path, {**review, "answer_sha256": "stale"})
    with pytest.raises(ValueError, match="changed"):
        import_reviews(path, directory / "human_reviews.jsonl", {row["answer_id"]: row})
    path.write_text(canonical(review) + "\n")
    resumed = score_runner.run_scoring(resume=directory, reviews_path=path)
    assert resumed["human_reviewed"] == 1 and resumed["human_minutes"] == 2
    assert not resumed["calibration_ready"]


def test_changed_answer_blocks_scoring_resume(saved_run, tmp_path):
    result = score_runner.run_scoring(saved_run, output_dir=tmp_path / "scores", rules_only=True)
    records = read_journal(saved_run / "results.jsonl")
    records[0]["final_answer"] += " changed"
    (saved_run / "results.jsonl").write_text("".join(canonical(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="answers changed"):
        score_runner.run_scoring(resume=Path(result["output_dir"]))


def test_judge_retries_only_errors_and_keeps_prior_costs(saved_run, judge_model, tmp_path, monkeypatch):
    original = score_runner.assess
    def failed(*args, **kwargs):
        row, payload = original(*args, **{**kwargs, "rules_only": True})
        row["status"] = "judge_error"
        row["model_calls"] = [{"call_id": kwargs["answer_id"] + ":failed", "request_id": kwargs["answer_id"], "kind": "llm", "usage": None}]
        return row, payload
    monkeypatch.setattr(score_runner, "assess", failed)
    first = score_runner.run_scoring(saved_run, output_dir=tmp_path / "scores", client_factory=judge_model[1])
    assert first["judge_errors"] == 4
    monkeypatch.setattr(score_runner, "assess", original)
    second = score_runner.run_scoring(resume=Path(first["output_dir"]), retry_errors=True, client_factory=judge_model[1])
    assert second["judge_errors"] == 0
    assert second["billing"]["judging"]["unique_attempts"] == 5
    assert second["billing"]["judging"]["unknown_usage_attempts"] == 4


def test_calibration_blocks_unresolved_model_human_disagreements():
    sample = {"answer_ids": [str(i) for i in range(8)], "status": "awaiting_human_review"}
    model = [{"answer_id": str(i), "judgment": {"points": [{"point_id": "p1", "status": "correct"}], "claims": [],
             "task": {"usability": "usable", "unsupported_answer": False, "over_refusal": False}}} for i in range(8)]
    human = [{**copy.deepcopy(row), "human_review": {"reviewer": "fixture"}, "pending": False} for row in model]
    assert calibration_status(sample, model, human)["ready_for_product_conclusions"]
    human[0]["judgment"]["points"][0]["status"] = "incorrect"
    result = calibration_status(sample, model, human)
    assert not result["ready_for_product_conclusions"] and len(result["differences"]) == 1
    human[0]['pending'] = True
    human[1]['judgment']['task']['serious_errors'] = [{'kind':'unsupported_conclusion'}]
    result = calibration_status(sample, model, human)
    assert len(result['differences']) == 2 and result['pending_human_reviews'] == ['0']


def test_human_can_resolve_derived_evidence_with_located_quotes(fixture):
    from aika.eval.answer_judge import apply_coverage_review, coverage_materials
    from aika.eval.judge_schema import HumanReview
    source = fixture.case.sources[0]
    summary = " ".join(fact.statement for fact in fixture.case.facts)
    derived = {**fixture.doc, "text": summary}
    coverage = {"retrieved": evidence_coverage(fixture.case, [derived], fixture.originals), "selected": []}
    assert all(item["status"] == "pending" for item in coverage["retrieved"])
    materials = {"retrieved": coverage_materials(fixture.case, [derived], fixture.originals), "selected": []}
    decisions = [{"unit_id": unit.unit_id, "status": "matched", "reason": "Human verified summary against original", "support": [{
        "source_id": source.source_id, "original_quote": source.quotes[0], "material_id": materials["retrieved"][0]["material_id"],
        "material_quote": summary}]} for unit in fixture.case.evidence_units if unit.required]
    review = HumanReview.model_validate({"answer_id": "fixture", "answer_sha256": "a", "judgment_sha256": "b",
        "reviewer": "fixture-reviewer", "reviewed_at": "2026-11-22T21:00:00+08:00", "decision": "approved",
        "notes": "Verified original", "judgment": None, "evidence_coverage": {"retrieved": decisions}})
    result = apply_coverage_review(review, coverage, materials, fixture.case)
    assert all(item["status"] == "matched" for item in result["retrieved"])
    review.evidence_coverage["retrieved"][0].support[0].material_quote = "not in the material"
    with pytest.raises(ValueError, match="quotes do not exist"):
        apply_coverage_review(review, coverage, materials, fixture.case)


def test_refusal_does_not_require_optional_background_or_supplemental_point(fixture):
    case = load_benchmark_v2()[-1]
    row = {**fixture.row, "question": case.question, "final_answer": "未提供内部会议纪要，无法核实客户及金额，不能编造。", "evidence": [], "retrieved_materials": {}}
    payload = judge_input(row, case, fixture.originals)
    judgment = pending_judgment(payload, "Fixture review")
    for point in judgment.points:
        definition = next(p for p in case.answer_points if p.point_id == point.point_id)
        point.status = "correct" if definition.critical else "missing"
        point.answer_quote = "未提供内部会议纪要" if definition.critical else ""
    for segment in judgment.segments:
        segment.kind = "nonfactual"
    judgment.task.usability = "usable"
    judgment.task.serious_errors_pending = False
    judgment.task.unsupported_answer = False
    judgment.task.over_refusal = False
    judgment = validate_judgment(judgment, payload)
    metric = score_metrics(case, row, judgment, {"retrieved": [], "selected": [], "segments": payload["segments"]})
    assert metric["answer_point_coverage"]["value"] == 1
    assert metric["answer_point_coverage"]["total"] == 2
    assert metric["evidence_coverage"]["retrieved"]["value"] is None
    assert metric["citation_correctness"]["value"] is None
    assert metric["task_usability"] == "usable"


def test_bad_quote_holds_only_affected_items(fixture):
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    decision = response_for(payload)
    decision["points"][0]["answer_quote"] = "network…other parts…not an actual quote"
    result = validate_judgment(Judgment.model_validate(decision), payload)
    assert result.points[0].status == "pending" and result.points[1].status == "correct"
    assert all(claim.status == "supported" for claim in result.claims)
    assert result.task.usability == "pending"


def judgment_with_local_defect(fixture, *, status='unsupported'):
    fixture.row['final_answer'] += '双方均有CPO布局[E1]。'
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    data = response_for(payload)
    local = data['claims'][-1]
    local.update(status=status, support=[], reason='Only the silicon/LPO part has source support')
    for citation in data['citations']:
        if citation['claim_id'] == local['claim_id']:
            citation.update(status='incorrect', reason='Does not support the full CPO claim')
    data['task'].update(usability='minor_revision', minor_revision_claim_ids=[local['claim_id']])
    return payload, data


def test_minor_revision_preserves_local_fact_and_citation_failures(fixture):
    payload, data = judgment_with_local_defect(fixture)
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert checked.task.usability == 'minor_revision'
    assert checked.claims[-1].status == 'unsupported'
    assert checked.citations[-1].status == 'incorrect'
    metric = score_metrics(fixture.case, fixture.row, checked,
                          {'retrieved': [], 'selected': [], 'segments': payload['segments']})
    assert metric['unsupported_count'] == 1
    assert metric['citation_correctness']['value'] < 1
    assert metric['task_usability'] == 'minor_revision'


def test_minor_revision_cannot_hide_an_incomplete_critical_point(fixture):
    payload, data = judgment_with_local_defect(fixture)
    data['points'][0]['status'] = 'missing'
    assert validate_judgment(Judgment.model_validate(data), payload).task.usability == 'major_revision'


def test_minor_revision_needs_explicit_localization_and_keeps_unresolved_metrics(fixture):
    from aika.eval.faithfulness import is_pending
    payload, data = judgment_with_local_defect(fixture, status='pending')
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert checked.task.usability == 'minor_revision'
    metric = score_metrics(fixture.case, fixture.row, checked,
                          {'retrieved': [], 'selected': [], 'segments': payload['segments']})
    assert is_pending(metric)  # Task minor revision never awards an unresolved fact credit.
    data['task']['minor_revision_claim_ids'] = []
    assert validate_judgment(Judgment.model_validate(data), payload).task.usability != 'minor_revision'
    data['task']['minor_revision_claim_ids'] = ['invented']
    with pytest.raises(ValueError, match='unknown claim'):
        validate_judgment(Judgment.model_validate(data), payload)


def test_program_constraints_do_not_mutate_the_original_human_decision(fixture):
    payload, data = judgment_with_local_defect(fixture)
    data['task'].update(usability='usable', minor_revision_claim_ids=[])
    original = Judgment.model_validate(data)
    checked = validate_judgment(original, payload)
    assert checked.task.usability == 'major_revision'
    assert original.task.usability == 'usable'
    assert original.model_dump() == Judgment.model_validate(data).model_dump()


@pytest.mark.parametrize('extra', [
    '二者不应相加，因为资本化部分已包含在研发投入总额中，相加会重复计算。',
    '匿名编号体系之间无公开映射。',
    '本回答仅基于截至2026-05-15已入库的资料。',
])
def test_method_and_global_scope_claims_cannot_be_hidden_as_nonfactual(fixture, extra):
    fixture.row['final_answer'] += extra
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    data = response_for(payload)
    omitted = data['claims'].pop()
    data['segments'][-1]['kind'] = 'nonfactual'
    data['citations'] = [c for c in data['citations'] if c['claim_id'] != omitted['claim_id']]
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert checked.segments[-1].kind == 'pending' and checked.task.usability == 'pending'


def test_prefixed_empty_input_and_conditional_inability_are_not_contradictions(fixture):
    fixture.row.update(final_answer='回答：无法给出具体金额。在缺少该表具体内容的情况下，订单金额属于无法确认事项。',
                       generation_context={'text': ''})
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    data = response_for(payload)
    for claim in data['claims']:
        claim['status'] = 'contradicted'
    for point in data['points']:
        point.update(status='missing', answer_quote='')
    data['task'].update(usability='unusable', over_refusal=True)
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert not checked.claims and all(s.kind == 'nonfactual' for s in checked.segments)
    assert checked.task.over_refusal is False and checked.task.usability == 'unusable'


@pytest.mark.parametrize('boundary', ['也无法仅凭该匿名表确认客户1就是NVIDIA', '也无法确认该客户是否为NVIDIA'])
def test_mixed_empty_input_and_identity_claim_retains_a_reviewable_boundary(fixture, boundary):
    fixture.row.update(final_answer='回答：无法给出订单金额，' + boundary + '。',
                       generation_context={'text': ''})
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    data = response_for(payload)
    data['claims'][0]['status'] = 'contradicted'
    for point in data['points']:
        point.update(status='missing', answer_quote='')
    data['task'].update(usability='unusable', over_refusal=True)
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert len(checked.claims) == 1 and checked.claims[0].status == 'pending'
    assert checked.claims[0].statement == boundary
    assert checked.task.over_refusal is False


def test_empty_input_does_not_excuse_a_real_enterprise_absence_claim(fixture):
    fixture.row.update(final_answer='年报未披露订单金额。', generation_context={'text': ''})
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    data = response_for(payload)
    data['claims'][0]['status'] = 'contradicted'
    data['task']['usability'] = 'unusable'
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert len(checked.claims) == 1 and checked.claims[0].status == 'contradicted'


@pytest.mark.parametrize('statement,quote', [
    ('所给资料仅提及人工智能服务器，未列示服务器的完整类型划分。', '未列示服务器的完整类型划分'),
    ('本次冻结资料中未提供胜宏科技2025年报的任何片段。', '本次冻结资料中未提供胜宏科技2025年报的任何片段'),
    ('所给资料不能确认该表是否包含股份支付费用。', '所给资料不能确认'),
    ('资料中可确认的服务器类型表述仅为人工智能服务器一项。', '资料中可确认的服务器类型表述仅为人工智能服务器一项'),
])
def test_supplied_material_scope_does_not_become_confirmed_serious_error(fixture, statement, quote):
    fixture.row.update(final_answer=statement,
                       generation_context={'text': '仅有人工智能服务器的片段'})
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    data = response_for(payload)
    data['task'].update(usability='unusable', over_refusal=True, serious_errors_pending=False,
        serious_errors=[{'kind':'unsupported_conclusion','answer_quote':quote,'reason':'标准原文有完整清单',
                         'support':data['claims'][0]['support']}])
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert checked.task.usability == 'unusable'
    assert not checked.task.serious_errors and checked.task.serious_errors_pending
    assert checked.task.over_refusal is None


@pytest.mark.parametrize('kind', ['factual_without_claim', 'nonfactual_with_claim'])
def test_conflicting_segment_inventory_remains_pending(fixture, kind):
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    data = response_for(payload)
    first = data['segments'][0]
    if kind == 'factual_without_claim':
        removed = {c['claim_id'] for c in data['claims'] if c['segment_id'] == first['segment_id']}
        data['claims'] = [c for c in data['claims'] if c['claim_id'] not in removed]
        data['citations'] = [c for c in data['citations'] if c['claim_id'] not in removed]
    else:
        first['kind'] = 'nonfactual'
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert checked.segments[0].kind == 'pending'
    assert checked.task.usability == 'pending'


def test_empty_input_cannot_turn_unconfirmed_amount_into_serious_error(fixture):
    fixture.row.update(final_answer='无法确认中际旭创2025年报招投标订单表中“客户1”的订单金额。',
                       generation_context={'text':''})
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    data = response_for(payload)
    data['task'].update(usability='unusable', over_refusal=True, serious_errors_pending=False,
        serious_errors=[{'kind':'unsupported_conclusion','answer_quote':fixture.row['final_answer'].rstrip('。'),
                         'reason':'标准原文有金额','support':data['claims'][0]['support']}])
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert checked.task.usability == 'unusable' and checked.task.over_refusal is False
    assert not checked.task.serious_errors


def test_removing_input_limits_does_not_clear_other_unfounded_conclusions(fixture):
    fixture.row.update(final_answer='无法给出具体金额。公司没有服务器业务。', generation_context={'text': ''})
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    data = response_for(payload)
    for claim in data['claims']:
        claim['status'] = 'contradicted'
    data['task'].update(usability='unusable', unsupported_answer=True)
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert len(checked.claims) == 1 and checked.claims[0].statement == '公司没有服务器业务。'
    assert checked.task.unsupported_answer is True


def test_minor_revision_cannot_hide_confirmed_serious_errors(fixture):
    payload, data = judgment_with_local_defect(fixture)
    data['task']['serious_errors'] = [{'kind': 'unsupported_conclusion',
        'answer_quote': data['claims'][-1]['statement'], 'reason': 'Fixture confirmed serious conclusion',
        'support': data['claims'][0]['support']}]
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert checked.task.usability == 'major_revision'
    assert len(checked.task.serious_errors) == 1


@pytest.mark.parametrize('boundary', [
    '输入缺口：所给片段未包含速率内容，不能据此概括为整份年报未披露。',
    '回答：无法给出研发费用、研发投入总额、资本化研发投入的具体数值。',
    '回答：无法给出浪潮信息2025年研发费用、研发投入总额、资本化研发投入及投入占营收比例的具体数字。',
    '资料未披露的部分不据此断言“年报未披露”。',
    '边界：该结论仅反映本次输入缺口，不表示该年报未披露相关表格或金额。',
    '回答：无法依据“截至2026-05-15已入库资料”完成两家公司年报的比较，因为本次未提供任何资料片段。',
])
def test_valid_input_boundaries_do_not_create_phantom_fact_inventory(fixture, boundary):
    fixture.row.update(final_answer=boundary, generation_context={'text': ''})
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    j = pending_judgment(payload, 'Awaiting review')
    for segment in j.segments:
        segment.kind = 'nonfactual'
    checked = validate_judgment(j, payload)
    assert all(s.kind == 'nonfactual' for s in checked.segments)


def test_source_quote_recovers_a_unique_pdf_line_break_without_inventing_columns(fixture):
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    original = '产\n品服务于云计算数据中心、数据通信、\n5G 无线网络。'
    payload['sources'].append({'id': 'layout', 'text': original})
    data = response_for(payload)
    data['claims'][0]['support'].append({'source_id': 'layout', 'quote': '产品服务于云计算数据中心、数据通信、5G无线网络'})
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert checked.claims[0].status == 'supported'
    assert checked.claims[0].support[-1].quote in original
    assert data['claims'][0]['support'][-1]['quote'] not in original
    payload['sources'][-1]['text'] = '| 研发费用 | 100 | 80 |\n| 研发投入 | 120 | 90 |'
    data['claims'][0]['support'][-1]['quote'] = '研发费用100研发投入120'
    assert validate_judgment(Judgment.model_validate(data), payload).claims[0].status == 'pending'


def collective_payload(fixture, beta_text, *, statement=None):
    fixture.row['final_answer'] += statement or '两家公司均披露硅光平台和LPO、CPO布局[E10][E11]。'
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    for cid, sid, company, text in [
        ('E10', 'alpha', '甲公司', '甲公司已构建硅光平台及LPO、CPO产品技术体系，服务云数据中心及AI算力集群。'),
        ('E11', 'beta', '乙公司', beta_text),
    ]:
        payload['sources'].append({'id': sid, 'company': company, 'text': text})
        payload['citations'].append({'citation_id': cid, 'original_source_ids': [sid]})
    return payload


@pytest.mark.parametrize('beta_text', [
    '乙公司已推出硅光平台及LPO产品。',
    '乙公司已推出硅光平台及LPO产品。另有行业介绍：CPO指的是光电共封装技术。',
])
def test_collective_technology_claim_cannot_borrow_another_companys_cpo(fixture, beta_text):
    payload = collective_payload(fixture, beta_text)
    checked = validate_judgment(Judgment.model_validate(response_for(payload)), payload)
    assert checked.claims[-1].status == 'pending'
    assert '乙公司：CPO' in checked.claims[-1].reason
    assert all(c.status == 'incorrect' for c in checked.citations if c.claim_id == checked.claims[-1].claim_id)


def test_collective_technology_claim_with_both_company_disclosures_stays_supported(fixture):
    payload = collective_payload(fixture, '乙公司已构建硅光平台及LPO、CPO技术体系。')
    checked = validate_judgment(Judgment.model_validate(response_for(payload)), payload)
    assert checked.claims[-1].status == 'supported'


def test_revenue_table_cannot_support_both_company_application_scenarios(fixture):
    payload = collective_payload(fixture, '乙公司光通信收发模块收入占97.95%。',
                                 statement='两家公司均面向数据中心与AI算力场景[E10][E11]。')
    checked = validate_judgment(Judgment.model_validate(response_for(payload)), payload)
    assert checked.claims[-1].status == 'pending'
    assert '乙公司：数据中心场景' in checked.claims[-1].reason


def test_collective_scope_does_not_cancel_a_local_minor_revision(fixture):
    payload = collective_payload(fixture, '乙公司已推出硅光平台及LPO产品。')
    data = response_for(payload)
    data['task'].update(usability='minor_revision', minor_revision_claim_ids=[data['claims'][-1]['claim_id']])
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert checked.task.usability == 'minor_revision'
    assert checked.claims[-1].status == 'pending'


def test_shared_industry_explanation_does_not_assert_shared_company_products(fixture):
    payload = collective_payload(fixture, '乙公司年报介绍CPO行业原理。',
        statement='两家公司均介绍CPO行业原理，不等同于自身产品布局[E10][E11]。')
    checked = validate_judgment(Judgment.model_validate(response_for(payload)), payload)
    assert checked.claims[-1].status == 'supported'


def rule_payload(case_id, answer):
    from aika.eval.answer_judge import split_segments
    case = next(c for c in load_benchmark_v2() if c.case_id == case_id)
    sources = [{'id': 'gold:' + s.source_id, 'text': '\n'.join(s.quotes), 'title': s.title}
               for s in case.sources]
    return {'question': case.question, 'answer': answer, 'sources': sources,
            'facts': [f.model_dump() for f in case.facts], 'answer_points': [p.model_dump() for p in case.answer_points],
            'answer_behavior': case.answer_behavior.model_dump(), 'generation_materials': {'has_materials': True},
            'segments': split_segments(answer), 'citations': [
                {'citation_id': f'E{i+1}', 'original_source_ids': [s['id']]} for i, s in enumerate(sources)]}


def test_rule_audit_rejects_wrong_rd_operands_even_when_model_says_usable():
    answer = ('研发费用381,198.60万元，研发投入总额387,745.73万元，资本化研发投入6,547.13万元[E1][E2]。'
              '研发投入总额=研发费用+资本化研发投入。'
              '把"研发费用"与"资本化研发投入"相加会重复计算[E1][E2]。')
    payload = rule_payload('pilot_011', answer)
    checked = validate_judgment(Judgment.model_validate(response_for(payload)), payload)
    assert next(p.status for p in checked.points if p.point_id == 'p3') == 'incorrect'
    assert checked.task.usability == 'major_revision'
    assert any('运算对象' in c.reason and c.status == 'contradicted' for c in checked.claims)
    assert any(e.kind == 'unsupported_conclusion' for e in checked.task.serious_errors)


def test_rule_audit_does_not_promote_wrong_group_parent_conversion_to_correct():
    payload = rule_payload('pilot_011', '研发费用381,198.60万元（即2,853,087,157.96元，尾差为四舍五入）[E1]。')
    checked = validate_judgment(Judgment.model_validate(response_for(payload)), payload)
    assert next(p.status for p in checked.points if p.point_id == 'p1') == 'incorrect'
    assert checked.task.usability == 'major_revision'
    assert any(e.kind == 'numeric_or_period' for e in checked.task.serious_errors)


def test_rule_audit_blocks_a_false_maximum_not_just_a_shared_band():
    payload = rule_payload('pilot_006', '两家最高披露速率均到1.6T[E1]。')
    payload['sources'].append({'id': 'product', 'company': '新易盛', 'title': '2025年年度报告',
        'text': '公司已推出6.4T NPO、12.8T XPO光模块。'})
    payload['citations'][0]['original_source_ids'].append('product')
    checked = validate_judgment(Judgment.model_validate(response_for(payload)), payload)
    assert next(p.status for p in checked.points if p.point_id == 'p1') == 'incorrect'
    assert checked.task.usability == 'major_revision'
    assert any('最高披露' in c.reason and c.status == 'contradicted' for c in checked.claims)


def test_rule_audit_does_not_drop_a_bad_method_as_nonfactual():
    payload = rule_payload('pilot_011', '研发投入总额含研发费用与资本化研发投入。研发费用与资本化研发投入相加会重复计算。')
    data = response_for(payload)
    removed = data['claims'].pop()
    data['segments'][-1]['kind'] = 'nonfactual'
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert checked.segments[-1].kind == 'factual'
    assert any(c.segment_id == removed['segment_id'] and c.status == 'contradicted' for c in checked.claims)


def test_parent_statement_context_requires_an_explicit_adjacent_heading(fixture):
    report = fixture.doc['report_id']
    heading = {'chunk_id': 'parent_heading', 'report_id': report, 'page': '63', 'text': '4、母公司利润表\n单位：元'}
    continuation = {'chunk_id': 'continuation', 'report_id': report, 'page': '64', 'text': '| 销售费用 | 100 |\n| 研发费用 | 200 |'}
    fixture.originals.by_report_page[(report, '63')].append(heading)
    result = fixture.originals.financial_context(continuation)
    assert result['financial_scope'] == 'parent'
    assert result['financial_scope_basis']['quote'] in heading['text']
    assert fixture.originals.financial_context({**continuation, 'page': '65'}) == {}
    fixture.originals.by_report_page[(report, '63')].append({**heading, 'text': '合并利润表'})
    assert fixture.originals.financial_context(continuation) == {}


def test_judge_hash_tracks_rule_implementation_dependencies(monkeypatch):
    change = {'value': 'before'}
    monkeypatch.setattr(score_runner, 'file_sha256', lambda p: change['value'] if p.name == 'semantic_evidence.py' else str(p))
    before = score_runner.judging_code_hash()
    change['value'] = 'after'
    assert before != score_runner.judging_code_hash()


def test_sampling_topic_difference_loses_task_credit_without_becoming_a_false_fact():
    answer = '在本次所给资料范围内，中际旭创的披露侧重经营数据，新易盛的披露侧重技术布局与研发项目。'
    payload = rule_payload('pilot_006', answer)
    checked = validate_judgment(Judgment.model_validate(response_for(payload)), payload)
    assert next(p.status for p in checked.points if p.point_id == 'p3') == 'missing'
    assert checked.claims[0].status == 'supported'
    assert not checked.task.serious_errors


def test_empty_input_does_not_excuse_an_affirmative_client_identity_claim(fixture):
    fixture.row.update(final_answer='无法给出订单金额，客户1就是NVIDIA。', generation_context={'text': ''})
    payload = judge_input(fixture.row, fixture.case, fixture.originals)
    data = response_for(payload)
    data['claims'][0]['status'] = 'unsupported'
    data['task']['usability'] = 'unusable'
    checked = validate_judgment(Judgment.model_validate(data), payload)
    assert checked.claims[0].status == 'unsupported' and '就是NVIDIA' in checked.claims[0].statement
