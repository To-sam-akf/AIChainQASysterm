import csv
import hashlib
import json
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from aika.eval.benchmark_v2 import (
    DEFAULT_BENCHMARK_V2,
    BenchmarkV2Case,
    BenchmarkV2Error,
    file_sha256,
    load_benchmark_v2,
    publication_end,
    require_approved,
    validate_collection,
    validate_pilot,
    validate_sources,
)
from scripts.validate_eval_benchmark import main, render_review


@pytest.fixture
def payload():
    row = json.loads(DEFAULT_BENCHMARK_V2.read_text(encoding="utf-8").splitlines()[0])
    row["review"].update(human_status="pending", human_reviewer=None,
                         human_reviewed_at=None, reviewed_content_sha256=None)
    return row


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_real_pilot_is_balanced_reviewed_and_does_not_expose_gold():
    cases = load_benchmark_v2()
    validate_pilot(cases)
    assert len(cases) == 20
    assert Counter(c.answer_behavior.mode for c in cases) == {"complete": 15, "partial": 3, "refuse": 2}
    assert Counter(c.review.human_status for c in cases) == {"approved": 20}
    for case in cases:
        assert case.split == "dev"
        if case.review.human_status == "approved":
            assert case.review.human_reviewer == "sam"
            assert case.review.reviewed_content_sha256 == case.content_sha256()
        else:
            assert case.review.human_reviewer is None
            assert case.review.human_reviewed_at is None
            assert case.review.reviewed_content_sha256 is None
        assert case.public_input() == {"question": case.question}
    assert len(load_benchmark_v2(require_human_review=True)) == 20


def test_numeric_gold_keeps_units_years_and_accounting_identities():
    cases = {case.case_id: case for case in load_benchmark_v2()}
    inspur = [Decimal(n.value) for f in cases["pilot_011"].facts for n in f.numeric_values]
    assert inspur[0] + inspur[2] == inspur[1]
    assert cases["pilot_011"].facts[0].numeric_values[0].unit == "万元"
    hygon = [Decimal(n.value) for n in cases["pilot_015"].facts[0].numeric_values]
    assert hygon[0] + hygon[1] == hygon[2]
    assert (hygon[1] / hygon[2] * 100).quantize(Decimal("0.01")) == Decimal("9.26")
    modules = cases["pilot_012"].facts
    assert Decimal(modules[0].numeric_values[1].value) - Decimal(modules[1].numeric_values[1].value) == Decimal("7.96")
    assert [fact.numeric_values[0].period for fact in cases["pilot_013"].facts] == ["2023", "2024", "2025"]
    cambricon = cases["pilot_014"]
    assert "不包含股份支付" in cambricon.facts[1].statement
    assert any("不包含股份支付" in q.replace(" ", "") for q in cambricon.sources[0].quotes)


@pytest.mark.parametrize("field", ["facts", "answer_points", "evidence_units", "sources", "forbidden_inferences"])
def test_missing_required_annotations_are_rejected(payload, field):
    payload[field] = []
    with pytest.raises(ValidationError):
        BenchmarkV2Case.model_validate(payload)


@pytest.mark.parametrize("change, message", [
    (lambda p: p["facts"][0].update(source_ids=["missing"]), "unknown fact source"),
    (lambda p: p["answer_points"][0].update(fact_ids=["missing"]), "unknown answer point fact"),
    (lambda p: p["evidence_units"][0].update(primary_sources=["missing"]), "unknown evidence source"),
    (lambda p: p["evidence_units"][0].update(alternative_sources=[[]]), "empty evidence alternative"),
    (lambda p: p["facts"].append(deepcopy(p["facts"][0])), "duplicate fact_id"),
    (lambda p: p["sources"][0].update(published_at="2026-05-16"), "exceeds as_of_date"),
    (lambda p: p.update(question="没有公开截止日期的问题"), "expose as_of_date"),
    (lambda p: p.update(split="test"), "pilot cases must remain in dev"),
    (lambda p: p.update(filters={"company": "leaked gold"}), "Extra inputs"),
    (lambda p: p["sources"][0]["pdf"].update(path="../secret.pdf"), "relative to the repository"),
])
def test_invalid_references_dates_and_hidden_filters_are_rejected(payload, change, message):
    change(payload)
    with pytest.raises(ValidationError, match=message):
        BenchmarkV2Case.model_validate(payload)


def test_partial_answer_requires_explicit_parts_and_boundary():
    payload = load_benchmark_v2()[15].model_dump(mode="json")
    payload["answer_behavior"]["unanswerable"] = []
    with pytest.raises(ValidationError, match="both answerable and unanswerable"):
        BenchmarkV2Case.model_validate(payload)
    payload = load_benchmark_v2()[15].model_dump(mode="json")
    payload["answer_points"] = payload["answer_points"][:1]
    with pytest.raises(ValidationError, match="boundary answer point"):
        BenchmarkV2Case.model_validate(payload)


def test_human_approval_requires_attribution_and_is_bound_to_current_content(payload, tmp_path):
    case = BenchmarkV2Case.model_validate(payload)
    payload["review"].update(human_status="approved")
    with pytest.raises(ValidationError, match="reviewer, timestamp and content hash"):
        BenchmarkV2Case.model_validate(payload)
    payload["review"].update(
        human_reviewer="test-fixture-reviewer", human_reviewed_at="2026-11-22T10:00:00+08:00",
        reviewed_content_sha256=case.content_sha256(),
    )
    path = tmp_path / "reviewed.jsonl"
    write_rows(path, [payload])
    assert len(load_benchmark_v2(path, require_human_review=True)) == 1
    approved = load_benchmark_v2(path)[0]
    approved.facts[0].statement += " changed after loading"
    with pytest.raises(BenchmarkV2Error, match="missing or stale"):
        require_approved([approved])
    payload["facts"][0]["statement"] += " changed"
    write_rows(path, [payload])
    with pytest.raises(BenchmarkV2Error, match="content hash is stale"):
        load_benchmark_v2(path, require_human_review=True)


def test_primary_evidence_must_cover_its_declared_fact_sources():
    payload = load_benchmark_v2()[5].model_dump(mode="json")
    payload["evidence_units"][0]["primary_sources"] = [payload["sources"][1]["source_id"]]
    with pytest.raises(ValidationError, match="does not cover"):
        BenchmarkV2Case.model_validate(payload)


def test_duplicate_ids_questions_and_cross_split_groups(payload, tmp_path):
    path = tmp_path / "cases.jsonl"
    write_rows(path, [payload, payload])
    with pytest.raises(BenchmarkV2Error, match="duplicate case_id"):
        load_benchmark_v2(path)
    second = deepcopy(payload)
    second["case_id"] = "second"
    write_rows(path, [payload, second])
    with pytest.raises(BenchmarkV2Error, match="duplicate question"):
        load_benchmark_v2(path)
    second.update(question=second["question"] + "换一种问法", pilot=False, split="test")
    with pytest.raises(BenchmarkV2Error, match="leakage_group crosses"):
        validate_collection([BenchmarkV2Case.model_validate(p) for p in [payload, second]])
    second["leakage_group"] = "different_group"
    with pytest.raises(BenchmarkV2Error, match="same standard fact crosses"):
        validate_collection([BenchmarkV2Case.model_validate(p) for p in [payload, second]])


def test_pilot_quota_and_invalid_json_are_rejected(tmp_path):
    with pytest.raises(BenchmarkV2Error, match="20 cases"):
        validate_pilot(load_benchmark_v2()[:-1])
    path = tmp_path / "bad.jsonl"
    path.write_text("{}\nnot JSON\n", encoding="utf-8")
    with pytest.raises(BenchmarkV2Error, match="line 1"):
        load_benchmark_v2(path)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(BenchmarkV2Error, match="no cases"):
        load_benchmark_v2(path)


@pytest.fixture
def source_case(payload, tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "report.pdf"
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), "2025 Report. Revenue 12.00 CNY.")
        pdf.save(pdf_path)
    source = payload["sources"][0]
    source.update(
        report_id="report", title="Report", source_url="https://example.com/report", published_at="2026-04-01",
        pdf={"path": "report.pdf", "sha256": file_sha256(pdf_path)}, pdf_page=1,
        quotes=["Revenue 12.00 CNY."], chunks={"chunk_a": hashlib.sha256(b"Revenue 12.00 CNY.").hexdigest()},
    )
    rag = tmp_path / "documents.jsonl"
    write_rows(rag, [{"chunk_id": "chunk_a", "report_id": "report", "page": "1", "text": "Revenue 12.00 CNY."}])
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["report_id", "title", "source_url", "published_at", "local_path", "sha256"])
        writer.writeheader()
        writer.writerow({"report_id": source["report_id"], "title": source["title"], "source_url": source["source_url"],
                         "published_at": source["published_at"], "local_path": source["pdf"]["path"], "sha256": source["pdf"]["sha256"]})
    payload["corpus_snapshot"] = {
        "rag_corpus": {"path": rag.name, "sha256": file_sha256(rag)},
        "reports_manifest": {"path": manifest.name, "sha256": file_sha256(manifest)},
    }
    return BenchmarkV2Case.model_validate(payload), tmp_path


def test_source_validation_checks_frozen_files_chunks_and_pdf_excerpts(source_case):
    case, root = source_case
    assert validate_sources([case], root=root) == {"cases": 1, "files": 3, "pdf_pages": 1, "excerpts": 1}
    (root / "documents.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(BenchmarkV2Error, match="source hash changed"):
        validate_sources([case], root=root)


@pytest.mark.parametrize("change, message", [
    (lambda p: p["sources"][0].update(quotes=["Fabricated quote."]), "quote not found"),
    (lambda p: p["sources"][0].update(pdf_page=2), "chunk location mismatch"),
    (lambda p: p["sources"][0].update(chunks={"missing": "0" * 64}), "unknown chunk_id"),
    (lambda p: p["sources"][0].update(chunks={"chunk_a": "0" * 64}), "chunk text hash changed"),
    (lambda p: p["sources"][0].update(title="Different report"), "manifest mismatch"),
])
def test_source_validation_rejects_false_locators_and_excerpts(source_case, change, message):
    case, root = source_case
    payload = case.model_dump(mode="json")
    change(payload)
    case = BenchmarkV2Case.model_validate(payload)
    with pytest.raises(BenchmarkV2Error, match=message):
        validate_sources([case], root=root)


def test_month_and_year_publication_dates_use_conservative_cutoffs():
    assert publication_end("2024").isoformat() == "2024-12-31"
    assert publication_end("2024-02").isoformat() == "2024-02-29"
    assert publication_end("2025-07").isoformat() == "2025-07-31"
    with pytest.raises(ValueError):
        publication_end("2025-02-30")


def test_cli_review_render_is_reproducible_and_human_gate_fails_closed(tmp_path, capsys, payload):
    cases = load_benchmark_v2()
    expected = render_review(cases, file_sha256(DEFAULT_BENCHMARK_V2))
    checked_in = DEFAULT_BENCHMARK_V2.with_name("benchmark_v2_review.md")
    assert checked_in.read_text(encoding="utf-8") == expected
    output = tmp_path / "review.md"
    assert main(["--pilot", "--review-output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == expected
    summary = json.loads(capsys.readouterr().out)
    assert summary["human_review"] == {"approved": 20}
    assert summary["source_checks"] == "not_requested"
    assert main(["--require-human-review"]) == 0
    capsys.readouterr()
    pending = tmp_path / "pending.jsonl"
    write_rows(pending, [payload])
    assert main(["--benchmark", str(pending), "--require-human-review"]) == 2
    assert "human review required" in capsys.readouterr().err


def test_sam_revisions_preserve_history_and_remove_unfair_requirements():
    cases = {c.case_id: c for c in load_benchmark_v2()}
    audit_path = DEFAULT_BENCHMARK_V2.with_name("benchmark_v2_reviews.jsonl")
    events = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert len(events) == 25
    assert Counter(e["decision"] for e in events) == {"approved": 20, "changes_requested": 5}
    for event in events:
        current = cases[event["case_id"]]
        if event["decision"] == "approved":
            assert event["reviewed_content_sha256"] == current.content_sha256()
        else:
            original = BenchmarkV2Case.model_validate(event["reviewed_case"])
            assert original.content_sha256() == event["reviewed_content_sha256"]
            assert original.content_sha256() != current.content_sha256()
            assert event["revision"]["current_content_sha256"] == current.content_sha256()
            assert current.review.human_status == "approved"
    rechecks = events[20:]
    assert {e["case_id"] for e in rechecks} == {"pilot_005", "pilot_006", "pilot_008", "pilot_019", "pilot_020"}
    assert all(e["review_round"] == 2 and e["decision"] == "approved" for e in rechecks)
    assert not cases["pilot_005"].evidence_units[0].alternative_sources
    comparison = cases["pilot_006"]
    assert {u.unit_id for u in comparison.evidence_units if u.required} == {"e_f1", "e_f2"}
    assert {f.fact_id for f in comparison.facts} >= {"f3", "f4", "f5"}
    assert "至少三个" not in cases["pilot_008"].answer_points[0].text
    for cid in ["pilot_019", "pilot_020"]:
        assert not any(u.required for u in cases[cid].evidence_units)
        assert all(not p.fact_ids for p in cases[cid].answer_points if p.critical)
        assert not cases[cid].answer_points[-1].critical


def test_only_pure_refusals_allow_no_required_retrieval(payload):
    for unit in payload["evidence_units"]:
        unit["required"] = False
    with pytest.raises(ValidationError, match="answerable cases need"):
        BenchmarkV2Case.model_validate(payload)
    for case in load_benchmark_v2()[-2:]:
        BenchmarkV2Case.model_validate(case.model_dump(mode="json"))
