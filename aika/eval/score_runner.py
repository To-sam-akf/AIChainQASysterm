"""Persisted, resumable judging and human review of already generated answers."""

from __future__ import annotations

import copy
import json
import os
import random
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from aika.eval.answer_judge import (
    JUDGE_VERSION, RUBRIC, OriginalSources, apply_coverage_review, coverage_materials, evidence_coverage, judge_input, pending_judgment, validate_judgment,
)
from aika.eval.batch_runner import append_jsonl, latest_results, read_journal, replace_jsonl, run_lock
from aika.eval.benchmark_v2 import ROOT_DIR, file_sha256, load_benchmark_v2, validate_sources
from aika.eval.experiment_artifacts import canonical, digest, source_snapshot, write_json
from aika.eval.experiments import ExperimentConfig, RequestRecorder, observed_client
from aika.eval.faithfulness import is_pending, score_metrics
from aika.eval.judge_schema import HumanReview, Judgment
from aika.eval.pricing import billing_summary, validate_prices
from aika.eval.task_judge import calibration_sample, calibration_status, import_reviews, review_queue
from aika.extraction_schema import parse_json_object
from aika.llm_client import OpenAICompatibleClient, load_dotenv
from aika.request_observation import request_role


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def judging_code_hash() -> str:
    directory = Path(__file__).parent
    return digest({name: file_sha256(directory / name) for name in (
        "answer_judge.py", "judge_schema.py", "faithfulness.py", "task_judge.py", "score_runner.py", "pricing.py",
        "rule_audit.py", "../numeric_evidence.py", "../semantic_evidence.py",
    )})


def create_judge_client(model: str | None = None):
    load_dotenv()
    client = OpenAICompatibleClient(
        model=model or os.getenv("JUDGE_MODEL") or None,
        base_url=os.getenv("JUDGE_BASE_URL") or None, api_key=os.getenv("JUDGE_API_KEY") or None,
        request_attempt_limit=3,
    )
    client.max_tokens = int(os.getenv("JUDGE_MAX_TOKENS", "8192"))
    client.thinking_enabled = os.getenv("JUDGE_THINKING_ENABLED", "false").lower() in {"1", "true", "yes"}
    return client


def client_settings(client) -> dict[str, Any]:
    url = urlsplit(client.base_url)
    return {"model": client.model, "provider": urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", "")),
            "max_tokens": client.max_tokens, "thinking_enabled": client.thinking_enabled,
            "reasoning_effort": client.reasoning_effort, "temperature": 0.0, "timeout": 60, "max_attempts": 3}


def load_generation(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    benchmark = directory / manifest["dataset"].get("artifact", "benchmark.jsonl")
    if file_sha256(benchmark) != manifest["dataset"]["sha256"]:
        raise ValueError("Saved benchmark hash does not match generation manifest")
    all_cases = load_benchmark_v2(benchmark, require_human_review=True)
    cases = {case.case_id: case for case in all_cases if case.case_id in manifest["dataset"]["case_ids"]}
    rows = latest_results(read_journal(directory / "results.jsonl"))
    expected = {(case_id, variant) for case_id in manifest["dataset"]["case_ids"] for variant in manifest["variants"]}
    if set(rows) != expected:
        raise ValueError("Generation journal is incomplete or contains unexpected case/variant slots; resume generation first")
    for (case_id, _), row in rows.items():
        if row.get("question") != cases[case_id].question or not isinstance(row.get("final_answer"), str):
            raise ValueError("Saved generation input/output does not match the benchmark")
    return manifest, cases, list(rows.values())


def _failure_judgment(payload):
    judgment = pending_judgment(payload, "Generation failed; task was not completed")
    for point in judgment.points:
        point.status = "missing"
    judgment.segments = []
    judgment.task.usability = "unusable"
    judgment.task.serious_errors_pending = False
    return judgment


def assess(row, case, originals, *, answer_id, client=None, rules_only=False, request_sink=None):
    payload = judge_input(row, case, originals)
    retrieved = list(material_rows_safe(row))
    coverage = {"retrieved": evidence_coverage(case, retrieved, originals),
                "selected": evidence_coverage(case, row.get("evidence", []), originals), "segments": payload["segments"]}
    recorder = RequestRecorder([getattr(client, "api_key", "")], sink=request_sink)
    error = None
    raw_text = None
    if row["status"] != "success":
        judgment = _failure_judgment(payload)
        status = "generation_failed"
    elif rules_only:
        judgment = pending_judgment(payload, "Rules-only mode: semantic judgment awaits human/model review")
        status = "pending"
    else:
        try:
            observed = observed_client(client, recorder, ExperimentConfig())
            with request_role("independent_judge"):
                response = observed.chat_text_with_metadata(
                    system_prompt=RUBRIC, user_prompt=canonical(payload), temperature=0.0,
                    thinking_enabled=client.thinking_enabled, reasoning_effort=client.reasoning_effort,
                )
            raw_text = response.content
            judgment = validate_judgment(Judgment.model_validate(parse_json_object(raw_text)), payload)
            status = "judged"
        except Exception as exc:
            error = str(exc).replace(getattr(client, "api_key", "") or "\0", "[REDACTED]")
            judgment = pending_judgment(payload, "Judge request or structured judgment validation failed: " + error)
            status = "judge_error"
    metrics = score_metrics(case, row, judgment, coverage)
    answer_hash = digest(row)
    assessment = {
        "schema_version": JUDGE_VERSION, "answer_id": answer_id, "answer_sha256": answer_hash,
        "case_id": case.case_id, "variant": row["variant"], "category": case.category,
        "created_at": now(), "status": status, "error": error, "judge_input_sha256": digest(payload),
        "judgment": judgment.model_dump(), "judgment_sha256": digest(judgment.model_dump()), "metrics": metrics,
        "coverage": coverage, "numeric_audit": payload["numeric_audit"], "model_calls": recorder.calls(),
        "coverage_materials": {"retrieved": coverage_materials(case, retrieved, originals),
                               "selected": coverage_materials(case, row.get("evidence", []), originals)},
        "pending": is_pending(metrics), "raw_response": raw_text,
    }
    return assessment, payload


def material_rows_safe(row):
    from aika.eval.answer_judge import material_rows
    yield from material_rows(row.get("retrieved_materials", {}))


def _human_rows(directory, rows, payloads, generated, cases):
    reviews = read_journal(directory / "human_reviews.jsonl")
    latest = {review["answer_id"]: HumanReview.model_validate(review) for review in reviews}
    effective = []
    for row in rows:
        review = latest.get(row["answer_id"])
        updated = copy.deepcopy(row)
        if review is not None:
            if review.answer_sha256 != row["answer_sha256"] or review.judgment_sha256 != row["judgment_sha256"]:
                # New judging rules require a new attributed review, not a stale approval.
                updated["stale_human_review"] = True
            else:
                judgment = review.judgment or Judgment.model_validate(row["judgment"])
                source = generated[(row["case_id"], row["variant"])]
                judgment = validate_judgment(judgment, payloads[row["answer_id"]]) if source["status"] == "success" else _failure_judgment(payloads[row["answer_id"]])
                updated["judgment"] = judgment.model_dump()
                source = generated[(row["case_id"], row["variant"])]
                updated["coverage"] = apply_coverage_review(review, row["coverage"], row["coverage_materials"], cases[row["case_id"]])
                updated["metrics"] = score_metrics(cases[row["case_id"]], source, judgment, updated["coverage"])
                updated["pending"] = is_pending(updated["metrics"])
                updated["human_review"] = review.model_dump(mode="json")
        effective.append(updated)
    return effective


def write_review_packet(directory, rows, payloads, queue):
    mapping = {row["answer_id"]: row for row in rows}
    packet = []
    templates = []
    for entry in queue:
        row = mapping[entry["answer_id"]]
        packet.append({**entry, "answer_sha256": row["answer_sha256"], "judgment_sha256": row["judgment_sha256"],
                       "input": payloads[row["answer_id"]], "model_judgment": row["judgment"],
                       "evidence_coverage": row["coverage"], "coverage_materials": row["coverage_materials"], "already_reviewed": bool(row.get("human_review"))})
        templates.append({"answer_id": row["answer_id"], "answer_sha256": row["answer_sha256"],
                          "judgment_sha256": row["judgment_sha256"], "reviewer": "", "reviewed_at": None,
                          "decision": "approved", "notes": "", "minutes": None, "judgment": None})
    for filename, items in (("review_packet.jsonl", packet), ("human_review_template.jsonl", templates)):
        # Regenerated views, never overwrite the imported human audit journal.
        path = directory / filename
        replace_jsonl(path, items)
    labels = {"correct": "正确", "missing": "遗漏", "incorrect": "错误", "pending": "待复核",
              "usable": "可用", "minor_revision": "可用，需小幅修订", "major_revision": "需大改", "unusable": "不可用"}
    lines = ["# 独立评分人工复核工作表", "",
             "答案已匿名；模型判分为暂定结果。请先核对答案和原文，再填写人工审核文件。", "",
             "复制 human_review_template.jsonl 为自己的审核文件，填写 reviewer、带时区的 reviewed_at 和 notes。",
             "判分无误用 approved；需要改分用 corrected，并在 judgment 填完整修正结果（可从 review_packet.jsonl 复制）。",
             "保留 answer_sha256 与 judgment_sha256；这些字段用于防止审核对应到旧答案。未决项保留 pending。", ""]
    for item in packet:
        row = mapping[item["answer_id"]]
        payload = item["input"]
        judgment = row["judgment"]
        lines += [f"## {item['answer_id']}", "", "复核原因：" + "、".join(item["reasons"]), "",
                  "问题：" + payload["question"], "", "### 保存的答案", "", payload["answer"], "",
                  "### 生成端实际收到的资料", ""]
        materials = payload["generation_materials"]
        if materials["status"] == "recorded":
            lines += ["以下是保存的实际资料输入；下方独立原文仅供判分，不代表生成端收到。", "",
                      "<details><summary>查看实际资料输入</summary>", "",
                      materials["text"] or "（本次没有提供资料）", "", "</details>", ""]
        else:
            lines += ["未发送最终生成请求。" if materials["status"] == "not_sent" else "未保存实际资料输入，不能由检索证据或标准原文反推。", ""]
        lines += ["### 暂定判分", "", "任务可用性：" + labels[judgment["task"]["usability"]],
                  judgment["task"]["reason"], "", "| 要点 | 是否关键 | 判定 | 理由 |", "|---|---|---|---|"]
        for point in judgment["points"]:
            definition = next(p for p in payload["answer_points"] if p["point_id"] == point["point_id"])
            lines.append(f"| {definition['text'].replace('|', '/')} | {'是' if definition['critical'] else '补充'} | {labels[point['status']]} | {point['reason'].replace('|', '/').replace(chr(10), ' ')} |")
        problematic = [claim for claim in judgment["claims"] if claim["status"] != "supported"]
        if problematic:
            lines += ["", "需核对的陈述：", ""]
            lines.extend(f"- {claim['statement']} — {claim['status']}：{claim['reason']}" for claim in problematic)
        if row["pending"]:
            lines += ["", "存在待复核项；当前分数不能作为最终产品结论。"]
        lines += ["", "<details><summary>独立原文与已审核要点（点击展开）</summary>", ""]
        for source in payload["sources"]:
            lines += [f"**{source['id']} · {source['title']} · 第 {source['page']} 页**", "",
                      "公开日期：" + source["published_at"], "", source["text"], ""]
        lines += ["</details>", ""]
    (directory / "review.md").write_text("\n".join(lines), encoding="utf-8")


def run_scoring(
    generation_dir: Path | None = None, *, output_dir: Path = ROOT_DIR / "data/eval_runs/scores",
    resume: Path | None = None, rules_only: bool | None = None, model: str | None = None,
    dry_run: bool = False, retry_errors: bool = False, seed: int | None = None,
    reviews_path: Path | None = None, prices: dict[str, Any] | None = None, client_factory=create_judge_client,
    calibration_from: Path | None = None,
) -> dict[str, Any]:
    previous = json.loads((resume / "manifest.json").read_text()) if resume is not None else None
    if previous:
        if generation_dir is not None and generation_dir.resolve() != Path(previous["generation_dir"]).resolve():
            raise ValueError("Cannot resume scoring against a different generation run")
        generation_dir = Path(previous["generation_dir"])
        rules_only = previous["rules_only"] if rules_only is None else rules_only
        seed = previous["seed"] if seed is None else seed
        prices = previous.get("prices") if prices is None else prices
        if (rules_only != previous["rules_only"] or seed != previous["seed"] or prices != previous.get("prices")
                or previous["code_hash"] != judging_code_hash() or previous["rubric_sha256"] != digest(RUBRIC)):
            raise ValueError("Scoring rules/configuration changed; create a new scoring run")
    if generation_dir is None:
        raise ValueError("--run-dir is required for a new scoring run")
    rules_only = bool(rules_only)
    seed = 20261122 if seed is None else seed
    validate_prices(prices)
    generation_manifest, cases, inputs = load_generation(generation_dir)
    input_hash = digest({"dataset": generation_manifest["dataset"]["sha256"], "results": sorted(inputs, key=lambda r: (r["case_id"], r["variant"]))})
    calibration_hashes = previous.get("calibration_answer_hashes") if previous else None
    if calibration_from is not None:
        prior = json.loads((calibration_from / "manifest.json").read_text())
        if prior["input_sha256"] != input_hash:
            raise ValueError("Previous calibration must use the same saved generation answers")
        selected_ids = json.loads((calibration_from / "calibration.json").read_text())["answer_ids"]
        prior_rows = {row["answer_id"]: row for row in read_journal(calibration_from / "scores.jsonl")}
        selected_hashes = [prior_rows[value]["answer_sha256"] for value in selected_ids]
        if previous and selected_hashes != calibration_hashes:
            raise ValueError("Cannot change calibration selection during resume")
        calibration_hashes = selected_hashes
        # Validate before spending judge calls, independently of new anonymous IDs.
        calibration_sample([{**row, "answer_id": digest(row), "answer_sha256": digest(row)} for row in inputs], seed, calibration_hashes)
    if previous and previous["input_sha256"] != input_hash:
        raise ValueError("Generation answers changed since scoring began")
    if dry_run:
        return {"dry_run": True, "planned_assessments": len(inputs), "rules_only": rules_only,
                "generation_run_id": generation_manifest["run_id"], "new_generation_calls": 0}
    validate_sources(list(cases.values()))
    originals = OriginalSources(list(cases.values()))
    run_id = previous["run_id"] if previous else "score_" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    directory = resume or output_dir / run_id
    directory.mkdir(parents=True, exist_ok=resume is not None)
    with run_lock(directory):
        manifest = previous or {
            "schema_version": JUDGE_VERSION, "run_id": run_id, "started_at": now(),
            "generation_dir": str(generation_dir.resolve()), "generation_run_id": generation_manifest["run_id"],
            "dataset_sha256": generation_manifest["dataset"]["sha256"], "input_sha256": input_hash,
            "rubric_sha256": digest(RUBRIC), "code_hash": judging_code_hash(), "rules_only": rules_only,
            "seed": seed, "prices": prices, "planned_assessments": len(inputs), "output_dir": str(directory.resolve()),
            "generation_valid_for_comparison": generation_manifest.get("valid_for_comparison", False),
            "calibration_answer_hashes": calibration_hashes,
            "calibration_from": str(calibration_from.resolve()) if calibration_from else None,
        }
        if previous is None:
            manifest["code"] = source_snapshot(directory)
            (directory / "rubric.txt").write_text(RUBRIC, encoding="utf-8")
        journal = read_journal(directory / "judgments.jsonl", repair_tail=True)
        current = {row["answer_id"]: row for row in journal}
        index = {digest(row): "answer_" + digest([run_id, digest(row)])[:20] for row in inputs}
        payloads = {index[digest(row)]: judge_input(row, cases[row["case_id"]], originals) for row in inputs}
        generated = {(row["case_id"], row["variant"]): row for row in inputs}
        shuffled = sorted(inputs, key=digest)
        random.Random(seed).shuffle(shuffled)
        needs_client = not rules_only and any(index[digest(row)] not in current or (retry_errors and current[index[digest(row)]]["status"] == "judge_error")
                                             for row in shuffled if row["status"] == "success")
        client = client_factory(model) if needs_client else None
        if client is not None:
            settings = client_settings(client)
            if previous and previous.get("judge_model_config") != settings:
                raise ValueError("Judge model configuration changed; create a new scoring run")
            manifest["judge_model_config"] = settings
            manifest["same_requested_model_as_generator"] = settings["model"] == generation_manifest["model_config"]["model"]
        elif model and previous and model != previous.get("judge_model_config", {}).get("model"):
            raise ValueError("Cannot change judge model during resume")
        manifest["status"] = "running"
        write_json(directory / "manifest.json", manifest)
        cache = {row["judge_input_sha256"]: row for row in journal if row["status"] == "judged"}
        try:
            for source in shuffled:
                answer_id = index[digest(source)]
                if answer_id in current and not (retry_errors and current[answer_id]["status"] == "judge_error"):
                    continue
                payload = payloads[answer_id]
                key = digest(payload)
                if key in cache and not rules_only and source["status"] == "success":
                    assessment = copy.deepcopy(cache[key])
                    assessment.update(answer_id=answer_id, answer_sha256=digest(source), case_id=source["case_id"],
                                      variant=source["variant"], reused_judgment_from=cache[key]["answer_id"], created_at=now())
                    case = cases[source["case_id"]]
                    assessment["coverage"] = {"retrieved": evidence_coverage(case, list(material_rows_safe(source)), originals),
                                              "selected": evidence_coverage(case, source.get("evidence", []), originals), "segments": payload["segments"]}
                    assessment["coverage_materials"] = {"retrieved": coverage_materials(case, list(material_rows_safe(source)), originals),
                                                        "selected": coverage_materials(case, source.get("evidence", []), originals)}
                    assessment["metrics"] = score_metrics(case, source, Judgment.model_validate(assessment["judgment"]), assessment["coverage"])
                    assessment["pending"] = is_pending(assessment["metrics"])
                else:
                    assessment, _ = assess(source, cases[source["case_id"]], originals, answer_id=answer_id, client=client, rules_only=rules_only,
                                           request_sink=lambda call: append_jsonl(directory / "model_calls.jsonl", call))
                append_jsonl(directory / "judgments.jsonl", assessment)
                journal.append(assessment)
                current[answer_id] = assessment
                if assessment["status"] == "judged":
                    cache[key] = assessment
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            manifest["finished_at"] = now()
            manifest["billing"] = {"judging": billing_summary(read_journal(directory / "model_calls.jsonl"), prices)}
            write_json(directory / "manifest.json", manifest)
            return manifest
        rows = [current[index[digest(source)]] for source in inputs]
        if reviews_path is not None:
            # Validate semantic constraints on all corrections before importing any audit entry.
            for item in read_journal(reviews_path):
                review = HumanReview.model_validate(item)
                existing = current.get(review.answer_id)
                if existing is None:
                    raise ValueError("Unknown answer in human review")
                apply_coverage_review(review, existing["coverage"], existing["coverage_materials"], cases[existing["case_id"]])
                if review.judgment is not None:
                    validate_judgment(review.judgment, payloads[review.answer_id])
            import_reviews(reviews_path, directory / "human_reviews.jsonl", current)
        effective = _human_rows(directory, rows, payloads, generated, cases)
        sample = calibration_sample(rows, seed, calibration_hashes)
        calibration = calibration_status(sample, rows, effective)
        queue = review_queue(effective, sample, seed)
        write_review_packet(directory, effective, payloads, queue)
        write_json(directory / "calibration.json", calibration)
        write_json(directory / "review_queue.json", queue)
        replace_jsonl(directory / "scores.jsonl", effective)
        generation_calls = [call for row in inputs for call in row.get("model_calls", [])]
        generation_calls.extend(read_journal(generation_dir / "model_calls.jsonl"))
        judge_calls = [call for row in journal for call in row["model_calls"]]
        judge_calls.extend(read_journal(directory / "model_calls.jsonl", repair_tail=True))
        manifest.update(
            status="needs_review" if any(row["pending"] for row in effective) or not calibration["ready_for_product_conclusions"] else "complete",
            finished_at=now(), assessed=len(rows), judge_errors=sum(row["status"] == "judge_error" for row in rows),
            pending_answers=sum(row["pending"] for row in effective), human_reviewed=sum(bool(row.get("human_review")) for row in effective),
            calibration_ready=calibration["ready_for_product_conclusions"],
            ready_for_product_conclusions=bool(calibration["ready_for_product_conclusions"] and generation_manifest.get("valid_for_comparison")
                                              and not any(row["pending"] for row in effective)
                                              and all(row.get("human_review") for row in effective if any(q["answer_id"] == row["answer_id"] for q in queue))),
            billing={"generation": billing_summary(generation_calls, prices), "judging": billing_summary(judge_calls, prices)},
            human_minutes=sum((row.get("human_review") or {}).get("minutes") or 0 for row in effective),
        )
        returned_models = sorted({call["response_model"] for call in judge_calls if call.get("response_model")})
        manifest["judge_models_returned"] = returned_models
        manifest["judge_configuration_consistent"] = len(returned_models) <= 1 and not any(call.get("parameter_downgrade") for call in judge_calls)
        manifest["ready_for_product_conclusions"] &= manifest["judge_configuration_consistent"]
        manifest["human_minutes_known"] = manifest["human_minutes"]
        manifest["human_time_unknown_reviews"] = sum(bool(row.get("human_review")) and row["human_review"].get("minutes") is None for row in effective)
        if manifest["human_time_unknown_reviews"]:
            manifest["human_minutes"] = None
        if not manifest["ready_for_product_conclusions"]:
            manifest["status"] = "needs_review"
        write_json(directory / "manifest.json", manifest)
        return manifest
