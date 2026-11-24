"""Generation-only A/B/E/F experiments. Gold annotations never enter this module."""

from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, fields, replace
from datetime import datetime
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable
from zoneinfo import ZoneInfo

from aika.eval.experiment_artifacts import canonical, digest
from aika.llm_client import OpenAICompatibleClient
from aika.professional_qa import EvidenceCard
from aika.request_observation import request_role


EXPERIMENT_VERSION = "experiments_v1"
PROMPT_VERSION = "experiment_answer_v4"
SYSTEM_PROMPT = (
    "你是严谨的资料问答助手。直接回答用户问题，区分事实与推断，保留数字、单位、年份及统计口径。"
    "回答结构为：回答、依据、边界。资料存在时仅依据所给资料，关键事实标注其证据编号如[E1]，"
    "不得引用不存在的编号。资料为空时遵循本次回答方式的限制，不得声称查阅或核实了未提供的资料，"
    "不得编造来源。无法确认时说明不确定性；可回答部分仍应回答。"
    "比较时对齐题目指定的公司、年度和来源类型；历史研报预测不能替代当年年报产品状态，行业技术原理不等于公司产品。"
    "收入或增速不能单独证明技术迭代。只拿到部分片段时说明输入缺口，不概括为整份年报未披露或整份报告的侧重点。"
    "同一公司的集团/合并与母公司报表须区分，两个金额各有来源不代表它们等价，须实际核算单位换算。"
    "舍入金额须注明近似，不能用来声称精确勾稽。解释相加时明确两个运算对象：费用化与资本化可组成总投入，总额再加其中的子项才重复。"
    "共同覆盖的速率档位不等于各公司的最高披露上限，不把单通道与聚合、不同架构的速率混作技术排名。"
    "共同客户/应用场景逐公司取原文交集；只有一方明确披露的AI算力集群等场景应单独归于该公司并引用其原文，不能扩展为双方共同场景。"
)


class Variant(str, Enum):
    A = "A"
    B = "B"
    E = "E"
    F = "F"


class ConfigurationMismatch(ValueError):
    """The requested experiment could not run under its frozen conditions."""


class ArtifactWriteError(RuntimeError):
    """A required candidate could not be persisted; checking it must not proceed."""


class ExperimentExecutionError(RuntimeError):
    def __init__(self, cause: Exception, partial: ExperimentResult) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.partial = partial


@dataclass(frozen=True)
class ExperimentConfig:
    context_max_chars: int = 12000
    top_k: int = 6
    temperature: float = 0.2
    timeout: int = 60
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.context_max_chars < 2 or self.top_k < 1 or self.timeout < 1 or self.max_attempts < 1:
            raise ValueError("Context budget must be >=2; top-k, timeout and attempts must be positive")


@dataclass
class ExperimentResult:
    variant: str
    status: str
    question: str
    raw_answer: str
    final_answer: str
    retrieved_materials: dict[str, Any]
    evidence: list[dict[str, Any]]
    generation_context: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    model_calls: list[dict[str, Any]]
    timings_ms: dict[str, float]
    errors: list[str]
    pair_id: str | None = None
    candidate_sha256: str | None = None
    verification: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)}
        calls = value["model_calls"]
        value.update(
            schema_version=EXPERIMENT_VERSION, prompt_version=PROMPT_VERSION,
            created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            model_usage={
                "attempts": len(calls),
                "logical_requests": len({call["request_id"] for call in calls}),
                "unknown_usage_attempts": sum(call.get("usage") is None for call in calls),
            },
            # No pricing assumptions. A deterministic check with no model calls costs zero.
            cost={"amount": None if calls else 0, "currency": None, "pricing": "unknown" if calls else "no_model_calls"},
            features={
                "configured": VARIANT_FEATURES[Variant(self.variant)],
                "executed_tools": sorted({call["tool"] for call in value["tool_calls"]}),
                "model_attempts": len(calls),
            },
        )
        return value


VARIANT_FEATURES = {
    Variant.A: {"retrieval": "none", "agent": False, "final_answer_check": False},
    Variant.B: {"retrieval": "raw_vector_once", "agent": False, "final_answer_check": False},
    Variant.E: {"retrieval": "agent", "agent": True, "final_answer_check": False},
    Variant.F: {"retrieval": "shared_E", "agent": True, "final_answer_check": True},
}


def normalize_variants(variants: Any) -> list[Variant]:
    requested = {Variant(value) for value in variants}
    if not requested:
        raise ValueError("Select at least one experiment variant")
    return [variant for variant in Variant if variant in requested]


def budget_context(cards: list[EvidenceCard], max_chars: int) -> tuple[list[EvidenceCard], dict[str, Any]]:
    """Truncate evidence text, never source identifiers or the surrounding JSON."""
    before = canonical([card.to_dict() for card in cards]) if cards else ""
    selected: list[EvidenceCard] = []
    for card in cards:
        trial = selected + [card]
        if len(canonical([item.to_dict() for item in trial])) <= max_chars:
            selected.append(card)
            continue
        # Keep only a prefix of the boundary card, using actual serialized character length.
        low, high = 0, len(card.evidence)
        best = None
        while low <= high:
            mid = (low + high) // 2
            clipped = replace(card, evidence=card.evidence[:mid], evidence_span="")
            size = len(canonical([item.to_dict() for item in selected + [clipped]]))
            if size <= max_chars:
                best = clipped
                low = mid + 1
            else:
                high = mid - 1
        if best is not None and best.evidence:
            selected.append(best)
        break
    text = canonical([card.to_dict() for card in selected]) if selected else ""
    return selected, {
        "text": text, "chars_before": len(before), "chars_after": len(text),
        "max_chars": max_chars, "truncated": before != text,
        "system_prompt": SYSTEM_PROMPT, "prompt_version": PROMPT_VERSION,
    }


class RequestRecorder:
    """One recorder per question/phase, including calls from query worker threads."""

    def __init__(self, secrets: list[str] | None = None, sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._calls: list[dict[str, Any]] = []
        self._secrets = [value for value in (secrets or []) if value]
        self._sink = sink

    def __call__(self, event: dict[str, Any]) -> None:
        event = copy.deepcopy(event)
        event["call_id"] = f"{event['request_id']}:{event['attempt']}"
        # Transport exceptions may contain a provider URL with a credential in it.
        for secret in self._secrets:
            if event.get("error"):
                event["error"] = event["error"].replace(secret, "[REDACTED]")
        with self._lock:
            self._calls.append(event)
            if self._sink is not None:
                self._sink(copy.deepcopy(event))

    def calls(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._calls)


def observed_client(client: Any, recorder: RequestRecorder, config: ExperimentConfig) -> Any:
    if not isinstance(client, OpenAICompatibleClient):
        # Injectable clients must implement the same observer contract; no invented usage.
        if not hasattr(client, "request_observer"):
            raise ConfigurationMismatch("LLM client does not support request observation")
    clone = copy.copy(client)
    clone.request_observer = recorder
    clone.request_attempt_limit = config.max_attempts
    clone.timeout = config.timeout
    clone._thinking_runtime_disabled = False
    return clone


class ExperimentRunner:
    """A/B need only their explicit clients; E/F require an existing QAEngine."""

    def __init__(
        self, *, llm_client: Any, semantic_index: Any = None, engine: Any = None,
        config: ExperimentConfig | None = None,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.llm_client = llm_client
        self.semantic_index = semantic_index
        self.engine = engine
        self._resolved_model: str | None = None
        if llm_client is None:
            raise ConfigurationMismatch("A generation model is required; offline fallback is disabled")

    def _recorder(self) -> RequestRecorder:
        embedding = getattr(self.semantic_index, "embedding_client", None)
        return RequestRecorder([getattr(self.llm_client, "api_key", ""), getattr(embedding, "api_key", "")],
                               sink=getattr(self, "request_sink", None))

    def _semantic(self, recorder: RequestRecorder) -> Any:
        if self.semantic_index is None:
            raise ConfigurationMismatch("Raw vector retrieval is unavailable")
        index = copy.copy(self.semantic_index)
        if getattr(index, "embedding_client", None) is not None:
            embedding = copy.copy(index.embedding_client)
            embedding.request_observer = recorder
            embedding.timeout = self.config.timeout
            embedding.max_retries = self.config.max_attempts - 1
            index.embedding_client = embedding
        return index

    def _generate(self, question: str, cards: list[EvidenceCard], client: Any, *, allow_prior: bool = False) -> tuple[str, str, list[EvidenceCard], dict[str, Any]]:
        selected, context = budget_context(cards, self.config.context_max_chars)
        user_prompt = (
            "请按回答、依据、边界组织答案。\n回答方式："
            + ("直接依据已有知识回答。" if allow_prior else "仅依据所给资料回答；资料不足时说明缺口，不补充资料外的实质事实。")
            + "\n问题：" + question + "\n资料：" + (context["text"] or "（未提供资料）")
        )
        context["user_prompt"] = user_prompt
        started = time.perf_counter()
        with request_role("answer"):
            response = client.chat_text_with_metadata(
                system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, temperature=self.config.temperature,
                thinking_enabled=getattr(self.llm_client, "thinking_enabled", False),
                reasoning_effort=getattr(self.llm_client, "reasoning_effort", ""),
            )
        if not response.content.strip():
            raise RuntimeError("Generation model returned an empty answer")
        context["generation_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return response.content, response.reasoning_content, selected, context

    def _status(self, result: ExperimentResult) -> None:
        if result.errors and result.status == "success":
            result.status = "failed"
        if any(call.get("parameter_downgrade") for call in result.model_calls):
            result.errors.append("Provider rejected frozen request parameters")
            result.status = "configuration_mismatch"
        for call in result.model_calls:
            model = call.get("response_model") if call.get("kind") == "llm" else None
            if model:
                if self._resolved_model is None:
                    self._resolved_model = model
                elif model != self._resolved_model:
                    result.errors.append("Provider changed the resolved model identifier")
                    result.status = "configuration_mismatch"
        result.errors = list(dict.fromkeys(result.errors))

    def _simple(self, question: str, variant: Variant, recorder: RequestRecorder) -> ExperimentResult:
        started = time.perf_counter()
        partial = ExperimentResult(variant.value, "failed", question, "", "", {}, [], {}, [], [], {}, [])
        try:
            return self._simple_impl(question, variant, recorder, partial)
        except Exception as exc:
            partial.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
            raise ExperimentExecutionError(exc, partial) from exc

    def _simple_impl(self, question: str, variant: Variant, recorder: RequestRecorder, partial: ExperimentResult) -> ExperimentResult:
        started = time.perf_counter()
        tool_calls = []
        partial.tool_calls = tool_calls
        retrieved = []
        cards = []
        if variant == Variant.B:
            index = self._semantic(recorder)
            retrieval_started = time.perf_counter()
            with request_role("retrieval"):
                hits = index.search(question, top_k=self.config.top_k, filters={"kinds": ["rag"]})
            if any(hit.kind != "rag" for hit in hits):
                raise ConfigurationMismatch("B retrieval returned non-raw documents")
            retrieved = [hit.to_dict() for hit in hits]
            cards = [EvidenceCard(
                citation_id=f"E{i}", kind="rag", title=hit.title, evidence=hit.text,
                source=hit.source, page=hit.page, section=hit.section, company=hit.company,
                semantic_ref_id=hit.ref_id or hit.doc_id, source_tier=hit.source_tier,
            ) for i, hit in enumerate(hits, 1)]
            tool_calls.append({
                "tool": "raw_vector_search", "args": {"question": question, "top_k": self.config.top_k, "filters": {"kinds": ["rag"]}},
                "result_count": len(hits), "elapsed_ms": round((time.perf_counter() - retrieval_started) * 1000, 2),
            })
            partial.retrieved_materials = {"semantic_hits": retrieved}
            partial.evidence = [card.to_dict() for card in cards]
        answer, _, selected, context = self._generate(
            question, cards, observed_client(self.llm_client, recorder, self.config), allow_prior=variant == Variant.A,
        )
        tool_calls.append({"tool": "generate_answer", "args": {"evidence_cards": len(selected)}, "result_count": 1,
                           "elapsed_ms": context["generation_ms"]})
        result = ExperimentResult(
            variant.value, "success", question, answer, answer, {"semantic_hits": retrieved},
            [card.to_dict() for card in selected], context, tool_calls, recorder.calls(),
            {"generation": context["generation_ms"], "total": round((time.perf_counter() - started) * 1000, 2)}, [],
        )
        return result

    def _candidate(self, question: str, recorder: RequestRecorder) -> tuple[ExperimentResult, dict[str, Any]]:
        from aika.agent_runner import should_refuse_metric_answer
        from aika.agent_tools import AgentToolCall
        from aika.agents.coverage import metric_gap_answer
        from aika.langgraph_runner import LangGraphAgentRunner, experiment_plain_value

        if self.engine is None:
            raise ConfigurationMismatch("E/F require a configured LangGraph QA engine")
        engine = copy.copy(self.engine)
        engine.llm_client = observed_client(self.llm_client, recorder, self.config)
        engine.semantic_index = self._semantic(recorder)
        engine.experiment_wait_for_workers = True
        generation_state = {}

        def generate(state: Any) -> dict[str, Any]:
            from aika.rd_evidence import matrix as rd_matrix
            generation_state.update(state)
            refused = should_refuse_metric_answer(state["coverage_report"])
            if refused:
                selected, context = budget_context(list(state["evidence_cards"]), self.config.context_max_chars)
                context.update(sent_to_model=False, user_prompt=None)
                answer, reasoning = metric_gap_answer(state["coverage_report"]), ""
            else:
                answer, reasoning, selected, context = self._generate(
                    question, list(state["evidence_cards"]), state["llm_budget"].client("answer"),
                )
                context["sent_to_model"] = True
            context['rd_evidence_matrix'] = rd_matrix(state['plan'], selected)
            return {
                "answer": answer, "reasoning_content": reasoning, "evidence_cards": selected,
                "generation_context": context, "refused_metric": refused,
                "answer_call": AgentToolCall(tool="generate_answer", args={"refused": refused, "evidence_cards": len(selected)}, result_count=1),
            }

        started = time.perf_counter()
        try:
            output = LangGraphAgentRunner(
                engine, max_steps=engine.agent_max_steps, verify_final_answer=False, answer_generator=generate,
            ).run(question, conversation_history=[], thinking_enabled=getattr(self.llm_client, "thinking_enabled", False),
                  reasoning_effort=getattr(self.llm_client, "reasoning_effort", ""))
        except Exception as exc:
            state = generation_state
            retrieval = state.get("retrieval_state")
            materials = {key: experiment_plain_value(getattr(retrieval, key, []))
                         for key in ("graph_records", "rag_hits", "research_hits", "semantic_hits")}
            materials["retrieval_rounds"] = state.get("retrieval_outputs", [])
            trace = [step.to_dict() for step in state.get("trace", [])]
            partial = ExperimentResult(
                "E", "failed", question, "", "", materials, experiment_plain_value(state.get("evidence_cards", [])),
                {}, [call for step in trace for call in step["tool_calls"]], [],
                {"total": round((time.perf_counter() - started) * 1000, 2)}, [],
            )
            raise ExperimentExecutionError(exc, partial) from exc
        diagnostics = output["diagnostics"]
        calls = [call for step in output["candidate_context"]["trace"] for call in step["tool_calls"]]
        retrieved = {key: output.get(key, []) for key in ("graph_records", "rag_hits", "research_hits")}
        # Include semantic hits and every candidate/rejection, not only summarized evidence.
        retrieved["multi_agent"] = diagnostics.get("multi_agent", {})
        retrieved["raw_cards"] = output["candidate_context"]["raw_cards"]
        retrieved["semantic_hits"] = output["candidate_context"]["semantic_hits"]
        retrieved["retrieval_rounds"] = output["candidate_context"]["retrieval_rounds"]
        snapshot = {
            "question": question, "contextual_question": output["contextual_question"], "plan": output["plan"],
            "task_plan": diagnostics["agent_plan"], "coverage": diagnostics["agent_coverage"],
            "retrieved_materials": retrieved, "evidence": output["evidence_cards"],
            "raw_answer": output["answer"], "generation_context": output["candidate_context"]["generation_context"],
            "refused_metric": output["candidate_context"]["refused_metric"],
        }
        result = ExperimentResult(
            "E", "success", question, output["answer"], output["answer"], retrieved,
            output["evidence_cards"], snapshot["generation_context"], calls, recorder.calls(),
            diagnostics["timings_ms"], list(output.get("errors") or []),
            pair_id=uuid.uuid4().hex, candidate_sha256=digest(snapshot),
            verification=output["verification"], diagnostics=diagnostics,
        )
        multi = diagnostics.get("multi_agent", {})
        budget = multi.get("llm", {})
        if (budget.get("fallback_count") or multi.get("summary", {}).get("degraded") or multi.get("review", {}).get("degraded")
                or diagnostics.get("agent_stop_reason") == "llm_budget_insufficient"):
            result.errors.append("Agent used a fallback or exhausted its LLM budget")
            result.status = "configuration_mismatch"
        self._status(result)
        return result, snapshot

    def _check(self, candidate: ExperimentResult, snapshot: dict[str, Any]) -> ExperimentResult:
        from aika.agents.verification import build_evidence_limited_answer, verify_answer_support
        from aika.question_planner import QuestionPlan

        started = time.perf_counter()
        if digest(snapshot) != candidate.candidate_sha256:
            raise ConfigurationMismatch("Candidate snapshot hash mismatch")
        result = copy.deepcopy(candidate)
        result.variant = "F"
        plan = QuestionPlan(**snapshot["plan"])
        allowed = {field.name for field in fields(EvidenceCard)}
        def cards(rows: list[dict[str, Any]]) -> list[EvidenceCard]:
            return [EvidenceCard(**{key: value for key, value in row.items() if key in allowed}) for row in rows]
        evidence = cards(snapshot["evidence"])
        raw = cards(snapshot["retrieved_materials"]["raw_cards"])
        def verify(answer: str) -> dict[str, Any]:
            check_start = time.perf_counter()
            verification = verify_answer_support(answer, plan, evidence, raw, question=snapshot["contextual_question"])
            result.tool_calls.append({"tool": "verify_answer_support", "args": {}, "result_count": 1,
                                      "elapsed_ms": round((time.perf_counter() - check_start) * 1000, 2)})
            return verification
        result.verification = verify(result.raw_answer)
        result.diagnostics = {"shared_generation": copy.deepcopy(candidate.diagnostics), "initial_verification": copy.deepcopy(result.verification)}
        if result.verification["status"] == "fail" and not snapshot["refused_metric"]:
            coverage = SimpleNamespace(gaps=snapshot["coverage"].get("gaps", []))
            result.final_answer = build_evidence_limited_answer(plan, evidence, result.verification, coverage, answer=result.raw_answer)
            result.tool_calls.append({"tool": "build_evidence_limited_answer", "args": {"reason": "verification_failed"}, "result_count": 1})
            result.verification = verify(result.final_answer)
        incremental = round((time.perf_counter() - started) * 1000, 2)
        result.timings_ms = {**candidate.timings_ms, "shared_generation": candidate.timings_ms["total"],
                             "check_increment": incremental, "total": round(candidate.timings_ms["total"] + incremental, 2)}
        return result

    def run(
        self, question: str, *, variants: Any = tuple(Variant),
        save_candidate: Callable[[dict[str, Any]], None] | None = None,
        save_result: Callable[[dict[str, Any]], None] | None = None,
        candidate_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        selected = normalize_variants(variants)
        results = []
        candidate = None
        snapshot = None
        if candidate_payload is not None:
            snapshot = copy.deepcopy(candidate_payload["candidate"])
            if snapshot["question"] != question or digest(snapshot) != candidate_payload["sha256"]:
                raise ConfigurationMismatch("Saved E/F candidate does not match the question or content hash")
            names = {item.name for item in fields(ExperimentResult)}
            candidate = ExperimentResult(**{key: value for key, value in candidate_payload["result"].items() if key in names})
            if candidate.candidate_sha256 != digest(snapshot) or candidate.pair_id != candidate_payload["pair_id"]:
                raise ConfigurationMismatch("Saved E/F result does not match its candidate")
            if (candidate.variant != "E" or candidate.raw_answer != snapshot["raw_answer"]
                    or candidate.final_answer != snapshot["raw_answer"] or candidate.question != snapshot["question"]
                    or candidate.evidence != snapshot["evidence"] or candidate.retrieved_materials != snapshot["retrieved_materials"]
                    or candidate.generation_context != snapshot["generation_context"]):
                raise ConfigurationMismatch("Saved E result fields differ from the hashed candidate")
        shared_failure: Exception | None = None
        shared_failed_result: ExperimentResult | None = None
        failed_pair_id = uuid.uuid4().hex
        for variant in selected:
            recorder = self._recorder()
            started = time.perf_counter()
            try:
                if variant in (Variant.A, Variant.B):
                    result = self._simple(question, variant, recorder)
                else:
                    if shared_failure is not None:
                        raise RuntimeError(f"Shared E generation failed; F was not run: {shared_failure}")
                    if candidate is None:
                        try:
                            candidate, snapshot = self._candidate(question, recorder)
                        except Exception as exc:
                            shared_failure = exc
                            raise
                        if save_candidate is not None:
                            try:
                                save_candidate({"pair_id": candidate.pair_id, "sha256": candidate.candidate_sha256,
                                                "candidate": snapshot, "result": candidate.to_dict()})
                            except Exception as exc:
                                raise ArtifactWriteError(f"Cannot save shared candidate: {exc}") from exc
                    result = candidate if variant == Variant.E else self._check(candidate, copy.deepcopy(snapshot))
            except Exception as exc:
                if isinstance(exc, ArtifactWriteError):
                    raise
                result = ExperimentResult(
                    variant.value, "configuration_mismatch" if isinstance(exc, ConfigurationMismatch) else "failed",
                    question, "", "", {}, [], {}, [], recorder.calls(),
                    {"total": round((time.perf_counter() - started) * 1000, 2)}, [str(exc)],
                )
                if isinstance(exc, ExperimentExecutionError):
                    result = copy.deepcopy(exc.partial)
                    result.variant = variant.value
                    result.errors = [str(exc)]
                    result.model_calls = recorder.calls()
                    result.status = "configuration_mismatch" if isinstance(exc.cause, ConfigurationMismatch) else "failed"
                if variant == Variant.F and candidate is not None:
                    result.raw_answer = candidate.raw_answer
                    result.pair_id = candidate.pair_id
                    result.candidate_sha256 = candidate.candidate_sha256
                    result.retrieved_materials = copy.deepcopy(candidate.retrieved_materials)
                    result.evidence = copy.deepcopy(candidate.evidence)
                    result.generation_context = copy.deepcopy(candidate.generation_context)
                    result.model_calls = copy.deepcopy(candidate.model_calls)
                    result.timings_ms["check_increment"] = result.timings_ms["total"]
                    result.timings_ms["shared_generation"] = candidate.timings_ms["total"]
                    result.timings_ms["total"] += candidate.timings_ms["total"]
                elif variant in (Variant.E, Variant.F):
                    result.pair_id = failed_pair_id
                    if shared_failed_result is not None:
                        result.model_calls = copy.deepcopy(shared_failed_result.model_calls)
                        result.timings_ms = copy.deepcopy(shared_failed_result.timings_ms)
                        result.retrieved_materials = copy.deepcopy(shared_failed_result.retrieved_materials)
                        result.evidence = copy.deepcopy(shared_failed_result.evidence)
                        result.generation_context = copy.deepcopy(shared_failed_result.generation_context)
                        result.tool_calls = copy.deepcopy(shared_failed_result.tool_calls)
                    else:
                        shared_failed_result = copy.deepcopy(result)
            self._status(result)
            payload = result.to_dict()
            results.append(payload)
            if save_result is not None:
                save_result(payload)
        return results
