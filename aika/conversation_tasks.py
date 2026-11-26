"""Detached, single-process conversation execution with durable event replay."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from typing import Any, Callable

from aika.agents.roles import ROLE_BY_ID, lead_question, resolve_role, role_context, role_step
from aika.conversation_store import ConversationStore, ConversationConflictError, now_iso


def empty_result(question: str, answer: str = "") -> dict[str, Any]:
    return {"question": question, "contextual_question": question, "answer": answer,
            "reasoning_content": "", "answer_type": "task", "plan": {}, "cypher": "",
            "cypher_params": {}, "cypher_source": "", "graph_records": [], "rag_hits": [],
            "evidence_cards": [], "evidence": [], "subgraph": [], "diagnostics": {}, "errors": []}


def knowledge_card(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """Persist metadata only, including PostgreSQL timestamps as ISO strings."""
    if not document:
        return None
    fields = {"document_id", "title", "source_type", "source_url", "filename", "status", "error",
              "semantic_status", "active_version", "pending_version", "version", "created_at", "updated_at", "fetched_at"}
    return {key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in document.items() if key in fields}


class PublicTaskError(RuntimeError):
    """An expected, safe-to-display task failure."""


class ConversationTaskRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workers: dict[tuple[str, str, str], tuple[threading.Thread, ConversationStore]] = {}
        self._closed = False

    def submit(self, store: ConversationStore, conversation_id: str, request: dict[str, Any],
               engine: Any, knowledge_handler: Callable, redact: Callable) -> dict[str, Any]:
        question, lead = resolve_role(request["question"].strip(), request.get("role_id"))
        if not question:
            raise ValueError("Question cannot be empty")
        signature = hashlib.sha256(json.dumps({**request, "question": question, "role_id": lead, "request_id": None}, sort_keys=True).encode()).hexdigest()
        turn = {"created_at": now_iso(), "task_id": uuid.uuid4().hex,
                "request_id": request.get("request_id") or uuid.uuid4().hex,
                "request_signature": signature, "question": question, "answer": "",
                "status": "pending", "lead_role_id": lead, "events": [],
                "thinking_enabled": request["thinking_enabled"], "reasoning_effort": request["reasoning_effort"],
                "result": empty_result(question)}
        with self._lock:
            if self._closed:
                raise ConversationConflictError("服务正在关闭，请稍后重试")
            turn, created = store.begin_task(conversation_id, turn)
            if created:
                # Persist the turn before any background work or browser subscription.
                key = (str(store.directory.resolve()), conversation_id, turn["task_id"])
                worker = threading.Thread(target=self._execute, args=(store, conversation_id, turn, engine, knowledge_handler, redact, key), daemon=True, name="conversation-task")
                self._workers[key] = worker, store
                worker.start()
        return turn

    def _execute(self, store, conversation_id, turn, engine, knowledge_handler, redact, key):
        task_id = turn["task_id"]
        pending_delta: list[str] = []
        last_flush = time.monotonic()
        sink_lock = threading.RLock()

        def flush():
            nonlocal last_flush
            if pending_delta:
                store.update_task(conversation_id, task_id, event={"type": "answer_delta", "content": "".join(pending_delta)})
                pending_delta.clear()
            last_flush = time.monotonic()

        def sink(event):
            # Multiple real retrieval workers share this sink, never its context state.
            with sink_lock:
                if event["type"] == "answer_delta":
                    pending_delta.append(event.get("content", ""))
                    if time.monotonic() - last_flush >= 0.25:
                        flush()
                else:
                    flush()
                    store.update_task(conversation_id, task_id, event=event)

        try:
            if store.update_task(conversation_id, task_id, status="running")["status"] == "failed":
                return
            sink({"type": "task", "status": "running", "lead_role_id": turn["lead_role_id"]})
            question = turn["question"]
            with role_context(sink, turn["lead_role_id"], fallback_tools=getattr(engine, "agent_runner", "") == "legacy"):
                # Management intent is classified before executing any write.
                from aika.api import is_knowledge_command
                management = is_knowledge_command(question)
                incompatible = (turn["lead_role_id"] == "librarian" and not management) or (turn["lead_role_id"] not in {"chief", "librarian"} and management)
                if incompatible:
                    suggested = "资料管理员" if management else "首席研究员或产业研究员"
                    answer = f"该任务不适合由{ROLE_BY_ID[turn['lead_role_id']]['name']}牵头，建议选择{suggested}。"
                    with role_step(turn["lead_role_id"], "检查角色适用范围") as outcome:
                        outcome.update(status="skipped", summary=answer)
                    result = empty_result(question, answer)
                elif management:
                    with role_step("librarian", "处理资料管理请求") as outcome:
                        result = knowledge_handler(question)
                        document = knowledge_card(result.get("knowledge_document"))
                        result["knowledge_document"] = document
                        if document and document.get("document_id"):
                            initial_status = document.get("status")
                            sink({"type": "role_event", "role_id": "librarian", "step_id": "knowledge-document",
                                  "status": "running" if initial_status in {"received", "processing"} else "failed" if initial_status == "failed" else "completed",
                                  "summary": "资料处理状态：" + {"received": "已接收", "processing": "处理中", "ready": "可检索", "failed": "失败"}.get(initial_status, str(initial_status)),
                                  "document": document})
                        if document and document.get("document_id") and document.get("status") in {"received", "processing"}:
                            from aika.api import knowledge_base
                            service = knowledge_base()
                            expected_version = document.get("pending_version") or document.get("active_version")
                            previous = None
                            while document.get("status") in {"received", "processing"}:
                                current = (document.get("status"), document.get("semantic_status"))
                                if current != previous:
                                    sink({"type": "role_event", "role_id": "librarian", "step_id": "knowledge-index", "status": "running",
                                          "summary": "资料处理中，尚未可检索", "document": document})
                                    previous = current
                                if store.task(conversation_id, task_id)["status"] == "failed":
                                    return
                                time.sleep(0.2)
                                document = knowledge_card(service.get(document["document_id"]))
                            if document.get("status") == "ready" and document.get("active_version") != expected_version:
                                raise PublicTaskError("资料版本已变化，请查看最新处理状态")
                            if previous is not None:
                                sink({"type": "role_event", "role_id": "librarian", "step_id": "knowledge-index", "status": "failed" if document.get("status") == "failed" else "completed",
                                      "summary": document.get("error") or "资料索引处理结束", "document": document})
                        if document and document.get("document_id") and document.get("status") not in {"received", "processing"}:
                            sink({"type": "role_event", "role_id": "librarian", "step_id": "knowledge-document",
                                  "status": "failed" if document.get("status") == "failed" else "completed",
                                  "summary": document.get("error") or "资料状态已更新", "document": document})
                        if document:
                            result["knowledge_document"] = document
                            if document.get("status") == "failed":
                                raise PublicTaskError(document.get("error") or "资料处理失败")
                            if document.get("status") == "ready":
                                semantic_label = {"ready": "已完成", "not_configured": "未配置，使用关键词检索", "failed": "失败，仍可使用关键词检索", "stale": "待更新"}.get(document.get("semantic_status"), "未就绪")
                                result["answer"] = f"资料《{document['title']}》已可检索。语义索引：{semantic_label}。"
                            outcome["document"] = document
                            if document.get("status") == "ready" and document.get("semantic_status") != "ready":
                                outcome.update(status="degraded", summary="关键词检索已可用；语义索引未就绪，请查看资料卡")
                        sink({"type": "answer_delta", "content": result["answer"]})
                else:
                    # Apply deletion redaction before sending stored evidence to an external model.
                    from aika.conversation_store import conversation_messages_from_turns
                    history = conversation_messages_from_turns(redact(store.get(conversation_id))["turns"])
                    kwargs = {"conversation_history": history, "thinking_enabled": turn["thinking_enabled"], "reasoning_effort": turn["reasoning_effort"] or None}
                    with role_step(turn["lead_role_id"], "牵头完成研究任务") as lead_outcome:
                        if hasattr(engine, "answer_question_stream"):
                            result = None
                            for event in engine.answer_question_stream(question, **kwargs):
                                if event["type"] == "final":
                                    result = event["result"]
                                elif event["type"] in {"progress", "answer_delta", "role_event"}:
                                    sink(event)
                            if result is None:
                                raise RuntimeError("未生成最终研究结果")
                        else:
                            with role_step(turn["lead_role_id"], "生成研究回答"):
                                result = engine.answer_question(lead_question(question), **kwargs)
                            sink({"type": "answer_delta", "content": result["answer"]})
                        lead_outcome.update(summary="研究任务已完成，详见最终回答", references=result.get("evidence_cards", []))
                failed = {event["role_id"]: event for event in store.task(conversation_id, task_id).get("events", [])
                          if event["type"] == "role_event" and event["status"] == "failed"}
                if failed:
                    limitation = "；".join(f"{ROLE_BY_ID[role]['name']}：{event['summary']}" for role, event in failed.items())
                    result["answer"] += "\n\n本轮协作限制：" + limitation
                    result.setdefault("diagnostics", {})["collaboration_limitations"] = limitation
                result.setdefault("diagnostics", {}).update(thinking_enabled=turn["thinking_enabled"], reasoning_effort=turn["reasoning_effort"])
                with sink_lock:
                    flush()
                    store.update_task(conversation_id, task_id, answer=result["answer"], result=result, status="completed")
        except Exception as exc:
            # Exception details can contain source text or secrets; use safe public messages.
            from fastapi import HTTPException
            message = str(exc.detail) if isinstance(exc, HTTPException) else "任务执行失败，请检查服务配置或重试"
            if isinstance(exc, PublicTaskError):
                message = str(exc)
            if isinstance(exc, (ValueError, KeyError)):
                message = "资料不存在或输入无效，请检查后重试"
            try:
                store.fail_task(conversation_id, task_id, message)
            except (OSError, KeyError):
                pass
        finally:
            with self._lock:
                self._workers.pop(key, None)

    def reopen(self):
        with self._lock:
            self._closed = False

    def shutdown(self):
        with self._lock:
            self._closed = True
            for (_, conversation_id, task_id), (_, store) in list(self._workers.items()):
                store.fail_task(conversation_id, task_id, "服务停止导致任务中断，请重新提交")


runtime = ConversationTaskRuntime()
