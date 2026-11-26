"""Behavioral coverage for durable role execution and reconnects (no network models)."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from aika import api
from aika.agents.roles import resolve_role, role_context
from aika.conversation_store import ConversationStore, ConversationConflictError, conversation_messages_from_turns
from aika.conversation_tasks import ConversationTaskRuntime, empty_result
from aika.qa_engine import QAEngine
from tests.test_api import FakeEngine, make_test_client
from tests.test_centralized_multi_agent import SingleRagIndex


@pytest.fixture
def anyio_backend():
    return "asyncio"


def wait_for(store, conversation_id, task_id, predicate=lambda t: t['status'] not in {'pending', 'running'}):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        turn = store.task(conversation_id, task_id)
        if predicate(turn):
            return turn
        time.sleep(0.01)
    pytest.fail("task did not reach the expected durable state")


def request(question="液冷是什么", **kwargs):
    return {"question": question, "thinking_enabled": False, "reasoning_effort": "", **kwargs}


def test_role_resolution_precedence_and_validation():
    assert resolve_role('@研报分析师 请总结报告', None) == ('请总结报告', 'analyst')
    assert resolve_role('@研报分析师 请比较公司', 'industry') == ('请比较公司', 'industry')
    with pytest.raises(ValueError):
        resolve_role('@任意Agent 执行代码', None)
    with pytest.raises(ValueError):
        resolve_role('你好', 'custom')


def test_real_roles_execute_without_thinking_and_keep_source_locations():
    events = []
    engine = QAEngine(rag_index=SingleRagIndex(), llm_client=None)
    with role_context(events.append):
        result = next(event['result'] for event in engine.answer_question_stream('浪潮信息有哪些AI服务器产品？', thinking_enabled=False) if event['type'] == 'final')
    assert {'chief', 'analyst', 'reviewer'} <= {e['role_id'] for e in events}
    assert 'librarian' not in {e['role_id'] for e in events}
    retrieved = next(e for e in events if e.get('tool') == 'rag' and e['status'] in {'completed', 'degraded'})
    assert retrieved['references'][0]['page'] == '8'
    assert retrieved['references'][0]['evidence'] == '浪潮信息布局AI服务器。'
    assert result['evidence_cards'][0]['citation_id']
    assert all(e['type'] == 'role_event' for e in events)


@pytest.mark.parametrize('agent_enabled,runner', [(True, 'legacy'), (False, 'langgraph')])
def test_compatibility_execution_emits_only_real_steps(agent_enabled, runner):
    engine = QAEngine(rag_index=SingleRagIndex(), llm_client=None, enable_agent=agent_enabled, agent_runner=runner)
    runtime = ConversationTaskRuntime()
    # Check instrumentation using actual fallback paths, not a synthetic trace adapter.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        store = ConversationStore(directory)
        cid = store.create()['id']
        turn = runtime.submit(store, cid, request('浪潮信息有哪些AI服务器产品？'), engine, lambda q: None, lambda c: c)
        saved = wait_for(store, cid, turn['task_id'])
        assert saved['status'] == 'completed', saved.get('error')
        roles = {e['role_id'] for e in saved['events'] if e['type'] == 'role_event'}
        assert {'chief', 'analyst', 'reviewer'} <= roles
        assert 'librarian' not in roles


def test_parallel_completion_events_follow_execution_order(monkeypatch):
    from aika.agents.centralized import CentralizedCoordinator, AgentAssignment, AgentResult, CentralLLMBudget
    from aika.question_planner import heuristic_plan_question
    graph_started, rag_completed = threading.Event(), threading.Event()
    events = []
    coordinator = CentralizedCoordinator(SimpleNamespace(multi_agent_max_workers=2), llm_budget=CentralLLMBudget(None, max_calls=0), llm_options={})
    def execute(assignment, plan, hyde):
        if assignment.agent_type == 'graph':
            graph_started.set()
            assert rag_completed.wait(2)
            return AgentResult(assignment, 'completed', {'records': []})
        assert graph_started.wait(2)
        return AgentResult(assignment, 'completed', [])
    def sink(event):
        events.append(event)
        if event.get('tool') == 'rag' and event['status'] == 'completed':
            rag_completed.set()
    monkeypatch.setattr(coordinator, '_run_assignment', execute)
    plan = heuristic_plan_question('比较浪潮信息与中科曙光')
    with role_context(sink):
        coordinator.execute_wave([AgentAssignment('graph', 'graph', 'q'), AgentAssignment('rag', 'rag', 'q')], plan=plan)
    completed = [e['tool'] for e in events if e['status'] == 'completed']
    assert completed == ['rag', 'graph']


def test_timeout_cannot_be_overwritten_by_late_worker(tmp_path, monkeypatch):
    from aika.agents.centralized import CentralizedCoordinator, AgentAssignment, AgentResult, CentralLLMBudget
    from aika.question_planner import heuristic_plan_question
    store = ConversationStore(tmp_path)
    cid = store.create()['id']
    turn, _ = store.begin_task(cid, {'task_id': 'task', 'request_id': 'request', 'request_signature': 'same', 'question': 'q', 'status': 'running', 'events': []})
    coordinator = CentralizedCoordinator(SimpleNamespace(multi_agent_max_workers=1), llm_budget=CentralLLMBudget(None, max_calls=0), llm_options={})
    coordinator.timeout_seconds = 0.03
    release, finished = threading.Event(), threading.Event()
    def execute(assignment, plan, hyde):
        assert release.wait(2)
        return AgentResult(assignment, 'completed', [])
    def sink(event):
        store.update_task(cid, 'task', event=event)
        if event['status'] == 'completed':
            finished.set()
    monkeypatch.setattr(coordinator, '_run_assignment', execute)
    with role_context(sink):
        results = coordinator.execute_wave([AgentAssignment('rag', 'rag', 'q')], plan=heuristic_plan_question('q'))
    assert results[0].timed_out
    release.set()
    deadline = time.monotonic() + 2
    while coordinator.diagnostics()['peak_concurrency'] and coordinator._active_workers and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not coordinator._active_workers
    assert not finished.is_set()  # The coordinator suppresses late terminal events too.
    terminal = [e['status'] for e in store.task(cid, 'task')['events'] if e['status'] in {'completed', 'failed'}]
    assert terminal == ['failed']


class GatedEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()
    def answer_question_stream(self, question, **kwargs):
        self.entered.set()
        assert self.release.wait(3)
        yield {'type': 'answer_delta', 'content': 'draft'}
        yield {'type': 'final', 'result': self.answer_question(question, **kwargs)}


def test_detached_job_idempotency_conflicts_and_restart(tmp_path):
    runtime = ConversationTaskRuntime()
    store = ConversationStore(tmp_path)
    cid = store.create()['id']
    engine = GatedEngine()
    turn = runtime.submit(store, cid, request(request_id='same'), engine, lambda q: None, lambda c: c)
    assert engine.entered.wait(1)
    same = runtime.submit(ConversationStore(tmp_path), cid, request(request_id='same'), engine, lambda q: None, lambda c: c)
    assert same['task_id'] == turn['task_id']
    with pytest.raises(ConversationConflictError):
        runtime.submit(store, cid, request('different', request_id='same'), engine, lambda q: None, lambda c: c)
    with pytest.raises(ConversationConflictError):
        runtime.submit(store, cid, request(request_id='new'), engine, lambda q: None, lambda c: c)
    with pytest.raises(ConversationConflictError):
        store.delete(cid)
    assert conversation_messages_from_turns(store.get(cid)['turns']) == []
    # A fresh store can read progress without any browser subscription.
    assert ConversationStore(tmp_path).task(cid, turn['task_id'])['events']
    engine.release.set()
    saved = wait_for(store, cid, turn['task_id'])
    assert saved['status'] == 'completed'
    assert saved['answer'].startswith('回答：')
    assert len(engine.calls) == 1
    assert len(store.get(cid)['turns']) == 1
    # Simulate a durable orphan after a process restart.
    orphan = {**turn, 'task_id': 'orphan', 'request_id': 'orphan', 'status': 'running', 'events': []}
    store.append_turn(cid, orphan)
    store.update_task(cid, 'orphan', event={'type': 'role_event', 'role_id': 'chief', 'step_id': 'work', 'status': 'running', 'summary': 'work'})
    assert ConversationStore(tmp_path).recover_tasks() == 1
    recovered = store.task(cid, 'orphan')
    assert recovered['status'] == 'failed'
    assert '重启' in recovered['error']
    assert recovered['events'][-2]['status'] == 'failed'
    assert store.recover_tasks() == 0


def test_role_incompatibility_does_not_write_or_invoke_engine(tmp_path):
    runtime, engine = ConversationTaskRuntime(), FakeEngine()
    store = ConversationStore(tmp_path)
    cid = store.create()['id']
    def forbidden(q):
        pytest.fail('incompatible role performed a write')
    turn = runtime.submit(store, cid, request('收录这个网址 https://example.com', role_id='analyst'), engine, forbidden, lambda c: c)
    saved = wait_for(store, cid, turn['task_id'])
    assert '资料管理员' in saved['answer']
    assert not engine.calls
    assert [e['status'] for e in saved['events'] if e['type'] == 'role_event'] == ['running', 'skipped']


@pytest.mark.parametrize('final_status,semantic', [('ready', 'ready'), ('ready', 'failed'), ('failed', 'not_configured')])
def test_librarian_tracks_actual_index_outcome(tmp_path, monkeypatch, final_status, semantic):
    runtime, engine = ConversationTaskRuntime(), FakeEngine()
    store = ConversationStore(tmp_path)
    cid = store.create()['id']
    from datetime import datetime, timezone
    doc = {'created_at': datetime.now(timezone.utc), 'raw_text': '不应复制到角色事件的全文', 'document_id': 'a'*32, 'title': '测试资料', 'status': 'received', 'pending_version': 1, 'active_version': None, 'semantic_status': 'not_configured'}
    calls = []
    def get(doc_id):
        calls.append(doc_id)
        return {**doc, 'status': final_status, 'active_version': 1 if final_status == 'ready' else None,
                'semantic_status': semantic, 'error': '解析失败' if final_status == 'failed' else ''}
    monkeypatch.setattr(api, 'knowledge_base', lambda: SimpleNamespace(get=get))
    turn = runtime.submit(store, cid, request('收录 https://example.com'), engine,
                          lambda q: {**empty_result(q, '已接收'), 'knowledge_document': doc}, lambda c: c)
    saved = wait_for(store, cid, turn['task_id'])
    role_events = [e for e in saved['events'] if e['type'] == 'role_event']
    assert {e['role_id'] for e in role_events} == {'librarian'}
    assert calls == ['a'*32]
    if final_status == 'failed':
        assert saved['status'] == 'failed'
        assert '解析失败' in saved['error']
        assert any(e.get('document', {}).get('status') == 'failed' for e in role_events)
    else:
        assert saved['status'] == 'completed'
        assert '已可检索' in saved['answer']
        assert role_events[-1]['status'] == ('completed' if semantic == 'ready' else 'degraded')
    assert '不应复制到角色事件的全文' not in str(saved)
    assert not engine.calls


@pytest.mark.anyio
async def test_api_stream_replay_cursor_role_catalog_and_failed_task(tmp_path):
    client = make_test_client(tmp_path, FakeEngine())
    try:
        roles = (await client.get('/api/agent/roles')).json()['roles']
        assert len(roles) == 5
        cid = (await client.post('/api/conversations', json={})).json()['id']
        response = await client.post(f'/api/conversations/{cid}/messages/stream', json={'question': '@研报分析师 液冷是什么', 'request_id': 'req'})
        assert response.status_code == 200
        assert 'event: role_event' in response.text
        conversation = (await client.get(f'/api/conversations/{cid}')).json()
        turn = conversation['turns'][0]
        assert turn['lead_role_id'] == 'analyst'
        assert turn['question'] == '液冷是什么'
        seqs = [e['seq'] for e in turn['events']]
        assert seqs == list(range(1, len(seqs) + 1))
        replay = await client.get(f'/api/conversations/{cid}/tasks/{turn["task_id"]}/events?after_seq=1')
        assert 'id: 1\n' not in replay.text
        assert 'event: final' in replay.text
        assert (await client.get(f'/api/conversations/{cid}/tasks/{turn["task_id"]}/events?after_seq=999')).status_code == 400
        again = await client.post(f'/api/conversations/{cid}/messages', json={'question': '@研报分析师 液冷是什么', 'request_id': 'req'})
        assert len(again.json()['conversation']['turns']) == 1
        bad = await client.post(f'/api/conversations/{cid}/messages', json={'question': 'q', 'role_id': 'invalid'})
        assert bad.status_code == 400
    finally:
        await client.aclose()
        api.app.dependency_overrides.clear()


def test_deleted_sources_redacted_from_role_results_and_split_deltas(monkeypatch):
    doc_id = 'a'*32
    class Connection:
        def execute(self, *args):
            return SimpleNamespace(fetchall=lambda: [{'document_id': doc_id}])
    @contextmanager
    def connection():
        yield Connection()
    monkeypatch.setattr(api, 'knowledge_base', lambda: SimpleNamespace(store=SimpleNamespace(pool=SimpleNamespace(connection=connection))))
    original = {'turns': [{'answer': '秘密内容', 'result': {'answer': '秘密内容'}, 'events': [
        {'type': 'role_event', 'references': [{'semantic_ref_id': f'knowledge:{doc_id}:1:0', 'evidence': '秘密内容'}]},
        {'type': 'answer_delta', 'content': '秘密'}, {'type': 'answer_delta', 'content': '内容'},
        {'type': 'role_event', 'document': {'document_id': doc_id, 'raw_text': '秘密内容'}}]}]}
    redacted = api.redact_deleted_sources(original)
    assert '秘密' not in str(redacted)
    assert redacted['turns'][0]['events'][0]['references'][0]['source_deleted']
    assert '秘密内容' in str(original)


def test_company_comparison_shows_actual_roles_and_final_citations(tmp_path):
    from tests.test_centralized_multi_agent import SlowGraphClient, ConcurrencyProbe
    runtime, store = ConversationTaskRuntime(), ConversationStore(tmp_path)
    engine = QAEngine(graph_client=SlowGraphClient(ConcurrencyProbe(delay=0)), rag_index=SingleRagIndex(), llm_client=None)
    cid = store.create()['id']
    turn = runtime.submit(store, cid, request('比较浪潮信息和中科曙光的AI服务器业务'), engine, lambda q: None, lambda c: c)
    saved = wait_for(store, cid, turn['task_id'])
    assert saved['status'] == 'completed'
    events = [e for e in saved['events'] if e['type'] == 'role_event']
    assert {'chief', 'analyst', 'industry', 'reviewer'} <= {e['role_id'] for e in events}
    assert any(e['role_id'] == 'chief' and '回答草稿' in e['summary'] for e in events)
    assert saved['result']['evidence_cards']
    assert saved['result']['evidence_cards'][0]['citation_id'] in saved['answer']


def test_partial_failure_stays_visible_in_final_answer(tmp_path):
    from aika.agents.roles import emit_role
    class PartialEngine(FakeEngine):
        def answer_question_stream(self, question, **kwargs):
            emit_role('industry', 'graph', 'running', '图谱查询')
            emit_role('industry', 'graph', 'failed', '图谱暂不可用')
            yield {'type': 'final', 'result': self.answer_question(question, **kwargs)}
    runtime, store = ConversationTaskRuntime(), ConversationStore(tmp_path)
    cid = store.create()['id']
    turn = runtime.submit(store, cid, request(), PartialEngine(), lambda q: None, lambda c: c)
    saved = wait_for(store, cid, turn['task_id'])
    assert saved['status'] == 'completed'
    assert '本轮协作限制' in saved['answer']
    assert '图谱暂不可用' in saved['answer']
    assert [e['status'] for e in saved['events'] if e.get('step_id') == 'graph'] == ['running', 'failed']


def test_knowledge_unavailable_and_deleted_processing_document_fail_truthfully(tmp_path, monkeypatch):
    from fastapi import HTTPException
    runtime, store = ConversationTaskRuntime(), ConversationStore(tmp_path)
    cid = store.create()['id']
    def unavailable(q):
        raise HTTPException(503, '动态知识库需要启用 PostgreSQL 检索服务')
    turn = runtime.submit(store, cid, request('收录 https://example.com'), FakeEngine(), unavailable, lambda c: c)
    saved = wait_for(store, cid, turn['task_id'])
    assert saved['status'] == 'failed'
    assert 'PostgreSQL' in saved['error']
    assert not saved['answer']
    doc = {'document_id': 'a'*32, 'title': '测试资料', 'status': 'processing', 'pending_version': 1}
    def deleted(_):
        raise KeyError('deleted')
    monkeypatch.setattr(api, 'knowledge_base', lambda: SimpleNamespace(get=deleted))
    turn = runtime.submit(store, cid, request('重新处理知识库资料 ' + 'a'*32), FakeEngine(),
                          lambda q: {**empty_result(q), 'knowledge_document': doc}, lambda c: c)
    saved = wait_for(store, cid, turn['task_id'])
    assert saved['status'] == 'failed'
    assert '可检索' not in saved['answer']
    assert any(e.get('document', {}).get('document_id') == 'a'*32 for e in saved['events'])


def test_selected_role_changes_answer_focus_but_not_retrieval_query():
    from tests.test_centralized_multi_agent import StructuredMultiAgentLLM
    class Capture(StructuredMultiAgentLLM):
        def chat_text(self, *, user_prompt, **kwargs):
            self.messages = user_prompt
            return super().chat_text(user_prompt=user_prompt, **kwargs)
        def chat_messages(self, *, messages, **kwargs):
            self.messages = messages
            return super().chat_messages(messages=messages, **kwargs)
    llm, queries = Capture(), []
    class Index(SingleRagIndex):
        def search(self, question, **kwargs):
            queries.append(question)
            return super().search(question, **kwargs)
    engine = QAEngine(rag_index=Index(), llm_client=llm)
    with role_context(lambda e: None, 'analyst'):
        list(engine.answer_question_stream('浪潮信息有哪些AI服务器产品？'))
    assert '由研报分析师牵头' in str(llm.messages)
    assert all('回答要求' not in query for query in queries)
