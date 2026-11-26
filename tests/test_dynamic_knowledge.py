from types import SimpleNamespace

import pytest

from aika import api
from aika.knowledge_base import parse_source, fetch_public_url


def test_text_and_pdf_parsing():
    import fitz
    assert parse_source('text', '冷板散热'.encode())[0] == [('', '冷板散热')]
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), 'Cooling report')
        content = pdf.tobytes()
    assert parse_source('pdf', content)[0] == [('1', 'Cooling report')]
    with pytest.raises(ValueError, match='未提取到文本'):
        parse_source('text', b'  ')


def test_webpage_parser_ignores_scripts():
    pages, title = parse_source('url', b'<html><head><title>Report</title><script>bad</script></head><body><p>Cooling</p></body></html>')
    assert title == 'Report'
    assert 'Cooling' in pages[0][1]
    assert 'bad' not in pages[0][1]


def test_fetch_rejects_private_address(monkeypatch):
    monkeypatch.setattr('socket.getaddrinfo', lambda *a, **k: [(2, 1, 6, '', ('127.0.0.1', 80))])
    with pytest.raises(ValueError, match='非公网'):
        fetch_public_url('http://example.com/')


def test_save_command_and_normal_question(monkeypatch):
    created = []
    service = SimpleNamespace(create=lambda *args: created.append(args) or {'title': '新资料'})
    monkeypatch.setattr(api, 'knowledge_base', lambda: service)
    result = api.knowledge_chat_result('把“冷板用于服务器散热”存入知识库')
    assert created == [('text', '冷板用于服务器散热')]
    assert '正在处理' in result['answer']
    assert api.knowledge_chat_result('液冷是什么') is None


def test_deleted_evidence_is_hidden_from_history(monkeypatch):
    from contextlib import contextmanager
    doc_id = 'a' * 32
    class Connection:
        def execute(self, *args):
            return SimpleNamespace(fetchall=lambda: [{'document_id': doc_id}])
    @contextmanager
    def connection():
        yield Connection()
    monkeypatch.setattr(api, 'knowledge_base', lambda: SimpleNamespace(store=SimpleNamespace(pool=SimpleNamespace(connection=connection))))
    original = {'turns': [{'answer': '原文是秘密内容', 'result': {'answer': '原文是秘密内容', 'evidence_cards': [{'semantic_ref_id': f'knowledge:{doc_id}:1:0', 'evidence': '秘密内容'}]}}]}
    redacted = api.redact_deleted_sources(original)
    assert '秘密内容' not in str(redacted)
    assert redacted['turns'][0]['result']['evidence_cards'][0]['source_deleted']
    assert '秘密内容' in str(original)
