import json

from src.llm_client import OpenAICompatibleClient


def test_deepseek_payload_enables_thinking_by_default(monkeypatch) -> None:
    monkeypatch.setattr("src.llm_client.load_dotenv", lambda path=None: None)
    monkeypatch.delenv("LLM_THINKING_ENABLED", raising=False)
    monkeypatch.delenv("LLM_ENABLE_THINKING", raising=False)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-pro",
    )

    payload = client._build_payload(messages=[{"role": "user", "content": "hi"}], temperature=0.2)

    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert "temperature" not in payload


def test_thinking_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("LLM_THINKING_ENABLED", "false")
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-pro",
    )

    payload = client._build_payload(messages=[{"role": "user", "content": "hi"}], temperature=0.2)

    assert "thinking" not in payload
    assert "reasoning_effort" not in payload
    assert payload["temperature"] == 0.2


def test_deepseek_chat_disables_thinking_by_default(monkeypatch) -> None:
    monkeypatch.setattr("src.llm_client.load_dotenv", lambda path=None: None)
    monkeypatch.delenv("LLM_THINKING_ENABLED", raising=False)
    monkeypatch.delenv("LLM_ENABLE_THINKING", raising=False)
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )

    payload = client._build_payload(messages=[{"role": "user", "content": "hi"}], temperature=0.2)

    assert "thinking" not in payload
    assert "reasoning_effort" not in payload
    assert payload["temperature"] == 0.2


def test_thinking_can_be_overridden_per_request(monkeypatch) -> None:
    monkeypatch.setenv("LLM_THINKING_ENABLED", "false")
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-pro",
    )

    payload = client._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        thinking_enabled=True,
        reasoning_effort="medium",
    )

    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "medium"
    assert "temperature" not in payload


def test_stream_chat_messages_parses_content_and_reasoning(monkeypatch) -> None:
    posted_payloads = []

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, decode_unicode: bool = False):
            del decode_unicode
            first = {"choices": [{"delta": {"reasoning_content": "分析证据。"}}], "model": "test-model"}
            second = {"choices": [{"delta": {"content": "结论"}}], "model": "test-model"}
            yield f"data: {json.dumps(first)}"
            yield f"data: {json.dumps(second)}"
            yield "data: [DONE]"

    def fake_post(url, *, headers, json, timeout, stream):
        del url, headers, timeout
        posted_payloads.append(json)
        assert stream is True
        return FakeResponse()

    monkeypatch.setattr("src.llm_client.requests.post", fake_post)
    client = OpenAICompatibleClient(api_key="test-key", base_url="https://example.com/v1", model="test-model")

    chunks = list(client.stream_chat_messages(messages=[{"role": "user", "content": "hi"}]))

    assert posted_payloads[0]["stream"] is True
    assert chunks[0].reasoning_content == "分析证据。"
    assert chunks[1].content == "结论"
