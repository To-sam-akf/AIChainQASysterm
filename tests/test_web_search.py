from __future__ import annotations

from src.web_search import DuckDuckGoSearchClient, parse_duckduckgo_html


def test_parse_duckduckgo_html_extracts_results_and_decodes_redirects() -> None:
    html = """
    <html>
      <body>
        <div class="result">
          <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fai%3Fx%3D1">AI 算力新闻</a>
          <a class="result__snippet">公开资料显示，AI 服务器需求增长。</a>
        </div>
        <div class="result">
          <a class="result__a" href="//example.org/report">液冷产业链</a>
          <div class="result__snippet">液冷、光模块和交换机是重要环节。</div>
        </div>
      </body>
    </html>
    """

    hits = parse_duckduckgo_html(html, limit=5)

    assert [hit.title for hit in hits] == ["AI 算力新闻", "液冷产业链"]
    assert hits[0].url == "https://example.com/ai?x=1"
    assert hits[1].url == "https://example.org/report"
    assert "AI 服务器" in hits[0].snippet


def test_duckduckgo_client_returns_empty_response_on_request_failure() -> None:
    class FailingSession:
        def get(self, *args, **kwargs):
            del args, kwargs
            raise TimeoutError("timeout")

    client = DuckDuckGoSearchClient(session=FailingSession(), timeout=0.01, top_k=3)

    response = client.search("AI 算力")

    assert response.hits == []
    assert "timeout" in response.error


def test_duckduckgo_client_parses_successful_response() -> None:
    class Response:
        text = '<div class="result"><a class="result__a" href="https://example.com">标题</a></div>'

        def raise_for_status(self) -> None:
            return None

    class Session:
        def get(self, *args, **kwargs):
            del args, kwargs
            return Response()

    client = DuckDuckGoSearchClient(session=Session(), top_k=1)

    response = client.search("AI 算力")

    assert len(response.hits) == 1
    assert response.hits[0].url == "https://example.com"
