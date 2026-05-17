"""Lightweight application-side web search for QA augmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


DUCKDUCKGO_HTML_URL = "https://duckduckgo.com/html/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


@dataclass(frozen=True)
class WebSearchHit:
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class WebSearchResponse:
    hits: list[WebSearchHit]
    error: str = ""


class DuckDuckGoSearchClient:
    """Fetches result snippets from DuckDuckGo's HTML endpoint.

    The client intentionally returns an empty response on failure so QA can
    continue from local graph/RAG evidence when public search is unavailable.
    """

    def __init__(
        self,
        *,
        timeout: float = 5.0,
        top_k: int = 5,
        session: Any | None = None,
    ) -> None:
        self.timeout = timeout
        self.top_k = max(int(top_k), 0)
        self.session = session or requests.Session()

    def search(self, query: str, *, top_k: int | None = None) -> WebSearchResponse:
        query = str(query or "").strip()
        limit = self.top_k if top_k is None else max(int(top_k), 0)
        if not query or limit <= 0:
            return WebSearchResponse(hits=[])
        try:
            response = self.session.get(
                DUCKDUCKGO_HTML_URL,
                params={"q": query},
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return WebSearchResponse(hits=parse_duckduckgo_html(response.text, limit=limit))
        except Exception as exc:  # pragma: no cover - exact requests errors vary by runtime
            return WebSearchResponse(hits=[], error=str(exc))


def parse_duckduckgo_html(html: str, *, limit: int = 5) -> list[WebSearchHit]:
    soup = BeautifulSoup(html or "", "html.parser")
    hits: list[WebSearchHit] = []
    seen_urls: set[str] = set()

    result_nodes = list(soup.select(".result"))
    if not result_nodes:
        result_nodes = [node.find_parent() or node for node in soup.select("a.result__a")]

    for result in result_nodes:
        link = result.select_one("a.result__a") or result.select_one("a[href]")
        if link is None:
            continue
        title = " ".join(link.get_text(" ", strip=True).split())
        url = normalize_duckduckgo_url(str(link.get("href") or ""))
        if not title or not url or url in seen_urls:
            continue
        snippet_node = result.select_one(".result__snippet") or result.select_one(".result__body")
        snippet = ""
        if snippet_node is not None:
            snippet = " ".join(snippet_node.get_text(" ", strip=True).split())
        hits.append(WebSearchHit(title=title, url=url, snippet=snippet))
        seen_urls.add(url)
        if len(hits) >= limit:
            break
    return hits


def normalize_duckduckgo_url(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("/"):
        url = urljoin("https://duckduckgo.com", url)

    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url
