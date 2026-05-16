"""Lightweight local RAG index over parsed report chunks."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.extraction_schema import load_jsonl, write_jsonl
from src.text_cleaner import CHUNKS_DIR


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAG_DIR = ROOT_DIR / "data" / "rag"
DOCUMENTS_FILE = "documents.jsonl"
METADATA_FILE = "metadata.json"
MAX_TEXT_CHARS = 1800


@dataclass(frozen=True)
class RagIndexMetadata:
    index_version: str
    built_at: str
    chunk_count: int
    source_dir: str
    index_dir: str


@dataclass(frozen=True)
class RagDocument:
    chunk_id: str
    report_id: str
    kind: str
    company: str
    source_title: str
    source_url: str
    source_tier: str
    source_type: str
    page: str
    section: str
    text: str
    token_counts: dict[str, int]
    token_count: int


@dataclass(frozen=True)
class RagHit:
    chunk_id: str
    report_id: str
    source_title: str
    source_tier: str
    source_type: str
    page: str
    section: str
    company: str
    text: str
    snippet: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(value: str) -> str:
    value = str(value or "").casefold()
    value = re.sub(r"\s+", "", value)
    return value


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English financial text without external deps."""
    text = str(text or "").casefold()
    tokens: list[str] = []
    tokens.extend(re.findall(r"[a-z0-9][a-z0-9_\-+.]{1,}", text))
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for run in cjk_runs:
        max_n = 4 if len(run) >= 4 else len(run)
        for n in range(2, max_n + 1):
            tokens.extend(run[index : index + n] for index in range(0, len(run) - n + 1))
    return tokens


def iter_chunk_records(chunks_dir: Path = CHUNKS_DIR) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not chunks_dir.exists():
        return records
    for path in sorted(chunks_dir.glob("*.jsonl")):
        records.extend(load_jsonl(path))
    return records


def document_from_chunk(chunk: dict[str, Any]) -> RagDocument | None:
    text = str(chunk.get("text") or "").strip()
    if not text:
        return None
    search_text = "\n".join(
        str(chunk.get(key, "") or "")
        for key in ("company", "source_title", "source_type", "section", "text")
    )
    counts = Counter(tokenize(search_text))
    if not counts:
        return None
    return RagDocument(
        chunk_id=str(chunk.get("chunk_id", "")),
        report_id=str(chunk.get("report_id", "")),
        kind=str(chunk.get("kind", "")),
        company=str(chunk.get("company", "")),
        source_title=str(chunk.get("source_title", "")),
        source_url=str(chunk.get("source_url", "")),
        source_tier=str(chunk.get("source_tier", "")),
        source_type=str(chunk.get("source_type", "")),
        page=str(chunk.get("page", "")),
        section=str(chunk.get("section", "")),
        text=text[:MAX_TEXT_CHARS],
        token_counts=dict(counts),
        token_count=sum(counts.values()),
    )


def build_rag_index(
    chunks_dir: Path = CHUNKS_DIR,
    output_dir: Path = DEFAULT_RAG_DIR,
) -> RagIndexMetadata:
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = [doc for chunk in iter_chunk_records(chunks_dir) if (doc := document_from_chunk(chunk))]
    write_jsonl(output_dir / DOCUMENTS_FILE, [asdict(document) for document in documents])
    metadata = RagIndexMetadata(
        index_version="lexical-v1",
        built_at=datetime.now(timezone.utc).isoformat(),
        chunk_count=len(documents),
        source_dir=str(chunks_dir),
        index_dir=str(output_dir),
    )
    (output_dir / METADATA_FILE).write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


class LocalRagIndex:
    def __init__(self, documents: list[RagDocument], *, index_dir: Path = DEFAULT_RAG_DIR) -> None:
        self.documents = documents
        self.index_dir = index_dir
        self.doc_freq = self._build_doc_freq(documents)

    @classmethod
    def load(cls, index_dir: Path = DEFAULT_RAG_DIR) -> "LocalRagIndex":
        path = index_dir / DOCUMENTS_FILE
        if not path.exists():
            raise FileNotFoundError(f"RAG index not found: {path}")
        documents = []
        for row in load_jsonl(path):
            documents.append(
                RagDocument(
                    chunk_id=str(row.get("chunk_id", "")),
                    report_id=str(row.get("report_id", "")),
                    kind=str(row.get("kind", "")),
                    company=str(row.get("company", "")),
                    source_title=str(row.get("source_title", "")),
                    source_url=str(row.get("source_url", "")),
                    source_tier=str(row.get("source_tier", "")),
                    source_type=str(row.get("source_type", "")),
                    page=str(row.get("page", "")),
                    section=str(row.get("section", "")),
                    text=str(row.get("text", "")),
                    token_counts={str(k): int(v) for k, v in dict(row.get("token_counts", {})).items()},
                    token_count=int(row.get("token_count") or 0),
                )
            )
        return cls(documents, index_dir=index_dir)

    @staticmethod
    def _build_doc_freq(documents: list[RagDocument]) -> Counter:
        doc_freq: Counter = Counter()
        for document in documents:
            doc_freq.update(document.token_counts.keys())
        return doc_freq

    def search(
        self,
        question: str,
        *,
        top_k: int = 6,
        filters: dict[str, str] | None = None,
    ) -> list[RagHit]:
        query_tokens = Counter(tokenize(question))
        if not query_tokens:
            return []
        candidates = self._filter_documents(filters or {})
        scored = [
            hit
            for document in candidates
            if (hit := self._score_document(document, question, query_tokens)).score > 0
        ]
        scored.sort(key=lambda hit: (-hit.score, hit.source_title, hit.page, hit.chunk_id))
        return scored[:top_k]

    def _filter_documents(self, filters: dict[str, str]) -> list[RagDocument]:
        documents = self.documents
        for key, expected in filters.items():
            if expected:
                documents = [doc for doc in documents if str(getattr(doc, key, "")) == expected]
        return documents

    def _score_document(self, document: RagDocument, question: str, query_tokens: Counter) -> RagHit:
        score = 0.0
        total_docs = max(len(self.documents), 1)
        for token, query_tf in query_tokens.items():
            doc_tf = document.token_counts.get(token, 0)
            if not doc_tf:
                continue
            idf = math.log((total_docs + 1) / (self.doc_freq.get(token, 0) + 1)) + 1.0
            tf = 1.0 + math.log(doc_tf)
            length_norm = math.sqrt(max(document.token_count, 1))
            score += query_tf * tf * idf / length_norm

        question_norm = normalize_text(question)
        searchable_norm = normalize_text(
            "\n".join([document.company, document.source_title, document.section, document.text])
        )
        if question_norm and question_norm in searchable_norm:
            score += 3.0
        for field in (document.company, document.section, document.source_title):
            if field and normalize_text(field) in question_norm:
                score += 0.8

        return RagHit(
            chunk_id=document.chunk_id,
            report_id=document.report_id,
            source_title=document.source_title,
            source_tier=document.source_tier,
            source_type=document.source_type,
            page=document.page,
            section=document.section,
            company=document.company,
            text=document.text,
            snippet=make_snippet(document.text, query_tokens.keys()),
            score=round(score, 6),
        )


def make_snippet(text: str, tokens: Any, *, radius: int = 110) -> str:
    text = str(text or "").strip()
    if len(text) <= radius * 2:
        return text
    normalized_tokens = [token for token in tokens if len(str(token)) >= 2]
    best_index = -1
    for token in sorted(normalized_tokens, key=len, reverse=True):
        best_index = text.casefold().find(str(token).casefold())
        if best_index >= 0:
            break
    if best_index < 0:
        return text[: radius * 2].strip()
    start = max(0, best_index - radius)
    end = min(len(text), best_index + radius)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"
