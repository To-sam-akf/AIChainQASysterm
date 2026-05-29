"""Local JSONL semantic index over RAG chunks, claims, and dossiers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.extraction_schema import load_jsonl, read_csv, write_jsonl
from src.rag_index import DEFAULT_RAG_DIR, DOCUMENTS_FILE
from src.research_claims import (
    CLAIMS_FILE,
    DEFAULT_RESEARCH_DIR,
    SEGMENT_DOSSIERS_FILE,
    claim_title,
    first_company,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SEMANTIC_DIR = ROOT_DIR / "data" / "semantic_index"
SEMANTIC_DOCUMENTS_FILE = "documents.jsonl"
SEMANTIC_VECTORS_FILE = "vectors.jsonl"
SEMANTIC_METADATA_FILE = "metadata.json"
SEMANTIC_INDEX_VERSION = "semantic-jsonl-v1"
MAX_SEMANTIC_TEXT_CHARS = 2200


@dataclass(frozen=True)
class SemanticDocument:
    doc_id: str
    kind: str
    title: str
    text: str
    source: str = ""
    page: str = ""
    section: str = ""
    topic: str = ""
    company: str = ""
    claim_type: str = ""
    exposure_level: str = ""
    ref_id: str = ""
    source_tier: str = ""
    source_type: str = ""
    confidence: str = ""
    as_of_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticHit:
    doc_id: str
    kind: str
    title: str
    text: str
    score: float
    source: str = ""
    page: str = ""
    section: str = ""
    topic: str = ""
    company: str = ""
    claim_type: str = ""
    exposure_level: str = ""
    ref_id: str = ""
    source_tier: str = ""
    source_type: str = ""
    confidence: str = ""
    as_of_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["semantic_score"] = self.score
        row["semantic_ref_id"] = self.ref_id or self.doc_id
        return row


@dataclass(frozen=True)
class SemanticIndexMetadata:
    index_version: str
    built_at: str
    document_count: int
    vector_count: int
    dimension: int
    embedding_model: str
    rag_dir: str
    research_dir: str
    index_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_semantic_documents(
    *,
    rag_dir: Path = DEFAULT_RAG_DIR,
    research_dir: Path = DEFAULT_RESEARCH_DIR,
    limit: int | None = None,
) -> list[SemanticDocument]:
    documents: list[SemanticDocument] = []
    documents.extend(documents_from_rag(rag_dir))
    documents.extend(documents_from_claims(research_dir))
    documents.extend(documents_from_dossiers(research_dir))
    if limit is not None and limit >= 0:
        return documents[:limit]
    return documents


def documents_from_rag(rag_dir: Path) -> list[SemanticDocument]:
    rows = load_jsonl(rag_dir / DOCUMENTS_FILE)
    documents: list[SemanticDocument] = []
    for index, row in enumerate(rows, start=1):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        chunk_id = str(row.get("chunk_id") or "")
        doc_id = f"rag:{chunk_id or index}"
        title = str(row.get("source_title") or row.get("section") or "RAG 原文片段")
        documents.append(
            SemanticDocument(
                doc_id=doc_id,
                kind="rag",
                title=title,
                text=text[:MAX_SEMANTIC_TEXT_CHARS],
                source=title,
                page=str(row.get("page") or ""),
                section=str(row.get("section") or ""),
                company=str(row.get("company") or ""),
                ref_id=chunk_id or doc_id,
                source_tier=str(row.get("source_tier") or ""),
                source_type=str(row.get("source_type") or ""),
            )
        )
    return documents


def documents_from_claims(research_dir: Path) -> list[SemanticDocument]:
    claims_path = research_dir / CLAIMS_FILE
    if not claims_path.exists():
        return []
    documents: list[SemanticDocument] = []
    for index, row in enumerate(read_csv(claims_path), start=1):
        claim_id = str(row.get("claim_id") or index)
        text = str(row.get("claim_text") or row.get("evidence_span") or "").strip()
        if not text:
            continue
        title = claim_title(row)
        documents.append(
            SemanticDocument(
                doc_id=f"claim:{claim_id}",
                kind="claim",
                title=title,
                text=text[:MAX_SEMANTIC_TEXT_CHARS],
                source=str(row.get("source_title") or ""),
                page=str(row.get("page") or ""),
                section=str(row.get("section") or ""),
                topic=str(row.get("topic") or ""),
                company=first_company(row.get("companies", "")),
                claim_type=str(row.get("claim_type") or ""),
                exposure_level=str(row.get("exposure_level") or ""),
                ref_id=claim_id,
                source_tier=str(row.get("source_tier") or ""),
                confidence=str(row.get("confidence") or ""),
                as_of_date=str(row.get("as_of_date") or ""),
            )
        )
    return documents


def documents_from_dossiers(research_dir: Path) -> list[SemanticDocument]:
    rows = load_jsonl(research_dir / SEGMENT_DOSSIERS_FILE)
    documents: list[SemanticDocument] = []
    for index, row in enumerate(rows, start=1):
        topic = str(row.get("topic") or f"dossier-{index}")
        text = dossier_to_semantic_text(row)
        if not text:
            continue
        documents.append(
            SemanticDocument(
                doc_id=f"dossier:{topic}",
                kind="dossier",
                title=f"{topic} 产业链投研摘要",
                text=text[:MAX_SEMANTIC_TEXT_CHARS],
                topic=topic,
                ref_id=topic,
            )
        )
    return documents


def dossier_to_semantic_text(row: dict[str, Any]) -> str:
    parts = [str(row.get("topic") or ""), str(row.get("summary") or "")]
    for key in (
        "mechanisms",
        "technology_mechanisms",
        "bottlenecks",
        "company_exposure",
        "indicators",
        "leading_indicators",
        "risks",
        "evidence_gaps",
    ):
        value = row.get(key)
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                parts.append(str(sub_key))
                parts.extend(flatten_text_values(sub_value))
        else:
            parts.extend(flatten_text_values(value))
    return "\n".join(part for part in parts if part).strip()


def flatten_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(flatten_text_values(item))
        return output
    if isinstance(value, dict):
        output = []
        for key, item in value.items():
            output.append(str(key))
            output.extend(flatten_text_values(item))
        return output
    return [str(value)]


def semantic_document_text(document: SemanticDocument) -> str:
    parts = [
        document.title,
        document.topic,
        document.company,
        document.claim_type,
        document.exposure_level,
        document.section,
        document.source,
        document.text,
    ]
    return "\n".join(part for part in parts if part)


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def dot_product(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def build_semantic_index(
    *,
    documents: list[SemanticDocument],
    embedding_client: Any,
    output_dir: Path = DEFAULT_SEMANTIC_DIR,
    rag_dir: Path = DEFAULT_RAG_DIR,
    research_dir: Path = DEFAULT_RESEARCH_DIR,
    progress_callback: Any | None = None,
) -> SemanticIndexMetadata:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [semantic_document_text(document) for document in documents]
    try:
        raw_vectors = embedding_client.embed_texts(texts, progress_callback=progress_callback)
    except TypeError:
        raw_vectors = embedding_client.embed_texts(texts)
    vectors = [l2_normalize(vector) for vector in raw_vectors]
    if len(vectors) != len(documents):
        raise ValueError(f"Embedding count mismatch: expected {len(documents)}, got {len(vectors)}")
    dimension = len(vectors[0]) if vectors else 0
    vector_rows = [
        {"doc_id": document.doc_id, "vector": vector}
        for document, vector in zip(documents, vectors)
    ]
    metadata = SemanticIndexMetadata(
        index_version=SEMANTIC_INDEX_VERSION,
        built_at=datetime.now(timezone.utc).isoformat(),
        document_count=len(documents),
        vector_count=len(vectors),
        dimension=dimension,
        embedding_model=str(getattr(embedding_client, "model", "")),
        rag_dir=str(rag_dir),
        research_dir=str(research_dir),
        index_dir=str(output_dir),
    )
    write_jsonl(output_dir / SEMANTIC_DOCUMENTS_FILE, [document.to_dict() for document in documents])
    write_jsonl(output_dir / SEMANTIC_VECTORS_FILE, vector_rows)
    with (output_dir / SEMANTIC_METADATA_FILE).open("w", encoding="utf-8") as file:
        json.dump(metadata.to_dict(), file, ensure_ascii=False, indent=2)
    return metadata


class SemanticIndex:
    def __init__(
        self,
        *,
        documents: list[SemanticDocument],
        vectors: list[list[float]],
        metadata: SemanticIndexMetadata,
        embedding_client: Any | None = None,
    ) -> None:
        self.documents = documents
        self.vectors = vectors
        self.metadata = metadata
        self.embedding_client = embedding_client

    @classmethod
    def load(cls, index_dir: Path = DEFAULT_SEMANTIC_DIR, *, embedding_client: Any | None = None) -> "SemanticIndex":
        metadata_path = index_dir / SEMANTIC_METADATA_FILE
        documents_path = index_dir / SEMANTIC_DOCUMENTS_FILE
        vectors_path = index_dir / SEMANTIC_VECTORS_FILE
        if not metadata_path.exists() or not documents_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(f"Semantic index not found in {index_dir}")

        with metadata_path.open(encoding="utf-8") as file:
            metadata = SemanticIndexMetadata(**json.load(file))
        documents_by_id = {
            str(row.get("doc_id")): SemanticDocument(**row)
            for row in load_jsonl(documents_path)
            if row.get("doc_id")
        }
        documents: list[SemanticDocument] = []
        vectors: list[list[float]] = []
        for row in load_jsonl(vectors_path):
            doc_id = str(row.get("doc_id") or "")
            vector = row.get("vector")
            if doc_id not in documents_by_id or not isinstance(vector, list):
                continue
            documents.append(documents_by_id[doc_id])
            vectors.append([float(value) for value in vector])
        if not documents:
            raise ValueError(f"Semantic index in {index_dir} is empty")
        return cls(documents=documents, vectors=vectors, metadata=metadata, embedding_client=embedding_client)

    def search(self, question: str, *, top_k: int = 8, filters: dict[str, list[str]] | None = None) -> list[SemanticHit]:
        if self.embedding_client is None:
            return []
        query_vectors = self.embedding_client.embed_texts([question])
        if not query_vectors:
            return []
        query_vector = l2_normalize(query_vectors[0])
        scored: list[SemanticHit] = []
        for document, vector in zip(self.documents, self.vectors):
            if filters and not document_matches_filters(document, filters):
                continue
            score = dot_product(query_vector, vector)
            scored.append(hit_from_document(document, score=round(score, 6)))
        scored.sort(key=lambda hit: (-hit.score, hit.kind, hit.topic, hit.company, hit.title))
        return scored[: max(0, top_k)]


def document_matches_filters(document: SemanticDocument, filters: dict[str, list[str]]) -> bool:
    kinds = filters.get("kinds")
    if kinds and document.kind not in kinds:
        return False
    companies = filters.get("companies")
    if companies and document.company and document.company not in companies:
        return False
    topics = filters.get("topics")
    if topics and document.topic and document.topic not in topics:
        return False
    return True


def hit_from_document(document: SemanticDocument, *, score: float) -> SemanticHit:
    return SemanticHit(
        doc_id=document.doc_id,
        kind=document.kind,
        title=document.title,
        text=document.text,
        score=score,
        source=document.source,
        page=document.page,
        section=document.section,
        topic=document.topic,
        company=document.company,
        claim_type=document.claim_type,
        exposure_level=document.exposure_level,
        ref_id=document.ref_id,
        source_tier=document.source_tier,
        source_type=document.source_type,
        confidence=document.confidence,
        as_of_date=document.as_of_date,
    )
