"""PostgreSQL-backed RAG, research-memory, and semantic retrieval."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from aika.rag_index import (
    RagDocument,
    RagHit,
    dedupe_hits,
    expand_query,
    is_low_value_hit,
    is_technical_question,
    make_snippet,
    normalize_text,
    source_priority,
    tokenize,
)
from aika.research_claims import (
    ResearchHit,
    ResearchMemory,
    apply_claim_reviews,
    build_segment_dossiers,
    claim_title,
    first_company,
    normalize_claim_review,
    normalize_claim_row,
    parse_companies,
    query_topics,
)
from aika.semantic_index import (
    SemanticHit,
    SemanticIndexMetadata,
    dossier_to_semantic_text,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = ROOT_DIR / "db" / "migrations"
SCHEMA_VERSION = "001_postgres_retrieval"
EMBEDDING_DIMENSIONS = 2048


class PostgresRetrievalError(RuntimeError):
    """Raised when the PostgreSQL retrieval backend is unavailable or invalid."""


def require_database_url(value: str | None = None) -> str:
    database_url = str(value or os.getenv("DATABASE_URL") or "").strip()
    if not database_url:
        raise PostgresRetrievalError("DATABASE_URL is required for PostgreSQL retrieval")
    return database_url


def migrate_database(database_url: str | None = None, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply versioned SQL migrations once, in filename order."""

    conninfo = require_database_url(database_url)
    migration_paths = sorted(migrations_dir.glob("*.sql"))
    if not migration_paths:
        raise PostgresRetrievalError(f"No SQL migrations found in {migrations_dir}")
    applied: list[str] = []
    with psycopg.connect(conninfo, autocommit=False) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_schema_migrations (
                version text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        known = {
            str(row[0])
            for row in connection.execute("SELECT version FROM retrieval_schema_migrations").fetchall()
        }
        for path in migration_paths:
            version = path.stem
            if version in known:
                continue
            connection.execute(path.read_text(encoding="utf-8"))
            connection.execute(
                "INSERT INTO retrieval_schema_migrations(version) VALUES (%s)",
                (version,),
            )
            applied.append(version)
        connection.commit()
    return applied


def _configure_connection(connection: psycopg.Connection[Any]) -> None:
    register_vector(connection)


def stable_content_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def corpus_hash(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values if str(value)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def rag_search_text(row: dict[str, Any]) -> str:
    return "\n".join(
        str(row.get(key) or "")
        for key in ("company", "source_title", "source_type", "section", "text")
        if row.get(key)
    )


def rag_semantic_text(row: dict[str, Any]) -> str:
    return "\n".join(
        str(row.get(key) or "")
        for key in ("source_title", "company", "section", "source_type", "text")
        if row.get(key)
    )


def claim_semantic_text(row: dict[str, Any]) -> str:
    companies = parse_companies(row.get("companies", []))
    parts = [
        claim_title(row),
        str(row.get("topic") or ""),
        first_company(companies),
        str(row.get("claim_type") or ""),
        str(row.get("exposure_level") or ""),
        str(row.get("section") or ""),
        str(row.get("source_title") or ""),
        str(row.get("claim_text") or row.get("evidence_span") or ""),
    ]
    return "\n".join(part for part in parts if part)


class PostgresRetrievalStore:
    """Shared synchronous connection pool and retrieval persistence API."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        min_size: int | None = None,
        max_size: int | None = None,
        hnsw_ef_search: int | None = None,
    ) -> None:
        self.database_url = require_database_url(database_url)
        configured_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", str(EMBEDDING_DIMENSIONS)))
        if configured_dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(f"EMBEDDING_DIMENSIONS must be {EMBEDDING_DIMENSIONS}")
        self.min_size = int(os.getenv("DB_POOL_MIN_SIZE", "1") if min_size is None else min_size)
        self.max_size = int(os.getenv("DB_POOL_MAX_SIZE", "8") if max_size is None else max_size)
        self.hnsw_ef_search = int(
            os.getenv("PG_HNSW_EF_SEARCH", "100") if hnsw_ef_search is None else hnsw_ef_search
        )
        if self.min_size < 0 or self.max_size < 1 or self.min_size > self.max_size:
            raise ValueError("invalid PostgreSQL pool size")
        if self.hnsw_ef_search <= 0:
            raise ValueError("PG_HNSW_EF_SEARCH must be positive")
        self.pool = ConnectionPool(
            conninfo=self.database_url,
            min_size=self.min_size,
            max_size=self.max_size,
            kwargs={"row_factory": dict_row},
            configure=_configure_connection,
            open=True,
        )

    @classmethod
    def from_env(cls) -> "PostgresRetrievalStore":
        return cls()

    def close(self) -> None:
        self.pool.close()

    def ensure_ready(self) -> None:
        try:
            with self.pool.connection() as connection:
                row = connection.execute(
                    """
                    SELECT
                        to_regclass('public.rag_chunks') IS NOT NULL AS has_rag,
                        to_regclass('public.research_claims') IS NOT NULL AS has_claims,
                        to_regclass('public.segment_dossiers') IS NOT NULL AS has_dossiers,
                        to_regclass('public.rag_chunks_bm25_idx') IS NOT NULL AS has_bm25,
                        to_regclass('public.rag_chunks_embedding_hnsw_idx') IS NOT NULL AS has_rag_hnsw,
                        to_regclass('public.research_claims_embedding_hnsw_idx') IS NOT NULL AS has_claim_hnsw,
                        to_regclass('public.segment_dossiers_embedding_hnsw_idx') IS NOT NULL AS has_dossier_hnsw,
                        EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') AS has_vector,
                        EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_search') AS has_search
                    """
                ).fetchone()
        except Exception as exc:
            raise PostgresRetrievalError(f"PostgreSQL retrieval backend unavailable: {exc}") from exc
        required = (
            "has_rag",
            "has_claims",
            "has_dossiers",
            "has_bm25",
            "has_rag_hnsw",
            "has_claim_hnsw",
            "has_dossier_hnsw",
            "has_vector",
            "has_search",
        )
        if not row or not all(bool(row[key]) for key in required):
            raise PostgresRetrievalError("PostgreSQL retrieval schema is not initialized; run migrate_postgres.py")

    def sync_rag_documents(self, documents: Iterable[RagDocument | dict[str, Any]]) -> dict[str, Any]:
        rows = [asdict(item) if isinstance(item, RagDocument) else dict(item) for item in documents]
        now = datetime.now(timezone.utc)
        values = []
        for row in rows:
            chunk_id = str(row.get("chunk_id") or "").strip()
            if not chunk_id:
                raise ValueError("RAG chunk_id is required")
            search_text = rag_search_text(row)
            semantic_text = rag_semantic_text(row)
            digest = stable_content_hash(
                {
                    key: row.get(key)
                    for key in (
                        "report_id",
                        "kind",
                        "company",
                        "source_title",
                        "source_url",
                        "source_tier",
                        "source_type",
                        "page",
                        "section",
                        "content_type",
                        "table_id",
                        "text",
                    )
                }
            )
            values.append(
                (
                    chunk_id,
                    str(row.get("report_id") or ""),
                    str(row.get("kind") or ""),
                    str(row.get("company") or ""),
                    str(row.get("source_title") or ""),
                    str(row.get("source_url") or ""),
                    str(row.get("source_tier") or ""),
                    str(row.get("source_type") or ""),
                    str(row.get("page") or ""),
                    str(row.get("section") or ""),
                    str(row.get("content_type") or "text"),
                    str(row.get("table_id") or ""),
                    str(row.get("text") or ""),
                    search_text,
                    semantic_text,
                    Jsonb(dict(row.get("token_counts") or {})),
                    int(row.get("token_count") or 0),
                    digest,
                    now,
                )
            )
        with self.pool.connection() as connection, connection.transaction():
            connection.execute("CREATE TEMP TABLE stage_rag_ids(chunk_id text PRIMARY KEY) ON COMMIT DROP")
            if values:
                connection.cursor().executemany(
                    """
                    INSERT INTO rag_chunks (
                        chunk_id, report_id, kind, company, source_title, source_url,
                        source_tier, source_type, page, section, content_type, table_id,
                        text, search_text, semantic_text, token_counts, token_count,
                        content_hash, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        report_id = EXCLUDED.report_id,
                        kind = EXCLUDED.kind,
                        company = EXCLUDED.company,
                        source_title = EXCLUDED.source_title,
                        source_url = EXCLUDED.source_url,
                        source_tier = EXCLUDED.source_tier,
                        source_type = EXCLUDED.source_type,
                        page = EXCLUDED.page,
                        section = EXCLUDED.section,
                        content_type = EXCLUDED.content_type,
                        table_id = EXCLUDED.table_id,
                        text = EXCLUDED.text,
                        search_text = EXCLUDED.search_text,
                        semantic_text = EXCLUDED.semantic_text,
                        token_counts = EXCLUDED.token_counts,
                        token_count = EXCLUDED.token_count,
                        embedding = CASE
                            WHEN rag_chunks.content_hash = EXCLUDED.content_hash THEN rag_chunks.embedding
                            ELSE NULL
                        END,
                        embedding_status = CASE
                            WHEN rag_chunks.content_hash = EXCLUDED.content_hash THEN rag_chunks.embedding_status
                            ELSE 'stale'
                        END,
                        content_hash = EXCLUDED.content_hash,
                        updated_at = EXCLUDED.updated_at
                    """,
                    values,
                )
                connection.cursor().executemany(
                    "INSERT INTO stage_rag_ids(chunk_id) VALUES (%s)",
                    [(value[0],) for value in values],
                )
            deleted = connection.execute(
                "DELETE FROM rag_chunks WHERE NOT EXISTS (SELECT 1 FROM stage_rag_ids s WHERE s.chunk_id = rag_chunks.chunk_id)"
            ).rowcount
            digest = corpus_hash(value[0] for value in values)
            connection.execute(
                """
                INSERT INTO retrieval_builds(
                    build_kind, corpus_hash, record_count, status, details,
                    schema_version, completed_at
                ) VALUES ('rag', %s, %s, 'completed', %s, %s, now())
                """,
                (digest, len(values), Jsonb({"deleted": deleted}), SCHEMA_VERSION),
            )
        return {"record_count": len(values), "deleted": deleted, "corpus_hash": digest}

    def latest_claim_reviews(self) -> dict[str, dict[str, Any]]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT ON (claim_id) claim_id, updates
                FROM claim_reviews
                ORDER BY claim_id, created_at DESC, id DESC
                """
            ).fetchall()
        return {str(row["claim_id"]): dict(row["updates"] or {}) for row in rows}

    def import_claim_reviews(self, reviews: Iterable[dict[str, Any]]) -> int:
        values = []
        for source in reviews:
            claim_id = str(source.get("claim_id") or "").strip()
            if not claim_id:
                continue
            review = normalize_claim_review(claim_id, dict(source))
            values.append(
                (
                    claim_id,
                    Jsonb(review),
                    stable_content_hash(review),
                    str(review["reviewer"]),
                    review["updated_at"],
                )
            )
        if not values:
            return 0
        with self.pool.connection() as connection, connection.transaction():
            before = int(connection.execute("SELECT count(*) AS count FROM claim_reviews").fetchone()["count"])
            connection.cursor().executemany(
                """
                INSERT INTO claim_reviews(claim_id, updates, review_hash, reviewer, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (review_hash) DO NOTHING
                """,
                values,
            )
            after = int(connection.execute("SELECT count(*) AS count FROM claim_reviews").fetchone()["count"])
        return after - before

    def sync_research(
        self,
        claims: Iterable[dict[str, Any]],
        dossiers: Iterable[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized = [normalize_claim_row(dict(row)) for row in claims]
        if any(not str(row.get("claim_id") or "").strip() for row in normalized):
            raise ValueError("research claim_id is required")
        effective_claims = apply_claim_reviews(normalized, self.latest_claim_reviews())
        if dossiers is not None and effective_claims == normalized:
            effective_dossiers = [dict(row) for row in dossiers]
        else:
            effective_dossiers = build_segment_dossiers(
                [row for row in effective_claims if str(row.get("review_status") or "") != "rejected"]
            )
        with self.pool.connection() as connection, connection.transaction():
            claim_count, deleted_claims = self._sync_claims(connection, effective_claims)
            dossier_count, deleted_dossiers = self._sync_dossiers(connection, effective_dossiers)
            digest = corpus_hash(
                [str(row.get("claim_id") or "") for row in effective_claims]
                + [str(row.get("topic") or "") for row in effective_dossiers]
            )
            connection.execute(
                """
                INSERT INTO retrieval_builds(
                    build_kind, corpus_hash, record_count, status, details,
                    schema_version, completed_at
                ) VALUES ('research', %s, %s, 'completed', %s, %s, now())
                """,
                (
                    digest,
                    claim_count + dossier_count,
                    Jsonb({"claims": claim_count, "dossiers": dossier_count}),
                    SCHEMA_VERSION,
                ),
            )
        return {
            "claims": claim_count,
            "dossiers": dossier_count,
            "deleted_claims": deleted_claims,
            "deleted_dossiers": deleted_dossiers,
            "corpus_hash": digest,
        }

    def _sync_claims(
        self,
        connection: psycopg.Connection[Any],
        claims: list[dict[str, Any]],
    ) -> tuple[int, int]:
        connection.execute("CREATE TEMP TABLE stage_claim_ids(claim_id text PRIMARY KEY) ON COMMIT DROP")
        values = []
        now = datetime.now(timezone.utc)
        for source in claims:
            row = normalize_claim_row(source)
            semantic_text = claim_semantic_text(row)
            digest = stable_content_hash(
                {
                    key: row.get(key)
                    for key in (
                        "claim_type",
                        "topic",
                        "claim_text",
                        "companies",
                        "mechanism",
                        "direction",
                        "horizon",
                        "metric",
                        "value",
                        "unit",
                        "source_report_id",
                        "source_title",
                        "page",
                        "section",
                        "source_tier",
                        "evidence_span",
                        "confidence",
                        "as_of_date",
                        "exposure_level",
                        "review_status",
                        "reviewer_note",
                        "quality_flags",
                        "conflict_group_id",
                    )
                }
            )
            reviewed_at = row.get("reviewed_at") or None
            values.append(
                (
                    str(row.get("claim_id") or ""),
                    str(row.get("claim_type") or ""),
                    str(row.get("topic") or ""),
                    str(row.get("claim_text") or ""),
                    parse_companies(row.get("companies", [])),
                    str(row.get("mechanism") or ""),
                    str(row.get("direction") or ""),
                    str(row.get("horizon") or ""),
                    str(row.get("metric") or ""),
                    str(row.get("value") or ""),
                    str(row.get("unit") or ""),
                    str(row.get("source_report_id") or ""),
                    str(row.get("source_title") or ""),
                    str(row.get("page") or ""),
                    str(row.get("section") or ""),
                    str(row.get("source_tier") or ""),
                    str(row.get("evidence_span") or ""),
                    str(row.get("confidence") or ""),
                    str(row.get("as_of_date") or ""),
                    str(row.get("exposure_level") or ""),
                    str(row.get("review_status") or "auto"),
                    str(row.get("reviewer_note") or ""),
                    str(row.get("quality_flags") or ""),
                    str(row.get("conflict_group_id") or ""),
                    reviewed_at,
                    str(row.get("reviewer") or ""),
                    semantic_text,
                    digest,
                    now,
                )
            )
        if values:
            connection.cursor().executemany(
                """
                INSERT INTO research_claims (
                    claim_id, claim_type, topic, claim_text, companies, mechanism,
                    direction, horizon, metric, value, unit, source_report_id,
                    source_title, page, section, source_tier, evidence_span,
                    confidence, as_of_date, exposure_level, review_status,
                    reviewer_note, quality_flags, conflict_group_id, reviewed_at,
                    reviewer, semantic_text, content_hash, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (claim_id) DO UPDATE SET
                    claim_type = EXCLUDED.claim_type,
                    topic = EXCLUDED.topic,
                    claim_text = EXCLUDED.claim_text,
                    companies = EXCLUDED.companies,
                    mechanism = EXCLUDED.mechanism,
                    direction = EXCLUDED.direction,
                    horizon = EXCLUDED.horizon,
                    metric = EXCLUDED.metric,
                    value = EXCLUDED.value,
                    unit = EXCLUDED.unit,
                    source_report_id = EXCLUDED.source_report_id,
                    source_title = EXCLUDED.source_title,
                    page = EXCLUDED.page,
                    section = EXCLUDED.section,
                    source_tier = EXCLUDED.source_tier,
                    evidence_span = EXCLUDED.evidence_span,
                    confidence = EXCLUDED.confidence,
                    as_of_date = EXCLUDED.as_of_date,
                    exposure_level = EXCLUDED.exposure_level,
                    review_status = EXCLUDED.review_status,
                    reviewer_note = EXCLUDED.reviewer_note,
                    quality_flags = EXCLUDED.quality_flags,
                    conflict_group_id = EXCLUDED.conflict_group_id,
                    reviewed_at = EXCLUDED.reviewed_at,
                    reviewer = EXCLUDED.reviewer,
                    semantic_text = EXCLUDED.semantic_text,
                    embedding = CASE
                        WHEN research_claims.content_hash = EXCLUDED.content_hash THEN research_claims.embedding
                        ELSE NULL
                    END,
                    embedding_status = CASE
                        WHEN research_claims.content_hash = EXCLUDED.content_hash THEN research_claims.embedding_status
                        ELSE 'stale'
                    END,
                    content_hash = EXCLUDED.content_hash,
                    updated_at = EXCLUDED.updated_at
                """,
                values,
            )
            connection.cursor().executemany(
                "INSERT INTO stage_claim_ids(claim_id) VALUES (%s)",
                [(value[0],) for value in values],
            )
        deleted = connection.execute(
            """
            DELETE FROM research_claims
            WHERE NOT EXISTS (
                SELECT 1 FROM stage_claim_ids s WHERE s.claim_id = research_claims.claim_id
            )
            """
        ).rowcount
        return len(values), deleted

    def _sync_dossiers(
        self,
        connection: psycopg.Connection[Any],
        dossiers: list[dict[str, Any]],
    ) -> tuple[int, int]:
        connection.execute("CREATE TEMP TABLE stage_dossier_topics(topic text PRIMARY KEY) ON COMMIT DROP")
        values = []
        now = datetime.now(timezone.utc)
        for row in dossiers:
            topic = str(row.get("topic") or "")
            if not topic:
                raise ValueError("segment dossier topic is required")
            semantic_text = dossier_to_semantic_text(row)
            digest = stable_content_hash(row)
            values.append(
                (
                    topic,
                    str(row.get("summary") or ""),
                    Jsonb(row),
                    semantic_text,
                    digest,
                    now,
                )
            )
        if values:
            connection.cursor().executemany(
                """
                INSERT INTO segment_dossiers(
                    topic, summary, payload, semantic_text, content_hash, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (topic) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    payload = EXCLUDED.payload,
                    semantic_text = EXCLUDED.semantic_text,
                    embedding = CASE
                        WHEN segment_dossiers.content_hash = EXCLUDED.content_hash THEN segment_dossiers.embedding
                        ELSE NULL
                    END,
                    embedding_status = CASE
                        WHEN segment_dossiers.content_hash = EXCLUDED.content_hash THEN segment_dossiers.embedding_status
                        ELSE 'stale'
                    END,
                    content_hash = EXCLUDED.content_hash,
                    updated_at = EXCLUDED.updated_at
                """,
                values,
            )
            connection.cursor().executemany(
                "INSERT INTO stage_dossier_topics(topic) VALUES (%s)",
                [(value[0],) for value in values],
            )
        deleted = connection.execute(
            """
            DELETE FROM segment_dossiers
            WHERE NOT EXISTS (
                SELECT 1 FROM stage_dossier_topics s WHERE s.topic = segment_dossiers.topic
            )
            """
        ).rowcount
        return len(values), deleted

    def pending_embeddings(self, *, force: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
        status_clause = "" if force else "WHERE embedding_status IN ('missing', 'stale', 'failed') OR embedding IS NULL"
        sql = f"""
            SELECT 'rag' AS kind, chunk_id AS ref_id, semantic_text, content_hash
            FROM rag_chunks
            {status_clause}
            UNION ALL
            SELECT 'claim' AS kind, claim_id AS ref_id, semantic_text, content_hash
            FROM research_claims
            {status_clause}
            UNION ALL
            SELECT 'dossier' AS kind, topic AS ref_id, semantic_text, content_hash
            FROM segment_dossiers
            {status_clause}
            ORDER BY kind, ref_id
        """
        params: tuple[Any, ...] = ()
        if limit is not None and limit >= 0:
            sql += " LIMIT %s"
            params = (limit,)
        with self.pool.connection() as connection:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

    def update_embeddings(
        self,
        rows: list[dict[str, Any]],
        vectors: list[list[float]],
        *,
        model: str,
    ) -> None:
        if len(rows) != len(vectors):
            raise ValueError("embedding row/vector count mismatch")
        table_map = {
            "rag": ("rag_chunks", "chunk_id"),
            "claim": ("research_claims", "claim_id"),
            "dossier": ("segment_dossiers", "topic"),
        }
        with self.pool.connection() as connection, connection.transaction():
            for row, vector in zip(rows, vectors):
                if len(vector) != EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        f"embedding dimension mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(vector)}"
                    )
                table, key = table_map[str(row["kind"])]
                connection.execute(
                    f"""
                    UPDATE {table}
                    SET embedding = %s,
                        embedding_status = 'ready',
                        embedding_model = %s,
                        embedded_at = now(),
                        updated_at = now()
                    WHERE {key} = %s AND content_hash = %s
                    """,
                    (vector, model, str(row["ref_id"]), str(row["content_hash"])),
                )

    def mark_embedding_failed(self, row: dict[str, Any]) -> None:
        table_map = {
            "rag": ("rag_chunks", "chunk_id"),
            "claim": ("research_claims", "claim_id"),
            "dossier": ("segment_dossiers", "topic"),
        }
        table, key = table_map[str(row["kind"])]
        with self.pool.connection() as connection:
            connection.execute(
                f"""
                UPDATE {table}
                SET embedding_status = 'failed', updated_at = now()
                WHERE {key} = %s AND content_hash = %s
                """,
                (str(row["ref_id"]), str(row["content_hash"])),
            )
            connection.commit()

    def record_embedding_build(self, *, model: str, count: int, status: str = "completed") -> None:
        with self.pool.connection() as connection:
            content_rows = connection.execute(
                """
                SELECT content_hash FROM rag_chunks
                UNION ALL SELECT content_hash FROM research_claims
                UNION ALL SELECT content_hash FROM segment_dossiers
                """
            ).fetchall()
            digest = corpus_hash(str(row["content_hash"]) for row in content_rows)
            connection.execute(
                """
                INSERT INTO retrieval_builds(
                    build_kind, corpus_hash, record_count, embedding_model,
                    embedding_dimensions, status, schema_version, completed_at
                ) VALUES ('embedding', %s, %s, %s, %s, %s, %s, now())
                """,
                (
                    digest,
                    count,
                    model,
                    EMBEDDING_DIMENSIONS,
                    status,
                    SCHEMA_VERSION,
                ),
            )
            connection.commit()

    def chunk_ids(self) -> set[str]:
        with self.pool.connection() as connection:
            return {str(row["chunk_id"]) for row in connection.execute("SELECT chunk_id FROM rag_chunks")}

    def semantic_rag_chunk_ids(self) -> set[str]:
        with self.pool.connection() as connection:
            return {
                str(row["chunk_id"])
                for row in connection.execute(
                    "SELECT chunk_id FROM rag_chunks WHERE embedding_status = 'ready' AND embedding IS NOT NULL"
                )
            }

    def corpus_metadata(self, kind: str) -> dict[str, Any]:
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT build_kind, corpus_hash, record_count, embedding_model,
                       embedding_dimensions, status, created_at, completed_at,
                       details, schema_version
                FROM retrieval_builds
                WHERE build_kind = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (kind,),
            ).fetchone()
        return dict(row) if row else {}

    def semantic_metadata(self) -> SemanticIndexMetadata:
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT count(*) FROM rag_chunks
                     WHERE embedding_status = 'ready' AND embedding IS NOT NULL) +
                    (SELECT count(*) FROM research_claims
                     WHERE embedding_status = 'ready' AND embedding IS NOT NULL) +
                    (SELECT count(*) FROM segment_dossiers
                     WHERE embedding_status = 'ready' AND embedding IS NOT NULL) AS vector_count,
                    (SELECT count(*) FROM rag_chunks) +
                    (SELECT count(*) FROM research_claims) +
                    (SELECT count(*) FROM segment_dossiers) AS document_count,
                    COALESCE((
                        SELECT embedding_model FROM retrieval_builds
                        WHERE build_kind = 'embedding' ORDER BY id DESC LIMIT 1
                    ), '') AS embedding_model,
                    COALESCE((
                        SELECT completed_at FROM retrieval_builds
                        WHERE build_kind = 'embedding' ORDER BY id DESC LIMIT 1
                    ), now()) AS built_at
                """
            ).fetchone()
        assert row is not None
        return SemanticIndexMetadata(
            index_version="postgres-pgvector-v1",
            built_at=str(row["built_at"]),
            document_count=int(row["document_count"] or 0),
            vector_count=int(row["vector_count"] or 0),
            dimension=EMBEDDING_DIMENSIONS,
            embedding_model=str(row["embedding_model"] or ""),
            rag_dir="postgresql",
            research_dir="postgresql",
            index_dir="postgresql",
        )


class PostgresRagIndex:
    """BM25 RAG adapter backed by a ParadeDB index."""

    def __init__(self, store: PostgresRetrievalStore) -> None:
        self.store = store

    def search(
        self,
        question: str,
        *,
        top_k: int = 6,
        filters: dict[str, str] | None = None,
    ) -> list[RagHit]:
        filters = dict(filters or {})
        expanded = expand_query(question)
        tokens = tokenize(expanded)
        if not tokens or top_k <= 0:
            return []
        clauses = ["search_text ||| %s"]
        params: list[Any] = [expanded]
        for field in ("company", "source_type", "source_tier", "kind", "content_type"):
            value = str(filters.get(field) or "")
            if value:
                clauses.append(f"{field} = %s")
                params.append(value)
        candidate_k = max(top_k * 8, 40)
        params.append(candidate_k)
        with self.store.pool.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT chunk_id, report_id, source_title, source_tier, source_type,
                       page, section, content_type, table_id, company, text,
                       pdb.score(id) AS bm25_score
                FROM rag_chunks
                WHERE {' AND '.join(clauses)}
                ORDER BY pdb.score(id) DESC, chunk_id
                LIMIT %s
                """,
                params,
            ).fetchall()
        hits = [self._hit_from_row(dict(row), question, tokens) for row in rows]
        hits = [hit for hit in hits if not is_low_value_hit(hit)]
        hits.sort(key=lambda hit: (-hit.score, source_priority(hit), hit.source_title, hit.page, hit.chunk_id))
        return dedupe_hits(hits)[: max(0, top_k)]

    @staticmethod
    def _hit_from_row(row: dict[str, Any], question: str, tokens: list[str]) -> RagHit:
        score = float(row.get("bm25_score") or 0.0)
        question_norm = normalize_text(question)
        searchable = normalize_text(
            "\n".join(
                str(row.get(key) or "")
                for key in ("company", "source_title", "section", "text")
            )
        )
        if question_norm and question_norm in searchable:
            score += 3.0
        for field in ("company", "section", "source_title"):
            value = str(row.get(field) or "")
            if value and normalize_text(value) in question_norm:
                score += 0.8
        if str(row.get("source_tier") or "") == "1":
            score += 0.6
        if str(row.get("source_type") or "") == "authority_whitepaper":
            score += 0.8
        if is_technical_question(question) and str(row.get("source_type") or "") in {
            "technical_roadmap",
            "open_specification",
            "manual_open_specification",
            "benchmark_methodology",
            "technical_paper",
            "model_technical_report",
            "authority_whitepaper",
        }:
            score += 0.9
        text = str(row.get("text") or "")
        return RagHit(
            chunk_id=str(row.get("chunk_id") or ""),
            report_id=str(row.get("report_id") or ""),
            source_title=str(row.get("source_title") or ""),
            source_tier=str(row.get("source_tier") or ""),
            source_type=str(row.get("source_type") or ""),
            page=str(row.get("page") or ""),
            section=str(row.get("section") or ""),
            content_type=str(row.get("content_type") or "text"),
            table_id=str(row.get("table_id") or ""),
            company=str(row.get("company") or ""),
            text=text,
            snippet=make_snippet(text, tokens),
            score=round(score, 6),
        )

    def chunk_ids(self) -> set[str]:
        return self.store.chunk_ids()

    def metadata_dict(self) -> dict[str, Any]:
        return self.store.corpus_metadata("rag")

    def corpus_hash(self) -> str:
        return str(self.metadata_dict().get("corpus_hash") or corpus_hash(self.chunk_ids()))


class PostgresResearchMemory:
    """Claim and dossier rule-retrieval adapter backed by PostgreSQL."""

    def __init__(self, store: PostgresRetrievalStore) -> None:
        self.store = store

    def _memory(
        self,
        *,
        topics: Iterable[str] | None = None,
        companies: Iterable[str] | None = None,
        claim_types: Iterable[str] | None = None,
    ) -> ResearchMemory:
        topic_values = [str(item) for item in (topics or []) if str(item)]
        company_values = [str(item) for item in (companies or []) if str(item)]
        type_values = [str(item) for item in (claim_types or []) if str(item)]
        claim_clauses = ["review_status <> 'rejected'"]
        claim_params: list[Any] = []
        if topic_values:
            claim_clauses.append("topic = ANY(%s::text[])")
            claim_params.append(topic_values)
        if company_values:
            claim_clauses.append("companies && %s::text[]")
            claim_params.append(company_values)
        if type_values:
            claim_clauses.append("claim_type = ANY(%s::text[])")
            claim_params.append(type_values)
        dossier_clauses: list[str] = []
        dossier_params: list[Any] = []
        if topic_values:
            dossier_clauses.append("topic = ANY(%s::text[])")
            dossier_params.append(topic_values)
        with self.store.pool.connection() as connection:
            claims = [
                normalize_claim_row(dict(row))
                for row in connection.execute(
                    f"""
                    SELECT claim_id, claim_type, topic, claim_text, companies,
                           mechanism, direction, horizon, metric, value, unit,
                           source_report_id, source_title, page, section, source_tier,
                           evidence_span, confidence, as_of_date, exposure_level,
                           review_status, reviewer_note, quality_flags,
                           conflict_group_id, reviewed_at, reviewer
                    FROM research_claims
                    WHERE {' AND '.join(claim_clauses)}
                    """,
                    claim_params,
                ).fetchall()
            ]
            dossier_sql = "SELECT payload FROM segment_dossiers"
            if dossier_clauses:
                dossier_sql += " WHERE " + " AND ".join(dossier_clauses)
            dossiers = [
                dict(row["payload"] or {})
                for row in connection.execute(dossier_sql, dossier_params).fetchall()
            ]
        return ResearchMemory(claims, dossiers)

    def search(self, question: str, plan: Any, *, limit: int = 8) -> list[ResearchHit]:
        topics = query_topics(question, getattr(plan, "topics", []), getattr(plan, "expanded_topics", []))
        return self._memory(topics=topics).search(question, plan, limit=limit)

    def search_global_dossiers(
        self,
        question: str,
        plan: Any,
        *,
        limit: int = 3,
        topics: list[str] | None = None,
    ) -> list[ResearchHit]:
        selected = topics or query_topics(question, getattr(plan, "topics", []), getattr(plan, "expanded_topics", []))
        return self._memory(topics=selected).search_global_dossiers(
            question,
            plan,
            limit=limit,
            topics=selected,
        )

    def search_local_claims(
        self,
        question: str,
        plan: Any,
        *,
        limit: int = 12,
        claim_types: list[str] | set[str] | tuple[str, ...] | None = None,
        companies: list[str] | set[str] | tuple[str, ...] | None = None,
        topics: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> list[ResearchHit]:
        selected_topics = list(topics or []) or query_topics(
            question,
            getattr(plan, "topics", []),
            getattr(plan, "expanded_topics", []),
        )
        selected_companies = list(companies or getattr(plan, "companies", []) or [])
        return self._memory(
            topics=selected_topics,
            companies=selected_companies,
            claim_types=claim_types,
        ).search_local_claims(
            question,
            plan,
            limit=limit,
            claim_types=claim_types,
            companies=companies,
            topics=selected_topics,
        )

    def get_claim(self, claim_id: str) -> dict[str, Any]:
        with self.store.pool.connection() as connection:
            row = connection.execute(
                "SELECT * FROM research_claims WHERE claim_id = %s",
                (claim_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Claim not found: {claim_id}")
        return normalize_claim_row(dict(row))

    def review_claim(
        self,
        claim_id: str,
        updates: dict[str, Any],
        *,
        reviewer: str = "frontend",
    ) -> dict[str, Any]:
        review = normalize_claim_review(claim_id, updates, reviewer=reviewer)
        with self.store.pool.connection() as connection, connection.transaction():
            current_row = connection.execute(
                "SELECT * FROM research_claims WHERE claim_id = %s FOR UPDATE",
                (claim_id,),
            ).fetchone()
            if current_row is None:
                raise KeyError(f"Claim not found: {claim_id}")
            current = normalize_claim_row(dict(current_row))
            updated = normalize_claim_row(apply_claim_reviews([current], {claim_id: review})[0])
            old_topic = str(current.get("topic") or "")
            semantic_text = claim_semantic_text(updated)
            digest = stable_content_hash(
                {
                    key: updated.get(key)
                    for key in (
                        "claim_type",
                        "topic",
                        "claim_text",
                        "companies",
                        "mechanism",
                        "direction",
                        "horizon",
                        "metric",
                        "value",
                        "unit",
                        "evidence_span",
                        "confidence",
                        "as_of_date",
                        "exposure_level",
                        "review_status",
                        "reviewer_note",
                        "quality_flags",
                        "conflict_group_id",
                    )
                }
            )
            connection.execute(
                """
                INSERT INTO claim_reviews(
                    claim_id, updates, review_hash, reviewer, created_at
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    claim_id,
                    Jsonb(review),
                    stable_content_hash(review),
                    str(review["reviewer"]),
                    review["updated_at"],
                ),
            )
            connection.execute(
                """
                UPDATE research_claims SET
                    claim_type = %s, topic = %s, claim_text = %s, companies = %s,
                    mechanism = %s, direction = %s, horizon = %s, metric = %s,
                    value = %s, unit = %s, evidence_span = %s, confidence = %s,
                    as_of_date = %s, exposure_level = %s, review_status = %s,
                    reviewer_note = %s, quality_flags = %s, conflict_group_id = %s,
                    reviewed_at = %s, reviewer = %s, semantic_text = %s,
                    content_hash = %s, embedding = NULL, embedding_status = 'stale',
                    updated_at = now()
                WHERE claim_id = %s
                """,
                (
                    str(updated.get("claim_type") or ""),
                    str(updated.get("topic") or ""),
                    str(updated.get("claim_text") or ""),
                    parse_companies(updated.get("companies", [])),
                    str(updated.get("mechanism") or ""),
                    str(updated.get("direction") or ""),
                    str(updated.get("horizon") or ""),
                    str(updated.get("metric") or ""),
                    str(updated.get("value") or ""),
                    str(updated.get("unit") or ""),
                    str(updated.get("evidence_span") or ""),
                    str(updated.get("confidence") or ""),
                    str(updated.get("as_of_date") or ""),
                    str(updated.get("exposure_level") or ""),
                    str(updated.get("review_status") or "revised"),
                    str(updated.get("reviewer_note") or ""),
                    str(updated.get("quality_flags") or ""),
                    str(updated.get("conflict_group_id") or ""),
                    review["updated_at"],
                    str(review["reviewer"]),
                    semantic_text,
                    digest,
                    claim_id,
                ),
            )
            self._rebuild_topics(connection, {old_topic, str(updated.get("topic") or "")})
        return self.get_claim(claim_id)

    def _rebuild_topics(
        self,
        connection: psycopg.Connection[Any],
        topics: set[str],
    ) -> None:
        for topic in {item for item in topics if item}:
            rows = [
                normalize_claim_row(dict(row))
                for row in connection.execute(
                    "SELECT * FROM research_claims WHERE topic = %s AND review_status <> 'rejected'",
                    (topic,),
                ).fetchall()
            ]
            dossiers = build_segment_dossiers(rows)
            if not dossiers:
                connection.execute("DELETE FROM segment_dossiers WHERE topic = %s", (topic,))
                continue
            dossier = dossiers[0]
            semantic_text = dossier_to_semantic_text(dossier)
            digest = stable_content_hash(dossier)
            connection.execute(
                """
                INSERT INTO segment_dossiers(topic, summary, payload, semantic_text, content_hash)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (topic) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    payload = EXCLUDED.payload,
                    semantic_text = EXCLUDED.semantic_text,
                    content_hash = EXCLUDED.content_hash,
                    embedding = NULL,
                    embedding_status = 'stale',
                    updated_at = now()
                """,
                (
                    topic,
                    str(dossier.get("summary") or ""),
                    Jsonb(dossier),
                    semantic_text,
                    digest,
                ),
            )

    def claim_stats(self) -> dict[str, Any]:
        with self.store.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    count(DISTINCT claim_id) AS claims,
                    count(DISTINCT claim_id) FILTER (
                        WHERE review_status IN ('approved', 'revised', 'rejected', 'needs_review')
                    ) AS reviewed_claims,
                    count(DISTINCT claim_id) FILTER (
                        WHERE review_status = 'rejected'
                    ) AS rejected_claims,
                    count(DISTINCT company) FILTER (
                        WHERE claim_type = 'company_exposure'
                          AND exposure_level IN ('core', 'direct')
                    ) AS direct_exposure_companies
                FROM research_claims
                LEFT JOIN LATERAL unnest(companies) AS company ON true
                """
            ).fetchone()
            dossier_count = int(
                connection.execute("SELECT count(*) AS count FROM segment_dossiers").fetchone()["count"]
            )
            type_rows = connection.execute(
                "SELECT claim_type, count(*) AS count FROM research_claims GROUP BY claim_type"
            ).fetchall()
        assert row is not None
        return {
            "claims": int(row["claims"] or 0),
            "dossiers": dossier_count,
            "reviewed_claims": int(row["reviewed_claims"] or 0),
            "rejected_claims": int(row["rejected_claims"] or 0),
            "direct_exposure_companies": int(row["direct_exposure_companies"] or 0),
            "claim_type_counts": {
                str(item["claim_type"] or "unknown"): int(item["count"] or 0)
                for item in type_rows
            },
        }


class PostgresSemanticIndex:
    """Dense semantic retrieval adapter backed by pgvector HNSW indexes."""

    def __init__(
        self,
        store: PostgresRetrievalStore,
        *,
        embedding_client: Any,
    ) -> None:
        self.store = store
        self.embedding_client = embedding_client
        self.metadata = store.semantic_metadata()

    def search(
        self,
        question: str,
        *,
        top_k: int = 8,
        filters: dict[str, list[str]] | None = None,
    ) -> list[SemanticHit]:
        if top_k <= 0:
            return []
        vectors = self.embedding_client.embed_texts([question])
        if not vectors:
            return []
        vector = [float(value) for value in vectors[0]]
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"query embedding dimension mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(vector)}"
            )
        selected_kinds = set((filters or {}).get("kinds") or ("rag", "claim", "dossier"))
        hits: list[SemanticHit] = []
        with self.store.pool.connection() as connection, connection.transaction():
            connection.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(self.store.hnsw_ef_search),),
            )
            if "rag" in selected_kinds:
                hits.extend(self._search_rag(connection, vector, top_k, filters or {}))
            if "claim" in selected_kinds:
                hits.extend(self._search_claims(connection, vector, top_k, filters or {}))
            if "dossier" in selected_kinds:
                hits.extend(self._search_dossiers(connection, vector, top_k, filters or {}))
        hits.sort(key=lambda hit: (-hit.score, hit.kind, hit.topic, hit.company, hit.title))
        return hits[: max(0, top_k)]

    def _search_rag(
        self,
        connection: psycopg.Connection[Any],
        vector: list[float],
        top_k: int,
        filters: dict[str, list[str]],
    ) -> list[SemanticHit]:
        clauses = ["embedding_status = 'ready'", "embedding IS NOT NULL"]
        params: list[Any] = [vector]
        companies = filters.get("companies") or []
        if companies:
            clauses.append("company = ANY(%s::text[])")
            params.append(companies)
        params.extend([vector, max(top_k, 1)])
        rows = connection.execute(
            f"""
            SELECT chunk_id, source_title, text, page, section, company,
                   source_tier, source_type,
                   1 - (embedding::halfvec(2048) <=> %s::vector::halfvec(2048)) AS score
            FROM rag_chunks
            WHERE {' AND '.join(clauses)}
            ORDER BY embedding::halfvec(2048) <=> %s::vector::halfvec(2048)
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [
            SemanticHit(
                doc_id=f"rag:{row['chunk_id']}",
                kind="rag",
                title=str(row["source_title"] or "RAG 原文片段"),
                text=str(row["text"] or ""),
                score=round(float(row["score"] or 0.0), 6),
                source=str(row["source_title"] or ""),
                page=str(row["page"] or ""),
                section=str(row["section"] or ""),
                company=str(row["company"] or ""),
                ref_id=str(row["chunk_id"] or ""),
                source_tier=str(row["source_tier"] or ""),
                source_type=str(row["source_type"] or ""),
            )
            for row in rows
        ]

    def _search_claims(
        self,
        connection: psycopg.Connection[Any],
        vector: list[float],
        top_k: int,
        filters: dict[str, list[str]],
    ) -> list[SemanticHit]:
        clauses = [
            "embedding_status = 'ready'",
            "embedding IS NOT NULL",
            "review_status <> 'rejected'",
        ]
        params: list[Any] = [vector]
        companies = filters.get("companies") or []
        topics = filters.get("topics") or []
        if companies:
            clauses.append("companies && %s::text[]")
            params.append(companies)
        if topics:
            clauses.append("topic = ANY(%s::text[])")
            params.append(topics)
        params.extend([vector, max(top_k, 1)])
        rows = connection.execute(
            f"""
            SELECT claim_id, claim_type, topic, claim_text, companies,
                   source_title, page, section, source_tier, exposure_level,
                   confidence, as_of_date,
                   1 - (embedding::halfvec(2048) <=> %s::vector::halfvec(2048)) AS score
            FROM research_claims
            WHERE {' AND '.join(clauses)}
            ORDER BY embedding::halfvec(2048) <=> %s::vector::halfvec(2048)
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [
            SemanticHit(
                doc_id=f"claim:{row['claim_id']}",
                kind="claim",
                title=claim_title(dict(row)),
                text=str(row["claim_text"] or ""),
                score=round(float(row["score"] or 0.0), 6),
                source=str(row["source_title"] or ""),
                page=str(row["page"] or ""),
                section=str(row["section"] or ""),
                topic=str(row["topic"] or ""),
                company=first_company(row["companies"] or []),
                claim_type=str(row["claim_type"] or ""),
                exposure_level=str(row["exposure_level"] or ""),
                ref_id=str(row["claim_id"] or ""),
                source_tier=str(row["source_tier"] or ""),
                confidence=str(row["confidence"] or ""),
                as_of_date=str(row["as_of_date"] or ""),
            )
            for row in rows
        ]

    def _search_dossiers(
        self,
        connection: psycopg.Connection[Any],
        vector: list[float],
        top_k: int,
        filters: dict[str, list[str]],
    ) -> list[SemanticHit]:
        clauses = ["embedding_status = 'ready'", "embedding IS NOT NULL"]
        params: list[Any] = [vector]
        topics = filters.get("topics") or []
        if topics:
            clauses.append("topic = ANY(%s::text[])")
            params.append(topics)
        params.extend([vector, max(top_k, 1)])
        rows = connection.execute(
            f"""
            SELECT topic, semantic_text,
                   1 - (embedding::halfvec(2048) <=> %s::vector::halfvec(2048)) AS score
            FROM segment_dossiers
            WHERE {' AND '.join(clauses)}
            ORDER BY embedding::halfvec(2048) <=> %s::vector::halfvec(2048)
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [
            SemanticHit(
                doc_id=f"dossier:{row['topic']}",
                kind="dossier",
                title=f"{row['topic']} 产业链投研摘要",
                text=str(row["semantic_text"] or ""),
                score=round(float(row["score"] or 0.0), 6),
                topic=str(row["topic"] or ""),
                ref_id=str(row["topic"] or ""),
            )
            for row in rows
        ]

    def rag_chunk_ids(self) -> set[str]:
        return self.store.semantic_rag_chunk_ids()

    def metadata_dict(self) -> dict[str, Any]:
        return self.store.semantic_metadata().to_dict()
