"""Read-only provenance checks and local artifacts for generation experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from aika.eval.benchmark_v2 import ROOT_DIR, file_sha256


def json_default(value: Any) -> str:
    if isinstance(value, (datetime, date, Path)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def rows_digest(rows: Iterable[Any]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        hasher.update((canonical(row) + "\n").encode("utf-8"))
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    import os
    import tempfile

    # A crash must not leave a truncated manifest or E/F candidate.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_snapshot(directory: Path) -> dict[str, Any]:
    """Include tracked sources and new implementation files, never ignored credentials."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT_DIR).decode("utf-8")

    names = set(git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0"))
    target = directory / "source.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            path = ROOT_DIR / name
            if not name or not path.is_file() or path.is_symlink():
                continue
            if name.startswith("data/") and not name.startswith(("data/eval/", "data/verified/", "data/curated/", "data/metadata/")):
                continue
            if any(part in {".env", ".git", "node_modules", ".venv"} for part in path.relative_to(ROOT_DIR).parts):
                continue
            archive.write(path, name)
    return {
        "git_commit": git("rev-parse", "HEAD").strip(),
        "dirty": bool(git("status", "--porcelain").strip()),
        "source_archive": target.name, "source_sha256": file_sha256(target),
    }


def raw_corpus_digest(rows: Iterable[dict[str, Any]]) -> str:
    return rows_digest(
        {key: str(row.get(key) or "") for key in ("chunk_id", "report_id", "page", "text")}
        for row in sorted(rows, key=lambda row: row["chunk_id"])
    )


def runtime_snapshot(engine: Any) -> dict[str, Any]:
    """Hash actual DB contents and vectors, rather than trusting import timestamps."""
    store = engine.retrieval_store
    if store is None:
        raise ValueError("Experiments B/E/F require PostgreSQL retrieval")
    tables = {}
    raw_rows = []
    with store.pool.connection() as connection, connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        for table, key in (("rag_chunks", "chunk_id"), ("research_claims", "claim_id"), ("segment_dossiers", "topic")):
            # Fixed identifiers above; no user SQL is interpolated.
            with connection.cursor(name=f"experiment_{table}") as cursor:
                cursor.execute(
                    f"SELECT to_jsonb(t) - 'embedding' AS record, md5(embedding::text) AS vector_hash "
                    f"FROM {table} t ORDER BY {key}"
                )
                records = list(cursor)
            tables[table] = {"count": len(records), "sha256": rows_digest(records)}
            if table == "rag_chunks":
                raw_rows = [row["record"] for row in records]
        builds = list(connection.execute("SELECT * FROM retrieval_builds ORDER BY id"))
        indexes = list(connection.execute(
            "SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' "
            "AND tablename = ANY(%s) ORDER BY tablename, indexname", (list(tables),),
        ))
    graph = {}
    if engine.csv_graph is not None:
        graph["csv"] = digest({"entities": engine.csv_graph.entities, "relations": engine.csv_graph.relations})
    if engine.graph_client is not None:
        # Direct read transaction avoids the normal answer query's LIMIT.
        with engine.graph_client._session() as session:
            def read_graph(tx: Any) -> dict[str, Any]:
                nodes = [record.data() for record in tx.run(
                    "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties ORDER BY id"
                )]
                edges = [record.data() for record in tx.run(
                    "MATCH (a)-[r]->(b) RETURN elementId(r) AS id, elementId(a) AS head, "
                    "elementId(b) AS tail, type(r) AS type, properties(r) AS properties ORDER BY id"
                )]
                return {"nodes": nodes, "edges": edges}
            graph["neo4j"] = digest(session.execute_read(read_graph))
    return {
        "tables": tables, "raw_corpus_sha256": raw_corpus_digest(raw_rows),
        "builds": builds, "indexes": indexes, "graph": graph,
        "semantic_index": engine.semantic_index.metadata_dict(),
        "hnsw_ef_search": store.hnsw_ef_search,
    }
