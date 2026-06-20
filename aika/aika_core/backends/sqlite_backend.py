"""SQLite FTS5 backend for the lightweight public AIKA Core path."""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from aika.report import build_report_spec, render_html, render_markdown, render_markdown_sections
from aika.aika_core.data_paths import (
    CLAIMS_FILE,
    EVIDENCE_SPANS_FILE,
    RELATIONS_FILE,
    SEGMENT_DOSSIERS_FILE,
)
from aika.aika_core.models import (
    ClaimRecord,
    CompanyComparison,
    CompanyProfile,
    EvidenceCard,
    EvidenceGap,
    GraphEdge,
    ResearchBackend,
    ResearchBrief,
)


SCHEMA_VERSION = "1"
DEFAULT_PROFILE = "sample"
AIKA_HOME_ENV = "AIKA_HOME"

CLAIM_COLUMNS = [
    "claim_id",
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
]

EVIDENCE_COLUMNS = [
    "evidence_id",
    "claim_id",
    "evidence",
    "source_report_id",
    "source_title",
    "page",
    "section",
    "source_tier",
    "as_of_date",
    "quality",
    "company",
    "topic",
    "claim_type",
    "confidence",
    "exposure_level",
]

DOSSIER_COLUMNS = ["dossier_id", "topic", "title", "content", "raw_json"]

RELATION_COLUMNS = [
    "relation_id",
    "head_type",
    "head_name",
    "relation",
    "tail_type",
    "tail_name",
    "evidence",
    "source_report_id",
    "source_title",
    "page",
    "section",
    "source_tier",
    "confidence",
    "review_status",
]


def resolve_aika_home(home: str | Path | None = None) -> Path:
    """Resolve the local AIKA home directory without creating it."""
    value = home or os.getenv(AIKA_HOME_ENV) or "~/.aika"
    return Path(value).expanduser().resolve()


def profile_knowledge_dir(home: str | Path | None = None, *, profile: str = DEFAULT_PROFILE) -> Path:
    return resolve_aika_home(home) / "knowledge" / profile


def profile_index_path(home: str | Path | None = None, *, profile: str = DEFAULT_PROFILE) -> Path:
    return resolve_aika_home(home) / "indexes" / f"{profile}.sqlite"


def sqlite_fts_status() -> dict[str, Any]:
    """Return SQLite FTS5/trigram support details for doctor output."""
    status: dict[str, Any] = {
        "sqlite_version": sqlite3.sqlite_version,
        "fts5": False,
        "trigram": False,
        "tokenizer": "",
        "error": "",
    }
    try:
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(x)")
            status["fts5"] = True
            try:
                connection.execute("CREATE VIRTUAL TABLE trigram_probe USING fts5(x, tokenize='trigram')")
                status["trigram"] = True
                status["tokenizer"] = "trigram"
            except sqlite3.OperationalError:
                status["tokenizer"] = "unicode61"
    except sqlite3.OperationalError as exc:
        status["error"] = str(exc)
    return status


def build_sqlite_index(knowledge_dir: str | Path, index_path: str | Path) -> dict[str, Any]:
    """Build a deterministic SQLite FTS5 index from local CSV/JSONL artifacts."""
    source_dir = Path(knowledge_dir).expanduser().resolve()
    target_path = Path(index_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    counts = {"claims": 0, "evidence_spans": 0, "dossiers": 0, "relations": 0}
    try:
        with sqlite3.connect(tmp_path) as connection:
            connection.row_factory = sqlite3.Row
            tokenizer = _select_tokenizer(connection)
            _create_schema(connection, tokenizer=tokenizer)
            claim_lookup = _load_claims(connection, source_dir / CLAIMS_FILE)
            counts["claims"] = len(claim_lookup)
            counts["evidence_spans"] = _load_evidence_spans(
                connection,
                source_dir / EVIDENCE_SPANS_FILE,
                claim_lookup,
            )
            counts["dossiers"] = _load_dossiers(connection, source_dir / SEGMENT_DOSSIERS_FILE)
            counts["relations"] = _load_relations(connection, source_dir / RELATIONS_FILE)
            _write_metadata(
                connection,
                {
                    "schema_version": SCHEMA_VERSION,
                    "built_at": datetime.now(timezone.utc).isoformat(),
                    "source_dir": str(source_dir),
                    "tokenizer": tokenizer,
                    **{f"{key}_count": str(value) for key, value in counts.items()},
                },
            )
            connection.commit()
        tmp_path.replace(target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return {"index_path": str(target_path), "knowledge_dir": str(source_dir), "counts": counts}


def inspect_sqlite_index(index_path: str | Path) -> dict[str, Any]:
    path = Path(index_path).expanduser().resolve()
    result: dict[str, Any] = {
        "exists": path.exists(),
        "path": str(path),
        "metadata": {},
        "counts": {},
        "error": "",
    }
    if not path.exists():
        return result
    try:
        with _connect(path) as connection:
            result["metadata"] = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata").fetchall()
            }
            for table in ("claims", "evidence_spans", "dossiers", "relations"):
                row = connection.execute(f"SELECT count(*) AS count FROM {table}").fetchone()
                result["counts"][table] = int(row["count"] or 0)
    except sqlite3.Error as exc:
        result["error"] = str(exc)
    return result


class SQLiteResearchBackend(ResearchBackend):
    """Research backend backed by a single local SQLite FTS5 index."""

    def __init__(self, index_path: str | Path | None = None, *, home: str | Path | None = None, profile: str = DEFAULT_PROFILE) -> None:
        self.index_path = Path(index_path).expanduser().resolve() if index_path else profile_index_path(home, profile=profile)

    @classmethod
    def from_home(cls, home: str | Path | None = None, *, profile: str = DEFAULT_PROFILE) -> "SQLiteResearchBackend":
        return cls(home=home, profile=profile)

    def search_evidence(self, query: str, *, top_k: int = 8, **filters: Any) -> list[EvidenceCard]:
        limit = _limit(top_k)
        if limit == 0 or not self.index_path.exists():
            return []
        try:
            rows = self._search_evidence_rows(query, limit=limit, filters=filters)
        except sqlite3.Error:
            return []
        cards = [_evidence_card_from_row(row, citation_id=f"E{index}") for index, row in enumerate(rows, start=1)]
        return cards[:limit]

    def search_claims(self, query: str, *, top_k: int = 8, **filters: Any) -> list[ClaimRecord]:
        limit = _limit(top_k)
        if limit == 0 or not self.index_path.exists():
            return []
        try:
            rows = self._search_claim_rows(query, limit=limit, filters=filters)
        except sqlite3.Error:
            return []
        return [ClaimRecord.from_row(dict(row), score=float(row["score"] or 0.0)) for row in rows[:limit]]

    def search_dossiers(self, query: str, *, top_k: int = 3, **filters: Any) -> list[EvidenceCard]:
        limit = _limit(top_k)
        if limit == 0 or not self.index_path.exists():
            return []
        try:
            rows = self._search_dossier_rows(query, limit=limit, filters=filters)
        except sqlite3.Error:
            return []
        return [_dossier_card_from_row(row, citation_id=f"D{index}") for index, row in enumerate(rows, start=1)]

    def query_graph(
        self,
        *,
        company: str = "",
        technology: str = "",
        relation_type: str = "",
        limit: int = 80,
    ) -> list[GraphEdge]:
        row_limit = _limit(limit)
        if row_limit == 0 or not self.index_path.exists():
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if company:
            clauses.append("(head_name LIKE ? OR tail_name LIKE ?)")
            params.extend([f"%{company}%", f"%{company}%"])
        if technology:
            clauses.append("(head_name LIKE ? OR tail_name LIKE ? OR evidence LIKE ?)")
            params.extend([f"%{technology}%", f"%{technology}%", f"%{technology}%"])
        if relation_type:
            clauses.append("relation = ?")
            params.append(relation_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        try:
            with _connect(self.index_path) as connection:
                rows = connection.execute(
                    f"""
                    SELECT * FROM relations
                    {where}
                    ORDER BY relation_id
                    LIMIT ?
                    """,
                    [*params, row_limit],
                ).fetchall()
        except sqlite3.Error:
            return []
        return [_graph_edge_from_row(row) for row in rows]

    def get_company_profile(self, company: str, *, topic: str = "") -> CompanyProfile:
        query = f"{company} {topic}".strip()
        evidence_cards = self.search_evidence(query, top_k=8, company=company, topic=topic or None)
        claims = self.search_claims(query, top_k=5, company=company, topic=topic or None)
        edges = self.query_graph(company=company, technology=topic, limit=12)
        summary = claims[0].claim_text if claims else ""
        return CompanyProfile(
            company=company,
            topic=topic,
            summary=summary,
            evidence_cards=evidence_cards,
            graph_edges=edges,
            research_outputs={
                "claims": [claim.to_dict() for claim in claims],
                "backend": "sqlite",
                "index_path": str(self.index_path),
            },
        )

    def compare_companies(self, companies: Iterable[str], *, topic: str = "") -> CompanyComparison:
        company_list = _dedupe(_text(company) for company in companies if _text(company))
        rows: list[dict[str, Any]] = []
        all_cards: list[EvidenceCard] = []
        all_gaps: list[EvidenceGap] = []
        for company in company_list:
            profile = self.get_company_profile(company, topic=topic)
            all_cards.extend(profile.evidence_cards)
            all_gaps.extend(profile.evidence_gaps)
            rows.append(
                {
                    "company": company,
                    "chain_segment": _infer_chain_segment(profile.graph_edges),
                    "exposure_level": _strongest_exposure(profile.evidence_cards),
                    "business_evidence": _summarize_cards(profile.evidence_cards, {"company_exposure", "mechanism", "supply_chain", ""}),
                    "leading_indicators": _summarize_cards(profile.evidence_cards, {"indicator"}),
                    "risks": _summarize_cards(profile.evidence_cards, {"risk", "bottleneck"}),
                    "citations": "、".join(_card_citations(profile.evidence_cards)[:6]),
                }
            )
            if not profile.evidence_cards:
                all_gaps.append(
                    EvidenceGap(
                        gap=f"{company} 缺少可用于公司对比的证据卡片。",
                        priority="高",
                        suggested_source="补充公司年报、公告、研报或投资者关系记录。",
                    )
                )
        return CompanyComparison(
            companies=company_list,
            topic=topic,
            columns=["company", "chain_segment", "exposure_level", "business_evidence", "leading_indicators", "risks", "citations"],
            rows=rows,
            evidence_cards=_dedupe_evidence_cards(all_cards),
            evidence_gaps=_dedupe_gaps(all_gaps),
            research_outputs={
                "backend": "sqlite",
                "index_path": str(self.index_path),
            },
        )

    def audit_evidence_gaps(
        self,
        query: str,
        *,
        companies: Iterable[str] | None = None,
        topic: str = "",
    ) -> list[EvidenceGap]:
        company_list = _dedupe(_text(company) for company in (companies or []) if _text(company))
        subject = _text(query) or "、".join(company_list) or _text(topic) or "AI算力产业链"
        filters: dict[str, Any] = {}
        if company_list:
            filters["company"] = company_list
        if topic:
            filters["topic"] = topic
        evidence_cards = self.search_evidence(subject, top_k=10, **filters)
        claims = self.search_claims(subject, top_k=10, **filters)
        claim_cards = _claim_cards_from_records(claims, start_index=len(evidence_cards) + 1)
        cards = [*evidence_cards, *claim_cards]
        gaps: list[EvidenceGap] = []
        if not cards:
            gaps.append(
                EvidenceGap(
                    gap="当前问题没有召回可用证据卡片。",
                    priority="高",
                    suggested_source="补充年报、研报原文或行业白皮书后重建本地索引。",
                )
            )
            return gaps
        for company in company_list:
            if not any(company in _card_support_text(card) for card in cards):
                gaps.append(
                    EvidenceGap(
                        gap=f"{company} 缺少直接证据，当前结论可能不完整。",
                        priority="高",
                        suggested_source="补充该公司产品、订单、客户导入或产业链位置证据。",
                    )
                )
        if not any(card.claim_type == "indicator" for card in cards):
            gaps.append(
                EvidenceGap(
                    gap=f"{subject} 缺少订单、收入、产能、客户导入或渗透率等领先指标证据。",
                    priority="高",
                    suggested_source="补充年报财务表、公告、研报指标表或产业数据库。",
                )
            )
        if not any(card.claim_type in {"risk", "bottleneck"} for card in cards):
            gaps.append(
                EvidenceGap(
                    gap=f"{subject} 缺少明确风险、反证或技术瓶颈证据。",
                    priority="中",
                    suggested_source="补充年报风险披露、行业竞争格局和技术路线替代证据。",
                )
            )
        if not any(card.citation_id for card in cards):
            gaps.append(
                EvidenceGap(
                    gap=f"{subject} 召回证据缺少 citation_id。",
                    priority="中",
                    suggested_source="重建 evidence spans 或 claims 索引，确保 citation_id 可追踪。",
                )
            )
        return _dedupe_gaps(gaps)[:12]

    def build_research_brief(self, query: str, *, topic: str = "") -> ResearchBrief:
        subject = _text(query) or (f"{topic}投研简报" if topic else "AI算力产业链投研简报")
        focus_topic = _text(topic)
        evidence_cards = self.search_evidence(subject, top_k=8, topic=focus_topic or None)
        claims = self.search_claims(subject, top_k=8, topic=focus_topic or None)
        claim_cards = _claim_cards_from_records(claims, start_index=len(evidence_cards) + 1)
        dossier_cards = self.search_dossiers(subject, top_k=2, topic=focus_topic or None)
        cards = _dedupe_evidence_cards(_ensure_citations([*dossier_cards, *evidence_cards, *claim_cards]))
        edges = self.query_graph(technology=focus_topic or subject, limit=12)
        gaps = self.audit_evidence_gaps(subject, topic=focus_topic)
        plan = SimpleNamespace(topics=[focus_topic] if focus_topic else [], companies=[])
        spec = build_report_spec(
            question=subject,
            plan=plan,
            evidence_cards=cards,
            graph_records=edges,
            gaps=gaps,
            verification={"status": "not_run", "checks": {}},
        )
        title = spec.title
        sections = render_markdown_sections(spec)
        markdown = render_markdown(spec)
        research_outputs = {
            "report": {
                "title": title,
                "markdown": markdown,
                "html": render_html(spec),
                "sections": sections,
                "report_type": spec.report_type,
                "report_type_label": spec.report_type_label,
                "coverage": spec.coverage.model_dump(),
                "spec": spec.model_dump(),
            },
            "evidence_gaps": [gap.to_dict() for gap in gaps],
            "verification": {"status": "not_run", "checks": {}},
            "meta": {
                "question": subject,
                "topic": focus_topic,
                "evidence_cards": len(cards),
                "coverage": spec.coverage.model_dump(),
                "report_type": spec.report_type,
                "report_type_label": spec.report_type_label,
                "backend": "sqlite",
                "index_path": str(self.index_path),
            },
        }
        return ResearchBrief(
            title=title,
            markdown=markdown,
            sections=sections,
            evidence_cards=cards,
            evidence_gaps=gaps,
            meta=research_outputs["meta"],
            research_outputs=research_outputs,
        )

    def _search_claim_rows(self, query: str, *, limit: int, filters: dict[str, Any]) -> list[sqlite3.Row]:
        match_query = _fts_query(query)
        filter_sql, filter_params = _claim_filter_sql(filters, alias="c")
        with _connect(self.index_path) as connection:
            if match_query:
                try:
                    candidate_limit = max(limit * 20, limit)
                    rows = connection.execute(
                        f"""
                        SELECT c.*, -bm25(claims_fts) AS score
                        FROM claims_fts
                        JOIN claims c ON c.id = claims_fts.rowid
                        WHERE claims_fts MATCH ?
                        {filter_sql}
                        ORDER BY bm25(claims_fts), c.topic, c.claim_type, c.claim_id
                        LIMIT ?
                        """,
                        [match_query, *filter_params, candidate_limit],
                    ).fetchall()
                    if rows:
                        fallback_rows = _fallback_claim_rows(connection, query, limit=candidate_limit, filters=filters)
                        ranked = _score_rows(
                            _dedupe_rows([*rows, *fallback_rows]),
                            query,
                            fields=("claim_text", "evidence_span", "topic", "companies"),
                        )
                        return ranked[:limit] or rows[:limit]
                except sqlite3.OperationalError:
                    pass
            return _fallback_claim_rows(connection, query, limit=limit, filters=filters)

    def _search_evidence_rows(self, query: str, *, limit: int, filters: dict[str, Any]) -> list[sqlite3.Row]:
        match_query = _fts_query(query)
        filter_sql, filter_params = _evidence_filter_sql(filters, alias="e")
        with _connect(self.index_path) as connection:
            if match_query:
                try:
                    candidate_limit = max(limit * 20, limit)
                    rows = connection.execute(
                        f"""
                        SELECT e.*, -bm25(evidence_fts) AS score
                        FROM evidence_fts
                        JOIN evidence_spans e ON e.id = evidence_fts.rowid
                        WHERE evidence_fts MATCH ?
                        {filter_sql}
                        ORDER BY bm25(evidence_fts), e.topic, e.company, e.evidence_id
                        LIMIT ?
                        """,
                        [match_query, *filter_params, candidate_limit],
                    ).fetchall()
                    if rows:
                        fallback_rows = _fallback_evidence_rows(connection, query, limit=candidate_limit, filters=filters)
                        ranked = _score_rows(
                            _dedupe_rows([*rows, *fallback_rows]),
                            query,
                            fields=("evidence", "source_title", "company", "topic"),
                        )
                        return ranked[:limit] or rows[:limit]
                except sqlite3.OperationalError:
                    pass
            return _fallback_evidence_rows(connection, query, limit=limit, filters=filters)

    def _search_dossier_rows(self, query: str, *, limit: int, filters: dict[str, Any]) -> list[sqlite3.Row]:
        match_query = _fts_query(query)
        filter_sql, filter_params = _topic_filter_sql(filters, alias="d")
        with _connect(self.index_path) as connection:
            if match_query:
                try:
                    candidate_limit = max(limit * 20, limit)
                    rows = connection.execute(
                        f"""
                        SELECT d.*, -bm25(dossiers_fts) AS score
                        FROM dossiers_fts
                        JOIN dossiers d ON d.id = dossiers_fts.rowid
                        WHERE dossiers_fts MATCH ?
                        {filter_sql}
                        ORDER BY bm25(dossiers_fts), d.topic
                        LIMIT ?
                        """,
                        [match_query, *filter_params, candidate_limit],
                    ).fetchall()
                    if rows:
                        fallback_rows = _fallback_dossier_rows(connection, query, limit=candidate_limit, filters=filters)
                        ranked = _score_rows(
                            _dedupe_rows([*rows, *fallback_rows]),
                            query,
                            fields=("title", "content", "topic"),
                        )
                        return ranked[:limit] or rows[:limit]
                except sqlite3.OperationalError:
                    pass
            return _fallback_dossier_rows(connection, query, limit=limit, filters=filters)


def _connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(path))
    connection.row_factory = sqlite3.Row
    return connection


def _select_tokenizer(connection: sqlite3.Connection) -> str:
    try:
        connection.execute("CREATE VIRTUAL TABLE tokenizer_probe USING fts5(x, tokenize='trigram')")
        connection.execute("DROP TABLE tokenizer_probe")
        return "trigram"
    except sqlite3.OperationalError:
        connection.execute("CREATE VIRTUAL TABLE tokenizer_probe USING fts5(x, tokenize='unicode61')")
        connection.execute("DROP TABLE tokenizer_probe")
        return "unicode61"


def _create_schema(connection: sqlite3.Connection, *, tokenizer: str) -> None:
    claim_definitions = ",\n                ".join(f"{column} TEXT" for column in CLAIM_COLUMNS)
    evidence_definitions = ",\n                ".join(f"{column} TEXT" for column in EVIDENCE_COLUMNS)
    dossier_definitions = ",\n                ".join(f"{column} TEXT" for column in DOSSIER_COLUMNS)
    relation_definitions = ",\n                ".join(f"{column} TEXT" for column in RELATION_COLUMNS)
    connection.executescript(
        f"""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE claims (
            id INTEGER PRIMARY KEY,
            {claim_definitions}
        );

        CREATE TABLE evidence_spans (
            id INTEGER PRIMARY KEY,
            {evidence_definitions}
        );

        CREATE TABLE dossiers (
            id INTEGER PRIMARY KEY,
            {dossier_definitions}
        );

        CREATE TABLE relations (
            id INTEGER PRIMARY KEY,
            {relation_definitions}
        );

        CREATE VIRTUAL TABLE claims_fts USING fts5(
            claim_text,
            evidence_span,
            topic,
            companies,
            tokenize='{tokenizer}'
        );

        CREATE VIRTUAL TABLE evidence_fts USING fts5(
            evidence,
            source_title,
            company,
            topic,
            tokenize='{tokenizer}'
        );

        CREATE VIRTUAL TABLE dossiers_fts USING fts5(
            title,
            content,
            topic,
            tokenize='{tokenizer}'
        );

        CREATE INDEX claims_topic_idx ON claims(topic);
        CREATE INDEX claims_type_idx ON claims(claim_type);
        CREATE INDEX claims_id_idx ON claims(claim_id);
        CREATE INDEX evidence_claim_idx ON evidence_spans(claim_id);
        CREATE INDEX evidence_topic_idx ON evidence_spans(topic);
        CREATE INDEX evidence_company_idx ON evidence_spans(company);
        CREATE INDEX relations_head_idx ON relations(head_name);
        CREATE INDEX relations_tail_idx ON relations(tail_name);
        """
    )


def _load_claims(connection: sqlite3.Connection, path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for index, row in enumerate(_read_csv_rows(path), start=1):
        normalized = {column: _text(row.get(column)) for column in CLAIM_COLUMNS}
        if not normalized["claim_id"]:
            normalized["claim_id"] = f"claim_row_{index}"
        record = ClaimRecord.from_row(normalized)
        normalized["companies"] = _serialize_cell(row.get("companies") or normalized["companies"])
        cursor = connection.execute(
            _insert_sql("claims", CLAIM_COLUMNS),
            [normalized[column] for column in CLAIM_COLUMNS],
        )
        rowid = int(cursor.lastrowid)
        connection.execute(
            "INSERT INTO claims_fts(rowid, claim_text, evidence_span, topic, companies) VALUES (?, ?, ?, ?, ?)",
            (
                rowid,
                normalized["claim_text"],
                normalized["evidence_span"],
                normalized["topic"],
                normalized["companies"],
            ),
        )
        lookup[normalized["claim_id"]] = {
            "company": record.companies[0] if record.companies else "",
            "topic": normalized["topic"],
            "claim_type": normalized["claim_type"],
            "confidence": normalized["confidence"],
            "exposure_level": normalized["exposure_level"],
        }
    return lookup


def _load_evidence_spans(connection: sqlite3.Connection, path: Path, claim_lookup: dict[str, dict[str, str]]) -> int:
    count = 0
    for index, row in enumerate(_read_csv_rows(path), start=1):
        claim_id = _text(row.get("claim_id"))
        claim = claim_lookup.get(claim_id, {})
        normalized = {
            "evidence_id": _text(row.get("evidence_id")) or f"evidence_row_{index}",
            "claim_id": claim_id,
            "evidence": _text(row.get("evidence") or row.get("text")),
            "source_report_id": _text(row.get("source_report_id")),
            "source_title": _text(row.get("source_title") or row.get("source")),
            "page": _text(row.get("page")),
            "section": _text(row.get("section")),
            "source_tier": _text(row.get("source_tier")),
            "as_of_date": _text(row.get("as_of_date")),
            "quality": _text(row.get("quality")),
            "company": _text(row.get("company") or claim.get("company")),
            "topic": _text(row.get("topic") or claim.get("topic")),
            "claim_type": _text(row.get("claim_type") or claim.get("claim_type")),
            "confidence": _text(row.get("confidence") or claim.get("confidence")),
            "exposure_level": _text(row.get("exposure_level") or claim.get("exposure_level")),
        }
        cursor = connection.execute(
            _insert_sql("evidence_spans", EVIDENCE_COLUMNS),
            [normalized[column] for column in EVIDENCE_COLUMNS],
        )
        rowid = int(cursor.lastrowid)
        connection.execute(
            "INSERT INTO evidence_fts(rowid, evidence, source_title, company, topic) VALUES (?, ?, ?, ?, ?)",
            (
                rowid,
                normalized["evidence"],
                normalized["source_title"],
                normalized["company"],
                normalized["topic"],
            ),
        )
        count += 1
    return count


def _load_dossiers(connection: sqlite3.Connection, path: Path) -> int:
    count = 0
    for index, dossier in enumerate(_read_jsonl_rows(path), start=1):
        topic = _text(dossier.get("topic"))
        title = _text(dossier.get("title")) or f"{topic} segment dossier".strip()
        content = _dossier_content(dossier)
        normalized = {
            "dossier_id": _text(dossier.get("dossier_id")) or f"dossier_{topic or index}",
            "topic": topic,
            "title": title,
            "content": content,
            "raw_json": json.dumps(dossier, ensure_ascii=False, sort_keys=True),
        }
        cursor = connection.execute(
            _insert_sql("dossiers", DOSSIER_COLUMNS),
            [normalized[column] for column in DOSSIER_COLUMNS],
        )
        rowid = int(cursor.lastrowid)
        connection.execute(
            "INSERT INTO dossiers_fts(rowid, title, content, topic) VALUES (?, ?, ?, ?)",
            (rowid, title, content, topic),
        )
        count += 1
    return count


def _load_relations(connection: sqlite3.Connection, path: Path) -> int:
    count = 0
    for index, row in enumerate(_read_csv_rows(path), start=1):
        normalized = {column: _text(row.get(column)) for column in RELATION_COLUMNS}
        if not normalized["relation_id"]:
            normalized["relation_id"] = f"relation_row_{index}"
        connection.execute(
            _insert_sql("relations", RELATION_COLUMNS),
            [normalized[column] for column in RELATION_COLUMNS],
        )
        count += 1
    return count


def _write_metadata(connection: sqlite3.Connection, values: dict[str, Any]) -> None:
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [(str(key), str(value)) for key, value in sorted(values.items())],
    )


def _insert_sql(table: str, columns: list[str]) -> str:
    names = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    return f"INSERT INTO {table}({names}) VALUES ({placeholders})"


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _fallback_claim_rows(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    filters: dict[str, Any],
) -> list[sqlite3.Row]:
    filter_sql, filter_params = _claim_filter_sql(filters, alias="c")
    search_sql, search_params = _like_search_sql(
        query,
        alias="c",
        columns=("claim_text", "evidence_span", "topic", "companies", "source_title"),
    )
    rows = connection.execute(
        f"""
        SELECT c.*, 0.0 AS score
        FROM claims c
        WHERE 1=1
        {filter_sql}
        {search_sql}
        ORDER BY c.topic, c.claim_type, c.claim_id
        LIMIT ?
        """,
        [*filter_params, *search_params, limit],
    ).fetchall()
    return _score_rows(rows, query, fields=("claim_text", "evidence_span", "topic", "companies"))


def _fallback_evidence_rows(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    filters: dict[str, Any],
) -> list[sqlite3.Row]:
    filter_sql, filter_params = _evidence_filter_sql(filters, alias="e")
    search_sql, search_params = _like_search_sql(
        query,
        alias="e",
        columns=("evidence", "source_title", "company", "topic"),
    )
    rows = connection.execute(
        f"""
        SELECT e.*, 0.0 AS score
        FROM evidence_spans e
        WHERE 1=1
        {filter_sql}
        {search_sql}
        ORDER BY e.topic, e.company, e.evidence_id
        LIMIT ?
        """,
        [*filter_params, *search_params, max(limit * 5, limit)],
    ).fetchall()
    return _score_rows(rows, query, fields=("evidence", "source_title", "company", "topic"))[:limit]


def _fallback_dossier_rows(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    filters: dict[str, Any],
) -> list[sqlite3.Row]:
    filter_sql, filter_params = _topic_filter_sql(filters, alias="d")
    search_sql, search_params = _like_search_sql(query, alias="d", columns=("title", "content", "topic"))
    rows = connection.execute(
        f"""
        SELECT d.*, 0.0 AS score
        FROM dossiers d
        WHERE 1=1
        {filter_sql}
        {search_sql}
        ORDER BY d.topic
        LIMIT ?
        """,
        [*filter_params, *search_params, max(limit * 5, limit)],
    ).fetchall()
    return _score_rows(rows, query, fields=("title", "content", "topic"))[:limit]


def _claim_filter_sql(filters: dict[str, Any], *, alias: str) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    companies = _values(filters.get("company") or filters.get("companies"))
    if companies:
        parts = []
        for company in companies:
            parts.append(f"({alias}.companies LIKE ? OR {alias}.claim_text LIKE ? OR {alias}.evidence_span LIKE ?)")
            params.extend([f"%{company}%", f"%{company}%", f"%{company}%"])
        clauses.append(f"({' OR '.join(parts)})")
    topics = _values(filters.get("topic") or filters.get("topics"))
    if topics:
        parts = [f"{alias}.topic LIKE ?" for _ in topics]
        clauses.append(f"({' OR '.join(parts)})")
        params.extend(f"%{topic}%" for topic in topics)
    claim_types = _values(filters.get("claim_type") or filters.get("claim_types"))
    if claim_types:
        placeholders = ", ".join("?" for _ in claim_types)
        clauses.append(f"{alias}.claim_type IN ({placeholders})")
        params.extend(claim_types)
    return _where_suffix(clauses), params


def _evidence_filter_sql(filters: dict[str, Any], *, alias: str) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    companies = _values(filters.get("company") or filters.get("companies"))
    if companies:
        parts = []
        for company in companies:
            parts.append(f"({alias}.company LIKE ? OR {alias}.evidence LIKE ?)")
            params.extend([f"%{company}%", f"%{company}%"])
        clauses.append(f"({' OR '.join(parts)})")
    topics = _values(filters.get("topic") or filters.get("topics"))
    if topics:
        parts = [f"{alias}.topic LIKE ?" for _ in topics]
        clauses.append(f"({' OR '.join(parts)})")
        params.extend(f"%{topic}%" for topic in topics)
    claim_types = _values(filters.get("claim_type") or filters.get("claim_types"))
    if claim_types:
        placeholders = ", ".join("?" for _ in claim_types)
        clauses.append(f"{alias}.claim_type IN ({placeholders})")
        params.extend(claim_types)
    return _where_suffix(clauses), params


def _topic_filter_sql(filters: dict[str, Any], *, alias: str) -> tuple[str, list[Any]]:
    topics = _values(filters.get("topic") or filters.get("topics"))
    if not topics:
        return "", []
    parts = [f"{alias}.topic LIKE ?" for _ in topics]
    return f"AND ({' OR '.join(parts)})", [f"%{topic}%" for topic in topics]


def _where_suffix(clauses: list[str]) -> str:
    return f"AND {' AND '.join(clauses)}" if clauses else ""


def _like_search_sql(query: str, *, alias: str, columns: Iterable[str]) -> tuple[str, list[Any]]:
    terms = _query_terms(query)
    if not terms:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    for term in terms:
        parts = [f"{alias}.{column} LIKE ?" for column in columns]
        clauses.append(f"({' OR '.join(parts)})")
        params.extend(f"%{term}%" for _ in columns)
    return f"AND ({' OR '.join(clauses)})", params


def _fts_query(query: str) -> str:
    terms = _query_terms(query)
    if not terms:
        return ""
    return " OR ".join(f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in terms)


def _query_terms(query: str) -> list[str]:
    text = _text(query)
    if not text:
        return []
    terms: list[str] = []
    for term in re.split(r"[\s,，;；]+", text):
        if not term.strip():
            continue
        terms.extend(_expand_query_term(term.strip()))
    return _dedupe(terms)


def _expand_query_term(term: str) -> list[str]:
    terms = [term]
    for suffix in ("产业链", "行业", "领域", "公司", "业务"):
        if term.endswith(suffix) and len(term) > len(suffix):
            stem = term[: -len(suffix)].strip()
            if stem:
                terms.append(stem)
            terms.append(suffix)
    return terms


def _score_rows(rows: list[sqlite3.Row], query: str, *, fields: tuple[str, ...]) -> list[sqlite3.Row]:
    if not query.strip():
        return rows
    scored = [(float(_text_score(" ".join(_text(row[field]) for field in fields), query)), row) for row in rows]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda item: -item[0])
    return [_row_with_score(row, score) for score, row in scored]


def _dedupe_rows(rows: Iterable[sqlite3.Row]) -> list[sqlite3.Row]:
    output: list[sqlite3.Row] = []
    seen: set[Any] = set()
    for row in rows:
        key = row["id"] if "id" in row.keys() else tuple(sorted(dict(row).items()))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _row_with_score(row: sqlite3.Row, score: float) -> sqlite3.Row:
    data = dict(row)
    data["score"] = round(score, 4)
    return _MappingRow(data)


class _MappingRow(dict):
    def keys(self) -> Any:
        return super().keys()


def _text_score(text: str, query: str) -> float:
    normalized = _normalize(text)
    score = 0.0
    full = _normalize(query)
    if full and full in normalized:
        score += 12.0
    for term in _query_terms(query):
        value = _normalize(term)
        if value and value in normalized:
            if value in {"产业链", "行业", "领域", "公司", "业务"}:
                score += 1.0
            else:
                score += 5.0 if len(value) >= 2 else 1.0
    return score


def _evidence_card_from_row(row: sqlite3.Row, *, citation_id: str) -> EvidenceCard:
    data = dict(row)
    return EvidenceCard(
        citation_id=citation_id,
        kind="evidence",
        title=_text(data.get("source_title") or data.get("topic")),
        evidence=_text(data.get("evidence")),
        claim_id=_text(data.get("claim_id")),
        source=_text(data.get("source_title")),
        page=_text(data.get("page")),
        section=_text(data.get("section")),
        company=_text(data.get("company")),
        source_tier=_text(data.get("source_tier")),
        score=_floatish(data.get("score")),
        reason="sqlite fts",
        topic=_text(data.get("topic")),
        claim_type=_text(data.get("claim_type")),
        exposure_level=_text(data.get("exposure_level")),
        confidence=_text(data.get("confidence")),
        as_of_date=_text(data.get("as_of_date")),
        evidence_span=_text(data.get("evidence")),
        raw=data,
    )


def _dossier_card_from_row(row: sqlite3.Row, *, citation_id: str) -> EvidenceCard:
    data = dict(row)
    return EvidenceCard(
        citation_id=citation_id,
        kind="dossier",
        title=_text(data.get("title")),
        evidence=_text(data.get("content")),
        score=_floatish(data.get("score")),
        reason="sqlite dossier fts",
        topic=_text(data.get("topic")),
        raw=data,
    )


def _graph_edge_from_row(row: sqlite3.Row) -> GraphEdge:
    data = dict(row)
    return GraphEdge(
        source=_text(data.get("head_name")),
        target=_text(data.get("tail_name")),
        relation=_text(data.get("relation")),
        label=_text(data.get("relation")),
        source_type=_text(data.get("head_type")),
        target_type=_text(data.get("tail_type")),
        evidence=_text(data.get("evidence")),
        source_title=_text(data.get("source_title")),
        page=_text(data.get("page")),
        section=_text(data.get("section")),
        source_tier=_text(data.get("source_tier")),
        report_id=_text(data.get("source_report_id")),
        raw=data,
    )


def _claim_cards_from_records(claims: Iterable[ClaimRecord], *, start_index: int = 1) -> list[EvidenceCard]:
    return [claim.to_evidence_card(citation_id=f"E{index}") for index, claim in enumerate(claims, start=start_index)]


def _ensure_citations(cards: Iterable[EvidenceCard]) -> list[EvidenceCard]:
    output: list[EvidenceCard] = []
    for index, card in enumerate(cards, start=1):
        if card.citation_id:
            output.append(card)
            continue
        data = card.to_dict()
        data["citation_id"] = f"E{index}"
        output.append(EvidenceCard.from_any(data))
    return output


def _dedupe_evidence_cards(cards: Iterable[EvidenceCard]) -> list[EvidenceCard]:
    output: list[EvidenceCard] = []
    seen: set[tuple[str, str, str, str]] = set()
    for card in cards:
        key = (
            card.claim_id,
            card.source,
            card.page,
            _short_text(card.evidence or card.evidence_span, 180),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(card)
    return output


def _dedupe_gaps(gaps: Iterable[EvidenceGap]) -> list[EvidenceGap]:
    output: list[EvidenceGap] = []
    seen: set[str] = set()
    for gap in gaps:
        key = _normalize(gap.gap)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(gap)
    return output


def _infer_chain_segment(edges: Iterable[GraphEdge]) -> str:
    for edge in edges:
        if edge.relation in {"HAS_PRODUCT", "PART_OF_CHAIN", "ENABLES", "SUPPLIES"} and edge.target:
            return edge.target
    for edge in edges:
        if edge.target:
            return edge.target
    return "当前证据不足"


def _strongest_exposure(cards: Iterable[EvidenceCard]) -> str:
    priority = {"core": 0, "direct": 1, "indirect": 2, "mentioned": 3, "": 9}
    labels = {
        "core": "核心敞口",
        "direct": "直接敞口",
        "indirect": "间接敞口",
        "mentioned": "仅提及",
        "": "未分级",
    }
    best = ""
    for card in cards:
        if priority.get(card.exposure_level, 9) < priority.get(best, 9):
            best = card.exposure_level
    return labels.get(best, best or "未分级")


def _summarize_cards(cards: Iterable[EvidenceCard], claim_types: set[str], *, limit: int = 2) -> str:
    lines: list[str] = []
    for card in cards:
        if card.claim_type not in claim_types:
            continue
        citation = f" [{card.citation_id}]" if card.citation_id else ""
        lines.append(f"{_short_text(card.evidence, 80)}{citation}")
        if len(lines) >= limit:
            break
    return "；".join(lines) if lines else "当前证据不足"


def _card_citations(cards: Iterable[EvidenceCard]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for card in cards:
        if card.citation_id and card.citation_id not in seen:
            seen.add(card.citation_id)
            output.append(card.citation_id)
    return output


def _card_support_text(card: EvidenceCard) -> str:
    return " ".join([card.company, card.topic, card.title, card.evidence, card.evidence_span, card.source])


def _core_judgment(cards: list[EvidenceCard], subject: str) -> str:
    if not cards:
        return f"{subject} 当前证据不足，无法形成可验证结论。"
    card = cards[0]
    citation = f" [{card.citation_id}]" if card.citation_id else ""
    return f"{_short_text(card.evidence, 220)}{citation}"


def _card_bullets(cards: Iterable[EvidenceCard], claim_types: set[str], *, limit: int) -> str:
    lines: list[str] = []
    for card in cards:
        if card.claim_type not in claim_types:
            continue
        citation = f" [{card.citation_id}]" if card.citation_id else ""
        lines.append(f"- {_short_text(card.evidence, 160)}{citation}")
        if len(lines) >= limit:
            break
    return "\n".join(lines) if lines else "当前证据不足。"


def _graph_bullets(edges: Iterable[GraphEdge], *, limit: int) -> str:
    lines: list[str] = []
    for edge in edges:
        if not edge.source and not edge.target:
            continue
        evidence = f"：{_short_text(edge.evidence, 100)}" if edge.evidence else ""
        lines.append(f"- {edge.source} - {edge.relation} - {edge.target}{evidence}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def _gap_bullets(gaps: Iterable[EvidenceGap]) -> str:
    lines = [f"- {gap.gap} 建议：{gap.suggested_source}" for gap in gaps]
    return "\n".join(lines) if lines else "当前证据包未识别出关键缺口。"


def _evidence_index(cards: Iterable[EvidenceCard], *, limit: int) -> str:
    lines: list[str] = []
    for card in list(cards)[:limit]:
        citation = card.citation_id or "uncited"
        source = card.source or card.title or "unknown source"
        page = f", p.{card.page}" if card.page else ""
        lines.append(f"- {citation}: {source}{page} - {_short_text(card.evidence, 120)}")
    return "\n".join(lines) if lines else "当前证据不足。"


def _dossier_content(dossier: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "summary",
        "technology_mechanism",
        "industry_chain",
        "bottlenecks",
        "leading_indicators",
        "risks",
        "policies",
        "gaps",
    ):
        value = dossier.get(key)
        if isinstance(value, list):
            parts.extend(_text(item) for item in value if _text(item))
        elif isinstance(value, dict):
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        elif _text(value):
            parts.append(_text(value))
    return "\n".join(part for part in parts if part)


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,，;；|]", value) if item.strip()]
    if isinstance(value, Iterable):
        return [_text(item) for item in value if _text(item)]
    return [_text(value)] if _text(value) else []


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        key = _normalize(text)
        if key and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def _serialize_cell(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _text(value)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", _text(value)).casefold()


def _floatish(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _short_text(value: str, limit: int) -> str:
    text = " ".join(_text(value).split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _limit(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
