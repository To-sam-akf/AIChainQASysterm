CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id text PRIMARY KEY,
    title text NOT NULL,
    source_type text NOT NULL,
    source_url text NOT NULL DEFAULT '',
    filename text NOT NULL DEFAULT '',
    content_hash text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'received',
    error text NOT NULL DEFAULT '',
    semantic_status text NOT NULL DEFAULT 'not_configured',
    active_version integer,
    pending_version integer,
    tags text[] NOT NULL DEFAULT '{}',
    topic text NOT NULL DEFAULT '',
    company text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    deleted_at timestamptz
);
CREATE TABLE IF NOT EXISTS knowledge_versions (
    document_id text NOT NULL REFERENCES knowledge_documents(document_id),
    version integer NOT NULL,
    source_data bytea,
    raw_text text NOT NULL DEFAULT '',
    content_hash text NOT NULL DEFAULT '',
    fetched_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, version)
);
CREATE INDEX IF NOT EXISTS knowledge_documents_hash_idx ON knowledge_documents(content_hash);
