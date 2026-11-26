"""User-managed documents and incremental retrieval indexing."""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import os
import socket
import ssl
import threading
import uuid
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from aika.postgres_retrieval import EMBEDDING_DIMENSIONS, rag_search_text, rag_semantic_text, stable_content_hash
from aika.text_cleaner import split_text

MAX_BYTES = int(os.getenv("KNOWLEDGE_MAX_BYTES", str(20 * 1024 * 1024)))
FETCH_TIMEOUT = float(os.getenv("KNOWLEDGE_FETCH_TIMEOUT", "15"))


def fetch_public_url(url: str) -> tuple[bytes, str, str]:
    """Fetch from a validated public IP, preserving TLS host verification on each redirect."""
    for _ in range(6):
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            raise ValueError("仅支持不含认证信息的公开 HTTP/HTTPS 网址")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        addresses = socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
        ips = {row[4][0] for row in addresses}
        if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
            raise ValueError("网址指向非公网地址，无法收录")
        ip = sorted(ips)[0]
        sock = socket.create_connection((ip, port), timeout=FETCH_TIMEOUT)
        try:
            if parts.scheme == "https":
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=parts.hostname)
            conn = http.client.HTTPConnection(parts.hostname, port, timeout=FETCH_TIMEOUT)
            conn.sock = sock
            path = parts.path or "/"
            if parts.query:
                path += "?" + parts.query
            conn.request("GET", path, headers={"Host": parts.netloc, "User-Agent": "AIKA-Knowledge/1.0", "Accept": "text/html,text/plain"})
            response = conn.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location:
                    raise ValueError("网页重定向缺少目标")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f"网页返回 HTTP {response.status}")
            content_type = response.getheader("Content-Type", "")
            if not any(t in content_type.lower() for t in ("text/html", "text/plain", "application/xhtml+xml")):
                raise ValueError("仅支持 HTML 或纯文本网页")
            raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("网页超过大小限制")
            return raw, content_type, url
        finally:
            sock.close()
    raise ValueError("网页重定向次数过多")


def parse_source(kind: str, data: bytes) -> tuple[list[tuple[str, str]], str]:
    if kind == "pdf":
        import fitz
        with fitz.open(stream=data, filetype="pdf") as pdf:
            pages = [(str(i + 1), page.get_text("text")) for i, page in enumerate(pdf)]
        title = ""
    elif kind == "url":
        soup = BeautifulSoup(data, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()
        pages = [("", soup.get_text("\n", strip=True))]
    else:
        pages, title = [("", data.decode("utf-8"))], ""
    pages = [(page, text.strip()) for page, text in pages if text.strip()]
    if not pages:
        raise ValueError("未提取到文本；扫描件需要 OCR，暂不支持")
    return pages, title


class KnowledgeBase:
    def __init__(self, store, embedding_client=None):
        self.store = store
        self.embedding_client = embedding_client

    def list(self, query="", include_deleted=False):
        with self.store.pool.connection() as c:
            return [dict(r) for r in c.execute("""SELECT * FROM knowledge_documents WHERE (%s OR deleted_at IS NULL)
                AND (title ILIKE %s OR source_url ILIKE %s) ORDER BY created_at DESC""", (include_deleted, f"%{query}%", f"%{query}%")).fetchall()]

    def get(self, doc_id, *, include_deleted=False, version=None):
        with self.store.pool.connection() as c:
            row = c.execute("SELECT * FROM knowledge_documents WHERE document_id=%s AND (%s OR deleted_at IS NULL)", (doc_id, include_deleted)).fetchone()
            if not row:
                raise KeyError(doc_id)
            result = dict(row)
            requested_version = version if version is not None else row['active_version']
            if requested_version:
                version_row = c.execute("SELECT raw_text, fetched_at, version FROM knowledge_versions WHERE document_id=%s AND version=%s", (doc_id, requested_version)).fetchone()
                if not version_row:
                    raise KeyError(doc_id)
                result.update(dict(version_row))
            return result

    def create(self, kind, content, title="", filename="", duplicate_action="check", document_id=None):
        if kind not in ("text", "pdf", "url"):
            raise ValueError("不支持的资料类型")
        data = content.encode() if isinstance(content, str) else content
        if not data or len(data) > MAX_BYTES:
            raise ValueError("内容为空或超过大小限制")
        if kind == "pdf" and not data.startswith(b"%PDF-"):
            raise ValueError("文件不是有效的 PDF")
        digest = hashlib.sha256(data).hexdigest()
        with self.store.pool.connection() as c, c.transaction():
            existing = c.execute("SELECT d.document_id,d.title FROM knowledge_documents d JOIN knowledge_versions v ON v.document_id=d.document_id WHERE v.content_hash=%s AND d.deleted_at IS NULL LIMIT 1", (digest,)).fetchone()
            if existing and duplicate_action == "check":
                return {"duplicate": dict(existing)}
            if duplicate_action == "version":
                if not document_id:
                    raise ValueError("创建版本需要指定资料")
                doc = c.execute("SELECT * FROM knowledge_documents WHERE document_id=%s AND deleted_at IS NULL FOR UPDATE", (document_id,)).fetchone()
                if not doc or doc['source_type'] != kind or doc['pending_version']:
                    raise ValueError("资料不存在、类型不一致或正在处理")
                doc_id = document_id
                version = c.execute("SELECT COALESCE(MAX(version),0)+1 AS n FROM knowledge_versions WHERE document_id=%s", (doc_id,)).fetchone()['n']
                c.execute("UPDATE knowledge_documents SET pending_version=%s,status='received',error='',updated_at=now() WHERE document_id=%s", (version, doc_id))
            else:
                doc_id, version = uuid.uuid4().hex, 1
                c.execute("INSERT INTO knowledge_documents(document_id,title,source_type,source_url,filename,pending_version) VALUES (%s,%s,%s,%s,%s,%s)", (doc_id, title.strip() or filename or (data.decode()[:80] if kind == 'url' else '未命名资料'), kind, data.decode() if kind == 'url' else '', filename, version))
            c.execute("INSERT INTO knowledge_versions(document_id,version,source_data,content_hash) VALUES (%s,%s,%s,%s)", (doc_id, version, data, digest))
        self.start(doc_id, version)
        return self.get(doc_id)

    def start(self, doc_id, version):
        threading.Thread(target=self.process, args=(doc_id, version), daemon=True).start()

    def retry(self, doc_id, refresh=False):
        with self.store.pool.connection() as c, c.transaction():
            doc = c.execute("SELECT * FROM knowledge_documents WHERE document_id=%s AND deleted_at IS NULL FOR UPDATE", (doc_id,)).fetchone()
            if not doc:
                raise KeyError(doc_id)
            if doc['status'] in ('received', 'processing'):
                raise ValueError('资料正在处理')
            version = doc['pending_version']
            if not version:
                version = c.execute("SELECT COALESCE(MAX(version),0)+1 AS n FROM knowledge_versions WHERE document_id=%s", (doc_id,)).fetchone()['n']
                c.execute("INSERT INTO knowledge_versions(document_id,version,source_data,content_hash) SELECT document_id,%s,source_data,content_hash FROM knowledge_versions WHERE document_id=%s AND version=%s", (version, doc_id, doc['active_version']))
            c.execute("UPDATE knowledge_documents SET pending_version=%s,status='received',error='',updated_at=now() WHERE document_id=%s", (version, doc_id))
        self.start(doc_id, version)
        return self.get(doc_id)

    def process(self, doc_id, version):
        try:
            with self.store.pool.connection() as c, c.transaction():
                doc = c.execute("SELECT * FROM knowledge_documents WHERE document_id=%s AND pending_version=%s AND deleted_at IS NULL FOR UPDATE", (doc_id, version)).fetchone()
                if not doc:
                    return
                c.execute("UPDATE knowledge_documents SET status='processing',updated_at=now() WHERE document_id=%s", (doc_id,))
                row = c.execute("SELECT source_data FROM knowledge_versions WHERE document_id=%s AND version=%s", (doc_id, version)).fetchone()
                data = bytes(row['source_data'])
            fetched = None
            if doc['source_type'] == 'url':
                data, _, fetched = fetch_public_url(data.decode())
            pages, suggested_title = parse_source(doc['source_type'], data)
            title = suggested_title if doc['title'] == doc['source_url'] and suggested_title else doc['title']
            raw_text = '\n\n'.join(text for _, text in pages)
            chunks = [(page, chunk) for page, text in pages for chunk in split_text(text)]
            vectors = None
            semantic_status = 'not_configured'
            if self.embedding_client:
                try:
                    vectors = self.embedding_client.embed_texts([chunk for _, chunk in chunks])
                    if len(vectors) != len(chunks) or any(len(vector) != EMBEDDING_DIMENSIONS for vector in vectors):
                        raise ValueError("向量数量不匹配")
                    semantic_status = 'ready'
                except Exception:
                    vectors = None
                    semantic_status = 'failed'
            with self.store.pool.connection() as c, c.transaction():
                current = c.execute("SELECT * FROM knowledge_documents WHERE document_id=%s FOR UPDATE", (doc_id,)).fetchone()
                if not current or current['deleted_at'] or current['pending_version'] != version:
                    return
                c.execute("DELETE FROM rag_chunks WHERE report_id=%s AND kind='user_knowledge'", (doc_id,))
                for i, (page, text) in enumerate(chunks):
                    row = dict(company=current['company'], source_title=title, source_type='user_knowledge', section='', text=text)
                    embedding = vectors[i] if vectors is not None else None
                    c.execute("""INSERT INTO rag_chunks(chunk_id,report_id,kind,company,source_title,source_url,source_type,page,text,search_text,semantic_text,content_hash,embedding,embedding_status,embedding_model,embedded_at)
                        VALUES (%s,%s,'user_knowledge',%s,%s,%s,'user_knowledge',%s,%s,%s,%s,%s,%s,%s,%s,CASE WHEN %s THEN now() ELSE NULL END)""",
                        (f'knowledge:{doc_id}:{version}:{i}', doc_id, current['company'], title, current['source_url'], page, text, rag_search_text(row), rag_semantic_text(row), stable_content_hash(row), embedding, 'ready' if embedding is not None else 'missing', getattr(self.embedding_client, 'model', '') if embedding is not None else '', embedding is not None))
                c.execute("UPDATE knowledge_versions SET raw_text=%s, fetched_at=CASE WHEN %s::text IS NOT NULL THEN now() ELSE NULL END WHERE document_id=%s AND version=%s", (raw_text, fetched, doc_id, version))
                c.execute("UPDATE knowledge_documents SET title=%s,content_hash=%s,active_version=%s,pending_version=NULL,status='ready',semantic_status=%s,error='',updated_at=now() WHERE document_id=%s", (title, hashlib.sha256(data).hexdigest(), version, semantic_status, doc_id))
        except Exception as exc:
            with self.store.pool.connection() as c, c.transaction():
                c.execute("UPDATE knowledge_documents SET status='failed',error=%s,updated_at=now() WHERE document_id=%s AND pending_version=%s AND deleted_at IS NULL", (str(exc)[:500], doc_id, version))

    def update(self, doc_id, title, tags, topic, company):
        with self.store.pool.connection() as c, c.transaction():
            row = c.execute("UPDATE knowledge_documents SET title=%s,tags=%s,topic=%s,company=%s,updated_at=now() WHERE document_id=%s AND deleted_at IS NULL RETURNING document_id", (title, tags, topic, company, doc_id)).fetchone()
            if not row:
                raise KeyError(doc_id)
            c.execute("UPDATE rag_chunks SET source_title=%s,company=%s,search_text=concat_ws(E'\\n', %s::text, %s::text, source_type, section, text), semantic_text=concat_ws(E'\\n', %s::text, %s::text, section, source_type, text), embedding=NULL, embedding_status='stale' WHERE report_id=%s AND kind='user_knowledge'", (title, company, company, title, title, company, doc_id))
            c.execute("UPDATE knowledge_documents SET semantic_status='stale' WHERE document_id=%s", (doc_id,))
        return self.get(doc_id)

    def delete(self, doc_id):
        with self.store.pool.connection() as c, c.transaction():
            row = c.execute("UPDATE knowledge_documents SET deleted_at=now(),status='deleted',pending_version=NULL,updated_at=now() WHERE document_id=%s AND deleted_at IS NULL RETURNING document_id", (doc_id,)).fetchone()
            if not row:
                raise KeyError(doc_id)
            c.execute("DELETE FROM rag_chunks WHERE report_id=%s AND kind='user_knowledge'", (doc_id,))
            c.execute("UPDATE knowledge_versions SET source_data=NULL,raw_text='' WHERE document_id=%s", (doc_id,))

    def recover(self):
        with self.store.pool.connection() as c, c.transaction():
            rows = c.execute("UPDATE knowledge_documents SET status='failed',error='处理因服务重启中断，请重试' WHERE status IN ('received','processing') AND deleted_at IS NULL RETURNING document_id").fetchall()
        return len(rows)
