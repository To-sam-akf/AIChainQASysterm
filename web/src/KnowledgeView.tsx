import { useCallback, useEffect, useState } from "react";

type Doc = { document_id: string; title: string; source_type: string; source_url: string; filename: string; status: string; semantic_status: string; error: string; active_version: number | null; tags: string[]; topic: string; company: string; raw_text?: string; created_at: string };
type Duplicate = { duplicate: { document_id: string; title: string } };

async function call<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api/knowledge${path}`, { ...options, headers: { "Content-Type": "application/json", ...options?.headers } });
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || "操作失败"); }
  return response.status === 204 ? undefined as T : response.json();
}

export default function KnowledgeView() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [selected, setSelected] = useState<Doc | null>(null);
  const [kind, setKind] = useState("text");
  const [content, setContent] = useState("");
  const [title, setTitle] = useState("");
  const [filename, setFilename] = useState("");
  const [query, setQuery] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const refresh = useCallback(async () => { try { const result = await call<{ documents: Doc[] }>(`/documents?q=${encodeURIComponent(query)}`); setDocs(result.documents); } catch (error) { setMessage(error instanceof Error ? error.message : "加载失败"); } }, [query]);
  useEffect(() => { void refresh(); const timer = window.setInterval(() => void refresh(), 3000); return () => window.clearInterval(timer); }, [refresh]);
  async function open(id: string) { try { setSelected(await call<Doc>(`/documents/${id}`)); } catch (error) { setMessage(String(error)); } }
  useEffect(() => { const params = new URLSearchParams(window.location.search); const id = params.get("knowledge_document"); const version = params.get("knowledge_version"); if (id) void call<Doc>(`/documents/${encodeURIComponent(id)}${version ? `?version=${encodeURIComponent(version)}` : ""}`).then(setSelected).catch(error => setMessage(String(error))); }, []);
  async function add() {
    if (!content || busy) return;
    setBusy(true); setMessage("");
    try {
      const payload = { kind, content, title, filename, duplicate_action: "check" };
      let result = await call<Doc | Duplicate>("/documents", { method: "POST", body: JSON.stringify(payload) });
      if ("duplicate" in result) {
        if (!window.confirm(`已存在《${result.duplicate.title}》。是否为其创建新版本？`)) return;
        result = await call<Doc>("/documents", { method: "POST", body: JSON.stringify({ ...payload, duplicate_action: "version", document_id: result.duplicate.document_id }) });
      }
      setMessage(`已接收《${result.title}》，请查看处理状态。`); setContent(""); setTitle(""); setFilename(""); await refresh();
    } catch (error) { setMessage(error instanceof Error ? error.message : "收录失败"); } finally { setBusy(false); }
  }
  async function remove(doc: Doc) { if (!window.confirm(`确定删除《${doc.title}》？原文及检索索引将被清理。`)) return; try { await call(`/documents/${doc.document_id}`, { method: "DELETE" }); setSelected(null); await refresh(); } catch (error) { setMessage(String(error)); } }
  async function retry(doc: Doc) { try { await call(`/documents/${doc.document_id}/retry`, { method: "POST" }); await refresh(); await open(doc.document_id); } catch (error) { setMessage(String(error)); } }
  async function save(doc: Doc) { try { const result = await call<Doc>(`/documents/${doc.document_id}`, { method: "PATCH", body: JSON.stringify({ title: doc.title, tags: doc.tags, topic: doc.topic, company: doc.company }) }); setSelected(result); await refresh(); } catch (error) { setMessage(String(error)); } }
  return <section className="page-panel knowledge-page">
    <div className="page-heading"><span className="eyebrow">KNOWLEDGE BASE</span><h1>知识库</h1><p>收录资料，完成索引后即可用于问答。</p></div>
    {message && <p role="status">{message}</p>}
    <div className="knowledge-grid"><div>
      <h2>添加资料</h2>
      <label>类型 <select value={kind} onChange={e => { setKind(e.target.value); setContent(""); }}><option value="text">文本</option><option value="url">公开网页</option><option value="pdf">PDF</option></select></label>
      <input placeholder="标题（可选）" value={title} onChange={e => setTitle(e.target.value)} />
      {kind === "pdf" ? <input type="file" accept="application/pdf,.pdf" onChange={e => { const file = e.target.files?.[0]; if (!file) return; setFilename(file.name); const reader = new FileReader(); reader.onload = () => setContent(String(reader.result).split(",")[1] || ""); reader.readAsDataURL(file); }} /> : <textarea placeholder={kind === "url" ? "https://..." : "粘贴文本"} value={content} onChange={e => setContent(e.target.value)} />}
      <button type="button" disabled={busy || !content} onClick={() => void add()}>收录到知识库</button>
      <h2>资料列表</h2><input placeholder="搜索标题或网址" value={query} onChange={e => setQuery(e.target.value)} />
      {docs.map(doc => <div className="knowledge-item" key={doc.document_id}><button type="button" onClick={() => void open(doc.document_id)}>{doc.title}</button><span>{({received:"已接收",processing:"处理中",ready:"可检索",failed:"失败"} as Record<string,string>)[doc.status] || doc.status}</span></div>)}
    </div><div><h2>资料详情</h2>{selected ? <>
      <label>标题<input value={selected.title} onChange={e => setSelected({ ...selected, title: e.target.value })} /></label>
      <label>标签（逗号分隔）<input value={selected.tags.join(",")} onChange={e => setSelected({ ...selected, tags: e.target.value.split(",") })} /></label>
      <label>主题<input value={selected.topic} onChange={e => setSelected({ ...selected, topic: e.target.value })} /></label>
      <label>公司<input value={selected.company} onChange={e => setSelected({ ...selected, company: e.target.value })} /></label>
      <p>状态：{selected.status}；语义索引：{selected.semantic_status}</p>{selected.error && <p role="alert">{selected.error}</p>}
      <p>来源：{selected.source_url ? <a href={selected.source_url} target="_blank" rel="noreferrer">{selected.source_url}</a> : selected.filename || "文本"}</p>
      <button type="button" onClick={() => void save(selected)}>保存信息</button>{" "}
      <button type="button" onClick={() => void retry(selected)}>重新处理 / 刷新</button>{" "}
      <button type="button" onClick={() => void remove(selected)}>删除</button>
      <pre className="knowledge-text">{selected.raw_text || "暂无可用原文"}</pre>
    </> : <p>选择一份资料查看内容和状态。</p>}</div></div>
  </section>;
}
