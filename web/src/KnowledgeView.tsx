import { ContextMenu } from "./workspace/ContextMenu";
import { Inspector } from "./workspace/Inspector";
import { useInspectorWidth } from "./workspace/useWorkspaceLayout";
import { useCallback, useEffect, useRef, useState } from "react";

type Doc = { document_id: string; title: string; source_type: string; source_url: string; filename: string; status: string; semantic_status: string; error: string; active_version: number | null; tags: string[]; topic: string; company: string; raw_text?: string; created_at: string };
type Duplicate = { duplicate: { document_id: string; title: string } };

async function call<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api/knowledge${path}`, { ...options, headers: { "Content-Type": "application/json", ...options?.headers } });
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || "操作失败"); }
  return response.status === 204 ? undefined as T : response.json();
}

export default function KnowledgeView({ active = true }: { active?: boolean }) {
  const [inspectorWidth, setInspectorWidth] = useInspectorWidth("knowledge");
  const [docs, setDocs] = useState<Doc[]>([]);
  const initialDocument = useRef<string | null>(null);
  if (initialDocument.current === null) { try { initialDocument.current = localStorage.getItem("aika.layout.knowledge.document") || ""; } catch { initialDocument.current = ""; } }
  const [selected, setSelected] = useState<Doc | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(() => { try { return new URLSearchParams(window.location.search).has("knowledge_document") || localStorage.getItem("aika.layout.knowledge.inspectorOpen") !== "false"; } catch { return true; } });
  useEffect(() => { try { localStorage.setItem("aika.layout.knowledge.inspectorOpen", String(inspectorOpen)); } catch { /* storage unavailable */ } }, [inspectorOpen]);
  useEffect(() => { try { if (selected) localStorage.setItem("aika.layout.knowledge.document", selected.document_id); } catch { /* storage unavailable */ } }, [selected]);
  const focusPrior = useRef(true);
  useEffect(() => {
    if (!active) return;
    const toggle = () => setInspectorOpen(open => !open);
    const reset = () => { setInspectorOpen(true); setListView("detailed"); };
    const focus = (event: Event) => { if ((event as CustomEvent<boolean>).detail) { setInspectorOpen(open => { focusPrior.current = open; return false; }); } else setInspectorOpen(focusPrior.current); };
    window.addEventListener("aika:toggle-inspector", toggle); window.addEventListener("aika:reset-layout", reset); window.addEventListener("aika:focus", focus);
    return () => { window.removeEventListener("aika:toggle-inspector", toggle); window.removeEventListener("aika:reset-layout", reset); window.removeEventListener("aika:focus", focus); };
  }, [active]);
  const [kind, setKind] = useState("text");
  const [content, setContent] = useState("");
  const [title, setTitle] = useState("");
  const [filename, setFilename] = useState("");
  const [query, setQuery] = useState("");
  const [message, setMessage] = useState("");
  const [loadError, setLoadError] = useState(false);
  const [busy, setBusy] = useState(false);
  const [adding, setAdding] = useState(false);
  const [filter, setFilter] = useState("all");
  const [listView, setListView] = useState<"detailed" | "compact">(() => { try { return localStorage.getItem("aika.layout.knowledge.listView") === "compact" ? "compact" : "detailed"; } catch { return "detailed"; } });
  useEffect(() => { try { localStorage.setItem("aika.layout.knowledge.listView", listView); } catch { /* storage unavailable */ } }, [listView]);
  const refresh = useCallback(async () => { try { const result = await call<{ documents: Doc[] }>(`/documents?q=${encodeURIComponent(query)}`); setDocs(result.documents); setLoadError(false); } catch (error) { setLoadError(true); setMessage(error instanceof Error ? error.message : "加载失败"); } }, [query]);
  useEffect(() => { if (!active) return; void refresh(); const timer = window.setInterval(() => void refresh(), 3000); return () => window.clearInterval(timer); }, [refresh, active]);
  async function open(id: string) { try { setSelected(await call<Doc>(`/documents/${id}`)); setInspectorOpen(true); } catch (error) { setMessage(String(error)); } }
  useEffect(() => { const params = new URLSearchParams(window.location.search); const id = params.get("knowledge_document") || (inspectorOpen ? initialDocument.current : ""); const version = params.get("knowledge_version"); if (id) void call<Doc>(`/documents/${encodeURIComponent(id)}${version ? `?version=${encodeURIComponent(version)}` : ""}`).then(setSelected).catch(error => setMessage(String(error))); }, []);
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
      setMessage(`已接收《${result.title}》，请查看处理状态。`); setContent(""); setTitle(""); setFilename(""); setAdding(false); await refresh();
    } catch (error) { setMessage(error instanceof Error ? error.message : "收录失败"); } finally { setBusy(false); }
  }
  async function remove(doc: Doc) { if (!window.confirm(`确定删除《${doc.title}》？原文及检索索引将被清理。`)) return; try { await call(`/documents/${doc.document_id}`, { method: "DELETE" }); setSelected(null); try { localStorage.removeItem("aika.layout.knowledge.document"); } catch { /* storage unavailable */ } await refresh(); } catch (error) { setMessage(String(error)); } }
  async function retry(doc: Doc) { try { await call(`/documents/${doc.document_id}/retry`, { method: "POST" }); await refresh(); await open(doc.document_id); } catch (error) { setMessage(String(error)); } }
  async function save(doc: Doc) { try { const result = await call<Doc>(`/documents/${doc.document_id}`, { method: "PATCH", body: JSON.stringify({ title: doc.title, tags: doc.tags, topic: doc.topic, company: doc.company }) }); setSelected(result); await refresh(); } catch (error) { setMessage(String(error)); } }
  useEffect(() => { if (!adding) return; const close = (event: KeyboardEvent) => { if (event.key === "Escape") setAdding(false); }; window.addEventListener("keydown", close); return () => window.removeEventListener("keydown", close); }, [adding]);
  const visibleDocs = docs.filter(doc => filter === "all" || doc.status === filter);
  return <section className="page-panel knowledge-page">
    <div className="knowledge-heading"><div className="page-heading"><span className="eyebrow">KNOWLEDGE BASE</span><h1>知识库</h1><p>收集研究资料，让每一个判断有据可循。</p></div><div><button className="knowledge-primary" onClick={() => setAdding(true)}>＋ 添加资料</button>{selected && !inspectorOpen && <button type="button" onClick={() => setInspectorOpen(true)}>显示详情</button>}</div></div>
    {message && <p className="knowledge-notice" role="status">{message}<button type="button" onClick={() => setMessage("")} aria-label="关闭提示">×</button></p>}
    <div className={`knowledge-grid ${selected && inspectorOpen ? "has-detail" : ""} list-${listView}`}><div className="knowledge-list-panel">
      <div className="knowledge-list-heading"><div><span className="eyebrow">YOUR LIBRARY</span><h2>研究资料 <small>{docs.length}</small></h2></div></div>
      <div className="knowledge-filters"><input aria-label="搜索资料" placeholder="搜索标题或网址" value={query} onChange={e => setQuery(e.target.value)} /><select aria-label="筛选索引状态" value={filter} onChange={e => setFilter(e.target.value)}><option value="all">全部状态</option><option value="ready">可检索</option><option value="processing">处理中</option><option value="failed">失败</option><option value="received">已接收</option></select><button type="button" className="list-view-toggle" onClick={() => setListView(listView === "compact" ? "detailed" : "compact")}>{listView === "compact" ? "详细视图" : "紧凑视图"}</button></div>
      <div className="knowledge-list-label"><span>资料 / 来源</span><span>索引状态</span></div>
      {visibleDocs.map(doc => <div className={`knowledge-item ${selected?.document_id === doc.document_id ? "selected" : ""}`} key={doc.document_id}><ContextMenu label={`资料操作：${doc.title}`} actions={[{label:"查看详情",run:()=>void open(doc.document_id)},{label:"重新处理",run:()=>void retry(doc)},{label:"删除",run:()=>void remove(doc)}]}><button type="button" onClick={() => void open(doc.document_id)}><strong>{doc.title}</strong><small>{doc.source_type?.toUpperCase()} · {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : ""}</small></button></ContextMenu><span className={`knowledge-status status-${doc.status}`}>{({received:"已接收",processing:"处理中",ready:"可检索",failed:"失败"} as Record<string,string>)[doc.status] || doc.status}</span></div>)}
      {!loadError && !visibleDocs.length && <div className="knowledge-empty"><strong>{query || filter !== "all" ? "没有匹配的资料" : "你的研究资料将从这里开始"}</strong><p>{query || filter !== "all" ? "尝试调整搜索或筛选条件。" : "添加文本、PDF 或公开网页，索引完成后即可在问答中引用。"}</p><button onClick={() => setAdding(true)}>添加资料 →</button></div>}
    </div>{selected && inspectorOpen && <Inspector title="资料详情" label="DOCUMENT INSPECTOR" width={inspectorWidth} onWidth={setInspectorWidth} onClose={() => setInspectorOpen(false)}><div className="drawer-body knowledge-detail">
      <label>标题<input value={selected.title} onChange={e => setSelected({ ...selected, title: e.target.value })} /></label>
      <label>标签（逗号分隔）<input value={selected.tags.join(",")} onChange={e => setSelected({ ...selected, tags: e.target.value.split(",") })} /></label>
      <label>主题<input value={selected.topic} onChange={e => setSelected({ ...selected, topic: e.target.value })} /></label>
      <label>公司<input value={selected.company} onChange={e => setSelected({ ...selected, company: e.target.value })} /></label>
      <p>状态：{selected.status}；语义索引：{selected.semantic_status}</p>{selected.error && <p role="alert">{selected.error}</p>}
      <p>来源：{selected.source_url ? <a href={selected.source_url} target="_blank" rel="noreferrer">{selected.source_url}</a> : selected.filename || "文本"}</p>
      <div className="knowledge-detail-actions"><button type="button" onClick={() => void save(selected)}>保存信息</button><button type="button" onClick={() => void retry(selected)}>重新处理 / 刷新</button><button type="button" onClick={() => void remove(selected)}>删除</button></div>
      <h3>原文</h3><pre className="knowledge-text">{selected.raw_text || "暂无可用原文"}</pre>
    </div></Inspector>}</div>
    {adding && <div className="command-backdrop" onMouseDown={() => setAdding(false)}><div className="knowledge-dialog" role="dialog" aria-modal="true" aria-label="添加资料" onMouseDown={e => e.stopPropagation()}><div className="knowledge-detail-header"><div><span className="eyebrow">ADD TO LIBRARY</span><h2>添加资料</h2></div><button type="button" onClick={() => setAdding(false)} aria-label="关闭">×</button></div><p>选择来源并提交，完成索引后即可检索。</p><label>资料类型<select value={kind} onChange={e => { setKind(e.target.value); setContent(""); }}><option value="text">文本</option><option value="url">公开网页</option><option value="pdf">PDF</option></select></label><label>标题（可选）<input value={title} onChange={e => setTitle(e.target.value)} /></label>{kind === "pdf" ? <label>选择 PDF<input type="file" accept="application/pdf,.pdf" onChange={e => { const file = e.target.files?.[0]; if (!file) { setContent(""); return; } setFilename(file.name); const reader = new FileReader(); reader.onload = () => setContent(String(reader.result).split(",")[1] || ""); reader.readAsDataURL(file); }} /></label> : <label>{kind === "url" ? "网页地址" : "资料内容"}<textarea placeholder={kind === "url" ? "https://..." : "粘贴文本"} value={content} onChange={e => setContent(e.target.value)} /></label>}<button className="knowledge-primary" type="button" disabled={busy || !content} onClick={() => void add()}>{busy ? "正在提交…" : "收录到知识库"}</button></div></div>}
  </section>;
}
