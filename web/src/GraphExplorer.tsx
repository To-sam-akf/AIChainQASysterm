import { ContextMenu } from "./workspace/ContextMenu";
import { Inspector } from "./workspace/Inspector";
import { useInspectorWidth } from "./workspace/useWorkspaceLayout";
import Graph3D, { nodeColor, type Graph3DHandle, type GraphNode as Node, type GraphRelation as Relation } from './Graph3D';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './graph-explorer.css';

type Branch = { node: Node; count: number };
type Selection = { node?: Node; edge?: Relation };
type Snapshot = { branches: Branch[]; selected: Selection; detailOpen: boolean; hidden: string[] };
type Page = { node: Node; items: Relation[]; total: number; offset: number };
const labels: Record<string, string> = { Company: '公司', Technology: '技术', Product: '产品', IndustryChain: '产业链', ValueChainSegment: '产业链环节', IndustryConcept: '产业概念', Metric: '指标', Risk: '风险', Report: '报告' };
const typeName = (t: string) => labels[t] || t;
async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`加载失败 (${response.status})`);
  return response.json() as Promise<T>;
}
function readState(): { branches: Branch[]; filter: string; hidden: string[]; view: 'graph' | 'list' } {
  try {
    const params = new URLSearchParams(location.search);
    const parsed: unknown = JSON.parse(params.get('graph') || '[]');
    const branches = Array.isArray(parsed) ? parsed.filter((x): x is Branch => x && typeof x.node?.id === 'string' && typeof x.node.name === 'string' && typeof x.node.type === 'string' && Number.isInteger(x.count) && x.count >= 20 && x.count <= 1000).slice(0, 30) : [];
    return { branches, filter: params.get('graph_relation') || '', hidden: params.getAll('graph_hide'), view: params.get('graph_mode') === 'list' ? 'list' : 'graph' };
  } catch { return { branches: [], filter: '', hidden: [], view: 'graph' }; }
}

export default function GraphExplorer({ active = true }: { active?: boolean }) {
  const [inspectorWidth, setInspectorWidth] = useInspectorWidth("graph");
  const initial = useRef(readState());
  const [branches, setBranches] = useState<Branch[]>(initial.current.branches);
  const [filter, setFilter] = useState(initial.current.filter);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Node[]>([]);
  const [pages, setPages] = useState<Page[]>([]);
  const [selected, setSelected] = useState<Selection>({});
  const [view, setView] = useState<'graph' | 'list'>(initial.current.view);
  const [hidden, setHidden] = useState<string[]>(initial.current.hidden);
  const [unavailable, setUnavailable] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [quick, setQuick] = useState<Node[]>([]);
  const [detailOpen, setDetailOpen] = useState(() => { try { return localStorage.getItem("aika.layout.graph.inspectorOpen") === "true"; } catch { return false; } });
  useEffect(() => { try { localStorage.setItem("aika.layout.graph.inspectorOpen", String(detailOpen)); } catch { /* storage unavailable */ } }, [detailOpen]);
  const [immersive, setImmersive] = useState(false);
  const focusPrior = useRef(true);
  useEffect(() => {
    if (!active) return;
    const toggle = () => setDetailOpen(open => !open);
    const reset = () => setDetailOpen(false);
    const focus = (event: Event) => { if ((event as CustomEvent<boolean>).detail) { setDetailOpen(open => { focusPrior.current = open; return false; }); } else setDetailOpen(focusPrior.current); };
    window.addEventListener("aika:toggle-inspector", toggle); window.addEventListener("aika:reset-layout", reset); window.addEventListener("aika:focus", focus);
    return () => { window.removeEventListener("aika:toggle-inspector", toggle); window.removeEventListener("aika:reset-layout", reset); window.removeEventListener("aika:focus", focus); };
  }, [active]);
  const [context, setContext] = useState<{ x: number; y: number; selection: Selection } | null>(null);
  const [steps, setSteps] = useState<Snapshot[]>([]);
  const [relationOptions, setRelationOptions] = useState<Record<string, string>>({});
  const graph = useRef<Graph3DHandle | null>(null);
  useEffect(() => { if (!context) return; const close = () => setContext(null); const key = (event: KeyboardEvent) => { if (event.key === "Escape") { event.stopPropagation(); close(); } }; document.addEventListener("pointerdown", close); document.addEventListener("keydown", key, true); return () => { document.removeEventListener("pointerdown", close); document.removeEventListener("keydown", key, true); }; }, [context]);
  useEffect(() => { if (!immersive) return; const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setImmersive(false); }; window.addEventListener("keydown", escape); return () => window.removeEventListener("keydown", escape); }, [immersive]);
  const writingHistory = useRef(false);

  useEffect(() => { Promise.all(['液冷', '光模块', 'AI服务器'].map(q => getJSON<{items: Node[]}>(`/api/graph/entities?q=${encodeURIComponent(q)}`).then(x => x.items.find(n => n.name === q) || x.items[0]).catch(() => undefined))).then(items => setQuick([...new Map(items.filter((n): n is Node => !!n).map(n => [n.id, n])).values()])); }, []);
  useEffect(() => { getJSON<{ relation_options: Record<string, string> }>('/api/graph/summary').then(x => setRelationOptions(x.relation_options)).catch(() => setError('无法加载图谱信息，请检查后端。')); }, []);
  useEffect(() => {
    if (!query.trim()) { setResults([]); return; }
    const controller = new AbortController();
    const timer = setTimeout(() => { fetch(`/api/graph/entities?q=${encodeURIComponent(query.trim())}`, { signal: controller.signal }).then(r => { if (!r.ok) throw new Error(); return r.json(); }).then(x => setResults(x.items)).catch(e => { if (e.name !== 'AbortError') setError('搜索失败，请检查后端。'); }); }, 200);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [query]);
  useEffect(() => {
    if (writingHistory.current) { writingHistory.current = false; return; }
    const url = new URL(location.href);
    url.searchParams.set('graph_view', '1');
    if (branches.length) url.searchParams.set('graph', JSON.stringify(branches)); else url.searchParams.delete('graph');
    if (filter) url.searchParams.set('graph_relation', filter); else url.searchParams.delete('graph_relation');
    url.searchParams.delete('graph_hide'); hidden.forEach(type => url.searchParams.append('graph_hide', type));
    if (view === 'list') url.searchParams.set('graph_mode', 'list'); else url.searchParams.delete('graph_mode');
    if (url.href !== location.href) history.pushState(null, '', url);
  }, [branches, filter, hidden, view]);
  useEffect(() => { const onPop = () => { const s = readState(); writingHistory.current = true; setBranches(s.branches); setFilter(s.filter); setHidden(s.hidden); setView(s.view); setSelected({}); setSteps([]); }; window.addEventListener('popstate', onPop); return () => window.removeEventListener('popstate', onPop); }, []);
  useEffect(() => {
    if (!branches.length) { setPages([]); setLoading(false); return; }
    let active = true;
    setLoading(true); setError('');
    Promise.all(branches.map(async branch => {
      const all: Relation[] = []; let total = 0; let node = branch.node;
      for (let offset = 0; offset < branch.count; offset += 100) {
        const params = new URLSearchParams({ offset: String(offset), limit: String(Math.min(100, branch.count - offset)), relation_type: filter });
        const page = await getJSON<Page>(`/api/graph/entities/${encodeURIComponent(branch.node.id)}/neighbors?${params}`);
        node = page.node; total = page.total; all.push(...page.items); if (all.length >= total) break;
      }
      return { node, items: all, total, offset: 0 };
    })).then(p => { if (active) setPages(p); }).catch(() => { if (active) { setPages([]); setError('无法加载图谱。节点可能已失效，请重新搜索。'); } }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [branches, filter]);
  const { nodes, edges } = useMemo(() => {
    const edgeMap = new Map<string, Relation>(); const nodeMap = new Map<string, Node>();
    for (const page of pages) { nodeMap.set(page.node.id, page.node); for (const e of page.items) { edgeMap.set(e.relation_id, e); nodeMap.set(e.source_id, { id: e.source_id, name: e.head_name, type: e.head_type }); nodeMap.set(e.target_id, { id: e.target_id, name: e.tail_name, type: e.tail_type }); } }
    return { nodes: [...nodeMap.values()], edges: [...edgeMap.values()] };
  }, [pages]);
  const visibleNodes = useMemo(() => nodes.filter(n => !hidden.includes(n.type)), [nodes, hidden]);
  const visibleEdges = useMemo(() => { const ids = new Set(visibleNodes.map(n => n.id)); return edges.filter(e => ids.has(e.source_id) && ids.has(e.target_id)); }, [edges, visibleNodes]);
  const typeCounts = useMemo(() => { const counts = new Map<string, number>(); nodes.forEach(n => counts.set(n.type, (counts.get(n.type) || 0) + 1)); return [...counts]; }, [nodes]);
  const remember = () => setSteps(old => [...old, { branches, selected, detailOpen, hidden }].slice(-100));
  const select = (next: Selection) => {
    if (next.node?.id === selected.node?.id && next.edge?.relation_id === selected.edge?.relation_id && detailOpen) return;
    remember(); setSelected(next); setDetailOpen(true);
  };
  const closeDetail = () => { if (!selected.node && !selected.edge && !detailOpen) return; remember(); setSelected({}); setDetailOpen(false); };
  const goBack = () => {
    const previous = steps[steps.length - 1]; if (!previous) return;
    setSteps(old => old.slice(0, -1)); setBranches(previous.branches); setSelected(previous.selected); setDetailOpen(previous.detailOpen); setHidden(previous.hidden);
  };
  const choose = (node: Node) => { remember(); setBranches([{ node, count: 20 }]); setHidden([]); setSelected({ node }); setDetailOpen(true); setQuery(''); setResults([]); };
  const expand = (node: Node) => { if (!branches.some(b => b.node.id === node.id)) { remember(); setBranches(old => [...old, { node, count: 20 }]); } };
  const onUnavailable = useCallback(() => { setUnavailable(true); setView('list'); }, []);
  const selectedPage = pages.find(p => p.node.id === selected.node?.id);
  return <section className={`page-panel graph-page explorer ${immersive ? "immersive" : ""}`}>
    <header className="explorer-header"><span className="eyebrow">AIKA / KNOWLEDGE UNIVERSE</span><h1>产业链图谱 <span>3D</span></h1><p>探索算力产业的连接，追溯每一条关系的证据。</p></header>
    <div className="explorer-search"><label htmlFor="graph-search">搜索知识图谱</label><input id="graph-search" value={query} onChange={e => setQuery(e.target.value)} placeholder="搜索公司、技术或产业链环节" autoComplete="off" />{query.trim() && <div className="explorer-results">{results.length ? results.map(n => <button key={n.id} onClick={() => choose(n)}><span>{n.name}</span><small>{typeName(n.type)}</small></button>) : <span>暂无匹配结果</span>}</div>}</div>
    <div className="explorer-quick">快速开始：{quick.map(node => <button key={node.id} onClick={() => choose(node)}>{node.name}</button>)}<button onClick={() => { remember(); setQuery(''); setBranches([]); setSelected({}); }}>重置</button></div>
    {error && <div className="warning" role="alert">{error}</div>}
    {context && <div className="graph-context-menu context-menu" role="menu" style={{ position:"fixed", left:Math.min(context.x, window.innerWidth - 165), top:Math.min(context.y, window.innerHeight - 100), right:"auto" }} onPointerDown={event => event.stopPropagation()}><button role="menuitem" onClick={() => { select(context.selection); setContext(null); }}>查看详情</button>{context.selection.node && <button role="menuitem" onClick={() => { expand(context.selection.node!); select(context.selection); setContext(null); }}>展开相关关系</button>}</div>}
    <div className={`explorer-workspace ${detailOpen ? "" : "detail-closed"}`}><div className="explorer-main"><div className="explorer-bar"><div><strong>关系图谱</strong><small>{branches.length ? `从 ${branches[0].node.name} 开始探索` : '搜索并选择节点开始探索'}</small></div><div className="explorer-controls"><button onClick={() => setImmersive(!immersive)}>{immersive ? "退出沉浸" : "沉浸探索"}</button><label>关系 <select value={filter} onChange={e => setFilter(e.target.value)}>{Object.entries(relationOptions).map(([label, value]) => <option key={label} value={value}>{label}</option>)}</select></label><button disabled={unavailable} onClick={() => setView(view === 'graph' ? 'list' : 'graph')}>{view === 'graph' ? '列表视图' : '图谱视图'}</button><button disabled={!steps.length} onClick={goBack}>上一步</button><button onClick={() => setDetailOpen(!detailOpen)}>{detailOpen ? '收起详情' : '显示详情'}</button><button onClick={() => navigator.clipboard.writeText(location.href).catch(() => setError('复制失败，请从地址栏复制链接。'))}>复制链接</button></div></div>
    {loading && <div className="explorer-loading">正在加载关系…</div>}
    {unavailable && <div className="explorer-fallback" role="status">当前设备无法使用 3D 图谱，已切换至列表视图。</div>}
    <div hidden={view !== 'graph' || unavailable} className="explorer-canvas-wrap"><Graph3D ref={graph} nodes={visibleNodes} edges={visibleEdges} selectedNode={selected.node?.id} selectedEdge={selected.edge?.relation_id} onNode={node => select({ node })} onEdge={edge => select({ edge })} onBackground={closeDetail} onNodeContext={(node,event) => { event.preventDefault(); setContext({ x:event.clientX, y:event.clientY, selection:{node} }); }} onEdgeContext={(edge,event) => { event.preventDefault(); setContext({ x:event.clientX, y:event.clientY, selection:{edge} }); }} onUnavailable={onUnavailable} />{!nodes.length && !loading && <div className="explorer-empty">搜索并选择一个节点，开始探索产业链关系。</div>}{!!nodes.length && <div className="explorer-zoom"><button onClick={() => graph.current?.zoom(0.8)}>＋</button><button onClick={() => graph.current?.zoom(1.2)}>−</button><button onClick={() => graph.current?.focus()}>聚焦</button><button onClick={() => graph.current?.reset()}>重置视角</button></div>}</div><div hidden={view === 'graph' && !unavailable} className="explorer-list"><h4>节点</h4><div className="explorer-list-nodes">{visibleNodes.map(n => <ContextMenu key={n.id} label={`${n.name}操作`} actions={[{label:"查看详情",run:()=>select({node:n})},{label:"展开相关关系",run:()=>expand(n)}]}><button onClick={() => select({ node: n })}>{n.name} · {typeName(n.type)}</button></ContextMenu>)}</div><h4>关系</h4>{visibleEdges.length ? visibleEdges.map(e => <ContextMenu key={e.relation_id} label={`${e.label}操作`} actions={[{label:"查看关系与证据",run:()=>select({edge:e})}]}><button onClick={() => select({ edge: e })}><span>{e.head_name}</span><small>{e.label} →</small><span>{e.tail_name}</span></button></ContextMenu>) : <p>暂无关系，请搜索节点或更换筛选条件。</p>}</div>
    <div className="explorer-footer"><span>{visibleNodes.length} 个节点 · {visibleEdges.length} 条关系</span><div className="explorer-legend">{typeCounts.map(([type, count]) => <button key={type} aria-pressed={!hidden.includes(type)} onClick={() => setHidden(old => old.includes(type) ? old.filter(t => t !== type) : [...old, type])}><i style={{ background: nodeColor(type) }} />{typeName(type)} <small>{count}</small></button>)}</div></div></div>
    {detailOpen && <Inspector title="详情" label="ENTITY INSPECTOR" width={inspectorWidth} onWidth={setInspectorWidth} onClose={closeDetail}><div className="drawer-body explorer-detail">{selected.node ? <><span className="explorer-badge">{typeName(selected.node.type)}</span><ContextMenu label={`${selected.node.name}操作`} actions={[{label:"展开相关关系",run:()=>selected.node && expand(selected.node)},{label:"聚焦节点",run:()=>graph.current?.focus()}]}><h3>{selected.node.name}</h3></ContextMenu><p>{selectedPage ? `已加载 ${selectedPage.items.length} / ${selectedPage.total} 条关系` : '选择展开以查看相关关系'}</p>{selectedPage ? <><button className="explorer-primary" disabled={selectedPage.items.length >= selectedPage.total} onClick={() => { remember(); setBranches(old => old.map(b => b.node.id === selected.node?.id ? { ...b, count: b.count + 20 } : b)); }}>加载更多关系</button>{branches.length > 1 && <button onClick={() => { remember(); setBranches(old => old.filter(b => b.node.id !== selected.node?.id)); setSelected({}); }}>收起此分支</button>}</> : <button className="explorer-primary" onClick={() => selected.node && expand(selected.node)}>展开相关关系</button>}<div className="explorer-node-list"><h4>当前关联节点</h4>{edges.filter(e => e.source_id === selected.node?.id || e.target_id === selected.node?.id).slice(0, 20).map(e => <button key={e.relation_id} onClick={() => select({ edge: e })}>{e.label} · {e.source_id === selected.node?.id ? e.tail_name : e.head_name}</button>)}</div></> : selected.edge ? <><span className="explorer-badge">关系</span><ContextMenu label={`${selected.edge.label}操作`} actions={[{label:"复制链接",run:()=>void navigator.clipboard.writeText(location.href).catch(()=>setError("复制失败，请从地址栏复制链接。"))}]}><h3>{selected.edge.label}</h3></ContextMenu><p>{selected.edge.head_name} → {selected.edge.tail_name}</p><h4>来源与证据</h4><p>{selected.edge.source_title || '暂无来源信息'}{selected.edge.page ? ` · 第 ${selected.edge.page} 页` : ''}</p><blockquote>{selected.edge.evidence || '暂无证据原文'}</blockquote></> : <p>选择节点或关系，查看详情与证据。</p>}</div></Inspector>}</div>
  </section>;
}
