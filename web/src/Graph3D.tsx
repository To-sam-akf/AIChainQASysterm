import ForceGraph3D, { type ForceGraph3DInstance, type NodeObject, type LinkObject } from '3d-force-graph';
import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';

export type GraphNode = { id: string; name: string; type: string };
export type GraphRelation = { relation_id: string; source_id: string; target_id: string; head_type: string; head_name: string; tail_type: string; tail_name: string; relation: string; label: string; evidence?: string; source_title?: string; page?: string; section?: string };
type RenderNode = NodeObject & GraphNode;
type RenderLink = LinkObject<RenderNode> & { id: string; relation: GraphRelation };
export type Graph3DHandle = { reset: () => void; zoom: (factor: number) => void; focus: () => void };
const palette: Record<string, string> = { Company: '#257e73', Technology: '#447db7', Product: '#8b66a8', IndustryChain: '#c18a41', ValueChainSegment: '#c18a41', IndustryConcept: '#378b96', Risk: '#ba6567', Metric: '#6c966b', Report: '#7989a4' };
export const nodeColor = (type: string) => palette[type] || '#76898b';

type Props = { nodes: GraphNode[]; edges: GraphRelation[]; selectedNode?: string; selectedEdge?: string; onNode: (node: GraphNode) => void; onEdge: (edge: GraphRelation) => void; onBackground: () => void; onUnavailable: () => void };

const Graph3D = forwardRef<Graph3DHandle, Props>(function Graph3D(props, ref) {
  const host = useRef<HTMLDivElement>(null);
  const overlay = useRef<HTMLDivElement>(null);
  const graph = useRef<ForceGraph3DInstance<RenderNode, RenderLink> | null>(null);
  const cache = useRef(new Map<string, RenderNode>());
  const current = useRef(props); current.current = props;
  const didFit = useRef(false);
  const duration = (ms: number) => window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : ms;
  useImperativeHandle(ref, () => ({
    reset: () => graph.current?.zoomToFit(duration(450), 35),
    zoom: factor => { const g = graph.current; if (!g) return; const p = g.cameraPosition(); g.cameraPosition({ x: p.x * factor, y: p.y * factor, z: p.z * factor }, undefined, duration(250)); },
    focus: () => { const g = graph.current; const n = current.current.nodes.find(n => n.id === current.current.selectedNode); const p = n && cache.current.get(n.id); if (!g || !p || p.x == null || p.y == null || p.z == null) return; g.cameraPosition({ x: p.x + 90, y: p.y + 65, z: p.z + 130 }, { x: p.x, y: p.y, z: p.z }, duration(500)); }
  }), []);
  useEffect(() => {
    const el = host.current; if (!el) return;
    let g: ForceGraph3DInstance<RenderNode, RenderLink>;
    try {
      const canvas = document.createElement('canvas');
      if (!canvas.getContext('webgl2') && !canvas.getContext('webgl')) throw new Error('WebGL unavailable');
      g = new ForceGraph3D(el, { controlType: 'orbit', rendererConfig: { antialias: true, alpha: false } }) as unknown as ForceGraph3DInstance<RenderNode, RenderLink>;
      graph.current = g;
      g.backgroundColor('#f6f7f4').showNavInfo(false).nodeId('id').nodeLabel(n => { const label = document.createElement('span'); label.textContent = n.name; return label; }).nodeRelSize(3.2).nodeResolution(12)
        .linkSource('source').linkTarget('target').linkWidth(0.7).linkDirectionalArrowLength(4).linkDirectionalArrowRelPos(0.86)
        .linkLabel(l => { const label = document.createElement('span'); label.textContent = `${l.relation.head_name} → ${l.relation.tail_name} · ${l.relation.label}`; return label; })
        .enableNodeDrag(false).cooldownTicks(110)
        .onNodeClick(n => current.current.onNode(n)).onLinkClick(l => current.current.onEdge(l.relation)).onBackgroundClick(() => current.current.onBackground());
      const linkForce = g.d3Force('link') as unknown as { distance: (value: number) => void } | undefined;
      const chargeForce = g.d3Force('charge') as unknown as { strength: (value: number) => void } | undefined;
      linkForce?.distance(60); chargeForce?.strength(-140);
      g.cameraPosition({ x: 100, y: 80, z: 260 });
      const resize = () => { g.width(el.clientWidth).height(el.clientHeight); };
      resize(); const observer = new ResizeObserver(resize); observer.observe(el);
      const onLost = (event: Event) => { event.preventDefault(); current.current.onUnavailable(); };
      g.renderer().domElement.addEventListener('webglcontextlost', onLost);
      let frame = 0;
      const labels = () => {
        if (overlay.current) {
          const children = overlay.current.children;
          for (let i = 0; i < children.length; i++) {
            const child = children[i] as HTMLElement; const node = cache.current.get(child.dataset.nodeId || '');
            if (!node || node.x == null || node.y == null || node.z == null) { child.style.display = 'none'; continue; }
            const point = g.graph2ScreenCoords(node.x, node.y, node.z); child.style.display = ''; child.style.transform = `translate(${point.x}px, ${point.y + 13}px) translate(-50%, 0)`;
          }
        }
        frame = requestAnimationFrame(labels);
      };
      frame = requestAnimationFrame(labels);
      return () => { cancelAnimationFrame(frame); observer.disconnect(); g.renderer().domElement.removeEventListener('webglcontextlost', onLost); g._destructor(); graph.current = null; cache.current.clear(); };
    } catch { current.current.onUnavailable(); }
  }, []);
  useEffect(() => {
    const g = graph.current; if (!g) return;
    const live = new Set(props.nodes.map(n => n.id));
    for (const id of cache.current.keys()) if (!live.has(id)) cache.current.delete(id);
    for (const n of props.nodes) { const existing = cache.current.get(n.id); if (existing) Object.assign(existing, n); else cache.current.set(n.id, { ...n }); }
    const renderNodes = props.nodes.map(n => cache.current.get(n.id)!);
    const renderEdges: RenderLink[] = props.edges.map(e => ({ id: e.relation_id, source: e.source_id, target: e.target_id, relation: e }));
    g.graphData({ nodes: renderNodes, links: renderEdges });
    if (!didFit.current && renderNodes.length) { didFit.current = true; const timer = window.setTimeout(() => g.zoomToFit(duration(500), 25), 900); return () => clearTimeout(timer); }
    if (!renderNodes.length) didFit.current = false;
  }, [props.nodes, props.edges]);
  useEffect(() => {
    const g = graph.current; if (!g) return;
    const focus = props.selectedNode;
    const related = new Set(props.edges.filter(e => e.source_id === focus || e.target_id === focus).flatMap(e => [e.source_id, e.target_id]));
    g.nodeColor(n => focus && n.id !== focus && !related.has(n.id) ? '#b1c0bd' : nodeColor(n.type))
      .nodeVal(n => n.id === focus ? 3 : 1)
      .linkColor(l => l.id === props.selectedEdge ? '#176f65' : focus && (l.relation.source_id === focus || l.relation.target_id === focus) ? '#438e83' : '#a5b8b4')
      .linkWidth(l => l.id === props.selectedEdge ? 2 : 0.7);
  }, [props.nodes, props.edges, props.selectedNode, props.selectedEdge]);
  const selectedNeighbors = new Set(props.edges.filter(e => e.source_id === props.selectedNode || e.target_id === props.selectedNode).flatMap(e => [e.source_id, e.target_id]));
  const shown = props.nodes.filter((n, i) => n.id === props.selectedNode || (props.selectedNode ? selectedNeighbors.has(n.id) : i < 8)).slice(0, 10);
  return <div className="space-graph"><div className="space-graph-renderer" ref={host} /><div className="space-graph-labels" ref={overlay} aria-hidden="true">{shown.map(n => <span key={n.id} data-node-id={n.id}>{n.name}</span>)}</div></div>;
});
export default Graph3D;
