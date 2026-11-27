import { useEffect, useRef, useState, type ReactNode } from "react";

export function Inspector({ title, label, children, onClose, width, onWidth }: {
  title: string; label: string; children: ReactNode; onClose: () => void;
  width: number; onWidth: (width: number) => void;
}) {
  const [viewport, setViewport] = useState(window.innerWidth);
  const [maximized, setMaximized] = useState(false);
  const host = useRef<HTMLElement>(null);
  const closeRef = useRef(onClose); closeRef.current = onClose;
  const drag = useRef<{ x: number; width: number } | null>(null);
  useEffect(() => { const update = () => setViewport(window.innerWidth); window.addEventListener("resize", update); return () => window.removeEventListener("resize", update); }, []);
  useEffect(() => {
    const prior = document.activeElement as HTMLElement | null;
    const key = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || document.querySelector('[role="dialog"], [role="menu"]') || !host.current || host.current.closest('[hidden]')) return;
      event.stopPropagation(); closeRef.current();
    };
    document.addEventListener("keydown", key, true);
    return () => { document.removeEventListener("keydown", key, true); if (host.current?.contains(document.activeElement)) prior?.focus(); };
  }, []);
  const min = Math.min(320, viewport);
  const max = Math.max(min, Math.min(720, viewport - (viewport > 860 ? 500 : 0)));
  const size = maximized ? Math.max(min, viewport - (viewport > 860 ? 240 : 0)) : Math.max(min, Math.min(max, width));
  const update = (next: number) => onWidth(Math.max(min, Math.min(max, next)));
  return <aside ref={host} className={`detail-drawer shared-inspector ${maximized ? "inspector-maximized" : ""}`} style={{ width: size }} aria-label={title}>
    <div className="drawer-resize-handle" role="separator" aria-label="调整详情面板宽度" aria-orientation="vertical" aria-valuemin={min} aria-valuemax={max} aria-valuenow={Math.min(size, max)} tabIndex={0}
      onDoubleClick={() => update(440)} onPointerDown={event => { if (event.button !== 0 || maximized) return; drag.current = { x: event.clientX, width: size }; event.currentTarget.setPointerCapture(event.pointerId); event.preventDefault(); }}
      onPointerMove={event => { if (drag.current) update(drag.current.width + drag.current.x - event.clientX); }} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }} onLostPointerCapture={() => { drag.current = null; }}
      onKeyDown={event => { if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return; event.preventDefault(); update(event.key === "Home" ? min : event.key === "End" ? max : size + (event.key === "ArrowLeft" ? 24 : -24)); }}><span /></div>
    <div className="drawer-header"><div><span className="eyebrow">{label}</span><h3>{title}</h3></div><div><button type="button" onClick={() => setMaximized(!maximized)} aria-label={maximized ? "还原详情" : "最大化详情"}>{maximized ? "↙" : "↗"}</button><button type="button" onClick={onClose} aria-label="关闭详情">×</button></div></div>
    {children}
  </aside>;
}
