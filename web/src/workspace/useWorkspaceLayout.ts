import { useEffect, useState } from "react";

export type SidebarMode = "expanded" | "collapsed" | "hidden";
const read = <T,>(key: string, fallback: T): T => {
  try { const value = localStorage.getItem(`aika.layout.${key}`); return value ? JSON.parse(value) as T : fallback; } catch { return fallback; }
};
export function useWorkspaceLayout() {
  const [mode, setMode] = useState<SidebarMode>(() => {
    const value = read<string>("sidebar", "expanded");
    return value === "hidden" || value === "collapsed" ? value : "expanded";
  });
  const [width, setWidth] = useState(() => Math.max(180, Math.min(340, Number(read("sidebarWidth", 240)) || 240)));
  const [inspectorWidth, setInspectorWidth] = useState(() => Math.max(320, Math.min(720, Number(read("inspectorWidth", 440)) || 440)));
  useEffect(() => { try { localStorage.setItem("aika.layout.sidebar", JSON.stringify(mode)); } catch { /* storage unavailable */ } }, [mode]);
  useEffect(() => { try { localStorage.setItem("aika.layout.sidebarWidth", JSON.stringify(width)); } catch { /* storage unavailable */ } }, [width]);
  useEffect(() => { try { localStorage.setItem("aika.layout.inspectorWidth", JSON.stringify(inspectorWidth)); } catch { /* storage unavailable */ } }, [inspectorWidth]);
  return { mode, setMode, width, setWidth, inspectorWidth, setInspectorWidth };
}

export function useInspectorWidth(page: string): [number, (width: number) => void] {
  const [width, setWidth] = useState(() => Math.max(320, Math.min(720, Number(read(`inspector.${page}`, 440)) || 440)));
  useEffect(() => { const reset = () => setWidth(440); window.addEventListener("aika:reset-layout", reset); return () => window.removeEventListener("aika:reset-layout", reset); }, []);
  useEffect(() => { try { localStorage.setItem(`aika.layout.inspector.${page}`, JSON.stringify(width)); } catch { /* storage unavailable */ } }, [page, width]);
  return [width, setWidth];
}
