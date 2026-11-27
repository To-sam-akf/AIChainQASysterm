import { useEffect, useRef, useState, type ReactNode } from "react";

type Action = { label: string; run: () => void; disabled?: boolean };
export function ContextMenu({ label, actions, children }: { label: string; actions: Action[]; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const outside = (event: MouseEvent) => { if (!host.current?.contains(event.target as Node)) setOpen(false); };
    const keyboard = (event: KeyboardEvent) => { if (event.key === "Escape") { event.stopPropagation(); setOpen(false); host.current?.querySelector<HTMLButtonElement>(".context-trigger")?.focus(); } };
    document.addEventListener("mousedown", outside); document.addEventListener("keydown", keyboard, true);
    return () => { document.removeEventListener("mousedown", outside); document.removeEventListener("keydown", keyboard, true); };
  }, [open]);
  return <div className="context-host" ref={host} onContextMenu={event => { event.preventDefault(); setOpen(true); }}>{children}<button className="context-trigger" type="button" aria-label={label} aria-expanded={open} onClick={() => setOpen(!open)}>⋯</button>{open && <div className="context-menu" role="menu">{actions.map(action => <button role="menuitem" type="button" disabled={action.disabled} key={action.label} onClick={() => { setOpen(false); action.run(); }}>{action.label}</button>)}</div>}</div>;
}
