import { RoleWorkspace } from "./RoleWorkspace";
import { ContextMenu } from "./workspace/ContextMenu";
import { Inspector } from "./workspace/Inspector";
import { useWorkspaceLayout } from "./workspace/useWorkspaceLayout";
import { lazy, Suspense } from "react";
const GraphExplorer = lazy(() => import("./GraphExplorer"));
const KnowledgeView = lazy(() => import("./KnowledgeView"));
import {
  BarChart3,
  Bot,
  Boxes,
  ChevronRight,
  CheckCircle2,
  ClipboardList,
  CircleSlash,
  Download,
  FileText,
  History,
  Loader2,
  Maximize2,
  Menu,
  MessageSquareText,
  Move,
  Pencil,
  Plus,
  Save,
  Send,
  Sparkles,
  Table2,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  X,
  ZoomIn,
  ZoomOut
} from "lucide-react";
import { FormEvent, KeyboardEvent, PointerEvent, WheelEvent, useEffect, useId, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  createConversation,
  deleteConversation,
  getConversation,
  getExamples,
  getStatus,
  listResearchRoles,
  resumeTask,
  listConversations,
  reviewClaim,
  streamMessage,
  submitFeedback,
  updateConversationPin,
  updateConversationTitle
} from "./api";
import type {
  ResearchRole,
  MessageStreamEvent,
  ApiStatus,
  ConflictGroup,
  Conversation,
  ConversationSummary,
  ConversationTurn,
  FeedbackCreateRequest,
  GraphEdge,
  ClaimReviewRequest,
  CompanyCoverageHeatmapChart,
  EvidenceStrengthBarChart,
  FlowMapChart,
  FreshnessTimelineChart,
  ReportCharts,
  ResearchOutputs,
  VerificationResult
} from "./types";

type View = "chat" | "overview" | "graph" | "knowledge";
type DetailTab = "collaboration" | "research" | "verification" | "evidence" | "cypher" | "diagnostics" | "graph";

const EMPTY_ARRAY: ConversationSummary[] = [];
const GRAPH_VIEWPORTS = {
  large: { width: 1040, height: 560 },
  compact: { width: 640, height: 360 }
} as const;
const ENTITY_ORDER = [
  "Company",
  "IndustryChain",
  "ValueChainSegment",
  "IndustryConcept",
  "Technology",
  "Product",
  "Metric",
  "Risk",
  "Policy",
  "Standard",
  "Report"
];
const ENTITY_STYLES: Record<string, { label: string; color: string }> = {
  Company: { label: "公司", color: "#1f6f70" },
  Technology: { label: "技术", color: "#2563eb" },
  Product: { label: "产品", color: "#7a4cc2" },
  IndustryChain: { label: "产业链", color: "#b86519" },
  ValueChainSegment: { label: "环节", color: "#c2410c" },
  IndustryConcept: { label: "概念", color: "#0e7490" },
  Metric: { label: "指标", color: "#477148" },
  Risk: { label: "风险", color: "#c43c39" },
  Policy: { label: "政策", color: "#4f5f9f" },
  Standard: { label: "标准", color: "#8a4f9f" },
  Report: { label: "报告", color: "#64748b" }
};
const FALLBACK_ENTITY_STYLE = { label: "实体", color: "#334155" };
const zhCollator = new Intl.Collator("zh-Hans-CN");

function createPendingTurn(question: string, thinkingEnabled: boolean, reasoningEffort: string): ConversationTurn {
  return {
    created_at: new Date().toISOString(),
    question,
    answer: "",
    thinking_enabled: thinkingEnabled,
    reasoning_effort: thinkingEnabled ? reasoningEffort : "",
    result: {
      question,
      contextual_question: question,
      answer: "",
      reasoning_content: "",
      answer_type: "streaming",
      plan: {},
      cypher: "",
      cypher_params: {},
      cypher_source: "",
      graph_records: [],
      rag_hits: [],
      evidence_cards: [],
      evidence: [],
      research_outputs: {
        report: { title: "", markdown: "", sections: [] },
        company_compare_table: { columns: [], rows: [] },
        risk_checklist: [],
        evidence_gaps: []
      },
      subgraph: [],
      diagnostics: { streaming: true },
      errors: []
    }
  };
}

function nextProgressItems(items: string[], message: string): string[] {
  const text = message.trim();
  if (!text || items[items.length - 1] === text) return items;
  return [...items, text].slice(-5);
}

function updateConversationClaimReview(
  conversation: Conversation | null,
  claimId: string,
  claim: Record<string, unknown>
): Conversation | null {
  if (!conversation) return conversation;
  return {
    ...conversation,
    turns: conversation.turns.map((turn) => updateTurnClaimReview(turn, claimId, claim) ?? turn)
  };
}

function updateTurnClaimReview(
  turn: ConversationTurn | null,
  claimId: string,
  claim: Record<string, unknown>
): ConversationTurn | null {
  if (!turn) return turn;
  const updateRows = (rows: Record<string, unknown>[] = []) =>
    rows.map((row) => {
      if (String(row.claim_id ?? "") !== claimId) return row;
      return {
        ...row,
        evidence: String(claim.claim_text ?? row.evidence ?? ""),
        text: String(claim.claim_text ?? row.text ?? ""),
        topic: String(claim.topic ?? row.topic ?? ""),
        claim_type: String(claim.claim_type ?? row.claim_type ?? ""),
        exposure_level: String(claim.exposure_level ?? row.exposure_level ?? ""),
        confidence: String(claim.confidence ?? row.confidence ?? ""),
        as_of_date: String(claim.as_of_date ?? row.as_of_date ?? ""),
        evidence_span: String(claim.evidence_span ?? row.evidence_span ?? ""),
        review_status: String(claim.review_status ?? row.review_status ?? ""),
        reviewer_note: String(claim.reviewer_note ?? row.reviewer_note ?? "")
      };
    });
  return {
    ...turn,
    result: {
      ...turn.result,
      evidence_cards: updateRows(turn.result.evidence_cards),
      evidence: updateRows(turn.result.evidence)
    }
  };
}

function applyTaskEvent(turn: ConversationTurn, event: MessageStreamEvent): ConversationTurn {
  if (event.type === "final") return event.turn;
  const seq = "seq" in event ? event.seq : undefined;
  if (seq && turn.events?.some(item => "seq" in item && item.seq === seq)) return turn;
  const next = { ...turn, events: [...(turn.events || []), event] };
  if (event.type === "task") return { ...next, task_id: event.task_id, lead_role_id: event.lead_role_id, status: "running" };
  if (event.type === "answer_delta") {
    const answer = turn.answer + event.content;
    return { ...next, answer, result: { ...turn.result, answer } };
  }
  if (event.type === "error") return { ...next, error: event.message };
  return next;
}

function App() {
  const [view, setView] = useState<View>(() => new URLSearchParams(window.location.search).has("knowledge_document") ? "knowledge" : new URLSearchParams(window.location.search).has("graph_view") ? "graph" : "chat");
  const [knowledgeVisited, setKnowledgeVisited] = useState(view === "knowledge");
  useEffect(() => { if (view === "knowledge") setKnowledgeVisited(true); }, [view]);
  const [graphVisited, setGraphVisited] = useState(view === "graph");
  useEffect(() => { if (view === "graph") setGraphVisited(true); }, [view]);
  const [status, setStatus] = useState<ApiStatus | null>(null);
  const [examples, setExamples] = useState<string[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>(EMPTY_ARRAY);
  const [current, setCurrent] = useState<Conversation | null>(null);
  const [input, setInput] = useState("");
  const [researchRoles, setResearchRoles] = useState<ResearchRole[]>([]);
  const currentIdRef = useRef<string | undefined>();
  currentIdRef.current = current?.id;
  const submissionAbort = useRef<AbortController | null>(null);
  const activeTask = current?.turns.find(turn => turn.status === "pending" || turn.status === "running");
  const [thinkingEnabled, setThinkingEnabled] = useState(false);
  const [reasoningEffort, setReasoningEffort] = useState("low");
  const [sending, setSending] = useState(false);
  const [streamingTurn, setStreamingTurn] = useState<ConversationTurn | null>(null);
  const [streamingProgress, setStreamingProgress] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [toast, setToast] = useState("");
  const [selectedTurn, setSelectedTurn] = useState<ConversationTurn | null>(null);
  const [detailTab, setDetailTab] = useState<DetailTab>("research");
  const [selectedCitation, setSelectedCitation] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const layout = useWorkspaceLayout();
  const [commandOpen, setCommandOpen] = useState(false);
  const [commandQuery, setCommandQuery] = useState("");
  const [focusMode, setFocusMode] = useState(false);
  const [commandIndex, setCommandIndex] = useState(0);
  const priorTurn = useRef<ConversationTurn | null>(null);
  const hiddenTurn = useRef<ConversationTurn | null>(null);
  const commandTrigger = useRef<HTMLElement | null>(null);
  const previousMode = useRef(layout.mode);
  const sidebarDrag = useRef<{ x: number; width: number } | null>(null);
  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); if (!commandOpen) commandTrigger.current = document.activeElement as HTMLElement; setCommandOpen(value => !value); }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "b") { event.preventDefault(); layout.setMode(layout.mode === "hidden" ? "expanded" : "hidden"); }
      if (event.key === "Escape") { if (document.querySelector(".knowledge-dialog")) return; if (commandOpen) { setCommandOpen(false); commandTrigger.current?.focus(); } else if (sidebarOpen) setSidebarOpen(false); }
    };
    window.addEventListener("keydown", onKey); return () => window.removeEventListener("keydown", onKey);
  }, [layout.mode, commandOpen, sidebarOpen, selectedTurn, view]);
  const [editingId, setEditingId] = useState("");
  const [editingTitle, setEditingTitle] = useState("");

  async function refreshConversations() {
    const items = await listConversations();
    setConversations(items);
  }

  async function loadInitial() {
    setLoading(true);
    try {
      const [statusPayload, examplePayload, conversationPayload] = await Promise.all([
        getStatus(),
        getExamples(),
        listConversations()
      ]);
      setStatus(statusPayload);
      setExamples(examplePayload);
      setConversations(conversationPayload);
      setThinkingEnabled(statusPayload.settings.thinking_enabled);
      setReasoningEffort(statusPayload.settings.reasoning_effort);
      if (conversationPayload.length > 0) {
        const latest = await getConversation(conversationPayload[0].id);
        setCurrent(latest);
      }
    } catch (error) {
      setToast(error instanceof Error ? error.message : "加载系统状态失败");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadInitial();
    void listResearchRoles().then(payload => setResearchRoles(payload.roles)).catch(() => setToast("角色资料暂不可用，请重试加载"));
  }, []);

  useEffect(() => {
    const conversationId = current?.id;
    const taskId = activeTask?.task_id;
    if (!conversationId || !taskId) return;
    const controller = new AbortController();
    let stopped = false;
    let cursor = Math.max(0, ...(activeTask.events || []).map(event => "seq" in event ? event.seq || 0 : 0));
    async function follow() {
      while (!stopped) {
        try {
          await resumeTask(conversationId!, taskId!, cursor, event => {
            if (stopped) return;
            if ("seq" in event && event.seq) cursor = Math.max(cursor, event.seq);
            if (event.type === "final") {
              setCurrent(previous => previous?.id === conversationId ? event.conversation : previous);
              void refreshConversations();
            } else {
              setCurrent(previous => previous && previous.id === conversationId ? { ...previous,
                turns: previous.turns.map(turn => turn.task_id === taskId ? applyTaskEvent(turn, event) : turn) } : previous);
            }
          }, controller.signal);
          return;
        } catch {
          if (stopped) return;
          setToast("连接中断，任务仍在后台执行，正在恢复连接…");
          await new Promise(resolve => window.setTimeout(resolve, 2000));
        }
      }
    }
    void follow();
    return () => { stopped = true; controller.abort(); };
    // Event updates keep the same task ID and must not restart subscriptions.
  }, [current?.id, activeTask?.task_id]);

  useEffect(() => () => { submissionAbort.current?.abort(); }, []);

  async function handleNewConversation() {
    submissionAbort.current?.abort();
    setSending(false);
    try {
      const conversation = await createConversation();
      setCurrent(conversation);
      setSelectedTurn(null);
      setInput("");
      setStreamingTurn(null);
      setStreamingProgress([]);
      setView("chat");
      setSidebarOpen(false);
      await refreshConversations();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "创建对话失败");
    }
  }

  async function handleLoadConversation(id: string) {
    submissionAbort.current?.abort();
    setSending(false);
    try {
      const conversation = await getConversation(id);
      setCurrent(conversation);
      setSelectedTurn(null);
      setStreamingTurn(null);
      setStreamingProgress([]);
      setView("chat");
      setSidebarOpen(false);
    } catch (error) {
      setToast(error instanceof Error ? error.message : "读取对话失败");
    }
  }

  async function handleDeleteConversation(id: string) {
    if (!window.confirm("删除这段对话？")) return;
    try {
      await deleteConversation(id);
      if (current?.id === id) {
        setCurrent(null);
        setSelectedTurn(null);
      }
      await refreshConversations();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "删除失败");
    }
  }

  async function handlePin(item: ConversationSummary) {
    try { const updated = await updateConversationPin(item.id, !item.pinned); if (current?.id === item.id) setCurrent(updated); await refreshConversations(); } catch (error) { setToast(error instanceof Error ? error.message : "更新置顶失败"); }
  }

  async function handleRename(id: string) {
    const title = editingTitle.trim();
    if (!title) {
      setEditingId("");
      return;
    }
    try {
      const updated = await updateConversationTitle(id, title);
      if (current?.id === id) setCurrent(updated);
      setEditingId("");
      await refreshConversations();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "重命名失败");
    }
  }

  async function handleSubmit(question = input) {
    const trimmed = question.trim();
    if (!trimmed || sending || activeTask) return;
    setSending(true);
    setToast("");
    const controller = new AbortController();
    submissionAbort.current = controller;
    let conversationId = current?.id;
    try {
      let conversation = current;
      if (!conversation) {
        conversation = await createConversation();
        setCurrent(conversation);
        currentIdRef.current = conversation.id;
      }
      conversationId = conversation.id;
      const requestId = crypto.randomUUID();
      setStreamingTurn({ ...createPendingTurn(trimmed, thinkingEnabled, thinkingEnabled ? reasoningEffort : ""),
        request_id: requestId, status: "running", lead_role_id: "chief", events: [] });
      setStreamingProgress([]);
      setSelectedTurn(null);
      setInput("");
      setView("chat");
      const updated = await streamMessage(conversationId, trimmed, thinkingEnabled,
        thinkingEnabled ? reasoningEffort : "", event => {
          if (currentIdRef.current !== conversationId) return;
          if (event.type === "final") {
            setCurrent(event.conversation);
            setStreamingTurn(null);
            setStreamingProgress([]);
          } else {
            setStreamingTurn(turn => turn ? applyTaskEvent(turn, event) : turn);
            if (event.type === "progress") setStreamingProgress(items => nextProgressItems(items, event.message));
            if (event.type === "error") setToast(event.message);
          }
        }, requestId, controller.signal);
      if (currentIdRef.current === conversationId) {
        setCurrent(updated);
        setStreamingTurn(null);
        setStreamingProgress([]);
      }
      await refreshConversations();
    } catch (error) {
      if (!controller.signal.aborted && currentIdRef.current === conversationId) {
        setToast(error instanceof Error ? error.message : "连接中断，正在恢复任务状态");
        // Reload the persisted turn; its active task triggers event resubscription.
        if (conversationId) {
          while (!controller.signal.aborted && currentIdRef.current === conversationId) {
            try {
              const saved = await getConversation(conversationId);
              if (currentIdRef.current === conversationId) setCurrent(saved);
              break;
            } catch {
              setToast("暂时无法连接，任务状态将在网络恢复后更新…");
              await new Promise(resolve => window.setTimeout(resolve, 2000));
            }
          }
        }
        setStreamingTurn(null);
        setStreamingProgress([]);
      }
    } finally {
      if (submissionAbort.current === controller) setSending(false);
    }
  }

  async function handleClaimReview(claimId: string, payload: ClaimReviewRequest) {
    try {
      const response = await reviewClaim(claimId, payload);
      setToast("Claim 修正已保存");
      setSelectedTurn((turn) => updateTurnClaimReview(turn, claimId, response.claim));
      setCurrent((conversation) => updateConversationClaimReview(conversation, claimId, response.claim));
    } catch (error) {
      setToast(error instanceof Error ? error.message : "保存 Claim 修正失败");
    }
  }

  async function handleFeedback(payload: FeedbackCreateRequest) {
    try {
      await submitFeedback(payload);
      setToast("反馈已保存");
    } catch (error) {
      setToast(error instanceof Error ? error.message : "保存反馈失败");
    }
  }

  const shellClassName = `app-shell sidebar-${layout.mode} view-${view} ${sidebarOpen ? "sidebar-open" : ""} ${selectedTurn && view === "chat" ? "has-inspector" : ""}`;
  const pageNames: Record<View, string> = { chat: "智能问答", knowledge: "知识库", overview: "数据概览", graph: "产业链图谱" };
  const pageEnglish: Record<View, string> = { chat: "RESEARCH WORKSPACE", knowledge: "KNOWLEDGE BASE", overview: "SYSTEM OVERVIEW", graph: "KNOWLEDGE UNIVERSE" };
  function toggleFocus() {
    if (!focusMode) { previousMode.current = layout.mode; priorTurn.current = selectedTurn; layout.setMode("hidden"); setSelectedTurn(null); }
    else { layout.setMode(previousMode.current); setSelectedTurn(priorTurn.current); }
    window.dispatchEvent(new CustomEvent("aika:focus", { detail: !focusMode }));
    setFocusMode(!focusMode);
  }

  const commands: { label: string; category: string; run: () => void }[] = [
    { label: "新建对话", category: "会话", run: () => void handleNewConversation() },
    ...([ ["智能问答", "chat"], ["知识库", "knowledge"], ["数据概览", "overview"], ["产业链图谱", "graph"] ] as [string, View][]).map(([label, target]) => ({ label, category: "导航", run: () => setView(target) })),
    { label: "搜索历史对话", category: "搜索", run: () => { layout.setMode("expanded"); window.setTimeout(() => document.querySelector<HTMLInputElement>(".history-search")?.focus(), 50); } },
    { label: "搜索资料", category: "搜索", run: () => { setView("knowledge"); window.setTimeout(() => document.querySelector<HTMLInputElement>('[aria-label="搜索资料"]')?.focus(), 200); } },
    { label: "搜索图谱实体", category: "搜索", run: () => { setView("graph"); window.setTimeout(() => document.querySelector<HTMLInputElement>("#graph-search")?.focus(), 200); } },
    { label: "导入资料", category: "资料", run: () => { setView("knowledge"); window.setTimeout(() => document.querySelector<HTMLButtonElement>(".knowledge-heading .knowledge-primary")?.click(), 200); } },
    { label: "展开侧栏", category: "布局", run: () => layout.setMode("expanded") },
    { label: "折叠侧栏", category: "布局", run: () => layout.setMode("collapsed") },
    { label: "隐藏侧栏", category: "布局", run: () => layout.setMode("hidden") },
    { label: "显示或隐藏详情", category: "布局", run: () => { if (view === "chat") { if (selectedTurn) { hiddenTurn.current = selectedTurn; setSelectedTurn(null); } else if (hiddenTurn.current) setSelectedTurn(hiddenTurn.current); } else window.dispatchEvent(new Event("aika:toggle-inspector")); } },
    { label: focusMode ? "退出专注模式" : "进入专注模式", category: "布局", run: toggleFocus },
    { label: "恢复默认布局", category: "布局", run: () => { layout.setMode("expanded"); layout.setWidth(240); layout.setInspectorWidth(440); window.dispatchEvent(new Event("aika:reset-layout")); setToast("已恢复默认布局"); } },
  ];
  const filteredCommands = commands.filter(command => `${command.label} ${command.category}`.toLowerCase().includes(commandQuery.trim().toLowerCase()));
  function executeCommand(index: number) { const command = filteredCommands[index]; if (!command) return; setCommandOpen(false); setCommandQuery(""); setCommandIndex(0); command.run(); }

  return (
    <div className={shellClassName} style={{ "--sidebar-width": `${layout.mode === "expanded" ? layout.width : layout.mode === "collapsed" ? 64 : 0}px` } as React.CSSProperties}>
      <button className="mobile-menu" type="button" onClick={() => setSidebarOpen(true)} aria-label="打开侧栏">
        <Menu size={20} />
      </button>
      {sidebarOpen && <button className="sidebar-scrim" type="button" aria-label="关闭侧栏遮罩" onClick={() => setSidebarOpen(false)} />}
      <Sidebar
        mode={layout.mode}
        onMode={layout.setMode}
        conversations={conversations}
        currentId={current?.id ?? ""}
        view={view}
        editingId={editingId}
        editingTitle={editingTitle}
        onClose={() => setSidebarOpen(false)}
        onNewConversation={handleNewConversation}
        onLoadConversation={handleLoadConversation}
        onDeleteConversation={handleDeleteConversation}
        onPin={handlePin}
        onStartRename={(item) => {
          setEditingId(item.id);
          setEditingTitle(item.title);
        }}
        onRename={handleRename}
        onEditTitle={setEditingTitle}
        onViewChange={(nextView) => {
          setView(nextView);
          setSidebarOpen(false);
        }}
      />
      {layout.mode === "expanded" && <div className="sidebar-resize" role="separator" aria-label="调整侧栏宽度" aria-orientation="vertical" aria-valuemin={180} aria-valuemax={340} aria-valuenow={layout.width} tabIndex={0}
        onPointerDown={event => { sidebarDrag.current = { x: event.clientX, width: layout.width }; event.currentTarget.setPointerCapture(event.pointerId); }}
        onPointerMove={event => { if (!sidebarDrag.current) return; const next = sidebarDrag.current.width + event.clientX - sidebarDrag.current.x; if (next < 145) { layout.setMode("collapsed"); sidebarDrag.current = null; } else layout.setWidth(Math.max(180, Math.min(340, next))); }}
        onPointerUp={() => { sidebarDrag.current = null; }} onPointerCancel={() => { sidebarDrag.current = null; }}
        onDoubleClick={() => layout.setWidth(240)} onKeyDown={event => { if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); layout.setWidth(Math.max(180, Math.min(340, layout.width + (event.key === "ArrowLeft" ? -10 : 10)))); } }} />}
      <main className="workspace">
        <header className="workspace-header">
          <div className="workspace-heading"><button type="button" className="header-icon" title="切换侧栏 (Ctrl+B)" aria-label="切换侧栏" onClick={() => { if (window.innerWidth <= 860) setSidebarOpen(true); else layout.setMode(layout.mode === "hidden" ? "expanded" : layout.mode === "expanded" ? "collapsed" : "expanded"); }}><Menu size={17} /></button><span className="header-name">{pageNames[view]}</span><span className="header-breadcrumb">/ {view === "chat" && current?.title ? current.title : pageEnglish[view]}</span></div>
          <div className="header-actions">{view === "chat" && current && <><a href={`/api/conversations/${current.id}/export?format=md`}>MD</a><a href={`/api/conversations/${current.id}/export?format=json`}>JSON</a></>}{view === "overview" && <button type="button" onClick={() => void getStatus().then(setStatus).catch(() => setToast("刷新系统状态失败"))}>刷新</button>}<button type="button" onClick={(event) => { commandTrigger.current = event.currentTarget; setCommandOpen(true); }} title="命令面板 Ctrl+K">搜索 <kbd>⌘ K</kbd></button><button type="button" onClick={toggleFocus}>{focusMode ? "退出专注" : "专注"}</button><button type="button" onClick={() => layout.setMode(layout.mode === "hidden" ? "expanded" : "hidden")}>{layout.mode === "hidden" ? "显示侧栏" : "隐藏侧栏"}</button></div>
        </header>
        <GeometricBackdrop />
        {toast && (
          <div className="toast" role="alert">
            <span>{toast}</span>
            <button type="button" onClick={() => setToast("")} aria-label="关闭提示">
              <X size={16} />
            </button>
          </div>
        )}
        {view === "chat" && (
          <ChatView
            loading={loading}
            conversation={current}
            examples={examples}
            input={input}
            sending={sending || Boolean(activeTask)}
            roles={researchRoles}
            streamingTurn={streamingTurn}
            streamingProgress={streamingProgress}
            thinkingEnabled={thinkingEnabled}
            reasoningEffort={reasoningEffort}
            reasoningEfforts={status?.settings.reasoning_efforts ?? ["low", "medium", "high"]}
            onInput={setInput}
            onSubmit={handleSubmit}
            onThinkingChange={setThinkingEnabled}
            onReasoningChange={setReasoningEffort}
            onOpenDetails={(turn, citation, tab) => {
              setSelectedCitation(citation || "");
              setSelectedTurn(turn);
              setDetailTab(tab || "evidence");
            }}
            onFeedback={handleFeedback}
            onExample={(example) => setInput(example)}
          />
        )}
        <div hidden={view !== "knowledge"}>{knowledgeVisited && <Suspense fallback={<p>正在加载知识库…</p>}><KnowledgeView active={view === "knowledge"} /></Suspense>}</div>
        {view === "overview" && <OverviewView status={status} />}
        <div hidden={view !== "graph"}>{graphVisited && <GraphView active={view === "graph"} />}</div>
      </main>
      {commandOpen && <div className="command-backdrop" onMouseDown={() => { setCommandOpen(false); commandTrigger.current?.focus(); }}><div className="command-dialog" role="dialog" aria-modal="true" aria-label="命令面板" onMouseDown={event => event.stopPropagation()}><input autoFocus placeholder="搜索页面或操作…" value={commandQuery} onChange={event => { setCommandQuery(event.target.value); setCommandIndex(0); }} onKeyDown={event => { if (event.key === "ArrowDown" || event.key === "ArrowUp") { event.preventDefault(); setCommandIndex(index => Math.max(0, Math.min(filteredCommands.length - 1, index + (event.key === "ArrowDown" ? 1 : -1)))); } else if (event.key === "Enter") { event.preventDefault(); executeCommand(commandIndex); } }} />{filteredCommands.map((command, index) => <button key={command.label} className={index === commandIndex ? "selected" : ""} type="button" onMouseEnter={() => setCommandIndex(index)} onClick={() => executeCommand(index)}>{command.label}<span>{command.category}</span></button>)}{!filteredCommands.length && <p>没有匹配的命令</p>}</div></div>}
      <DetailDrawer
        citation={selectedCitation}
        width={layout.inspectorWidth}
        onWidth={layout.setInspectorWidth}
        roles={researchRoles}
        onRetry={handleSubmit}
        turn={selectedTurn}
        open={Boolean(selectedTurn) && view === "chat"}
        tab={detailTab}
        onTab={setDetailTab}
        onClose={() => setSelectedTurn(null)}
        onClaimReview={handleClaimReview}
      />
    </div>
  );
}

function Sidebar(props: {
  mode: "expanded" | "collapsed" | "hidden";
  onMode: (mode: "expanded" | "collapsed" | "hidden") => void;
  conversations: ConversationSummary[];
  currentId: string;
  view: View;
  editingId: string;
  editingTitle: string;
  onClose: () => void;
  onNewConversation: () => void;
  onLoadConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onPin: (item: ConversationSummary) => void;
  onStartRename: (item: ConversationSummary) => void;
  onRename: (id: string) => void;
  onEditTitle: (title: string) => void;
  onViewChange: (view: View) => void;
}) {
  const [historyQuery, setHistoryQuery] = useState("");
  return (
    <aside className="sidebar">
      <button className="mobile-close" type="button" onClick={props.onClose} aria-label="关闭侧栏">
        <X size={18} />
      </button>
      <div className="brand">
        <div className="brand-mark">
          <span />
          <span />
        </div>
        <div className="brand-text">AIKA</div>
      </div>

      <button className="new-chat" title="新对话" type="button" onClick={props.onNewConversation}>
        <Plus size={19} />
        <span>新对话</span>
      </button>

      <nav className="nav-list" aria-label="主导航">
        <button className={props.view === "chat" ? "active" : ""} title="智能问答" type="button" onClick={() => props.onViewChange("chat")}>
          <MessageSquareText size={20} />
          <span>智能问答</span>
        </button>
        <button className={props.view === "knowledge" ? "active" : ""} title="知识库" type="button" onClick={() => props.onViewChange("knowledge")}>
          <FileText size={20} /><span>知识库</span>
        </button>
        <button
          className={props.view === "overview" ? "active" : ""}
          title="数据概览"
          type="button"
          onClick={() => props.onViewChange("overview")}
        >
          <BarChart3 size={20} />
          <span>数据概览</span>
        </button>
        <button className={props.view === "graph" ? "active" : ""} title="产业链图谱" type="button" onClick={() => props.onViewChange("graph")}>
          <Boxes size={20} />
          <span>产业链图谱</span>
        </button>
      </nav>

      <button className="collapsed-history" type="button" title="展开历史记录" aria-label="展开历史记录" onClick={() => props.onMode("expanded")}><History size={19} /></button>
      <section className="history-panel">
        <div className="section-title">
          <History size={16} />
          <span>问答历史</span>
        </div>
        <input className="history-search" aria-label="搜索历史对话" placeholder="搜索历史对话" value={historyQuery} onChange={event => setHistoryQuery(event.target.value)} />
        <div className="history-list">
          {props.conversations.length === 0 && <div className="empty-history">暂无历史对话</div>}
          {props.conversations.filter(item => `${item.title} ${item.preview}`.toLowerCase().includes(historyQuery.toLowerCase())).map((item) => (
            <article className={`history-item ${props.currentId === item.id ? "selected" : ""}`} key={item.id}><ContextMenu label={`对话操作：${item.title}`} actions={[{ label: item.pinned ? "取消置顶" : "置顶", run: () => props.onPin(item) }, { label: "重命名", run: () => props.onStartRename(item) }, { label: "删除", run: () => props.onDeleteConversation(item.id) }]}>
              {props.editingId === item.id ? (
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    props.onRename(item.id);
                  }}
                >
                  <input
                    value={props.editingTitle}
                    autoFocus
                    onChange={(event) => props.onEditTitle(event.target.value)}
                    onBlur={() => props.onRename(item.id)}
                  />
                </form>
              ) : (
                <button className="history-open" title={item.title} type="button" onClick={() => props.onLoadConversation(item.id)}>
                  <strong>{item.pinned ? "📌 " : ""}{item.title}</strong>
                  <span>{item.preview}</span>
                </button>
              )}
              </ContextMenu><div className="history-actions">
                <button type="button" title={item.pinned ? "取消置顶" : "置顶"} aria-label={item.pinned ? "取消置顶" : "置顶"} onClick={() => props.onPin(item)}>{item.pinned ? "★" : "☆"}</button>
                <button type="button" onClick={() => props.onStartRename(item)} aria-label="重命名">
                  <Pencil size={14} />
                </button>
                <button type="button" onClick={() => props.onDeleteConversation(item.id)} aria-label="删除">
                  <Trash2 size={14} />
                </button>
              </div>
            </article>
          ))}
        </div>
      </section>
      <div className="sidebar-bottom"><span>AIKA / RESEARCH</span><button type="button" title="折叠侧栏" onClick={() => props.onMode(props.mode === "expanded" ? "collapsed" : "expanded")}>↔</button></div>
    </aside>
  );
}

function ChatView(props: {
  roles: ResearchRole[];
  loading: boolean;
  conversation: Conversation | null;
  examples: string[];
  input: string;
  sending: boolean;
  streamingTurn: ConversationTurn | null;
  streamingProgress: string[];
  thinkingEnabled: boolean;
  reasoningEffort: string;
  reasoningEfforts: string[];
  onInput: (value: string) => void;
  onSubmit: (question?: string) => void;
  onThinkingChange: (value: boolean) => void;
  onReasoningChange: (value: string) => void;
  onOpenDetails: (turn: ConversationTurn, citation?: string, tab?: DetailTab) => void;
  onFeedback: (payload: FeedbackCreateRequest) => Promise<void>;
  onExample: (value: string) => void;
}) {
  const hasTurns = Boolean(props.conversation?.turns.length || props.streamingTurn);
  return (
    <section className={`chat-page ${hasTurns ? "with-turns" : "empty"}`}>
      {!hasTurns ? (
        <div className="hero-chat">
          <div className="hero-mark" aria-hidden="true"><span /><i /></div><div className="hero-kicker">AI RESEARCH WORKSPACE</div>
          <h1>AIKA</h1>
          <p>面向产业链图谱、研报证据与连续追问的专业问答工作台</p>
          <Composer
            value={props.input}
            sending={props.sending}
            thinkingEnabled={props.thinkingEnabled}
            reasoningEffort={props.reasoningEffort}
            reasoningEfforts={props.reasoningEfforts}
            placeholder="请输入你想了解的 AI 算力链产业问题..."
            onChange={props.onInput}
            onSubmit={props.onSubmit}
            onThinkingChange={props.onThinkingChange}
            onReasoningChange={props.onReasoningChange}
          />
          <ExampleRail examples={props.examples} onPick={props.onExample} onSubmit={props.onSubmit} />
          {props.loading && (
            <div className="loading-line">
              <Loader2 size={16} className="spin" />
              <span>正在连接知识图谱与本地证据库</span>
            </div>
          )}
        </div>
      ) : (
        <div className="conversation-layout">
          <div className="conversation-header">
            <div>
              <span className="eyebrow">当前对话</span>
              <h2>{props.conversation?.title}</h2>
            </div>
            {props.conversation && (
              <div className="export-group">
                <a className="export-link" href={`/api/conversations/${props.conversation.id}/export?format=md`}>
                  <Download size={16} />
                  <span>MD</span>
                </a>
                <a className="export-link" href={`/api/conversations/${props.conversation.id}/export?format=json`}>
                  <Download size={16} />
                  <span>JSON</span>
                </a>
              </div>
            )}
          </div>
          <div className="message-list">
            {props.conversation?.turns.map((turn, index) => (
              <MessagePair
                key={`${turn.created_at}-${index}`}
                index={index}
                turn={turn}
                roles={props.roles}
                onRetry={props.onSubmit}
                isStreaming={turn.status === "pending" || turn.status === "running"}
                conversationId={props.conversation?.id ?? ""}
                onOpenDetails={props.onOpenDetails}
                onFeedback={props.onFeedback}
              />
            ))}
            {props.streamingTurn && (
              <MessagePair
                index={props.conversation?.turns.length ?? 0}
                turn={props.streamingTurn}
                roles={props.roles}
                onRetry={props.onSubmit}
                isStreaming
                progressItems={props.streamingProgress}
                conversationId={props.conversation?.id ?? ""}
                onOpenDetails={props.onOpenDetails}
                onFeedback={props.onFeedback}
              />
            )}
            {props.sending && !props.streamingTurn && (
              <div className="assistant-thinking">
                <Loader2 size={18} className="spin" />
                <span>正在启动问答链路</span>
              </div>
            )}
          </div>
          <Composer
            compact
            value={props.input}
            sending={props.sending}
            thinkingEnabled={props.thinkingEnabled}
            reasoningEffort={props.reasoningEffort}
            reasoningEfforts={props.reasoningEfforts}
            placeholder="继续追问，系统会自动携带当前对话上下文..."
            onChange={props.onInput}
            onSubmit={props.onSubmit}
            onThinkingChange={props.onThinkingChange}
            onReasoningChange={props.onReasoningChange}
          />
        </div>
      )}
    </section>
  );
}

function verificationFromResult(result: ConversationTurn["result"]): VerificationResult | undefined {
  return (
    asVerification(result.verification) ??
    asVerification(result.research_outputs?.verification) ??
    asVerification(result.diagnostics?.agent_verification)
  );
}

function asVerification(value: unknown): VerificationResult | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const record = value as Record<string, unknown>;
  return typeof record.status === "string" ? (record as unknown as VerificationResult) : undefined;
}

function verificationStatusLabel(status?: string) {
  if (status === "pass") return "通过";
  if (status === "warn") return "需关注";
  if (status === "fail") return "未通过";
  return "未校验";
}

function Composer(props: {
  value: string;
  sending: boolean;
  thinkingEnabled: boolean;
  reasoningEffort: string;
  reasoningEfforts: string[];
  placeholder: string;
  compact?: boolean;
  onChange: (value: string) => void;
  onSubmit: (question?: string) => void;
  onThinkingChange: (value: boolean) => void;
  onReasoningChange: (value: string) => void;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 144)}px`;
  }, [props.value]);

  const reasoningEfforts = props.reasoningEfforts.length ? props.reasoningEfforts : ["low", "medium", "high"];
  const activeReasoningEffort = reasoningEfforts.includes(props.reasoningEffort)
    ? props.reasoningEffort
    : reasoningEfforts[0];

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      props.onSubmit();
    }
  }

  function toggleThinking() {
    props.onThinkingChange(!props.thinkingEnabled);
  }

  function cycleReasoningEffort() {
    if (!props.thinkingEnabled) {
      props.onReasoningChange(activeReasoningEffort);
      props.onThinkingChange(true);
      return;
    }
    const currentIndex = reasoningEfforts.indexOf(activeReasoningEffort);
    const nextEffort = reasoningEfforts[(currentIndex + 1) % reasoningEfforts.length];
    props.onReasoningChange(nextEffort);
  }

  return (
    <div className={`composer ${props.compact ? "compact" : ""}`}>
      <textarea
        ref={textareaRef}
        aria-label="输入问题"
        value={props.value}
        rows={1}
        placeholder={props.placeholder}
        onChange={(event) => props.onChange(event.target.value)}
        onKeyDown={onKeyDown}
      />
      <div className="composer-footer">
        <div className="composer-controls">
          <button
            className={props.thinkingEnabled ? "pill active" : "pill"}
            type="button"
            disabled={props.sending}
            aria-pressed={props.thinkingEnabled}
            onClick={toggleThinking}
          >
            思考模式：{props.thinkingEnabled ? "开" : "关"}
          </button>
          <button
            className={props.thinkingEnabled ? "pill active" : "pill"}
            type="button"
            disabled={props.sending || !props.thinkingEnabled}
            aria-pressed={props.thinkingEnabled}
            title={props.thinkingEnabled ? "点击切换思考强度" : "开启思考模式后可设置"}
            onClick={cycleReasoningEffort}
          >
            强度：{props.thinkingEnabled ? activeReasoningEffort : "不可用"}
          </button>
        </div>
        <button className="send-button" type="button" aria-label="发送问题" disabled={!props.value.trim() || props.sending} onClick={() => props.onSubmit()}>
          {props.sending ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
        </button>
      </div>
    </div>
  );
}

function ExampleRail(props: { examples: string[]; onPick: (value: string) => void; onSubmit: (value: string) => void }) {
  return (
    <div className="example-rail">
      {props.examples.slice(0, 4).map((example) => (
        <button key={example} type="button" onClick={() => props.onPick(example)} onDoubleClick={() => props.onSubmit(example)}>
          <Sparkles size={14} />
          <span>{example}</span>
        </button>
      ))}
    </div>
  );
}

function MessagePair(props: {
  roles: ResearchRole[];
  onRetry: (question: string) => void;
  index: number;
  turn: ConversationTurn;
  conversationId: string;
  isStreaming?: boolean;
  progressItems?: string[];
  onOpenDetails: (turn: ConversationTurn, citation?: string, tab?: DetailTab) => void;
  onFeedback: (payload: FeedbackCreateRequest) => Promise<void>;
}) {
  return (
    <article className="turn">
      <div className="message user-message" aria-label="用户问题">
        <div className="bubble">
          <p>{props.turn.question}</p>
        </div>
      </div>
      <RoleWorkspace turn={props.turn} roles={props.roles} onEvidence={props.onOpenDetails} onCollaboration={() => props.onOpenDetails(props.turn, "", "collaboration")} onRetry={props.onRetry} />
      <div className="message assistant-message">
        <div className="avatar">
          <Bot size={18} />
        </div>
        <div className="bubble answer">
          <div className="answer-meta">
            <span>第 {props.index + 1} 轮</span>
            <span>{props.turn.thinking_enabled ? `思考 ${props.turn.reasoning_effort}` : "快速回答"}</span>
            {props.isStreaming ? <span>草稿，待核查</span> : <span>{props.turn.status === "failed" ? "任务失败，以下为未完成内容" : "最终回答"}</span>}
          </div>
          {props.turn.thinking_enabled && Boolean(props.progressItems?.length) && (
            <ThoughtProgress items={props.progressItems ?? []} />
          )}
          <div className="answer-text" onClick={(event) => { const target = event.target as HTMLElement; const citation = target.closest<HTMLElement>("[data-citation]"); if (citation) props.onOpenDetails(props.turn, citation.dataset.citation); }}>
            {props.turn.answer ? (
              <>
                {renderAnswerMarkdown(props.turn.answer)}
                {props.isStreaming && <span className="stream-caret" />}
              </>
            ) : props.isStreaming ? (
              <div className="stream-placeholder">
                <Loader2 size={16} className="spin" />
                <span>正在生成答案</span>
              </div>
            ) : (
              <div className="stream-placeholder">本轮未生成答案，请重试；可在证据与诊断中查看详情。</div>
            )}
          </div>
          {!props.isStreaming && props.turn.status !== "failed" && (
            <>
              <div className="answer-action-row">
                <button className="details-button" type="button" onClick={() => props.onOpenDetails(props.turn)}>
                  <FileText size={15} />
                  <span>查看证据与诊断</span>
                  <ChevronRight size={15} />
                </button>
              </div>
              <FeedbackControls
                conversationId={props.conversationId}
                turnIndex={props.index}
                turn={props.turn}
                onFeedback={props.onFeedback}
              />
            </>
          )}
        </div>
      </div>
    </article>
  );
}

function FeedbackControls(props: {
  conversationId: string;
  turnIndex: number;
  turn: ConversationTurn;
  onFeedback: (payload: FeedbackCreateRequest) => Promise<void>;
}) {
  const [helpful, setHelpful] = useState<boolean | null>(null);
  const [evidenceSupported, setEvidenceSupported] = useState<boolean | null>(null);
  const [missingAnswer, setMissingAnswer] = useState<boolean | null>(null);
  const [humanScore, setHumanScore] = useState("");
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    try {
      await props.onFeedback({
        conversation_id: props.conversationId,
        turn_index: props.turnIndex,
        question: props.turn.question,
        answer_hash: hashText(props.turn.answer),
        helpful,
        evidence_supported: evidenceSupported,
        missing_answer: missingAnswer,
        human_score: humanScore ? Number(humanScore) : null,
        note,
        citation_ids: citationIdsFromTurn(props.turn)
      });
      setSaved(true);
    } finally {
      setSaving(false);
    }
  }

  return (
    <form className="feedback-panel" onSubmit={submit}>
      <div className="feedback-toggles">
        <button className={helpful === true ? "active" : ""} type="button" onClick={() => setHelpful(helpful === true ? null : true)}>
          <ThumbsUp size={14} />
          <span>有帮助</span>
        </button>
        <button className={helpful === false ? "active negative" : ""} type="button" onClick={() => setHelpful(helpful === false ? null : false)}>
          <ThumbsDown size={14} />
          <span>无帮助</span>
        </button>
        <button
          className={evidenceSupported === true ? "active" : ""}
          type="button"
          onClick={() => setEvidenceSupported(evidenceSupported === true ? null : true)}
        >
          <CheckCircle2 size={14} />
          <span>证据支持</span>
        </button>
        <button
          className={evidenceSupported === false ? "active negative" : ""}
          type="button"
          onClick={() => setEvidenceSupported(evidenceSupported === false ? null : false)}
        >
          <CircleSlash size={14} />
          <span>证据不足</span>
        </button>
        <button
          className={missingAnswer === true ? "active negative" : ""}
          type="button"
          onClick={() => setMissingAnswer(missingAnswer === true ? null : true)}
        >
          <FileText size={14} />
          <span>有遗漏</span>
        </button>
      </div>
      <div className="feedback-note-row">
        <select value={humanScore} onChange={(event) => setHumanScore(event.target.value)} aria-label="人工评分">
          <option value="">人工评分</option>
          {[5, 4, 3, 2, 1].map((score) => (
            <option key={score} value={score}>
              {score} 分
            </option>
          ))}
        </select>
        <input value={note} placeholder="补充反馈" onChange={(event) => setNote(event.target.value)} />
        <button type="submit" disabled={saving || saved}>
          {saving ? <Loader2 size={14} className="spin" /> : saved ? <CheckCircle2 size={14} /> : <Save size={14} />}
          <span>{saved ? "已保存" : "保存反馈"}</span>
        </button>
      </div>
    </form>
  );
}

function citationIdsFromTurn(turn: ConversationTurn) {
  const rows = turn.result.evidence_cards?.length ? turn.result.evidence_cards : turn.result.evidence;
  return rows.map((row) => String(row.citation_id ?? "")).filter(Boolean);
}

function hashText(text: string) {
  let hash = 5381;
  for (let index = 0; index < text.length; index += 1) {
    hash = ((hash << 5) + hash) ^ text.charCodeAt(index);
  }
  return (hash >>> 0).toString(16);
}

function ThoughtProgress(props: { items: string[] }) {
  return (
    <div className="thought-progress" aria-live="polite">
      {props.items.map((item, index) => (
        <div className="thought-step" key={`${item}-${index}`}>
          <span>{index + 1}</span>
          <p>{item}</p>
        </div>
      ))}
    </div>
  );
}

function DetailDrawer(props: {
  turn: ConversationTurn | null;
  citation: string;
  roles: ResearchRole[];
  onRetry: (question: string) => void;
  width: number;
  onWidth: (width: number) => void;
  open: boolean;
  tab: DetailTab;
  onTab: (tab: DetailTab) => void;
  onClose: () => void;
  onClaimReview: (claimId: string, payload: ClaimReviewRequest) => Promise<void>;
}) {
  useEffect(() => {
    if (!props.open || props.tab !== "evidence" || !props.citation) return;
    const timer = window.setTimeout(() => { const cards = document.querySelectorAll<HTMLElement>(".evidence-card[data-citation-id]"); [...cards].find(card => card.dataset.citationId === props.citation)?.scrollIntoView({ block: "nearest" }); }, 50);
    return () => window.clearTimeout(timer);
  }, [props.open, props.tab, props.citation, props.turn]);

  if (!props.turn || !props.open) return null;
  const result = props.turn.result;
  const evidence = result.evidence_cards?.length ? result.evidence_cards : result.evidence;
  const verification = verificationFromResult(result);

  return (
    <Inspector title={result.answer_type || "问答诊断"} label="证据详情" width={props.width} onWidth={props.onWidth} onClose={props.onClose}>
      {result.contextual_question && result.contextual_question !== result.question && (
        <div className="context-note">上下文改写：{result.contextual_question}</div>
      )}
      <div className="drawer-tabs">
        {[
          ["collaboration", "协作"],
          ["research", "投研"],
          ["verification", "验证"],
          ["evidence", "证据"],
          ["cypher", "查询"],
          ["diagnostics", "诊断"],
          ["graph", "子图"]
        ].map(([key, label]) => (
          <button
            key={key}
            type="button"
            className={props.tab === key ? "active" : ""}
            onClick={() => props.onTab(key as DetailTab)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="drawer-body">
        {props.tab === "collaboration" && <RoleWorkspace turn={props.turn} roles={props.roles} onEvidence={() => props.onTab("evidence")} onRetry={props.onRetry} defaultExpanded />}
        {props.tab === "research" && <ResearchOutputsView outputs={result.research_outputs} />}
        {props.tab === "verification" && <VerificationPanel verification={verification} />}
        {props.tab === "evidence" && (
          <EvidenceCards rows={evidence} empty="当前没有证据卡片" onClaimReview={props.onClaimReview} />
        )}
        {props.tab === "cypher" && (
          <div className="code-stack">
            <pre>{result.cypher || "无 Cypher 查询"}</pre>
            <JsonBlock value={result.cypher_params} />
          </div>
        )}
        {props.tab === "diagnostics" && (
          <div className="code-stack">
            <JsonBlock value={result.plan} title="问题规划" />
            <JsonBlock value={result.diagnostics} title="诊断信息" />
            {result.errors?.map((error) => (
              <div className="warning" key={error}>
                {error}
              </div>
            ))}
          </div>
        )}
        {props.tab === "graph" && <SubgraphMini edges={result.subgraph} />}
      </div>
    </Inspector>
  );
}

function renderAnswerMarkdown(markdown: string): ReactNode[] {
  const lines = markdown.split(/\r?\n/);
  const rendered: ReactNode[] = [];
  const cells = (line: string) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, "|"));
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const text = line.trim();
    const separator = lines[index + 1]?.trim();
    if (text.includes("|") && separator?.includes("|") && cells(separator).every((cell) => /^:?-{3,}:?$/.test(cell))) {
      const headers = cells(text);
      const rows: string[][] = [];
      const tableIndex = index;
      index += 2;
      while (index < lines.length && lines[index].trim().includes("|") && lines[index].trim()) {
        rows.push(cells(lines[index]));
        index += 1;
      }
      index -= 1;
      rendered.push(<div className="table-wrap markdown-table-wrap" key={tableIndex} tabIndex={0} aria-label="表格，可横向滚动">
        <table className="markdown-table"><thead><tr>{headers.map((header, i) => <th scope="col" key={i}>{renderInlineMarkdown(header)}</th>)}</tr></thead>
          <tbody>{rows.map((row, i) => <tr key={i}>{headers.map((_, j) => <td key={j}>{renderInlineMarkdown(row[j] ?? "")}</td>)}</tr>)}</tbody>
        </table>
      </div>);
      continue;
    }
    if (!text) { rendered.push(<div className="answer-space" key={index} />); continue; }
    if (text.startsWith("### ")) {
      rendered.push(<h3 key={index}>{renderInlineMarkdown(text.slice(4))}</h3>); continue;
    }
    if (text.startsWith("## ")) {
      rendered.push(<h3 key={index}>{renderInlineMarkdown(text.slice(3))}</h3>); continue;
    }
    const bulletMatch = text.match(/^[-*]\s+(.+)$/);
    if (bulletMatch) {
      rendered.push(
        <p className="answer-bullet" key={index}>
          <span>•</span>
          <span>{renderInlineMarkdown(bulletMatch[1])}</span>
        </p>
      ); continue;
    }
    const orderedMatch = text.match(/^(\d+)\.\s+(.+)$/);
    if (orderedMatch) {
      rendered.push(
        <p className="answer-bullet" key={index}>
          <span>{orderedMatch[1]}.</span>
          <span>{renderInlineMarkdown(orderedMatch[2])}</span>
        </p>
      ); continue;
    }
    rendered.push(<p key={index}>{renderInlineMarkdown(text)}</p>);
  }
  return rendered;
}

function renderInlineMarkdown(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|\[E\d+\])/g).filter(Boolean).map((part, index) => {
    if (/^\[E\d+\]$/.test(part)) return <button type="button" className="inline-citation" data-citation={part.slice(1, -1)} key={index} title="查看证据">{part}</button>;
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    return <span key={index}>{part}</span>;
  });
}

function OverviewView(props: { status: ApiStatus | null }) {
  const stats = props.status?.stats;
  const researchStats = stats?.research;
  return (
    <section className="page-panel">
      <div className="page-header">
        <span className="eyebrow">SYSTEM OVERVIEW</span>
        <h1>数据概览</h1>
        <p>图谱、RAG、Embedding 和 LLM 状态集中展示，便于演示前快速确认系统健康度。</p>
      </div>
      <div className="metric-grid">
        <MetricCard label="公司" value={stats?.companies} />
        <MetricCard label="报告" value={stats?.reports} />
        <MetricCard label="实体" value={stats?.entities} />
        <MetricCard label="关系" value={stats?.relations} />
        <MetricCard label="Claim" value={researchStats?.claims} />
        <MetricCard label="Dossier" value={researchStats?.dossiers} />
        <MetricCard label="已审校 Claim" value={researchStats?.reviewed_claims} />
        <MetricCard label="直接敞口公司" value={researchStats?.direct_exposure_companies} />
      </div>
      <div className="status-strip">
        <StatusPill label="图谱后端" value={props.status?.graph_backend?.toUpperCase() ?? "UNKNOWN"} />
        <StatusPill label="Neo4j" value={props.status?.neo4j_enabled ? "可用" : "降级/未启用"} />
        <StatusPill label="本地 RAG" value={props.status?.rag_enabled ? "就绪" : "未构建"} />
        <StatusPill label="投研层" value={props.status?.research_enabled ? "就绪" : "未构建"} />
        <StatusPill label="LLM" value={props.status?.llm_enabled ? "就绪" : "未配置"} />
        <StatusPill label="Embedding" value={props.status?.embedding_enabled ? "就绪" : "未启用"} />
      </div>
      <div className="chart-grid">
        <BarList title="实体分布" data={stats?.entity_counts ?? {}} />
        <BarList title="关系分布" data={stats?.relation_counts ?? {}} />
      </div>
    </section>
  );
}

function GraphView(props: { active: boolean }) {
  return <Suspense fallback={<div>正在加载图谱…</div>}><GraphExplorer active={props.active} /></Suspense>;
}

function MetricCard(props: { label: string; value?: number }) {
  return (
    <div className="metric-card">
      <span>{props.label}</span>
      <strong>{props.value?.toLocaleString() ?? "—"}</strong>
    </div>
  );
}

function StatusPill(props: { label: string; value: string }) {
  return (
    <div className="status-pill">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}

type GraphVariant = keyof typeof GRAPH_VIEWPORTS;

type GraphTransform = {
  x: number;
  y: number;
  scale: number;
};

type GraphNodeLayout = {
  id: string;
  type: string;
  degree: number;
  sourceCount: number;
  targetCount: number;
  x: number;
  y: number;
  r: number;
  lane: string;
};

type GraphColumnLayout = {
  id: string;
  label: string;
  type: string;
  x: number;
  count: number;
};

type GraphLayoutEdge = GraphEdge & {
  key: string;
};

type GraphLayout = {
  width: number;
  height: number;
  nodes: GraphNodeLayout[];
  edges: GraphLayoutEdge[];
  columns: GraphColumnLayout[];
  legendTypes: string[];
  showEdgeLabels: boolean;
};

function entityStyle(entityType: string) {
  return ENTITY_STYLES[entityType] ?? FALLBACK_ENTITY_STYLE;
}

function entityOrderIndex(entityType: string) {
  const index = ENTITY_ORDER.indexOf(entityType);
  return index >= 0 ? index : ENTITY_ORDER.length;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function splitSvgLabel(value: string, length: number) {
  const text = value.replace(/\s+/g, " ").trim();
  if (text.length <= length) return [text];
  const second = text.slice(length, length * 2);
  return [text.slice(0, length), text.length > length * 2 ? `${second}...` : second];
}

function graphEdgeKey(edge: GraphEdge, index: number) {
  return `${edge.source}-${edge.label}-${edge.target}-${index}`;
}

function createGraphLayout(edges: GraphEdge[], variant: GraphVariant): GraphLayout {
  const compact = variant === "compact";
  const viewport = GRAPH_VIEWPORTS[variant];
  const visibleEdges = edges
    .filter((edge) => edge.source?.trim() && edge.target?.trim())
    .slice(0, compact ? 70 : 140)
    .map((edge, index) => ({ ...edge, key: graphEdgeKey(edge, index) }));
  const nodeMap = new Map<string, GraphNodeLayout>();

  function ensureNode(id: string, entityType: string) {
    let node = nodeMap.get(id);
    if (!node) {
      node = {
        id,
        type: entityType,
        degree: 0,
        sourceCount: 0,
        targetCount: 0,
        x: 0,
        y: 0,
        r: compact ? 9 : 11,
        lane: ""
      };
      nodeMap.set(id, node);
    } else if (!node.type && entityType) {
      node.type = entityType;
    }
    return node;
  }

  visibleEdges.forEach((edge) => {
    const source = ensureNode(edge.source, edge.source_type);
    const target = ensureNode(edge.target, edge.target_type);
    source.degree += 1;
    source.sourceCount += 1;
    target.degree += 1;
    target.targetCount += 1;
  });

  const nodes = Array.from(nodeMap.values());
  nodes.forEach((node) => {
    node.r = clamp((compact ? 7 : 8) + Math.sqrt(node.degree) * (compact ? 1.9 : 2.2), compact ? 8 : 9, compact ? 15 : 18);
  });

  function sortNodes(a: GraphNodeLayout, b: GraphNodeLayout) {
    return (
      entityOrderIndex(a.type) - entityOrderIndex(b.type) ||
      b.degree - a.degree ||
      zhCollator.compare(a.id, b.id)
    );
  }

  let sourceNodes = nodes
    .filter((node) => node.type === "Company" || (node.sourceCount > 0 && node.sourceCount >= node.targetCount))
    .sort(sortNodes);
  if (!sourceNodes.length) {
    sourceNodes = nodes.filter((node) => node.sourceCount > 0).sort(sortNodes);
  }
  const sourceSet = new Set(sourceNodes.map((node) => node.id));
  const targetGroups = new Map<string, GraphNodeLayout[]>();
  nodes
    .filter((node) => !sourceSet.has(node.id))
    .sort(sortNodes)
    .forEach((node) => {
      const key = node.type || "Entity";
      const group = targetGroups.get(key) ?? [];
      group.push(node);
      targetGroups.set(key, group);
    });

  const columns: Array<{ id: string; label: string; type: string; nodes: GraphNodeLayout[] }> = [];
  const maxRowsPerColumn = compact ? 12 : 24;

  function pushColumns(id: string, label: string, type: string, columnNodes: GraphNodeLayout[]) {
    for (let start = 0; start < columnNodes.length; start += maxRowsPerColumn) {
      const chunk = columnNodes.slice(start, start + maxRowsPerColumn);
      const suffix = columnNodes.length > maxRowsPerColumn ? ` ${Math.floor(start / maxRowsPerColumn) + 1}` : "";
      columns.push({ id: `${id}-${start}`, label: `${label}${suffix}`, type, nodes: chunk });
    }
  }

  if (sourceNodes.length) {
    pushColumns("source", "起点", "", sourceNodes);
  }
  Array.from(targetGroups.entries())
    .sort(([a], [b]) => entityOrderIndex(a) - entityOrderIndex(b) || zhCollator.compare(entityStyle(a).label, entityStyle(b).label))
    .forEach(([type, group]) => {
      pushColumns(type, entityStyle(type).label, type, group);
    });
  if (!columns.length && nodes.length) {
    columns.push({ id: "entities", label: "实体", type: "", nodes });
  }

  const nodeGap = compact ? 46 : 54;
  const columnGap = compact ? 214 : 282;
  const leftPadding = compact ? 64 : 88;
  const topPadding = compact ? 58 : 68;
  const rightPadding = compact ? 188 : 240;
  const bottomPadding = compact ? 72 : 88;
  const maxRows = Math.max(...columns.map((column) => column.nodes.length), 1);
  const width = Math.max(viewport.width, leftPadding + Math.max(columns.length - 1, 0) * columnGap + rightPadding);
  const height = Math.max(viewport.height, topPadding + Math.max(maxRows - 1, 0) * nodeGap + bottomPadding);
  const columnLayouts: GraphColumnLayout[] = [];

  columns.forEach((column, columnIndex) => {
    const x = leftPadding + columnIndex * columnGap;
    columnLayouts.push({ id: column.id, label: column.label, type: column.type, x, count: column.nodes.length });
    column.nodes.forEach((node, nodeIndex) => {
      node.x = x;
      node.y = topPadding + nodeIndex * nodeGap;
      node.lane = column.id;
    });
  });

  return {
    width,
    height,
    nodes,
    edges: visibleEdges,
    columns: columnLayouts,
    legendTypes: Array.from(new Set(nodes.map((node) => node.type))).sort((a, b) => entityOrderIndex(a) - entityOrderIndex(b)),
    showEdgeLabels: visibleEdges.length <= (compact ? 10 : 18) && nodes.length <= (compact ? 22 : 34)
  };
}

function fitGraphTransform(layout: GraphLayout, viewport: { width: number; height: number }): GraphTransform {
  if (!layout.nodes.length) return { x: 0, y: 0, scale: 1 };
  const padding = 42;
  const scale = Math.min(
    (viewport.width - padding * 2) / Math.max(layout.width, 1),
    (viewport.height - padding * 2) / Math.max(layout.height, 1),
    1.18
  );
  const safeScale = clamp(scale, 0.08, 1.18);
  return {
    x: (viewport.width - layout.width * safeScale) / 2,
    y: (viewport.height - layout.height * safeScale) / 2,
    scale: safeScale
  };
}

function graphPath(source: GraphNodeLayout, target: GraphNodeLayout, index: number) {
  const direction = target.x >= source.x ? 1 : -1;
  const startX = source.x + source.r * direction;
  const endX = target.x - target.r * direction;
  const startY = source.y;
  const endY = target.y;
  const offset = ((index % 9) - 4) * 3.5;
  if (Math.abs(endX - startX) < 28) {
    const loop = 44 + (index % 5) * 10;
    return `M ${startX.toFixed(1)} ${startY.toFixed(1)} C ${(startX + loop).toFixed(1)} ${(startY - 22).toFixed(1)}, ${(endX + loop).toFixed(1)} ${(endY + 22).toFixed(1)}, ${endX.toFixed(1)} ${endY.toFixed(1)}`;
  }
  const curve = Math.max(72, Math.abs(endX - startX) * 0.42);
  return `M ${startX.toFixed(1)} ${startY.toFixed(1)} C ${(startX + curve * direction).toFixed(1)} ${(startY + offset).toFixed(1)}, ${(endX - curve * direction).toFixed(1)} ${(endY - offset).toFixed(1)}, ${endX.toFixed(1)} ${endY.toFixed(1)}`;
}

function edgeLabelPoint(source: GraphNodeLayout, target: GraphNodeLayout, index: number) {
  return {
    x: (source.x + target.x) / 2,
    y: (source.y + target.y) / 2 - 7 + ((index % 5) - 2) * 4
  };
}

function GraphCanvas(props: { edges: GraphEdge[]; variant?: GraphVariant; empty: string }) {
  const variant = props.variant ?? "large";
  const viewport = GRAPH_VIEWPORTS[variant];
  const layout = useMemo(() => createGraphLayout(props.edges, variant), [props.edges, variant]);
  const [transform, setTransform] = useState<GraphTransform>({ x: 0, y: 0, scale: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
    unitX: number;
    unitY: number;
  } | null>(null);
  const reactId = useId();
  const arrowId = useMemo(() => `graph-arrow-${reactId.replace(/:/g, "")}`, [reactId]);
  const nodeById = useMemo(() => new Map(layout.nodes.map((node) => [node.id, node])), [layout.nodes]);

  useEffect(() => {
    setTransform(fitGraphTransform(layout, viewport));
  }, [layout, viewport.height, viewport.width]);

  if (!layout.edges.length) return <div className="empty-table">{props.empty}</div>;

  function zoomAt(point: { x: number; y: number }, scale: number) {
    setTransform((current) => {
      const nextScale = clamp(scale, 0.08, 4.5);
      const localX = (point.x - current.x) / current.scale;
      const localY = (point.y - current.y) / current.scale;
      return {
        scale: nextScale,
        x: point.x - localX * nextScale,
        y: point.y - localY * nextScale
      };
    });
  }

  function zoomBy(factor: number) {
    setTransform((current) => {
      const point = { x: viewport.width / 2, y: viewport.height / 2 };
      const nextScale = clamp(current.scale * factor, 0.08, 4.5);
      const localX = (point.x - current.x) / current.scale;
      const localY = (point.y - current.y) / current.scale;
      return {
        scale: nextScale,
        x: point.x - localX * nextScale,
        y: point.y - localY * nextScale
      };
    });
  }

  function fitGraph() {
    setTransform(fitGraphTransform(layout, viewport));
  }

  function eventPoint(event: WheelEvent<SVGSVGElement> | PointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * viewport.width,
      y: ((event.clientY - rect.top) / rect.height) * viewport.height
    };
  }

  function onWheel(event: WheelEvent<SVGSVGElement>) {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.16 : 0.86;
    zoomAt(eventPoint(event), transform.scale * factor);
  }

  function onPointerDown(event: PointerEvent<SVGSVGElement>) {
    if (event.button !== 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: transform.x,
      originY: transform.y,
      unitX: viewport.width / rect.width,
      unitY: viewport.height / rect.height
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setIsPanning(true);
  }

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    setTransform((current) => ({
      ...current,
      x: drag.originX + (event.clientX - drag.startX) * drag.unitX,
      y: drag.originY + (event.clientY - drag.startY) * drag.unitY
    }));
  }

  function stopPanning(event: PointerEvent<SVGSVGElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    if (event.currentTarget.hasPointerCapture(drag.pointerId)) {
      event.currentTarget.releasePointerCapture(drag.pointerId);
    }
    dragRef.current = null;
    setIsPanning(false);
  }

  return (
    <div className={`graph-canvas ${variant}`}>
      <div className="graph-toolbar">
        <span className="graph-count">{layout.nodes.length} 节点 · {layout.edges.length} 关系 · {Math.round(transform.scale * 100)}%</span>
        <div className="graph-actions" aria-label="图谱控制">
          <span className="graph-tool-hint" title="拖拽画布可平移" aria-label="拖拽画布可平移">
            <Move size={16} />
          </span>
          <button type="button" title="缩小" aria-label="缩小" onClick={() => zoomBy(0.78)}>
            <ZoomOut size={16} />
          </button>
          <button type="button" title="放大" aria-label="放大" onClick={() => zoomBy(1.28)}>
            <ZoomIn size={16} />
          </button>
          <button type="button" title="适配窗口" aria-label="适配窗口" onClick={fitGraph}>
            <Maximize2 size={16} />
          </button>
        </div>
      </div>
      <svg
        className={isPanning ? "is-panning" : ""}
        viewBox={`0 0 ${viewport.width} ${viewport.height}`}
        role="img"
        aria-label="产业链知识图谱"
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={stopPanning}
        onPointerCancel={stopPanning}
      >
        <rect width={viewport.width} height={viewport.height} fill="#f8fafc" />
        <defs>
          <marker id={arrowId} markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto">
            <path d="M0,0 L0,7 L7,3.5 z" fill="#94a3b8" />
          </marker>
        </defs>
        <g transform={`translate(${transform.x.toFixed(2)} ${transform.y.toFixed(2)}) scale(${transform.scale.toFixed(4)})`}>
          {layout.columns.map((column) => (
            <g className="graph-column-label" key={column.id}>
              <text x={column.x} y="32" textAnchor="middle">
                {column.label}
              </text>
              <text x={column.x} y="49" textAnchor="middle">
                {column.count}
              </text>
            </g>
          ))}
          {layout.edges.map((edge, index) => {
            const source = nodeById.get(edge.source);
            const target = nodeById.get(edge.target);
            if (!source || !target) return null;
            const labelPoint = edgeLabelPoint(source, target, index);
            return (
              <g className="graph-edge-group" key={edge.key}>
                <path className="graph-edge" d={graphPath(source, target, index)} markerEnd={`url(#${arrowId})`}>
                  <title>
                    {edge.source} - {edge.label} - {edge.target}
                    {edge.citation_id ? ` · ${edge.citation_id}` : ""}
                  </title>
                </path>
                {layout.showEdgeLabels && (
                  <text className="graph-edge-label" x={labelPoint.x} y={labelPoint.y} textAnchor="middle">
                    {edge.label}
                  </text>
                )}
              </g>
            );
          })}
          {layout.nodes.map((node) => {
            const style = entityStyle(node.type);
            const lines = splitSvgLabel(node.id, variant === "compact" ? 8 : 10);
            return (
              <g className="graph-node" key={node.id}>
                <circle cx={node.x} cy={node.y} r={node.r} fill={style.color}>
                  <title>
                    {node.id} · {style.label}
                  </title>
                </circle>
                <text className="graph-node-label" x={node.x + node.r + 8} y={node.y - (lines.length - 1) * 7 + 4}>
                  {lines.map((line, index) => (
                    <tspan key={line} x={node.x + node.r + 8} dy={index === 0 ? 0 : 14}>
                      {line}
                    </tspan>
                  ))}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
      <div className="graph-legend">
        {layout.legendTypes.map((type) => {
          const style = entityStyle(type);
          return (
            <span key={type}>
              <i style={{ background: style.color }} />
              {style.label}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function BarList(props: { title: string; data: Record<string, number> }) {
  const entries = useMemo(() => Object.entries(props.data).sort((a, b) => b[1] - a[1]).slice(0, 10), [props.data]);
  const max = Math.max(...entries.map(([, value]) => value), 1);
  return (
    <div className="bar-card">
      <h3>{props.title}</h3>
      {entries.map(([label, value]) => (
        <div className="bar-row" key={label}>
          <span>{label}</span>
          <div>
            <i style={{ width: `${Math.max((value / max) * 100, 4)}%` }} />
          </div>
          <b>{value}</b>
        </div>
      ))}
    </div>
  );
}

function ResearchOutputsView(props: { outputs?: ResearchOutputs }) {
  const outputs = props.outputs;
  if (!outputs) return <div className="empty-table">当前回答没有生成投研产物</div>;
  const report = outputs.report;
  const charts = report?.spec?.charts;
  const hasCharts = hasReportChartData(charts);
  const table = outputs.company_compare_table;
  const risks = outputs.risk_checklist ?? [];
  const gaps = outputs.evidence_gaps ?? [];
  return (
    <div className="research-output">
      <TaskOutputsView outputs={outputs.task_outputs} />
      <ReportChartsView charts={charts} />
      <section className="research-section">
        <div className="research-section-title">
          <FileText size={16} />
          {report?.markdown ? <ContextMenu label="研究报告操作" actions={[{ label: "复制报告", run: () => { void navigator.clipboard.writeText(report.markdown).catch(() => window.alert("复制失败，请检查浏览器权限")); } }]}><span>{report.title || "投研简报"}</span></ContextMenu> : <span>{report?.title || "投研简报"}</span>}
        </div>
        <div className="research-report">
          {report?.sections?.length ? (
            report.sections.filter((section) => !(hasCharts && section.title === "图表")).map((section) => (
              <div className="research-report-section" key={section.title}>
                <h4>{section.title}</h4>
                <div>{renderAnswerMarkdown(section.content)}</div>
              </div>
            ))
          ) : (
            <div className="empty-table">当前证据不足以生成报告</div>
          )}
        </div>
      </section>
      <section className="research-section">
        <div className="research-section-title">
          <Table2 size={16} />
          <span>公司对比表</span>
        </div>
        <ResearchTable rows={table?.rows ?? []} empty="当前证据不足以生成公司对比表" />
      </section>
      <section className="research-section">
        <div className="research-section-title">
          <ClipboardList size={16} />
          <span>风险清单</span>
        </div>
        <ResearchList rows={risks} primaryKey="risk" secondaryKeys={["scope", "priority", "follow_up", "citation_id"]} empty="当前证据不足以生成风险清单" />
      </section>
      <section className="research-section">
        <div className="research-section-title">
          <FileText size={16} />
          <span>证据缺口</span>
        </div>
        <ResearchList rows={gaps} primaryKey="gap" secondaryKeys={["priority", "suggested_source"]} empty="当前证据包未识别出关键缺口" />
      </section>
    </div>
  );
}

function ReportChartsView(props: { charts?: ReportCharts }) {
  const charts = props.charts;
  if (!hasReportChartData(charts)) return null;
  return (
    <section className="research-section report-chart-section">
      <div className="research-section-title">
        <BarChart3 size={16} />
        <span>证据覆盖图表</span>
      </div>
      <div className="report-chart-grid">
        <EvidenceStrengthChart chart={charts?.evidence_strength_bar} />
        <FreshnessTimelineChartView chart={charts?.source_freshness_timeline} />
        <CompanyHeatmapChart chart={charts?.company_coverage_heatmap} />
        <FlowMapChartView chart={charts?.flow_map} />
      </div>
    </section>
  );
}

function hasReportChartData(charts?: ReportCharts): boolean {
  if (!charts) return false;
  return (
    Boolean(charts.company_coverage_heatmap?.cells?.length) ||
    Boolean(charts.evidence_strength_bar?.counts && Object.values(charts.evidence_strength_bar.counts).some(Boolean)) ||
    Boolean(charts.flow_map?.links?.length) ||
    Boolean(charts.source_freshness_timeline?.items?.length)
  );
}

function EvidenceStrengthChart(props: { chart?: EvidenceStrengthBarChart }) {
  const counts = props.chart?.counts ?? {};
  const entries = Object.entries(counts);
  if (!entries.some(([, value]) => Number(value) > 0)) return <ChartEmpty title={props.chart?.title ?? "证据强度柱状图"} message={props.chart?.empty_message} />;
  const max = Math.max(...entries.map(([, value]) => Number(value)), 1);
  const labels: Record<string, string> = {
    direct: "直接证据",
    indirect: "间接证据",
    mentioned: "仅提及",
    unsupported: "未支撑",
    no_evidence: "无证据"
  };
  return (
    <article className="report-chart-card">
      <div className="report-chart-head">
        <strong>{props.chart?.title ?? "证据强度柱状图"}</strong>
        <span>数量</span>
      </div>
      <div className="evidence-bar-chart">
        {entries.map(([key, value]) => (
          <div className="evidence-bar-row" key={key}>
            <span>{labels[key] ?? key}</span>
            <div>
              <i style={{ width: `${(Number(value) / max) * 100}%` }} />
            </div>
            <b>{value}</b>
          </div>
        ))}
      </div>
      <p className="chart-note">敞口级别按证据卡统计；未支撑和无证据按审计条目统计，统计口径不同。</p>
    </article>
  );
}

function FreshnessTimelineChartView(props: { chart?: FreshnessTimelineChart }) {
  const items = props.chart?.items ?? [];
  if (!items.length) return <ChartEmpty title="来源年份分布" message={props.chart?.empty_message} />;
  const max = Math.max(...items.map((item) => Number(item.count)), 1);
  return (
    <article className="report-chart-card">
      <div className="report-chart-head">
        <strong>来源年份分布</strong>
        <span>证据卡数</span>
      </div>
      <p className="chart-note">仅统计标注日期的证据卡；新鲜度为报告生成时的状态。</p>
      <div className="freshness-timeline-chart">
        {items.map((item) => (
          <div className={`freshness-year ${item.freshness || "unknown"}`} key={item.year}>
            <span>{item.year}</span>
            <div>
              <i style={{ height: `${Math.max((Number(item.count) / max) * 100, 12)}%` }} />
            </div>
            <b>{item.count}</b>
            <em>{freshnessLabel(item.freshness)}</em>
          </div>
        ))}
      </div>
    </article>
  );
}

function CompanyHeatmapChart(props: { chart?: CompanyCoverageHeatmapChart }) {
  const chart = props.chart;
  const rows = chart?.rows ?? [];
  const columns = chart?.columns ?? [];
  const cells = chart?.cells ?? [];
  if (!rows.length || !columns.length || !cells.length) return <ChartEmpty title={chart?.title ?? "公司覆盖热力图"} message={chart?.empty_message} />;
  const cellMap = new Map(cells.map((cell) => [`${cell.company}\u0000${cell.segment}`, cell]));
  return (
    <article className="report-chart-card wide">
      <div className="report-chart-head">
        <strong>{chart?.title ?? "公司覆盖热力图"}</strong>
        <span>0 无证据 · 5 高覆盖</span>
      </div>
      <div className="heatmap-wrap">
        <table className="coverage-heatmap">
          <thead>
            <tr>
              <th>公司</th>
              {columns.map((column) => <th key={column}>{column}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row}>
                <th>{row}</th>
                {columns.map((column) => {
                  const cell = cellMap.get(`${row}\u0000${column}`);
                  const score = clampNumber(Number(cell?.score ?? 0), 0, 5);
                  return (
                    <td key={column}>
                      <span className={`heatmap-score score-${score}`} title={`${cell?.reason || "无证据"} ${cell?.evidence_ids?.join("、") || ""}`}>
                        {score}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

function FlowMapChartView(props: { chart?: FlowMapChart }) {
  const links = props.chart?.links ?? [];
  if (!links.length) return <ChartEmpty title="证据关联" message={props.chart?.empty_message} />;
  return (
    <article className="report-chart-card wide">
      <div className="report-chart-head">
        <strong>证据关联</strong>
        <span>共 {links.length} 条</span>
      </div>
      <p className="chart-note">展示记录中的起点、关联对象和本地权重，不能据此判断供应链关系或市场规模。历史报告的起点可能是文档名称。</p>
      <div className="table-wrap flow-relations-wrap" tabIndex={0} aria-label="证据关联表，可滚动查看">
        <table className="markdown-table flow-relations">
          <thead><tr><th scope="col">记录起点</th><th scope="col">关联对象</th><th scope="col">权重</th><th scope="col">证据编号</th></tr></thead>
          <tbody>{links.map((link, index) => <tr key={`${link.source}-${link.target}-${index}`}>
            <td>{link.source}</td><td>{link.target}</td><td>{link.value}</td><td>{link.evidence_ids?.join("、") || "未标注"}</td>
          </tr>)}</tbody>
        </table>
      </div>
    </article>
  );
}

function ChartEmpty(props: { title: string; message?: string }) {
  return (
    <article className="report-chart-card empty">
      <div className="report-chart-head">
        <strong>{props.title}</strong>
        <span>No chart data</span>
      </div>
      <div className="empty-table">{props.message || "当前证据不足"}</div>
    </article>
  );
}

function freshnessLabel(value: string) {
  return { fresh: "新鲜", aging: "较旧", stale: "过期", unknown: "未知" }[value] ?? value;
}

function clampNumber(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, Math.round(value)));
}

const CONFLICT_TYPE_LABELS: Record<string, string> = {
  optimistic_vs_risk: "乐观判断 vs 风险披露",
  old_vs_new: "新旧证据冲突",
  exposure_level_conflict: "敞口等级冲突",
  technical_route_conflict: "技术路线冲突"
};

function VerificationPanel(props: { verification?: VerificationResult }) {
  const verification = props.verification;
  if (!verification) return <div className="empty-table">当前回答没有生成验证结果</div>;
  const gaps = arrayRecords(verification.evidence_gaps ?? []);
  const conflicts = verification.conflict_groups ?? [];
  return (
    <div className="verification-panel">
      <section className="research-section">
        <div className="research-section-title">
          <ClipboardList size={16} />
          <span>验证状态</span>
        </div>
        <div className={`verification-summary ${verification.status || "unknown"}`}>
          <strong>{verificationStatusLabel(verification.status)}</strong>
          <span>{gaps.length} 个证据缺口</span>
          <span>{conflicts.length} 组冲突证据</span>
        </div>
      </section>
      <section className="research-section">
        <div className="research-section-title">
          <FileText size={16} />
          <span>证据缺口</span>
        </div>
        <ResearchList rows={gaps} primaryKey="gap" secondaryKeys={["priority", "suggested_source"]} empty="验证未发现关键证据缺口" />
      </section>
      <section className="research-section">
        <div className="research-section-title">
          <ClipboardList size={16} />
          <span>冲突证据</span>
        </div>
        <ConflictGroupsView groups={conflicts} />
      </section>
    </div>
  );
}

function ConflictGroupsView(props: { groups: ConflictGroup[] }) {
  if (!props.groups.length) return <div className="empty-table">当前未识别出冲突证据</div>;
  return (
    <div className="conflict-list">
      {props.groups.slice(0, 8).map((group) => (
        <article className="conflict-card" key={group.conflict_group_id}>
          <div className="conflict-card-head">
            <strong>{CONFLICT_TYPE_LABELS[group.conflict_type] ?? group.conflict_type}</strong>
            <span>{[group.company, group.topic].filter(Boolean).join(" · ") || group.conflict_group_id}</span>
          </div>
          <div className="conflict-claims">
            <ConflictClaimView label="证据 A" claim={group.claim_a} />
            <ConflictClaimView label="证据 B" claim={group.claim_b} />
          </div>
          <p className="conflict-resolution">{group.resolution}</p>
        </article>
      ))}
    </div>
  );
}

function ConflictClaimView(props: { label: string; claim: ConflictGroup["claim_a"] }) {
  const claim = props.claim ?? {};
  return (
    <div className="conflict-claim">
      <span>{props.label}</span>
      <strong>{claim.citation_id || claim.claim_id || "未编号"}</strong>
      <p>{claim.evidence || claim.title || "无证据文本"}</p>
      <div>
        {claim.source && <em>{claim.source}</em>}
        {claim.page && <em>p.{claim.page}</em>}
        {claim.as_of_date && <em>{claim.as_of_date}</em>}
        {claim.claim_type && <em>{CLAIM_TYPE_LABELS[claim.claim_type] ?? claim.claim_type}</em>}
        {claim.exposure_level && <em>{EXPOSURE_LABELS[claim.exposure_level] ?? claim.exposure_level}</em>}
      </div>
    </div>
  );
}

function TaskOutputsView(props: { outputs?: Record<string, unknown> }) {
  const outputs = props.outputs;
  if (!outputs) return null;
  const schemaType = String(outputs.schema_type ?? "");
  const taskLabel = String(outputs.task_label ?? schemaType);
  const cards = taskOutputCards(outputs);
  if (!cards.length) return null;
  return (
    <section className="research-section task-output-section">
      <div className="research-section-title">
        <Boxes size={16} />
        <span>{taskLabel || "任务结构化输出"}</span>
      </div>
      <div className="task-output-grid">
        {cards.map((card) => (
          <article className="task-output-card" key={card.title}>
            <strong>{card.title}</strong>
            <ResearchList rows={card.rows} primaryKey={card.primaryKey} secondaryKeys={card.secondaryKeys} empty="当前证据不足" />
          </article>
        ))}
      </div>
    </section>
  );
}

function taskOutputCards(outputs: Record<string, unknown>) {
  const specs = [
    { key: "common_drivers", title: "共同驱动", primaryKey: "evidence", secondaryKeys: ["scope", "citation_id"] },
    { key: "differences", title: "差异点", primaryKey: "business_position", secondaryKeys: ["company", "leading_indicators", "citations"] },
    { key: "risk_differences", title: "风险差异", primaryKey: "risks", secondaryKeys: ["company", "citations"] },
    { key: "technology_products", title: "产品/技术", primaryKey: "evidence", secondaryKeys: ["scope", "citation_id"] },
    { key: "indicators", title: "指标证据", primaryKey: "evidence", secondaryKeys: ["scope", "citation_id"] },
    { key: "counter_evidence", title: "反证/边界", primaryKey: "evidence", secondaryKeys: ["scope", "citation_id"] },
    { key: "follow_up_indicators", title: "跟踪指标", primaryKey: "indicator", secondaryKeys: ["scope", "citation_id"] },
    { key: "missing_companies", title: "缺失公司证据", primaryKey: "gap", secondaryKeys: ["priority", "suggested_source"] },
    { key: "missing_metrics", title: "缺失指标证据", primaryKey: "gap", secondaryKeys: ["priority", "suggested_source"] },
    { key: "missing_risks", title: "缺失风险证据", primaryKey: "gap", secondaryKeys: ["priority", "suggested_source"] },
    { key: "evidence_index", title: "证据索引", primaryKey: "title", secondaryKeys: ["citation_id", "source", "page"] }
  ];
  return specs
    .map((spec) => ({
      ...spec,
      rows: arrayRecords(outputs[spec.key])
    }))
    .filter((item) => item.rows.length > 0)
    .slice(0, 4);
}

function arrayRecords(value: unknown): Record<string, string>[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (item && typeof item === "object" && !Array.isArray(item)) {
        return Object.fromEntries(Object.entries(item as Record<string, unknown>).map(([key, val]) => [key, String(val ?? "")]));
      }
      return { value: String(item ?? "") };
    })
    .filter((row) => Object.values(row).some(Boolean));
}

function ResearchTable(props: { rows: Record<string, string>[]; empty: string }) {
  if (!props.rows.length) return <div className="empty-table">{props.empty}</div>;
  return (
    <div className="research-table">
      {props.rows.slice(0, 10).map((row) => (
        <article className="research-row" key={row.company}>
          <div>
            <strong>{row.company || "公司"}</strong>
            <span>{row.chain_segment || "未识别"} · {row.exposure_level || "未分级"}</span>
          </div>
          <p>{row.business_evidence || "当前证据不足"}</p>
          <dl>
            <dt>指标</dt>
            <dd>{row.leading_indicators || "当前证据不足"}</dd>
            <dt>风险</dt>
            <dd>{row.risks || "当前证据不足"}</dd>
            <dt>证据</dt>
            <dd>{row.citations || "无"}</dd>
          </dl>
        </article>
      ))}
    </div>
  );
}

function ResearchList(props: {
  rows: Record<string, string>[];
  primaryKey: string;
  secondaryKeys: string[];
  empty: string;
}) {
  if (!props.rows.length) return <div className="empty-table">{props.empty}</div>;
  return (
    <div className="research-list">
      {props.rows.slice(0, 12).map((row, index) => (
        <article className="research-list-item" key={`${row[props.primaryKey]}-${index}`}>
          <p>{row[props.primaryKey]}</p>
          <div>
            {props.secondaryKeys.map((key) => (
              row[key] ? <span key={key}>{row[key]}</span> : null
            ))}
          </div>
        </article>
      ))}
    </div>
  );
}

const EVIDENCE_KIND_LABELS: Record<string, string> = {
  dossier: "Dossier",
  claim: "Claim",
  graph: "Graph",
  rag: "RAG"
};

const CLAIM_TYPE_LABELS: Record<string, string> = {
  company_exposure: "公司敞口",
  mechanism: "技术机理",
  bottleneck: "瓶颈",
  indicator: "领先指标",
  risk: "风险",
  supply_chain: "产业传导",
  trend: "趋势",
  policy: "政策"
};

const EXPOSURE_LABELS: Record<string, string> = {
  core: "核心敞口",
  direct: "直接敞口",
  indirect: "间接敞口",
  mentioned: "仅提及"
};

const CLAIM_REVIEW_STATUSES = [
  ["revised", "已修正"],
  ["approved", "已确认"],
  ["needs_review", "待复核"],
  ["rejected", "噪声/弃用"]
];

function ClaimReviewPanel(props: {
  row: Record<string, unknown>;
  claimId: string;
  onSubmit: (claimId: string, payload: ClaimReviewRequest) => Promise<void>;
}) {
  const [claimText, setClaimText] = useState(fieldText(props.row, "evidence"));
  const [topic, setTopic] = useState(fieldText(props.row, "topic"));
  const [claimType, setClaimType] = useState(fieldText(props.row, "claim_type"));
  const [exposureLevel, setExposureLevel] = useState(fieldText(props.row, "exposure_level"));
  const [evidenceSpan, setEvidenceSpan] = useState(fieldText(props.row, "evidence_span"));
  const [confidence, setConfidence] = useState(fieldText(props.row, "confidence"));
  const [asOfDate, setAsOfDate] = useState(fieldText(props.row, "as_of_date"));
  const [reviewStatus, setReviewStatus] = useState(fieldText(props.row, "review_status") || "revised");
  const [reviewerNote, setReviewerNote] = useState(fieldText(props.row, "reviewer_note"));
  const [saving, setSaving] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    try {
      await props.onSubmit(props.claimId, {
        claim_text: claimText,
        topic,
        claim_type: claimType,
        exposure_level: exposureLevel,
        evidence_span: evidenceSpan,
        confidence,
        as_of_date: asOfDate,
        review_status: reviewStatus,
        reviewer_note: reviewerNote
      });
    } finally {
      setSaving(false);
    }
  }

  return (
    <details className="claim-review">
      <summary>
        <Pencil size={14} />
        <span>修正 Claim</span>
      </summary>
      <form onSubmit={handleSubmit}>
        <label>
          Claim
          <textarea value={claimText} rows={4} onChange={(event) => setClaimText(event.target.value)} />
        </label>
        <div className="review-grid">
          <label>
            主题
            <input value={topic} onChange={(event) => setTopic(event.target.value)} />
          </label>
          <label>
            类型
            <select value={claimType} onChange={(event) => setClaimType(event.target.value)}>
              <option value="">未分类</option>
              {Object.entries(CLAIM_TYPE_LABELS).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>
          <label>
            敞口
            <select value={exposureLevel} onChange={(event) => setExposureLevel(event.target.value)}>
              <option value="">未分级</option>
              {Object.entries(EXPOSURE_LABELS).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>
          <label>
            状态
            <select value={reviewStatus} onChange={(event) => setReviewStatus(event.target.value)}>
              {CLAIM_REVIEW_STATUSES.map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>
          <label>
            置信度
            <input value={confidence} onChange={(event) => setConfidence(event.target.value)} />
          </label>
          <label>
            时点
            <input value={asOfDate} onChange={(event) => setAsOfDate(event.target.value)} />
          </label>
        </div>
        <label>
          原文证据
          <textarea value={evidenceSpan} rows={3} onChange={(event) => setEvidenceSpan(event.target.value)} />
        </label>
        <label>
          备注
          <textarea value={reviewerNote} rows={2} onChange={(event) => setReviewerNote(event.target.value)} />
        </label>
        <button type="submit" disabled={saving}>
          {saving ? <Loader2 size={15} className="spin" /> : <Save size={15} />}
          <span>保存修正</span>
        </button>
      </form>
    </details>
  );
}

function EvidenceCards(props: {
  rows: Record<string, unknown>[];
  empty: string;
  onClaimReview: (claimId: string, payload: ClaimReviewRequest) => Promise<void>;
}) {
  if (!props.rows.length) return <div className="empty-table">{props.empty}</div>;
  const groups = props.rows.reduce<Record<string, Record<string, unknown>[]>>((acc, row) => {
    const kind = fieldText(row, "kind") || "other";
    const label = EVIDENCE_KIND_LABELS[kind] ?? kind;
    acc[label] = acc[label] ?? [];
    acc[label].push(row);
    return acc;
  }, {});
  return (
    <div className="evidence-groups">
      {Object.entries(groups).map(([label, rows]) => (
        <section className="evidence-group" key={label}>
          <div className="evidence-group-title">
            <span>{label}</span>
            <b>{rows.length}</b>
          </div>
          {rows.map((row, index) => (
            <EvidenceCardView
              row={row}
              key={`${fieldText(row, "citation_id") || label}-${index}`}
              onClaimReview={props.onClaimReview}
            />
          ))}
        </section>
      ))}
    </div>
  );
}

function EvidenceCardView(props: {
  row: Record<string, unknown>;
  onClaimReview: (claimId: string, payload: ClaimReviewRequest) => Promise<void>;
}) {
  const row = props.row;
  const citationId = fieldText(row, "citation_id");
  const claimId = fieldText(row, "claim_id");
  const title = fieldText(row, "title") || fieldText(row, "source") || "证据";
  const evidence = fieldText(row, "evidence");
  const evidenceSpan = fieldText(row, "evidence_span");
  const source = fieldText(row, "source");
  const page = fieldText(row, "page");
  const section = fieldText(row, "section");
  const claimType = fieldText(row, "claim_type");
  const exposureLevel = fieldText(row, "exposure_level");
  const confidence = fieldText(row, "confidence");
  const score = fieldText(row, "score");
  const semanticScore = fieldText(row, "semantic_score");
  const semanticRefId = fieldText(row, "semantic_ref_id");
  const knowledgeId = fieldText(row, "document_id") || /^knowledge:([0-9a-f]{32}):/.exec(semanticRefId)?.[1] || "";
  const knowledgeVersion = /^knowledge:[0-9a-f]{32}:(\d+):/.exec(semanticRefId)?.[1];
  const reviewStatus = fieldText(row, "review_status");
  const reviewerNote = fieldText(row, "reviewer_note");
  return (
    <article className="evidence-card" data-citation-id={citationId}>
      <div className="evidence-card-head">
        <div>
          <span className="citation-pill">{citationId || "E?"}</span>
          <strong>{title}</strong>
        </div>
        <div className="evidence-tags">
          {claimType && <span>{CLAIM_TYPE_LABELS[claimType] ?? claimType}</span>}
          {exposureLevel && <span>{EXPOSURE_LABELS[exposureLevel] ?? exposureLevel}</span>}
          {reviewStatus && <span>{reviewStatus}</span>}
        </div>
      </div>
      <p>{evidence || "无证据文本"}</p>
      {evidenceSpan && evidenceSpan !== evidence && <p className="evidence-span">原文：{evidenceSpan}</p>}
      <div className="evidence-meta">
        {source && <span>{source}</span>}
        {page && <span>p.{page}</span>}
        {section && <span>{section}</span>}
        {confidence && <span>置信度 {confidence}</span>}
        {semanticScore && semanticScore !== "0" && <span>semantic {semanticScore}</span>}
        {semanticRefId && <span>{semanticRefId}</span>}
        {knowledgeId && !row.source_deleted && <a href={`/?knowledge_document=${encodeURIComponent(knowledgeId)}${knowledgeVersion ? `&knowledge_version=${knowledgeVersion}` : ""}`} target="_blank" rel="noreferrer">查看资料原文</a>}
        {score && <span>score {score}</span>}
      </div>
      {reviewerNote && <div className="review-note">审校备注：{reviewerNote}</div>}
      {fieldText(row, "kind") === "claim" && claimId && (
        <ClaimReviewPanel row={row} claimId={claimId} onSubmit={props.onClaimReview} />
      )}
    </article>
  );
}

function fieldText(row: Record<string, unknown>, key: string) {
  const value = row[key];
  if (value === null || value === undefined) return "";
  return String(value);
}

function JsonBlock(props: { value: unknown; title?: string }) {
  return (
    <div className="json-block">
      {props.title && <h4>{props.title}</h4>}
      <pre>{JSON.stringify(props.value ?? {}, null, 2)}</pre>
    </div>
  );
}

function SubgraphMini(props: { edges: GraphEdge[] }) {
  return <GraphCanvas edges={props.edges} variant="compact" empty="当前答案没有可展示的子图" />;
}

function GeometricBackdrop() {
  return (
    <div className="geometric-backdrop" aria-hidden="true">
      <span className="arc arc-left" />
      <span className="arc arc-right" />
      <span className="paper-block top-block" />
      <span className="paper-block bottom-block" />
      <span className="line line-one" />
      <span className="line line-two" />
      <span className="square-dot" />
    </div>
  );
}

export default App;
