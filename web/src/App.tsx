import {
  BarChart3,
  Bot,
  Boxes,
  ChevronRight,
  ClipboardList,
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
  Trash2,
  X,
  ZoomIn,
  ZoomOut
} from "lucide-react";
import { FormEvent, KeyboardEvent, PointerEvent, WheelEvent, useEffect, useId, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  createAgentTask,
  createConversation,
  deleteConversation,
  exportAgentTaskUrl,
  getAgentTask,
  getConversation,
  getExamples,
  getGraphSubgraph,
  getGraphSummary,
  getStatus,
  listAgentTasks,
  listConversations,
  reviewClaim,
  streamMessage,
  updateConversationTitle
} from "./api";
import type {
  AgentTask,
  AgentTaskSummary,
  AgentTaskType,
  ApiStatus,
  Conversation,
  ConversationSummary,
  ConversationTurn,
  GraphEdge,
  GraphSubgraph,
  GraphSummary,
  ClaimReviewRequest,
  ResearchOutputs
} from "./types";

type View = "chat" | "overview" | "graph" | "agents";
type DetailTab = "research" | "evidence" | "cypher" | "diagnostics" | "graph";

const EMPTY_ARRAY: ConversationSummary[] = [];
const AGENT_TASK_TYPES: Array<{ value: AgentTaskType; label: string; description: string }> = [
  { value: "research_brief", label: "投研简报", description: "主题研究、公司排序、指标、风险与缺口" },
  { value: "company_compare", label: "公司对比", description: "业务卡位、差异点、指标和风险差异" },
  { value: "company_profile", label: "公司画像", description: "公司业务、产品技术、产业链位置和风险" },
  { value: "risk_review", label: "风险审查", description: "主要风险、反证、不确定性和跟踪指标" },
  { value: "evidence_gap_audit", label: "证据缺口", description: "缺失公司、指标、风险和建议数据源" }
];
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

function App() {
  const [view, setView] = useState<View>("chat");
  const [status, setStatus] = useState<ApiStatus | null>(null);
  const [examples, setExamples] = useState<string[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>(EMPTY_ARRAY);
  const [agentTasks, setAgentTasks] = useState<AgentTaskSummary[]>([]);
  const [selectedAgentTask, setSelectedAgentTask] = useState<AgentTask | null>(null);
  const [agentGoal, setAgentGoal] = useState("");
  const [agentTaskType, setAgentTaskType] = useState<AgentTaskType>("research_brief");
  const [agentRunning, setAgentRunning] = useState(false);
  const [current, setCurrent] = useState<Conversation | null>(null);
  const [input, setInput] = useState("");
  const [thinkingEnabled, setThinkingEnabled] = useState(false);
  const [reasoningEffort, setReasoningEffort] = useState("low");
  const [sending, setSending] = useState(false);
  const [streamingTurn, setStreamingTurn] = useState<ConversationTurn | null>(null);
  const [streamingProgress, setStreamingProgress] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [toast, setToast] = useState("");
  const [selectedTurn, setSelectedTurn] = useState<ConversationTurn | null>(null);
  const [detailTab, setDetailTab] = useState<DetailTab>("research");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [editingId, setEditingId] = useState("");
  const [editingTitle, setEditingTitle] = useState("");

  async function refreshConversations() {
    const items = await listConversations();
    setConversations(items);
  }

  async function refreshAgentTasks() {
    const items = await listAgentTasks();
    setAgentTasks(items);
  }

  async function loadInitial() {
    setLoading(true);
    try {
      const [statusPayload, examplePayload, conversationPayload, agentTaskPayload] = await Promise.all([
        getStatus(),
        getExamples(),
        listConversations(),
        listAgentTasks()
      ]);
      setStatus(statusPayload);
      setExamples(examplePayload);
      setConversations(conversationPayload);
      setAgentTasks(agentTaskPayload);
      setThinkingEnabled(statusPayload.settings.thinking_enabled);
      setReasoningEffort(statusPayload.settings.reasoning_effort);
      if (conversationPayload.length > 0) {
        const latest = await getConversation(conversationPayload[0].id);
        setCurrent(latest);
      }
      if (agentTaskPayload.length > 0) {
        const latestTask = await getAgentTask(agentTaskPayload[0].task_id);
        setSelectedAgentTask(latestTask);
      }
    } catch (error) {
      setToast(error instanceof Error ? error.message : "加载系统状态失败");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadInitial();
  }, []);

  async function handleNewConversation() {
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

  async function handleLoadAgentTask(id: string) {
    try {
      const task = await getAgentTask(id);
      setSelectedAgentTask(task);
      setView("agents");
      setSidebarOpen(false);
    } catch (error) {
      setToast(error instanceof Error ? error.message : "读取 Agent 任务失败");
    }
  }

  async function handleCreateAgentTask(goal = agentGoal) {
    const trimmed = goal.trim();
    if (!trimmed || agentRunning) return;
    setAgentRunning(true);
    setToast("");
    try {
      const task = await createAgentTask({
        task_type: agentTaskType,
        goal: trimmed,
        thinking_enabled: thinkingEnabled,
        reasoning_effort: thinkingEnabled ? reasoningEffort : ""
      });
      setSelectedAgentTask(task);
      setAgentGoal("");
      setView("agents");
      await refreshAgentTasks();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "创建 Agent 任务失败");
    } finally {
      setAgentRunning(false);
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
    if (!trimmed || sending) return;
    setSending(true);
    setToast("");
    const submittedThinking = thinkingEnabled;
    const submittedReasoningEffort = submittedThinking ? reasoningEffort : "";
    try {
      let conversation = current;
      if (!conversation) {
        conversation = await createConversation();
        setCurrent(conversation);
      }
      const pendingTurn = createPendingTurn(trimmed, submittedThinking, submittedReasoningEffort);
      setStreamingTurn(pendingTurn);
      setStreamingProgress([]);
      setSelectedTurn(null);
      setInput("");
      setView("chat");
      const updated = await streamMessage(
        conversation.id,
        trimmed,
        submittedThinking,
        submittedReasoningEffort,
        (event) => {
          if (event.type === "progress") {
            setStreamingProgress((items) => nextProgressItems(items, event.message));
          } else if (event.type === "answer_delta") {
            setStreamingTurn((turn) => {
              if (!turn) return turn;
              const answer = `${turn.answer}${event.content}`;
              return {
                ...turn,
                answer,
                result: {
                  ...turn.result,
                  answer
                }
              };
            });
          } else if (event.type === "final") {
            setCurrent(event.conversation);
            setStreamingTurn(null);
            setStreamingProgress([]);
          } else if (event.type === "error") {
            setToast(event.message);
          }
        }
      );
      setCurrent(updated);
      setStreamingTurn(null);
      setStreamingProgress([]);
      await refreshConversations();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "发送失败");
      setStreamingTurn(null);
      setStreamingProgress([]);
    } finally {
      setSending(false);
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

  const shellClassName = `app-shell ${sidebarOpen ? "sidebar-open" : ""}`;

  return (
    <div className={shellClassName}>
      <button className="mobile-menu" type="button" onClick={() => setSidebarOpen(true)} aria-label="打开侧栏">
        <Menu size={20} />
      </button>
      <Sidebar
        conversations={conversations}
        currentId={current?.id ?? ""}
        view={view}
        editingId={editingId}
        editingTitle={editingTitle}
        onClose={() => setSidebarOpen(false)}
        onNewConversation={handleNewConversation}
        onLoadConversation={handleLoadConversation}
        onDeleteConversation={handleDeleteConversation}
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
      <main className="workspace">
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
            sending={sending}
            streamingTurn={streamingTurn}
            streamingProgress={streamingProgress}
            thinkingEnabled={thinkingEnabled}
            reasoningEffort={reasoningEffort}
            reasoningEfforts={status?.settings.reasoning_efforts ?? ["low", "medium", "high"]}
            onInput={setInput}
            onSubmit={handleSubmit}
            onThinkingChange={setThinkingEnabled}
            onReasoningChange={setReasoningEffort}
            onOpenDetails={(turn) => {
              setSelectedTurn(turn);
              setDetailTab("research");
            }}
            onExample={(example) => setInput(example)}
          />
        )}
        {view === "agents" && (
          <AgentTasksView
            tasks={agentTasks}
            selectedTask={selectedAgentTask}
            goal={agentGoal}
            taskType={agentTaskType}
            running={agentRunning}
            thinkingEnabled={thinkingEnabled}
            reasoningEffort={reasoningEffort}
            reasoningEfforts={status?.settings.reasoning_efforts ?? ["low", "medium", "high"]}
            onGoal={setAgentGoal}
            onTaskType={setAgentTaskType}
            onCreate={handleCreateAgentTask}
            onSelectTask={handleLoadAgentTask}
            onThinkingChange={setThinkingEnabled}
            onReasoningChange={setReasoningEffort}
          />
        )}
        {view === "overview" && <OverviewView status={status} />}
        {view === "graph" && <GraphView />}
      </main>
      <DetailDrawer
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
  conversations: ConversationSummary[];
  currentId: string;
  view: View;
  editingId: string;
  editingTitle: string;
  onClose: () => void;
  onNewConversation: () => void;
  onLoadConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onStartRename: (item: ConversationSummary) => void;
  onRename: (id: string) => void;
  onEditTitle: (title: string) => void;
  onViewChange: (view: View) => void;
}) {
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

      <button className="new-chat" type="button" onClick={props.onNewConversation}>
        <Plus size={19} />
        <span>新对话</span>
      </button>

      <nav className="nav-list" aria-label="主导航">
        <button className={props.view === "chat" ? "active" : ""} type="button" onClick={() => props.onViewChange("chat")}>
          <MessageSquareText size={20} />
          <span>智能问答</span>
        </button>
        <button
          className={props.view === "agents" ? "active" : ""}
          type="button"
          onClick={() => props.onViewChange("agents")}
        >
          <Bot size={20} />
          <span>Agent 任务</span>
        </button>
        <button
          className={props.view === "overview" ? "active" : ""}
          type="button"
          onClick={() => props.onViewChange("overview")}
        >
          <BarChart3 size={20} />
          <span>数据概览</span>
        </button>
        <button className={props.view === "graph" ? "active" : ""} type="button" onClick={() => props.onViewChange("graph")}>
          <Boxes size={20} />
          <span>产业链图谱</span>
        </button>
      </nav>

      <section className="history-panel">
        <div className="section-title">
          <History size={16} />
          <span>问答历史</span>
        </div>
        <div className="history-list">
          {props.conversations.length === 0 && <div className="empty-history">暂无历史对话</div>}
          {props.conversations.map((item) => (
            <article className={`history-item ${props.currentId === item.id ? "selected" : ""}`} key={item.id}>
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
                <button className="history-open" type="button" onClick={() => props.onLoadConversation(item.id)}>
                  <strong>{item.title}</strong>
                  <span>{item.preview}</span>
                </button>
              )}
              <div className="history-actions">
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
    </aside>
  );
}

function ChatView(props: {
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
  onOpenDetails: (turn: ConversationTurn) => void;
  onExample: (value: string) => void;
}) {
  const hasTurns = Boolean(props.conversation?.turns.length || props.streamingTurn);
  return (
    <section className={`chat-page ${hasTurns ? "with-turns" : "empty"}`}>
      {!hasTurns ? (
        <div className="hero-chat">
          <div className="hero-kicker">AI 算力链产业智能问答系统</div>
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
              <MessagePair key={`${turn.created_at}-${index}`} index={index} turn={turn} onOpenDetails={props.onOpenDetails} />
            ))}
            {props.streamingTurn && (
              <MessagePair
                index={props.conversation?.turns.length ?? 0}
                turn={props.streamingTurn}
                isStreaming
                progressItems={props.streamingProgress}
                onOpenDetails={props.onOpenDetails}
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

function AgentTasksView(props: {
  tasks: AgentTaskSummary[];
  selectedTask: AgentTask | null;
  goal: string;
  taskType: AgentTaskType;
  running: boolean;
  thinkingEnabled: boolean;
  reasoningEffort: string;
  reasoningEfforts: string[];
  onGoal: (value: string) => void;
  onTaskType: (value: AgentTaskType) => void;
  onCreate: (goal?: string) => void;
  onSelectTask: (id: string) => void;
  onThinkingChange: (value: boolean) => void;
  onReasoningChange: (value: string) => void;
}) {
  const reasoningEfforts = props.reasoningEfforts.length ? props.reasoningEfforts : ["low", "medium", "high"];
  const activeReasoningEffort = reasoningEfforts.includes(props.reasoningEffort)
    ? props.reasoningEffort
    : reasoningEfforts[0];
  const task = props.selectedTask;

  function submit(event: FormEvent) {
    event.preventDefault();
    props.onCreate();
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
    props.onReasoningChange(reasoningEfforts[(currentIndex + 1) % reasoningEfforts.length]);
  }

  return (
    <section className="page-panel agent-page">
      <div className="page-heading">
        <span className="eyebrow">Agent Workspace</span>
        <h2>投研任务工作台</h2>
      </div>
      <div className="agent-layout">
        <aside className="agent-list-panel">
          <form className="agent-create" onSubmit={submit}>
            <label htmlFor="agent-task-type">任务类型</label>
            <select
              id="agent-task-type"
              value={props.taskType}
              disabled={props.running}
              onChange={(event) => props.onTaskType(event.target.value as AgentTaskType)}
            >
              {AGENT_TASK_TYPES.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
            <p className="agent-task-type-hint">
              {AGENT_TASK_TYPES.find((item) => item.value === props.taskType)?.description}
            </p>
            <label htmlFor="agent-goal">研究目标</label>
            <textarea
              id="agent-goal"
              value={props.goal}
              rows={5}
              placeholder="例如：液冷产业链投研简报"
              onChange={(event) => props.onGoal(event.target.value)}
            />
            <div className="agent-controls">
              <button
                className={props.thinkingEnabled ? "pill active" : "pill"}
                type="button"
                disabled={props.running}
                onClick={toggleThinking}
              >
                思考模式：{props.thinkingEnabled ? "开" : "关"}
              </button>
              <button
                className={props.thinkingEnabled ? "pill active" : "pill"}
                type="button"
                disabled={props.running}
                onClick={cycleReasoningEffort}
              >
                强度：{props.thinkingEnabled ? activeReasoningEffort : "关闭"}
              </button>
            </div>
            <button className="agent-run-button" type="submit" disabled={!props.goal.trim() || props.running}>
              {props.running ? <Loader2 size={17} className="spin" /> : <Sparkles size={17} />}
              <span>创建{agentTaskTypeLabel(props.taskType)}任务</span>
            </button>
          </form>

          <div className="agent-task-list">
            <div className="section-title">
              <ClipboardList size={16} />
              <span>任务历史</span>
            </div>
            {props.tasks.length === 0 && <div className="empty-table">暂无 Agent 任务</div>}
            {props.tasks.map((item) => (
              <button
                className={`agent-task-item ${task?.task_id === item.task_id ? "selected" : ""}`}
                key={item.task_id}
                type="button"
                onClick={() => props.onSelectTask(item.task_id)}
              >
                <strong>{item.title}</strong>
                <span>{item.preview || item.goal}</span>
                <em>{agentStatusLabel(item.status)} · {item.evidence_card_count} 证据 · {item.evidence_gap_count} 缺口</em>
              </button>
            ))}
          </div>
        </aside>

        <div className="agent-detail-panel">
          {!task ? (
            <div className="agent-empty">
              <Bot size={34} />
              <span>选择一个历史任务，或创建新的投研简报任务</span>
            </div>
          ) : (
            <>
              <div className="agent-detail-header">
                <div>
                  <span className={`task-status ${task.status}`}>{agentStatusLabel(task.status)}</span>
                  <span className="task-type-label">{task.final_outputs.task_label || agentTaskTypeLabel(task.task_type as AgentTaskType)}</span>
                  <h3>{task.final_outputs.report_title || task.title}</h3>
                  <p>{task.goal}</p>
                </div>
                <div className="export-group">
                  <a className="export-link" href={exportAgentTaskUrl(task.task_id, "md")}>
                    <Download size={16} />
                    <span>MD</span>
                  </a>
                  <a className="export-link" href={exportAgentTaskUrl(task.task_id, "json")}>
                    <Download size={16} />
                    <span>JSON</span>
                  </a>
                </div>
              </div>

              <div className="agent-stat-grid">
                <StatCard label="证据卡" value={String(task.final_outputs.evidence_card_count ?? task.evidence_cards.length)} />
                <StatCard label="证据缺口" value={String(task.final_outputs.evidence_gap_count ?? 0)} />
                <StatCard label="执行步骤" value={String(task.steps.length)} />
                <StatCard label="工具调用" value={String(task.tool_calls.length)} />
              </div>

              <ResearchOutputsView outputs={task.research_outputs} />

              <section className="agent-section">
                <div className="research-section-title">
                  <ClipboardList size={16} />
                  <span>执行轨迹</span>
                </div>
                <div className="agent-step-list">
                  {task.steps.map((step, index) => (
                    <article className="agent-step" key={`${step.phase ?? "step"}-${index}`}>
                      <span>{String(step.step ?? index + 1)}</span>
                      <div>
                        <strong>{String(step.phase ?? "step")}</strong>
                        <p>{String(step.observation ?? step.thought ?? "")}</p>
                      </div>
                      <ChevronRight size={16} />
                    </article>
                  ))}
                </div>
              </section>

              <section className="agent-section">
                <div className="research-section-title">
                  <FileText size={16} />
                  <span>QA 答案摘要</span>
                </div>
                <div className="answer-text">
                  {renderAnswerMarkdown(task.final_outputs.qa_answer || "当前任务没有生成问答摘要")}
                </div>
              </section>
            </>
          )}
        </div>
      </div>
    </section>
  );
}

function agentStatusLabel(status: AgentTask["status"]) {
  const labels = {
    pending: "等待中",
    running: "执行中",
    completed: "已完成",
    failed: "失败"
  };
  return labels[status] ?? status;
}

function agentTaskTypeLabel(taskType: AgentTaskType | string) {
  return AGENT_TASK_TYPES.find((item) => item.value === taskType)?.label ?? String(taskType || "Agent");
}

function StatCard(props: { label: string; value: string }) {
  return (
    <article className="stat-card">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </article>
  );
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
        value={props.value}
        rows={3}
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
            disabled={props.sending}
            aria-pressed={props.thinkingEnabled}
            title={props.thinkingEnabled ? "点击切换思考强度" : "点击开启思考模式"}
            onClick={cycleReasoningEffort}
          >
            强度：{props.thinkingEnabled ? activeReasoningEffort : "关闭"}
          </button>
        </div>
        <button className="send-button" type="button" disabled={!props.value.trim() || props.sending} onClick={() => props.onSubmit()}>
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
  index: number;
  turn: ConversationTurn;
  isStreaming?: boolean;
  progressItems?: string[];
  onOpenDetails: (turn: ConversationTurn) => void;
}) {
  return (
    <article className="turn">
      <div className="message user-message">
        <div className="avatar">U</div>
        <div className="bubble">
          <p>{props.turn.question}</p>
        </div>
      </div>
      <div className="message assistant-message">
        <div className="avatar">
          <Bot size={18} />
        </div>
        <div className="bubble answer">
          <div className="answer-meta">
            <span>第 {props.index + 1} 轮</span>
            <span>{props.turn.thinking_enabled ? `思考 ${props.turn.reasoning_effort}` : "快速回答"}</span>
            {props.isStreaming && <span>生成中</span>}
          </div>
          {props.turn.thinking_enabled && Boolean(props.progressItems?.length) && (
            <ThoughtProgress items={props.progressItems ?? []} />
          )}
          <div className="answer-text">
            {props.turn.answer ? (
              <>
                {renderAnswerMarkdown(props.turn.answer)}
                {props.isStreaming && <span className="stream-caret" />}
              </>
            ) : (
              <div className="stream-placeholder">
                <Loader2 size={16} className="spin" />
                <span>正在生成答案</span>
              </div>
            )}
          </div>
          {!props.isStreaming && (
            <button className="details-button" type="button" onClick={() => props.onOpenDetails(props.turn)}>
              <FileText size={15} />
              <span>查看证据与诊断</span>
              <ChevronRight size={15} />
            </button>
          )}
        </div>
      </div>
    </article>
  );
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
  open: boolean;
  tab: DetailTab;
  onTab: (tab: DetailTab) => void;
  onClose: () => void;
  onClaimReview: (claimId: string, payload: ClaimReviewRequest) => Promise<void>;
}) {
  if (!props.turn || !props.open) return null;
  const result = props.turn.result;
  const evidence = result.evidence_cards?.length ? result.evidence_cards : result.evidence;

  return (
    <aside className="detail-drawer">
      <div className="drawer-header">
        <div>
          <span className="eyebrow">证据详情</span>
          <h3>{result.answer_type || "问答诊断"}</h3>
        </div>
        <button type="button" onClick={props.onClose} aria-label="关闭详情">
          <X size={18} />
        </button>
      </div>
      {result.contextual_question && result.contextual_question !== result.question && (
        <div className="context-note">上下文改写：{result.contextual_question}</div>
      )}
      <div className="drawer-tabs">
        {[
          ["research", "投研"],
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
        {props.tab === "research" && <ResearchOutputsView outputs={result.research_outputs} />}
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
    </aside>
  );
}

function renderAnswerMarkdown(markdown: string): ReactNode[] {
  return markdown.split(/\r?\n/).map((line, index) => {
    const text = line.trim();
    if (!text) return <div className="answer-space" key={index} />;
    if (text.startsWith("### ")) {
      return <h3 key={index}>{renderInlineMarkdown(text.slice(4))}</h3>;
    }
    if (text.startsWith("## ")) {
      return <h3 key={index}>{renderInlineMarkdown(text.slice(3))}</h3>;
    }
    const bulletMatch = text.match(/^[-*]\s+(.+)$/);
    if (bulletMatch) {
      return (
        <p className="answer-bullet" key={index}>
          <span>•</span>
          <span>{renderInlineMarkdown(bulletMatch[1])}</span>
        </p>
      );
    }
    const orderedMatch = text.match(/^(\d+)\.\s+(.+)$/);
    if (orderedMatch) {
      return (
        <p className="answer-bullet" key={index}>
          <span>{orderedMatch[1]}.</span>
          <span>{renderInlineMarkdown(orderedMatch[2])}</span>
        </p>
      );
    }
    return <p key={index}>{renderInlineMarkdown(text)}</p>;
  });
}

function renderInlineMarkdown(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean).map((part, index) => {
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
        <span className="eyebrow">Knowledge Base</span>
        <h1>数据概览</h1>
        <p>图谱、RAG、Embedding 和 LLM 状态集中展示，便于演示前快速确认系统健康度。</p>
      </div>
      <div className="metric-grid">
        <MetricCard label="公司" value={stats?.companies ?? 0} />
        <MetricCard label="报告" value={stats?.reports ?? 0} />
        <MetricCard label="实体" value={stats?.entities ?? 0} />
        <MetricCard label="关系" value={stats?.relations ?? 0} />
        <MetricCard label="Claim" value={researchStats?.claims ?? 0} />
        <MetricCard label="Dossier" value={researchStats?.dossiers ?? 0} />
        <MetricCard label="已审校 Claim" value={researchStats?.reviewed_claims ?? 0} />
        <MetricCard label="直接敞口公司" value={researchStats?.direct_exposure_companies ?? 0} />
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

function GraphView() {
  const [summary, setSummary] = useState<GraphSummary | null>(null);
  const [subgraph, setSubgraph] = useState<GraphSubgraph | null>(null);
  const [company, setCompany] = useState("");
  const [technology, setTechnology] = useState("");
  const [relation, setRelation] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadSummary() {
      try {
        const payload = await getGraphSummary();
        setSummary(payload);
      } catch (err) {
        setError(err instanceof Error ? err.message : "加载图谱摘要失败");
      }
    }
    void loadSummary();
  }, []);

  useEffect(() => {
    async function loadSubgraph() {
      setLoading(true);
      setError("");
      try {
        const payload = await getGraphSubgraph({ company, technology, relation_type: relation });
        setSubgraph(payload);
      } catch (err) {
        setError(err instanceof Error ? err.message : "加载子图失败");
      } finally {
        setLoading(false);
      }
    }
    void loadSubgraph();
  }, [company, technology, relation]);

  const relationEntries = Object.entries(summary?.relation_options ?? {});

  return (
    <section className="page-panel graph-page">
      <div className="page-header">
        <span className="eyebrow">Industry Graph</span>
        <h1>产业链图谱</h1>
        <p>按公司、技术或关系筛选局部图谱，用于解释答案背后的结构化证据。</p>
      </div>
      <div className="filter-row">
        <label>
          公司
          <select value={company} onChange={(event) => setCompany(event.target.value)}>
            <option value="">全部公司</option>
            {summary?.companies_options.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label>
          技术
          <select value={technology} onChange={(event) => setTechnology(event.target.value)}>
            <option value="">全部技术</option>
            {summary?.technologies_options.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label>
          关系
          <select value={relation} onChange={(event) => setRelation(event.target.value)}>
            {relationEntries.length === 0 && <option value="">全部关系</option>}
            {relationEntries.map(([label, value]) => (
              <option key={label} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
      </div>
      {error && <div className="warning">{error}</div>}
      <div className="graph-surface">
        {loading ? (
          <div className="center-loading">
            <Loader2 size={18} className="spin" />
            <span>加载图谱中</span>
          </div>
        ) : (
          <GraphCanvas edges={subgraph?.edges ?? []} empty="当前筛选条件下没有可展示的子图" />
        )}
      </div>
      <DataTable rows={subgraph?.rows ?? []} empty="当前筛选条件下没有关系" />
    </section>
  );
}

function MetricCard(props: { label: string; value: number }) {
  return (
    <div className="metric-card">
      <span>{props.label}</span>
      <strong>{props.value.toLocaleString()}</strong>
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
  const table = outputs.company_compare_table;
  const risks = outputs.risk_checklist ?? [];
  const gaps = outputs.evidence_gaps ?? [];
  return (
    <div className="research-output">
      <TaskOutputsView outputs={outputs.task_outputs} />
      <section className="research-section">
        <div className="research-section-title">
          <FileText size={16} />
          <span>{report?.title || "投研简报"}</span>
        </div>
        <div className="research-report">
          {report?.sections?.length ? (
            report.sections.map((section) => (
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
  const reviewStatus = fieldText(row, "review_status");
  const reviewerNote = fieldText(row, "reviewer_note");
  return (
    <article className="evidence-card">
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

function DataTable(props: { rows: Record<string, unknown>[]; empty: string }) {
  if (!props.rows.length) return <div className="empty-table">{props.empty}</div>;
  const columns = Object.keys(props.rows[0]).slice(0, 7);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {props.rows.slice(0, 24).map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>{String(row[column] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
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
