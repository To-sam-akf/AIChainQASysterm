import {
  BarChart3,
  Bot,
  Boxes,
  ChevronRight,
  Download,
  FileText,
  History,
  Loader2,
  Menu,
  MessageSquareText,
  Pencil,
  Plus,
  Send,
  Settings2,
  Sparkles,
  Trash2,
  X
} from "lucide-react";
import { KeyboardEvent, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  createConversation,
  deleteConversation,
  getConversation,
  getExamples,
  getGraphSubgraph,
  getGraphSummary,
  getStatus,
  listConversations,
  streamMessage,
  updateConversationTitle
} from "./api";
import type {
  ApiStatus,
  Conversation,
  ConversationSummary,
  ConversationTurn,
  GraphEdge,
  GraphSubgraph,
  GraphSummary
} from "./types";

type View = "chat" | "overview" | "graph";
type DetailTab = "evidence" | "cypher" | "diagnostics" | "graph";

const EMPTY_ARRAY: ConversationSummary[] = [];

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

function App() {
  const [view, setView] = useState<View>("chat");
  const [status, setStatus] = useState<ApiStatus | null>(null);
  const [examples, setExamples] = useState<string[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>(EMPTY_ARRAY);
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
  const [detailTab, setDetailTab] = useState<DetailTab>("evidence");
  const [sidebarOpen, setSidebarOpen] = useState(false);
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
        thinkingEnabled={thinkingEnabled}
        reasoningEffort={reasoningEffort}
        reasoningEfforts={status?.settings.reasoning_efforts ?? ["low", "medium", "high"]}
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
        onThinkingChange={setThinkingEnabled}
        onReasoningChange={setReasoningEffort}
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
            onInput={setInput}
            onSubmit={handleSubmit}
            onOpenDetails={(turn) => {
              setSelectedTurn(turn);
              setDetailTab("evidence");
            }}
            onExample={(example) => setInput(example)}
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
      />
    </div>
  );
}

function Sidebar(props: {
  conversations: ConversationSummary[];
  currentId: string;
  view: View;
  thinkingEnabled: boolean;
  reasoningEffort: string;
  reasoningEfforts: string[];
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
  onThinkingChange: (value: boolean) => void;
  onReasoningChange: (value: string) => void;
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

      <section className="settings-card" aria-label="模型设置">
        <div className="section-title">
          <Settings2 size={16} />
          <span>模型设置</span>
        </div>
        <label className="toggle-row">
          <span>思考模式</span>
          <input
            type="checkbox"
            checked={props.thinkingEnabled}
            onChange={(event) => props.onThinkingChange(event.target.checked)}
          />
        </label>
        <label className="select-row">
          <span>思考强度</span>
          <select
            value={props.reasoningEffort}
            disabled={!props.thinkingEnabled}
            onChange={(event) => props.onReasoningChange(event.target.value)}
          >
            {props.reasoningEfforts.map((effort) => (
              <option key={effort} value={effort}>
                {effort}
              </option>
            ))}
          </select>
        </label>
      </section>

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
  onInput: (value: string) => void;
  onSubmit: (question?: string) => void;
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
            placeholder="请输入你想了解的 AI 算力链产业问题..."
            onChange={props.onInput}
            onSubmit={props.onSubmit}
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
            placeholder="继续追问，系统会自动携带当前对话上下文..."
            onChange={props.onInput}
            onSubmit={props.onSubmit}
          />
        </div>
      )}
    </section>
  );
}

function Composer(props: {
  value: string;
  sending: boolean;
  thinkingEnabled: boolean;
  reasoningEffort: string;
  placeholder: string;
  compact?: boolean;
  onChange: (value: string) => void;
  onSubmit: (question?: string) => void;
}) {
  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      props.onSubmit();
    }
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
          <span className={props.thinkingEnabled ? "pill active" : "pill"}>思考模式</span>
          <span className="pill">强度：{props.thinkingEnabled ? props.reasoningEffort : "关闭"}</span>
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
        {props.tab === "evidence" && <DataTable rows={evidence} empty="当前没有证据卡片" />}
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
  return (
    <section className="page-panel">
      <div className="page-header">
        <span className="eyebrow">Knowledge Base</span>
        <h1>数据概览</h1>
        <p>图谱、RAG 和 LLM 状态集中展示，便于演示前快速确认系统健康度。</p>
      </div>
      <div className="metric-grid">
        <MetricCard label="公司" value={stats?.companies ?? 0} />
        <MetricCard label="报告" value={stats?.reports ?? 0} />
        <MetricCard label="实体" value={stats?.entities ?? 0} />
        <MetricCard label="关系" value={stats?.relations ?? 0} />
      </div>
      <div className="status-strip">
        <StatusPill label="图谱后端" value={props.status?.graph_backend?.toUpperCase() ?? "UNKNOWN"} />
        <StatusPill label="Neo4j" value={props.status?.neo4j_enabled ? "可用" : "降级/未启用"} />
        <StatusPill label="本地 RAG" value={props.status?.rag_enabled ? "就绪" : "未构建"} />
        <StatusPill label="LLM" value={props.status?.llm_enabled ? "就绪" : "未配置"} />
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
          <div dangerouslySetInnerHTML={{ __html: subgraph?.svg ?? "" }} />
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
  if (!props.edges.length) return <div className="empty-table">当前答案没有可展示的子图</div>;
  const nodes = Array.from(new Set(props.edges.flatMap((edge) => [edge.source, edge.target]).filter(Boolean)));
  const width = 560;
  const height = 360;
  const radius = 128;
  const cx = width / 2;
  const cy = height / 2;
  const positions = new Map<string, { x: number; y: number }>();
  nodes.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(nodes.length, 1);
    positions.set(node, { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius });
  });
  return (
    <svg className="mini-graph" viewBox={`0 0 ${width} ${height}`} role="img">
      <defs>
        <marker id="mini-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#8b8174" />
        </marker>
      </defs>
      {props.edges.slice(0, 40).map((edge, index) => {
        const source = positions.get(edge.source);
        const target = positions.get(edge.target);
        if (!source || !target) return null;
        return (
          <g key={`${edge.source}-${edge.target}-${index}`}>
            <line
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke="#8b8174"
              strokeWidth="1.2"
              markerEnd="url(#mini-arrow)"
            />
            <text x={(source.x + target.x) / 2} y={(source.y + target.y) / 2 - 4} textAnchor="middle">
              {edge.label}
            </text>
          </g>
        );
      })}
      {nodes.map((node) => {
        const position = positions.get(node)!;
        return (
          <g key={node}>
            <circle cx={position.x} cy={position.y} r="22" />
            <text x={position.x} y={position.y + 39} textAnchor="middle">
              {node.length > 12 ? `${node.slice(0, 12)}...` : node}
            </text>
          </g>
        );
      })}
    </svg>
  );
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
