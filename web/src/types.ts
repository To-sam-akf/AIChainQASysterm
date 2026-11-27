export type Role = "user" | "assistant";

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turn_count: number;
  preview: string;
  pinned: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turns: ConversationTurn[];
  pinned: boolean;
}

export interface ResearchRole {
  id: string;
  name: string;
  avatar: string;
  responsibility: string;
  capabilities: string[];
}

export interface RoleEvent {
  type: "role_event";
  event_id: string;
  seq: number;
  task_id: string;
  conversation_id: string;
  created_at: string;
  role_id: string;
  step_id: string;
  status: "pending" | "running" | "completed" | "failed" | "skipped" | "degraded";
  summary: string;
  reason?: string;
  references?: Record<string, unknown>[];
  document?: KnowledgeDocument;
  evidence_gaps?: Record<string, unknown>[];
  conflicts?: Record<string, unknown>[];
}

export interface KnowledgeDocument {
  document_id: string;
  title: string;
  status: string;
  semantic_status?: string;
  error?: string;
  active_version?: number;
  pending_version?: number;
  source_deleted?: boolean;
}

export interface ConversationTurn {
  task_id?: string;
  request_id?: string;
  lead_role_id?: string;
  status?: "pending" | "running" | "completed" | "failed";
  error?: string;
  events?: MessageStreamEvent[];
  created_at: string;
  question: string;
  answer: string;
  thinking_enabled: boolean;
  reasoning_effort: string;
  result: QAResult;
}

export interface QAResult {
  question: string;
  contextual_question: string;
  answer: string;
  reasoning_content: string;
  answer_type: string;
  plan: Record<string, unknown>;
  cypher: string;
  cypher_params: Record<string, unknown>;
  cypher_source: string;
  graph_records: Record<string, unknown>[];
  rag_hits: Record<string, unknown>[];
  evidence_cards: Record<string, unknown>[];
  evidence: Record<string, unknown>[];
  research_outputs?: ResearchOutputs;
  verification?: VerificationResult;
  subgraph: GraphEdge[];
  diagnostics: Record<string, unknown>;
  errors: string[];
}

export interface ResearchOutputs {
  report?: {
    title: string;
    markdown: string;
    sections: { title: string; content: string }[];
    report_type?: string;
    report_type_label?: string;
    coverage?: Record<string, unknown>;
    spec?: ReportSpec;
    html?: string;
  };
  company_compare_table?: {
    columns: string[];
    rows: Record<string, string>[];
  };
  risk_checklist?: Record<string, string>[];
  evidence_gaps?: Record<string, string>[];
  verification?: VerificationResult;
  task_outputs?: Record<string, unknown>;
  meta?: Record<string, unknown>;
}

export interface ReportSpec {
  report_type: string;
  topic: string;
  title: string;
  report_type_label?: string;
  usability?: string;
  coverage?: Record<string, unknown>;
  executive_summary?: Record<string, unknown>;
  charts?: ReportCharts;
  appendix?: Record<string, unknown>;
}

export interface ReportCharts {
  flow_map?: FlowMapChart;
  company_coverage_heatmap?: CompanyCoverageHeatmapChart;
  evidence_strength_bar?: EvidenceStrengthBarChart;
  source_freshness_timeline?: FreshnessTimelineChart;
}

export interface CompanyCoverageHeatmapChart {
  title?: string;
  columns?: string[];
  rows?: string[];
  cells?: Array<{
    company: string;
    segment: string;
    score: number;
    evidence_ids?: string[];
    reason?: string;
  }>;
  empty_message?: string;
}

export interface EvidenceStrengthBarChart {
  title?: string;
  counts?: Record<string, number>;
  empty_message?: string;
}

export interface FlowMapChart {
  title?: string;
  nodes?: Array<{ id: string; label: string }>;
  links?: Array<{ source: string; target: string; value: number; evidence_ids?: string[] }>;
  caption?: string;
  empty_message?: string;
}

export interface FreshnessTimelineChart {
  title?: string;
  items?: Array<{ year: string; count: number; freshness: string }>;
  empty_message?: string;
}

export interface ConflictClaim {
  citation_id?: string;
  claim_id?: string;
  title?: string;
  evidence?: string;
  source?: string;
  page?: string;
  as_of_date?: string;
  claim_type?: string;
  exposure_level?: string;
}

export interface ConflictGroup {
  conflict_group_id: string;
  conflict_type: string;
  topic?: string;
  company?: string;
  claim_a: ConflictClaim;
  claim_b: ConflictClaim;
  resolution: string;
  confidence?: string;
}

export interface VerificationResult {
  status: "pass" | "warn" | "fail" | string;
  checks: Record<string, unknown>;
  evidence_gaps?: Record<string, string>[];
  conflict_groups?: ConflictGroup[];
}

export interface ClaimReviewRequest {
  claim_text?: string;
  claim_type?: string;
  topic?: string;
  companies?: string[] | string;
  evidence_span?: string;
  exposure_level?: string;
  confidence?: string;
  as_of_date?: string;
  review_status?: string;
  reviewer_note?: string;
}

export interface ClaimReviewResponse {
  claim: Record<string, unknown>;
  review: Record<string, unknown>;
}

export interface FeedbackCreateRequest {
  conversation_id: string;
  turn_index: number;
  question: string;
  answer_hash: string;
  helpful?: boolean | null;
  evidence_supported?: boolean | null;
  missing_answer?: boolean | null;
  human_score?: number | null;
  note?: string;
  citation_ids?: string[];
}

export interface FeedbackResponse {
  feedback: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  target: string;
  label: string;
  source_type: string;
  target_type: string;
  source_kind?: string;
  citation_id?: string;
  claim_type?: string;
  exposure_level?: string;
}

export interface ApiStatus {
  graph_backend: string;
  neo4j_enabled: boolean;
  rag_enabled: boolean;
  research_enabled: boolean;
  embedding_enabled: boolean;
  llm_enabled: boolean;
  csv_graph_enabled: boolean;
  graph_data_dir: string;
  errors: {
    graph: string;
    rag: string;
    research: string;
    embedding: string;
    llm: string;
  };
  stats: GraphStats;
  settings: {
    thinking_enabled: boolean;
    reasoning_effort: string;
    reasoning_efforts: string[];
    agent_enabled: boolean;
    agent_max_steps: number;
    hyde_enabled?: boolean;
    hyde_query_mode?: string;
    hyde_max_chars?: number;
  };
}

export interface GraphStats {
  companies: number;
  reports: number;
  entities: number;
  relations: number;
  entity_counts: Record<string, number>;
  relation_counts: Record<string, number>;
  research?: {
    claims?: number;
    dossiers?: number;
    reviewed_claims?: number;
    rejected_claims?: number;
    direct_exposure_companies?: number;
    claim_type_counts?: Record<string, number>;
  };
}

export interface GraphSummary extends GraphStats {
  companies_options: string[];
  technologies_options: string[];
  relation_options: Record<string, string>;
}

export interface GraphSubgraph {
  rows: Record<string, unknown>[];
  edges: GraphEdge[];
  svg: string;
}

export type MessageStreamEvent =
  | RoleEvent
  | { type: "task"; status: string; task_id: string; lead_role_id: string; seq: number }
  | { type: "progress"; stage: string; message: string; seq?: number }
  | { type: "answer_delta"; content: string; seq?: number }
  | { type: "final"; conversation: Conversation; turn: ConversationTurn }
  | { type: "error"; message: string; seq?: number };
