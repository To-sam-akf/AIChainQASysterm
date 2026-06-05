export type Role = "user" | "assistant";

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turn_count: number;
  preview: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turns: ConversationTurn[];
}

export interface ConversationTurn {
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

export interface EvalRunSummary {
  run_id: string;
  created_at: string;
  dataset_name: string;
  dataset_hash: string;
  cases: number;
  passed: number;
  failed: number;
  overall_score: number;
  metrics: Record<string, number | null>;
}

export interface EvalRun {
  run_id: string;
  created_at: string;
  dataset: {
    name: string;
    path: string;
    hash: string;
    version: string;
    cases: number;
  };
  environment: Record<string, unknown>;
  summary: {
    cases: number;
    passed: number;
    warned: number;
    failed: number;
    pass_rate: number;
    overall_score: number;
    metrics: Record<string, number | null>;
  };
  category_scores: EvalCategoryScore[];
  failed_examples: EvalFailedExample[];
  results: EvalCaseResult[];
}

export interface EvalCategoryScore {
  category: string;
  cases: number;
  overall_score: number;
  pass_rate: number;
  metrics: Record<string, number | null>;
}

export interface EvalFailedExample {
  case_id: string;
  category: string;
  question: string;
  score: number;
  failures: string[];
  evidence_gaps: Record<string, string>[];
  answer_preview: string;
}

export interface EvalCaseResult {
  case_id: string;
  category: string;
  question: string;
  expected_answer_type: string;
  answer_type: string;
  refusal_expected: boolean;
  metrics: Record<string, number | null>;
  score: number;
  status: "pass" | "warn" | "fail" | string;
  failures: string[];
  evidence_gaps: Record<string, string>[];
  answer: string;
  answer_preview: string;
  evidence_cards: Record<string, unknown>[];
  evidence_card_count: number;
  expected: Record<string, unknown>;
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

export interface AgentTaskSummary {
  task_id: string;
  task_type: string;
  title: string;
  goal: string;
  status: "pending" | "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  evidence_card_count: number;
  evidence_gap_count: number;
  preview: string;
}

export interface AgentTask {
  task_id: string;
  task_type: string;
  goal: string;
  title: string;
  status: "pending" | "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  plan: Record<string, unknown>;
  steps: Record<string, unknown>[];
  tool_calls: Record<string, unknown>[];
  evidence_cards: Record<string, unknown>[];
  research_outputs?: ResearchOutputs;
  diagnostics: Record<string, unknown>;
  errors: string[];
  final_outputs: {
    report_markdown?: string;
    report_title?: string;
    task_type?: AgentTaskType;
    task_label?: string;
    task_schema_type?: string;
    evidence_gap_count?: number;
    evidence_card_count?: number;
    verification_status?: string;
    conflict_group_count?: number;
    qa_answer?: string;
    contextual_question?: string;
    answer_type?: string;
  };
}

export type AgentTaskType =
  | "research_brief"
  | "company_compare"
  | "company_profile"
  | "risk_review"
  | "evidence_gap_audit";

export interface AgentTaskCreateRequest {
  task_type: AgentTaskType;
  goal: string;
  thinking_enabled?: boolean;
  reasoning_effort?: string;
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
  | { type: "progress"; stage: string; message: string }
  | { type: "answer_delta"; content: string }
  | { type: "final"; conversation: Conversation; turn: ConversationTurn }
  | { type: "error"; message: string };
