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
  web_search_enabled?: boolean;
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
  web_search_hits?: Record<string, unknown>[];
  evidence_cards: Record<string, unknown>[];
  evidence: Record<string, unknown>[];
  subgraph: GraphEdge[];
  diagnostics: Record<string, unknown>;
  errors: string[];
}

export interface GraphEdge {
  source: string;
  target: string;
  label: string;
  source_type: string;
  target_type: string;
}

export interface ApiStatus {
  graph_backend: string;
  neo4j_enabled: boolean;
  rag_enabled: boolean;
  llm_enabled: boolean;
  csv_graph_enabled: boolean;
  graph_data_dir: string;
  errors: {
    graph: string;
    rag: string;
    llm: string;
  };
  stats: GraphStats;
  settings: {
    thinking_enabled: boolean;
    reasoning_effort: string;
    reasoning_efforts: string[];
    web_search_enabled: boolean;
  };
}

export interface GraphStats {
  companies: number;
  reports: number;
  entities: number;
  relations: number;
  entity_counts: Record<string, number>;
  relation_counts: Record<string, number>;
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
