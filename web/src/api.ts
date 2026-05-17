import type {
  ApiStatus,
  Conversation,
  ConversationSummary,
  GraphSubgraph,
  GraphSummary
} from "./types";

async function request<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {})
    },
    ...options
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      // Keep the HTTP status message when the response is not JSON.
    }
    throw new Error(message);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function getStatus(): Promise<ApiStatus> {
  return request<ApiStatus>("/api/status");
}

export async function getExamples(): Promise<string[]> {
  const payload = await request<{ examples: string[] }>("/api/examples");
  return payload.examples;
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const payload = await request<{ conversations: ConversationSummary[] }>("/api/conversations");
  return payload.conversations;
}

export function createConversation(title = ""): Promise<Conversation> {
  return request<Conversation>("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ title })
  });
}

export function getConversation(id: string): Promise<Conversation> {
  return request<Conversation>(`/api/conversations/${encodeURIComponent(id)}`);
}

export function updateConversationTitle(id: string, title: string): Promise<Conversation> {
  return request<Conversation>(`/api/conversations/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify({ title })
  });
}

export function deleteConversation(id: string): Promise<void> {
  return request<void>(`/api/conversations/${encodeURIComponent(id)}`, {
    method: "DELETE"
  });
}

export async function sendMessage(
  id: string,
  question: string,
  thinkingEnabled: boolean,
  reasoningEffort: string
): Promise<Conversation> {
  const payload = await request<{ conversation: Conversation }>(
    `/api/conversations/${encodeURIComponent(id)}/messages`,
    {
      method: "POST",
      body: JSON.stringify({
        question,
        thinking_enabled: thinkingEnabled,
        reasoning_effort: reasoningEffort
      })
    }
  );
  return payload.conversation;
}

export function getGraphSummary(): Promise<GraphSummary> {
  return request<GraphSummary>("/api/graph/summary");
}

export function getGraphSubgraph(params: {
  company?: string;
  technology?: string;
  relation_type?: string;
}): Promise<GraphSubgraph> {
  const query = new URLSearchParams();
  if (params.company) query.set("company", params.company);
  if (params.technology) query.set("technology", params.technology);
  if (params.relation_type) query.set("relation_type", params.relation_type);
  return request<GraphSubgraph>(`/api/graph/subgraph?${query.toString()}`);
}
