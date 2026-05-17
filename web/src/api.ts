import type {
  ApiStatus,
  Conversation,
  ConversationSummary,
  GraphSubgraph,
  GraphSummary,
  MessageStreamEvent
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

export async function streamMessage(
  id: string,
  question: string,
  thinkingEnabled: boolean,
  reasoningEffort: string,
  onEvent: (event: MessageStreamEvent) => void
): Promise<Conversation> {
  const response = await fetch(`/api/conversations/${encodeURIComponent(id)}/messages/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      question,
      thinking_enabled: thinkingEnabled,
      reasoning_effort: reasoningEffort
    })
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
  if (!response.body) {
    throw new Error("当前浏览器不支持流式响应");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalConversation: Conversation | null = null;
  let streamError: Error | null = null;

  function dispatchBlock(block: string) {
    const lines = block.split("\n");
    let eventType = "message";
    const dataLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (dataLines.length === 0) return;
    const payload = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
    const event = { type: eventType, ...payload } as MessageStreamEvent;
    onEvent(event);
    if (event.type === "final") {
      finalConversation = event.conversation;
    } else if (event.type === "error") {
      streamError = new Error(event.message || "生成失败");
    }
  }

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    buffer = buffer.replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);
      if (block) dispatchBlock(block);
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }
  const tail = buffer.trim();
  if (tail) dispatchBlock(tail);
  if (streamError) throw streamError;
  if (!finalConversation) throw new Error("流式响应未返回最终结果");
  return finalConversation;
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
