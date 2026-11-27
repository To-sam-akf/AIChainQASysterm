import { useState } from "react";
import type { ConversationTurn, ResearchRole, RoleEvent } from "./types";

export const statusLabels: Record<string, string> = {
  pending: "等待", running: "执行中", completed: "已完成", failed: "失败", skipped: "跳过", degraded: "降级"
};

export function RoleWorkspace({ turn, roles, onEvidence, onRetry, onCollaboration, defaultExpanded = false }: {
  turn: ConversationTurn; roles: ResearchRole[]; onEvidence: (turn: ConversationTurn) => void;
  onRetry: (question: string) => void; onCollaboration?: () => void; defaultExpanded?: boolean;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const events = (turn.events ?? []).filter((event): event is RoleEvent => event.type === "role_event");
  if (!events.length && !turn.task_id) return null;
  const latest = new Map<string, RoleEvent>();
  for (const event of events) latest.set(event.step_id, event);
  const active = [...latest.values()].filter(e => e.status === "running" || e.status === "pending");
  const failed = [...latest.values()].filter(e => e.status === "failed");
  const retryDocument = [...events].reverse().find(event => event.document?.document_id)?.document;
  const rolesById = new Map(roles.map(role => [role.id, role]));
  return <section className="role-workspace" aria-label="角色协作">
    <button className="role-workspace-toggle" type="button" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
      <strong>角色协作</strong><span>{active.length ? `${active.length} 项执行中` : statusLabels[turn.status || "completed"]}
        {failed.length > 0 ? ` · ${failed.length} 项失败` : ""} · {expanded ? "收起" : "展开"}</span>
    </button>
    {onCollaboration && <button type="button" className="role-inspector-link" onClick={onCollaboration}>在研究详情中查看协作记录 →</button>}
    {!expanded && <div className="role-active" aria-live="polite">{active.map(event => <span key={event.step_id}>
      {rolesById.get(event.role_id)?.name || event.role_id}：{event.summary}
    </span>)}</div>}
    {expanded && <ol className="role-timeline">{events.map(event => {
      const role = rolesById.get(event.role_id);
      const doc = event.document;
      return <li key={event.event_id} className={`role-event role-${event.status}`}>
        <span className="role-avatar" aria-hidden="true">{role?.avatar || "研"}</span>
        <div className="role-event-body">
          <div className="role-event-heading"><strong>{role?.name || event.role_id}</strong>
            <span className="role-status">{statusLabels[event.status]}</span>
            <time dateTime={event.created_at}>{new Date(event.created_at).toLocaleTimeString()}</time></div>
          {role && <small>{role.responsibility}</small>}
          <p>{event.summary}</p>{event.reason && <p className="role-reason">{event.reason}</p>}
          {event.evidence_gaps && event.evidence_gaps.length > 0 && <p>待补充证据：{event.evidence_gaps.map(g => String(g.reason || g.coverage || "证据不足")).join("；")}</p>}
          {event.conflicts && event.conflicts.length > 0 && <p>发现 {event.conflicts.length} 项来源冲突，详见证据与诊断。</p>}
          {doc && <div className="role-document">
            <strong>{doc.title}</strong><span>{doc.source_deleted ? "来源已删除" : ({received: "已接收", processing: "处理中", ready: "可检索", failed: "失败"} as Record<string, string>)[doc.status] || doc.status}</span>
            <small>资料 ID：{doc.document_id}</small>
            <small>语义索引：{({ready: "已完成", not_configured: "未配置", failed: "失败", stale: "待更新"} as Record<string, string>)[doc.semantic_status || ""] || doc.semantic_status || "未就绪"}</small>
            {doc.error && <p role="alert">{doc.error}</p>}
            {!doc.source_deleted && <a href={`/?knowledge_document=${encodeURIComponent(doc.document_id)}`} target="_blank" rel="noreferrer">查看资料</a>}
            {doc.status === "failed" && !doc.source_deleted && <button type="button" onClick={() => onRetry(`重新处理知识库资料 ${doc.document_id}`)}>重新处理资料</button>}
          </div>}
          {event.references && event.references.length > 0 && <details className="role-references"><summary>关联证据（{event.references.length}）</summary>
            {event.references.map((ref, i) => {
              const id = String(ref.document_id || ref.report_id || String(ref.semantic_ref_id || "").split(":")[1] || "");
              return <div key={i}><strong>{String(ref.citation_id || "原文")} · {String(ref.title || ref.source || "来源")}</strong>
                <small>{String(ref.source || "")} {ref.page ? `第 ${String(ref.page)} 页` : ""}</small>
                <p>{ref.source_deleted ? "来源已删除" : String(ref.evidence || ref.snippet || "")}</p>
                {/^[0-9a-f]{32}$/.test(id) && !ref.source_deleted && <a href={`/?knowledge_document=${id}`} target="_blank" rel="noreferrer">查看资料原文</a>}
              </div>;
            })}
            <button type="button" onClick={() => onEvidence(turn)}>查看全部证据与诊断</button>
          </details>}
        </div>
      </li>;
    })}</ol>}
    {turn.status === "failed" && <div className="role-task-error" role="alert"><p>{turn.error || "任务中断，请重试"}</p>
      <button type="button" onClick={() => onRetry(retryDocument ? `重新处理知识库资料 ${retryDocument.document_id}` : turn.question)}>{retryDocument ? "重新处理资料" : "重新提交任务"}</button></div>}
  </section>;
}
