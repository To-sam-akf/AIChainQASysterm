from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.curated_graph import DEFAULT_CURATED_DIR
from src.frontend_data import (
    LocalKnowledgeGraph,
    RELATION_LABELS,
    render_svg_graph,
    subgraph_edges,
)
from src.llm_client import load_dotenv
from src.qa_engine import QAEngine


load_dotenv()
st.set_page_config(page_title="AI算力产业链知识图谱问答系统", layout="wide")
CONVERSATION_DIR = Path("data/conversations")
REASONING_EFFORTS = ["low", "medium", "high"]


@st.cache_data(show_spinner=False)
def load_graph() -> LocalKnowledgeGraph:
    data_dir = Path(os.getenv("KG_DATA_DIR", str(DEFAULT_CURATED_DIR)))
    if not (data_dir / "entities.csv").exists():
        return LocalKnowledgeGraph.from_csvs()
    return LocalKnowledgeGraph.from_dir(data_dir)


@st.cache_resource(show_spinner=False)
def load_qa_engine() -> QAEngine:
    return QAEngine.from_env()


def relation_label_options() -> dict[str, str]:
    return {"全部关系": "", **{label: rel for rel, label in RELATION_LABELS.items()}}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off", "disabled"}


def ensure_conversation_state() -> None:
    st.session_state.setdefault("qa_turns", [])
    st.session_state.setdefault("saved_conversation_path", "")
    st.session_state.setdefault(
        "llm_thinking_enabled",
        env_bool("LLM_THINKING_ENABLED", "deepseek" in os.getenv("LLM_BASE_URL", "").casefold()),
    )
    effort = os.getenv("LLM_REASONING_EFFORT", "high").strip() or "high"
    st.session_state.setdefault("llm_reasoning_effort", effort if effort in REASONING_EFFORTS else "high")
    st.session_state.setdefault("qa_question_input", "")


def conversation_messages_from_turns(turns: list[dict]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        question = str(turn.get("question") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if question:
            messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})
    return messages


def short_text(value: str, limit: int = 42) -> str:
    value = " ".join(str(value or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def conversation_markdown(turns: list[dict]) -> str:
    lines = [
        "# AI算力产业链知识图谱问答记录",
        "",
        f"- 保存时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 对话轮次：{len(turns)}",
        "",
    ]
    for index, turn in enumerate(turns, start=1):
        result = turn.get("result") or {}
        contextual_question = result.get("contextual_question", turn.get("question", ""))
        lines.extend(
            [
                f"## 第 {index} 轮",
                "",
                f"**用户问题**：{turn.get('question', '')}",
                "",
            ]
        )
        if contextual_question and contextual_question != turn.get("question"):
            lines.extend([f"**上下文改写**：{contextual_question}", ""])
        lines.extend([f"**助手回答**：{turn.get('answer', '')}", ""])
    return "\n".join(lines)


def conversation_json(turns: list[dict]) -> str:
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "turns": turns,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def save_conversation(turns: list[dict]) -> Path:
    CONVERSATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    markdown_path = CONVERSATION_DIR / f"qa_conversation_{timestamp}.md"
    json_path = CONVERSATION_DIR / f"qa_conversation_{timestamp}.json"
    markdown_path.write_text(conversation_markdown(turns), encoding="utf-8")
    json_path.write_text(conversation_json(turns), encoding="utf-8")
    return markdown_path


def list_saved_conversations(limit: int = 12) -> list[Path]:
    if not CONVERSATION_DIR.exists():
        return []
    return sorted(CONVERSATION_DIR.glob("qa_conversation_*.md"), reverse=True)[:limit]


def current_reasoning_effort() -> str | None:
    if not st.session_state.get("llm_thinking_enabled", False):
        return None
    return str(st.session_state.get("llm_reasoning_effort") or "high")


def render_conversation_sidebar() -> None:
    ensure_conversation_state()
    turns = st.session_state["qa_turns"]
    with st.sidebar:
        st.header("模型设置")
        st.toggle(
            "启用思考模式",
            key="llm_thinking_enabled",
            help="开启后请求 DeepSeek thinking，并读取 reasoning_content 用于页面展示。",
        )
        st.selectbox(
            "思考强度",
            options=REASONING_EFFORTS,
            key="llm_reasoning_effort",
            disabled=not st.session_state.get("llm_thinking_enabled", False),
        )

        st.header("对话记录")
        st.caption(f"当前对话 {len(turns)} 轮")
        if turns:
            for index, turn in enumerate(turns, start=1):
                st.markdown(f"**{index}.** {short_text(turn.get('question', ''))}")
        else:
            st.info("暂无对话。")

        if st.button("新建对话", key="sidebar_new_conversation", disabled=not turns, use_container_width=True):
            st.session_state["qa_turns"] = []
            st.session_state.pop("qa_result", None)
            st.session_state["saved_conversation_path"] = ""
            st.rerun()

        if st.button("保存当前对话", key="sidebar_save_conversation", disabled=not turns, use_container_width=True):
            try:
                saved_path = save_conversation(turns)
                st.session_state["saved_conversation_path"] = str(saved_path)
                st.success(f"已保存：{saved_path.name}")
            except Exception as exc:
                st.error(f"保存失败：{exc}")

        if st.session_state.get("saved_conversation_path"):
            st.caption(st.session_state["saved_conversation_path"])

        st.download_button(
            "下载 Markdown",
            data=conversation_markdown(turns) if turns else "",
            file_name=f"qa_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            disabled=not turns,
            use_container_width=True,
        )

        saved_paths = list_saved_conversations()
        with st.expander("已保存记录", expanded=bool(saved_paths)):
            if saved_paths:
                for path in saved_paths:
                    st.caption(str(path))
            else:
                st.caption("暂无本地保存记录。")
        st.download_button(
            "下载 JSON",
            data=conversation_json(turns) if turns else "",
            file_name=f"qa_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            disabled=not turns,
            use_container_width=True,
        )


def page_overview(graph: LocalKnowledgeGraph) -> None:
    entity_counts = graph.entity_counts()
    relation_counts = graph.relation_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("公司", entity_counts.get("Company", 0))
    col2.metric("报告", graph.reports_count())
    col3.metric("实体", len(graph.entities))
    col4.metric("关系", len(graph.relations))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("实体分布")
        st.bar_chart(dict(entity_counts))
    with right:
        st.subheader("关系分布")
        st.bar_chart({RELATION_LABELS.get(k, k): v for k, v in relation_counts.items()})

    st.subheader("可演示问题")
    examples = [
        "哪些公司涉及AI服务器？",
        "液冷产业链有哪些上市公司，各自处于什么环节？",
        "中际旭创和新易盛在光模块业务上的差异是什么？",
        "英维克液冷业务进展和主要风险是什么？",
        "AI算力产业链当前最大的瓶颈是什么？",
    ]
    st.write(" / ".join(f"`{item}`" for item in examples))


def render_qa_details(result: dict) -> None:
    if result.get("contextual_question") and result["contextual_question"] != result.get("question"):
        st.caption(f"结合历史对话改写后的检索问题：{result['contextual_question']}")

    if result.get("reasoning_content"):
        with st.expander("模型思考过程", expanded=False):
            st.write(result["reasoning_content"])

    cols = st.columns(3)
    cols[0].metric("答案类型", result.get("answer_type", ""))
    diagnostics = result.get("diagnostics", {})
    cols[1].metric("图谱证据", diagnostics.get("graph_records", 0))
    cols[2].metric("证据卡片", diagnostics.get("evidence_cards", 0))

    with st.expander("问题规划", expanded=False):
        st.json(result.get("plan", {}))

    st.subheader("Cypher")
    st.code(result["cypher"], language="cypher")
    if result.get("cypher_params"):
        st.json(result["cypher_params"])

    st.subheader("证据链")
    evidence_cards = result.get("evidence_cards") or result["evidence"]
    if evidence_cards:
        st.dataframe(evidence_cards, width="stretch", hide_index=True)
    else:
        st.info("当前知识图谱中未找到相关证据。")

    st.subheader("Neo4j 查询结果")
    if result["graph_records"]:
        st.dataframe(result["graph_records"], width="stretch", hide_index=True)
    else:
        st.info("graph_records 为空。")

    st.subheader("本地 RAG 命中")
    if result["rag_hits"]:
        st.dataframe(result["rag_hits"], width="stretch", hide_index=True)
    else:
        st.info("rag_hits 为空。")

    st.subheader("子图")
    st.html(render_svg_graph(result["subgraph"], height=480))

    if result["errors"]:
        st.subheader("运行状态")
        for error in result["errors"]:
            st.warning(error)
    with st.expander("诊断信息", expanded=False):
        st.json(result.get("diagnostics", {}))


def set_example_question(question: str) -> None:
    st.session_state["qa_question_input"] = question


def submit_question(engine: QAEngine, question: str) -> None:
    history = conversation_messages_from_turns(st.session_state["qa_turns"])
    thinking_enabled = bool(st.session_state.get("llm_thinking_enabled", False))
    reasoning_effort = current_reasoning_effort()
    result = engine.answer_question(
        question,
        conversation_history=history,
        thinking_enabled=thinking_enabled,
        reasoning_effort=reasoning_effort,
    )
    result["diagnostics"]["thinking_enabled"] = thinking_enabled
    result["diagnostics"]["reasoning_effort"] = reasoning_effort or ""
    st.session_state["qa_result"] = result
    st.session_state["qa_turns"].append(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "answer": result["answer"],
            "thinking_enabled": thinking_enabled,
            "reasoning_effort": reasoning_effort or "",
            "result": result,
        }
    )


def page_qa(engine: QAEngine) -> None:
    ensure_conversation_state()
    status = engine.status
    status_cols = st.columns(4)
    status_cols[0].metric("图谱后端", status.graph_backend.upper())
    status_cols[1].metric("Neo4j", "可用" if status.neo4j_enabled else "降级/未启用")
    status_cols[2].metric("本地 RAG", "就绪" if status.rag_enabled else "未构建")
    status_cols[3].metric("LLM", "就绪" if status.llm_enabled else "未配置")
    if status.graph_error:
        st.caption(f"图谱：{status.graph_error}")
    if status.rag_error:
        st.caption(f"RAG：{status.rag_error}")
    if status.llm_error:
        st.caption(f"LLM：{status.llm_error}")

    examples = [
        "液冷产业链有哪些上市公司，各自处于什么环节？",
        "中际旭创和新易盛在光模块业务上的差异是什么？",
        "继续说它们的主要风险",
        "AI算力产业链当前最大的瓶颈是什么？",
    ]
    example_cols = st.columns(len(examples))
    for index, example in enumerate(examples):
        example_cols[index].button(
            short_text(example, 18),
            key=f"qa_example_{index}",
            use_container_width=True,
            on_click=set_example_question,
            args=(example,),
        )

    with st.form("qa_form", clear_on_submit=False):
        question = st.text_area(
            "问题",
            key="qa_question_input",
            height=88,
            placeholder="输入公司、技术、产品、产业链、风险或对比类问题；也可以直接追问“继续说它们的主要风险”。",
        )
        submitted = st.form_submit_button("发送问题", use_container_width=True)

    if submitted:
        question = question.strip()
        if not question:
            st.warning("请输入问题。")
        else:
            st.session_state["saved_conversation_path"] = ""
            with st.spinner("正在结合历史对话、规划问题、检索图谱与本地文档..."):
                submit_question(engine, question)

    st.subheader("连续对话")
    if not st.session_state["qa_turns"]:
        st.info("当前还没有问答记录。输入问题后，本轮和后续追问都会保留在这里。")
        return

    for index, turn in enumerate(st.session_state["qa_turns"]):
        st.markdown(f"**用户 {index + 1}：** {turn['question']}")
        thinking_label = "开启" if turn.get("thinking_enabled") else "关闭"
        effort = turn.get("reasoning_effort") or "无"
        st.caption(f"思考模式：{thinking_label}；思考强度：{effort}")
        st.markdown("**助手：**")
        st.write(turn["answer"])
        with st.expander("证据、图谱与诊断", expanded=index == len(st.session_state["qa_turns"]) - 1):
            render_qa_details(turn["result"])
        st.divider()


def page_graph(graph: LocalKnowledgeGraph) -> None:
    companies = [""] + graph.names_by_type("Company")
    technologies = [""] + graph.names_by_type("Technology")
    label_to_relation = relation_label_options()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        company = st.selectbox("公司筛选", options=companies, format_func=lambda x: x or "全部公司")
    with col2:
        technology = st.selectbox("技术筛选", options=technologies, format_func=lambda x: x or "全部技术")
    with col3:
        relation_label = st.selectbox("关系类型", options=list(label_to_relation.keys()))

    rows = graph.subgraph_relations(
        company=company,
        technology=technology,
        relation_type=label_to_relation[relation_label],
        limit=80,
    )
    st.caption(f"当前子图关系数：{len(rows)}")
    st.html(render_svg_graph(subgraph_edges(rows)))

    st.subheader("子图关系明细")
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("当前筛选条件下没有关系。")


def main() -> None:
    ensure_conversation_state()
    graph = load_graph()
    qa_engine = load_qa_engine()
    st.title("AI算力产业链知识图谱问答系统")

    if not graph.entities or not graph.relations:
        st.error("未找到可用图谱数据。请先运行 scripts/build_verified_graph.py。")
        render_conversation_sidebar()
        return

    tab_overview, tab_qa, tab_graph = st.tabs(["数据概览", "智能问答", "图谱展示"])
    with tab_overview:
        page_overview(graph)
    with tab_qa:
        page_qa(qa_engine)
    with tab_graph:
        page_graph(graph)
    render_conversation_sidebar()


if __name__ == "__main__":
    main()
