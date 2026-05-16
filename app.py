from __future__ import annotations

import streamlit as st

from src.frontend_data import (
    LocalKnowledgeGraph,
    RELATION_LABELS,
    render_svg_graph,
    subgraph_edges,
)
from src.qa_engine import QAEngine


st.set_page_config(page_title="AI算力产业链知识图谱问答系统", layout="wide")


@st.cache_data(show_spinner=False)
def load_graph() -> LocalKnowledgeGraph:
    return LocalKnowledgeGraph.from_csvs()


@st.cache_resource(show_spinner=False)
def load_qa_engine() -> QAEngine:
    return QAEngine.from_env()


def relation_label_options() -> dict[str, str]:
    return {"全部关系": "", **{label: rel for rel, label in RELATION_LABELS.items()}}


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
        "天孚通信涉及哪些技术？",
        "新易盛披露了哪些风险？",
        "寒武纪有哪些财务指标？",
        "什么是智能算力？",
    ]
    st.write(" / ".join(f"`{item}`" for item in examples))


def page_qa(engine: QAEngine) -> None:
    status = engine.status
    status_cols = st.columns(3)
    status_cols[0].metric("Neo4j", "启用" if status.neo4j_enabled else "未启用")
    status_cols[1].metric("本地 RAG", "就绪" if status.rag_enabled else "未构建")
    status_cols[2].metric("LLM", "就绪" if status.llm_enabled else "未配置")
    if status.rag_error:
        st.caption(f"RAG：{status.rag_error}")
    if status.llm_error:
        st.caption(f"LLM：{status.llm_error}")

    with st.form("qa_form"):
        question = st.text_input("问题", value="哪些公司涉及AI服务器？", placeholder="输入公司、技术、产品、风险等问题")
        submitted = st.form_submit_button("提问")

    if submitted and question.strip():
        with st.spinner("正在检索 Neo4j 与本地文档..."):
            st.session_state["qa_result"] = engine.answer_question(question)

    result = st.session_state.get("qa_result")
    if not result:
        return

    st.subheader("回答")
    st.write(result["answer"])

    st.subheader("Cypher")
    st.code(result["cypher"], language="cypher")
    if result.get("cypher_params"):
        st.json(result["cypher_params"])

    st.subheader("证据链")
    evidence = result["evidence"]
    if evidence:
        st.dataframe(evidence, width="stretch", hide_index=True)
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
    graph = load_graph()
    qa_engine = load_qa_engine()
    st.title("AI算力产业链知识图谱问答系统")

    if not graph.entities or not graph.relations:
        st.error("未找到可用图谱数据。请先运行 scripts/build_verified_graph.py。")
        return

    tab_overview, tab_qa, tab_graph = st.tabs(["数据概览", "智能问答", "图谱展示"])
    with tab_overview:
        page_overview(graph)
    with tab_qa:
        page_qa(qa_engine)
    with tab_graph:
        page_graph(graph)


if __name__ == "__main__":
    main()
