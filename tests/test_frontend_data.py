from src.frontend_data import LocalKnowledgeGraph, infer_question, render_svg_graph, subgraph_edges


def sample_graph() -> LocalKnowledgeGraph:
    return LocalKnowledgeGraph(
        entities=[
            {"type": "Company", "name": "浪潮信息", "normalized_name": "浪潮信息"},
            {"type": "Company", "name": "天孚通信", "normalized_name": "天孚通信"},
            {"type": "Technology", "name": "AI服务器", "normalized_name": "ai服务器"},
            {"type": "Report", "name": "AI服务器与光模块产业链专题", "normalized_name": "research_1"},
        ],
        relations=[
            {
                "head_type": "Company",
                "head_name": "浪潮信息",
                "relation": "USES_TECHNOLOGY",
                "tail_type": "Technology",
                "tail_name": "AI服务器",
                "evidence": "浪潮信息位于AI服务器产业链。",
                "source_title": "AI服务器与光模块产业链专题",
                "page": "11",
            },
            {
                "head_type": "Company",
                "head_name": "天孚通信",
                "relation": "USES_TECHNOLOGY",
                "tail_type": "Technology",
                "tail_name": "高速光器件",
                "evidence": "AI技术推动公司高速光器件市场持续增长。",
                "source_title": "AI服务器与光模块产业链专题",
                "page": "41",
            },
        ],
    )


def test_infer_question_technology_to_company() -> None:
    result = infer_question(sample_graph(), "哪些公司涉及AI服务器？")

    assert "MATCH" in result["cypher"]
    assert "浪潮信息" in result["answer"]
    assert result["evidence"][0]["source"] == "AI服务器与光模块产业链专题"


def test_infer_question_topic_to_company_from_evidence() -> None:
    graph = LocalKnowledgeGraph(
        entities=[
            {"type": "Company", "name": "浪潮信息", "normalized_name": "浪潮信息"},
            {"type": "Product", "name": "服务器", "normalized_name": "服务器"},
        ],
        relations=[
            {
                "head_type": "Company",
                "head_name": "浪潮信息",
                "relation": "HAS_PRODUCT",
                "tail_type": "Product",
                "tail_name": "服务器",
                "evidence": "公司布局AI服务器整机产品。",
                "source_title": "研报",
                "page": "1",
            }
        ],
    )

    result = infer_question(graph, "哪些公司涉及AI服务器？")

    assert "浪潮信息" in result["answer"]
    assert result["records"][0]["relation"] == "HAS_PRODUCT"


def test_graph_filter_by_company() -> None:
    graph = sample_graph()

    rows = graph.subgraph_relations(company="天孚通信")

    assert len(rows) == 1
    assert rows[0]["head_name"] == "天孚通信"


def test_render_svg_graph_outputs_svg() -> None:
    edges = subgraph_edges(sample_graph().relations)

    svg = render_svg_graph(edges)

    assert "<svg" in svg
    assert "浪潮信息" in svg
