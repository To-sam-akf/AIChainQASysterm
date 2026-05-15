# AI 算力产业链知识图谱问答系统完成计划

## Summary
- 目标：完成一个可答辩展示的系统，流程为“PDF 报告 → LLM 抽取实体关系 → Neo4j 图谱 → LLM 生成 Cypher 查询 → 图谱检索 → LLM 基于证据回答 → Streamlit 展示”。
- 第一版规模：10 家 AI 算力产业链公司，约 15 份报告，30 个评测问题。
- 技术栈：Python + Neo4j + Streamlit + 可配置云端 LLM API。
- 当前仓库只有技术文档，需要从零搭建工程目录、数据目录、后端模块、前端页面和评测材料。

## Key Changes
- 建立项目结构：
  - `data/raw_pdfs/` 保存财报和研报 PDF。
  - `data/parsed_text/` 保存解析后的文本。
  - `data/extracted/` 保存 LLM 抽取的实体关系 JSON。
  - `data/verified/` 保存人工校验后的 CSV/JSON。
  - `src/` 放 PDF 解析、LLM 抽取、Neo4j 导入、问答引擎代码。
  - `app.py` 作为 Streamlit 前端入口。
- 定义图谱 Schema：
  - 节点：`Company`、`Technology`、`Product`、`IndustryChain`、`Metric`、`Risk`、`Report`。
  - 关系：`USES_TECHNOLOGY`、`HAS_PRODUCT`、`BELONGS_TO_CHAIN`、`HAS_METRIC`、`DISCLOSES_RISK`、`MENTIONED_IN`。
  - 每条关系保留 `evidence`、`source`、`page` 或 `section` 字段，保证回答可溯源。
- 建立端到端后端流程：
  - `pdf_parser.py`：用 PyMuPDF 或 pdfplumber 提取 PDF 文本。
  - `text_cleaner.py`：清洗页眉页脚、目录、乱码和空行，按章节/段落切块。
  - `llm_extractor.py`：调用 LLM 抽取实体和关系，输出严格 JSON。
  - `kg_loader.py`：把校验后的实体关系导入 Neo4j。
  - `cypher_generator.py`：根据用户问题和 Schema 生成只读 Cypher。
  - `neo4j_client.py`：执行 Neo4j 查询。
  - `answer_generator.py`：让 LLM 仅基于查询结果生成中文答案。
  - `qa_engine.py`：串联“问题解析 → Cypher → 查询 → 回答”。
- 建立 Cypher 安全策略：
  - 只允许 `MATCH`、`OPTIONAL MATCH`、`WITH`、`WHERE`、`RETURN`。
  - 禁止 `CREATE`、`MERGE`、`DELETE`、`SET`、`REMOVE`、`DROP`、`CALL dbms` 等修改或管理语句。
  - 查询结果为空时固定回答：“当前知识图谱中未找到相关证据。”
- 建立 Streamlit 页面：
  - 数据概览：公司数、报告数、实体数、关系数。
  - 图谱展示：按公司、技术、产业链环节筛选子图。
  - 智能问答：展示答案、证据、Cypher、查询结果和子图。
  - 评测结果：展示 30 个问题的对比评分和准确率。

## Implementation Steps
- 第 1 阶段：数据准备
  - 选定 10 家公司：浪潮信息、中科曙光、工业富联、中际旭创、新易盛、天孚通信、英维克、申菱环境、寒武纪、海光信息。
  - 收集每家公司最新可用年报，加 5 份 AI 算力产业链研报。
  - 统一命名，例如 `浪潮信息_2024年报.pdf`、`AI算力产业链研报_2024.pdf`。
- 第 2 阶段：知识抽取
  - PDF 转文本后，只保留业务概要、核心竞争力、管理层讨论、研发投入、风险因素、财务指标、产业链分析等相关章节。
  - 每段文本调用 LLM 抽取 JSON，要求“不编造、必须有 evidence、必须有 source”。
  - 把所有 JSON 汇总为实体表和关系表，再人工校验去重。
- 第 3 阶段：图谱构建
  - 在 Neo4j 中为 `Company.name`、`Technology.name`、`Product.name` 等字段建立唯一约束或索引。
  - 使用 `MERGE` 导入节点和关系，关系属性保留证据来源。
  - 准备 5 到 8 条固定 Cypher 示例，用于调试和答辩演示。
- 第 4 阶段：问答系统
  - LLM 根据用户问题生成 Cypher，但后端必须先做安全检查。
  - Neo4j 返回结构化 records。
  - LLM 只能根据 records 生成答案，答案中必须包含来源或证据。
  - 对投资相关问题加限制：不提供股票买卖建议，只回答图谱事实。
- 第 5 阶段：前端与展示
  - Streamlit 首页直接进入系统，不做营销页。
  - 问答页保留完整证据链：问题、答案、Cypher、证据、来源、子图。
  - 图谱页支持按公司或技术筛选，便于答辩现场演示。
- 第 6 阶段：评测与材料
  - 准备 30 个问题，覆盖单跳查询、多跳查询、比较问题、风险问题、空结果问题。
  - 对比三种方法：纯 LLM、关键词检索、LLM + 知识图谱。
  - 按 2/1/0 分评分，输出准确率和案例分析。
  - 最终整理实验报告和答辩 PPT，突出“LLM 不直接瞎答，而是基于图谱证据回答”。

## Public Interfaces
- `answer_question(question: str) -> dict`
  - 返回：`question`、`cypher`、`records`、`answer`、`evidence`、`subgraph`。
- `extract_from_text(text: str, source: str) -> dict`
  - 返回：`entities`、`relations`。
- `load_verified_graph(entity_file, relation_file) -> None`
  - 将人工校验后的实体和关系导入 Neo4j。
- `.env` 配置：
  - `LLM_PROVIDER`
  - `LLM_API_KEY`
  - `LLM_BASE_URL`
  - `LLM_MODEL`
  - `NEO4J_URI`
  - `NEO4J_USER`
  - `NEO4J_PASSWORD`

## Test Plan
- 单元测试：
  - PDF 解析能输出非空文本。
  - LLM 抽取结果能通过 JSON Schema 校验。
  - Cypher 安全检查能拦截写入、删除和管理语句。
  - Neo4j 查询为空时返回固定兜底回答。
- 集成测试：
  - 用 1 家公司和 1 份报告跑通“PDF → 抽取 → 导入 → 问答”。
  - 验证典型问题：“浪潮信息涉及哪些技术？”能返回答案、证据和来源。
  - 验证空问题：“哪些公司涉及不存在的技术？”不会编造。
- 评测测试：
  - 30 个问题分别跑纯 LLM、关键词检索、LLM + 图谱。
  - 记录每题得分、错误类型和最终准确率。
  - 至少准备 3 个可展示案例：正确回答、复杂关系回答、空结果拒答。

## Assumptions
- 按“答辩完整版”完成，而不是只做原型。
- LLM 使用云端 API，并通过环境变量配置，后续可切换通义、DeepSeek、OpenAI 或其他兼容接口。
- 第一版数据规模固定为 10 家公司和约 15 份报告，优先保证质量和可解释性。
- 项目重点是知识图谱问答与证据展示，不做股票推荐或投资决策系统。
