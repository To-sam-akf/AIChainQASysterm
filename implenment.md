# AI 算力产业链智能问答系统整体流程框架

本文档梳理当前项目的整体问答流程。系统不是让大模型直接回答产业链问题，而是先把年报、研报、白皮书、技术规范和论文构建成可检索的知识资产，再由问答引擎基于图谱、投研 Claim/Dossier 和 RAG 原文证据生成答案。

## 1. 总体架构

```text
数据源
  ├─ 上市公司年报
  ├─ AI 算力产业链研报
  ├─ 行业白皮书/政策/标准
  └─ 技术路线图、开放规范、benchmark、论文、模型技术报告
        ↓
PDF 下载与清单管理
        ↓
PDF 解析、文本清洗、chunk 切分
        ↓
LLM 抽取实体、关系、指标、风险和技术机理
        ↓
verified 图谱 CSV
        ↓
curated 专业图谱 + Claim + EvidenceSpan + Segment Dossier
        ↓
Neo4j/CSV 图谱检索 + 本地 BM25 RAG + 投研 Claim/Dossier 检索
        ↓
问题改写、问题规划、证据包构造、答案生成
        ↓
FastAPI/Streamlit/React 前端展示答案、证据和子图
```

核心原则：

- LLM 参与知识抽取、问题理解和答案组织，但最终答案必须受知识库证据约束。
- 所有关键判断都应能追溯到 Claim、Dossier、图谱关系或 RAG 原文片段。
- 面向投研问题时，答案优先组织为“核心判断、技术机理、产业传导、公司排序、领先指标、反证/边界、证据”。
- 对“哪些公司/谁受益”类问题，必须按 `core/direct/indirect/mentioned` 敞口分层，避免把直接受益公司和间接受益环节混为一谈。

## 2. 离线知识构建流程

### 2.1 数据准备

入口脚本：`scripts/prepare_stage1_data.py`

输入配置：

- `data/metadata/companies_extended.csv`：核心上市公司、别名、产业链环节。
- `data/metadata/research_keywords.csv`：研报检索关键词。
- `data/metadata/industry_sources.csv`：行业白皮书、技术规范、benchmark、论文等资料源。

输出产物：

- `data/raw_pdfs/annual/`：年报 PDF。
- `data/raw_pdfs/research/`：研报 PDF。
- `data/raw_pdfs/industry/`：行业和技术资料 PDF。
- `data/metadata/reports_manifest.csv`：报告清单、来源、状态、页数、SHA256、source_tier、source_type。

典型命令：

```bash
python scripts/prepare_stage1_data.py --kind all --max-research 10
```

### 2.2 PDF 解析与 chunk 切分

入口脚本：`scripts/parse_pdfs.py`

处理步骤：

1. 读取 `reports_manifest.csv` 中已下载或可用的 PDF。
2. 使用 PDF 解析器逐页提取文本。
3. 清洗页眉页脚、免责声明、空白文本等噪声。
4. 按报告、页码、章节切分为适合 LLM 抽取和 RAG 检索的 chunk。

输出产物：

- `data/parsed_text/*.jsonl`：逐页解析文本。
- `data/chunks/*.jsonl`：结构化文本块，保留 `report_id`、`kind`、`company`、`source_title`、`page`、`section` 等元数据。

典型命令：

```bash
python scripts/parse_pdfs.py --manifest data/metadata/reports_manifest.csv
```

### 2.3 LLM 知识抽取

入口脚本：`scripts/extract_knowledge.py`

核心模块：`src/llm_extractor.py`

抽取对象：

- 实体：`Company`、`Technology`、`Product`、`IndustryChain`、`Metric`、`Risk`、`IndustryConcept`、`Policy`、`Standard`、`ValueChainSegment`、`Workload`、`Architecture`、`Bottleneck`、`LeadingIndicator` 等。
- 关系：`USES_TECHNOLOGY`、`HAS_PRODUCT`、`BELONGS_TO_CHAIN`、`HAS_METRIC`、`DISCLOSES_RISK`、`UPSTREAM_OF`、`DOWNSTREAM_OF`、`ENABLES`、`CONSTRAINS`、`DEFINES`、`SUPPORTED_BY_POLICY`、`DRIVES`、`DEPENDS_ON`、`RELIEVES`、`HAS_EXPOSURE`、`HAS_INDICATOR`、`BENEFITS_FROM` 等。

抽取约束：

- 只抽取文本中明确出现的信息。
- 每条关系必须带原文 `evidence`。
- 年报优先抽公司业务、产品、技术、财务指标和风险。
- 研报优先抽产业链映射、公司所处环节、竞争格局和技术路线。
- 专业技术源默认不抽公司，除非原文明确出现核心 A 股上市公司；重点抽技术机理、瓶颈、指标和标准。

输出产物：

- `data/extracted/extractions.jsonl`：LLM 原始结构化抽取结果。
- `data/extracted/extraction_errors.csv`：抽取错误记录。

典型命令：

```bash
python scripts/extract_knowledge.py --kind research --contains 算力 --limit-chunks 20 --sleep 0.3
python scripts/extract_knowledge.py --resume --sleep 0.3
```

### 2.4 verified 图谱构建

入口脚本：`scripts/build_verified_graph.py`

核心模块：`src/graph_builder.py`

处理步骤：

1. 读取 LLM 抽取 JSONL。
2. 标准化实体名称和别名。
3. 合并重复实体与关系。
4. 自动补充报告来源关系。
5. 生成可人工校验的实体表和关系表。

输出产物：

- `data/verified/entities.csv`
- `data/verified/relations.csv`

典型命令：

```bash
python scripts/build_verified_graph.py
```

### 2.5 curated 投研图谱与研究层构建

入口脚本：`scripts/build_curated_graph.py`

核心模块：

- `src/curated_graph.py`
- `src/research_claims.py`

处理步骤：

1. 从 verified 图谱中过滤非核心公司噪声、低价值关系和目录/免责声明噪声。
2. 生成面向问答的 curated 图谱。
3. 从 curated 关系派生投研 Claim。
4. 从专业技术源 chunk 中直抽技术机理、瓶颈、指标、风险等原文级 Claim。
5. 为 Claim 生成 EvidenceSpan。
6. 按产业主题生成 Segment Dossier。

输出产物：

- `data/curated/entities.csv`
- `data/curated/relations.csv`
- `data/curated/claims.csv`
- `data/curated/evidence_spans.csv`
- `data/curated/segment_dossiers.jsonl`

典型命令：

```bash
python scripts/build_curated_graph.py
```

### 2.6 RAG 索引构建

入口脚本：`scripts/build_rag_index.py`

核心模块：`src/rag_index.py`

处理步骤：

1. 读取 `data/chunks/*.jsonl`。
2. 对中英文混合文本进行分词和领域词增强。
3. 构建本地 BM25/倒排索引。
4. 保留 chunk 的报告来源、页码、章节、公司和 source_type。

输出产物：

- `data/rag/documents.jsonl`
- `data/rag/metadata.json`

典型命令：

```bash
python scripts/build_rag_index.py
```

### 2.7 Neo4j 导入

入口脚本：`scripts/load_neo4j.py`

核心模块：`src/kg_loader.py`

处理方式：

- 默认优先导入 curated 图谱。
- 如果 Neo4j 不可用，问答引擎会回退到本地 CSV 图谱检索。

典型命令：

```bash
docker compose up -d neo4j
python scripts/load_neo4j.py --clear
```

只校验 CSV：

```bash
python scripts/load_neo4j.py --dry-run
```

## 3. 在线问答运行流程

在线问答主入口是 `src/qa_engine.py` 中的 `QAEngine.answer_question()` 和 `QAEngine.answer_question_stream()`。

调用来源：

- Streamlit 页面：`app.py`
- FastAPI 后端：`src/api.py`
- React 前端：`web/src/App.tsx`、`web/src/api.ts`

### 3.1 用户问题进入系统

用户通过前端输入问题，例如：

```text
液冷产业链有哪些上市公司，各自处于什么环节？
中际旭创和新易盛在光模块业务上的差异是什么？
继续说它们的主要风险
AI 算力产业链当前最大的瓶颈是什么？
```

FastAPI 路径：

- `POST /api/conversations/{conversation_id}/messages`
- `POST /api/conversations/{conversation_id}/messages/stream`

系统会先读取当前对话历史，用于处理追问和代词指代。

### 3.2 历史对话压缩与追问改写

模块：`src/qa_engine.py`

相关方法：

- `normalize_conversation_history()`
- `_contextualize_question()`
- `heuristic_contextual_question()`

作用：

- 截取最近若干轮历史，避免上下文过长。
- 判断问题是否需要结合历史，例如“继续说它们的风险”。
- 使用启发式或 LLM 将追问改写成可独立检索的问题。

示例：

```text
原问题：继续说它们的主要风险
历史主题：中际旭创和新易盛的光模块业务比较
改写后：中际旭创和新易盛在光模块业务上的主要风险是什么？
```

### 3.3 问题规划

模块：`src/question_planner.py`

核心方法：

- `heuristic_plan_question()`
- `plan_question()`

规划输出：`QuestionPlan`

关键字段：

- `answer_type`：答案类型。
- `companies`：问题涉及的核心公司。
- `topics`：主题，例如液冷、光模块、AI 服务器、国产算力。
- `expanded_topics`：同义词和扩展词。
- `relations`：需要检索的图谱关系。
- `needs_comparison`、`needs_risk`、`needs_metrics`、`needs_chain`：问题意图标记。

当前支持的答案类型：

- `topic_to_company`：主题找公司/受益公司。
- `company_compare`：公司对比。
- `risk_analysis`：风险分析。
- `industry_bottleneck`：产业瓶颈。
- `company_profile`：公司画像。
- `thematic_research`：主题研究。

### 3.4 Cypher 生成与图谱检索

模块：

- `src/cypher_generator.py`
- `src/professional_qa.py`
- `src/neo4j_client.py`
- `src/frontend_data.py`

处理逻辑：

1. 如果当前图谱后端是 CSV，生成展示用 pseudo Cypher。
2. 如果启用 Neo4j 且允许 LLM Cypher，则由 LLM 生成只读 Cypher，并通过安全规则校验。
3. 默认情况下使用模板 Cypher 或启发式 Cypher。
4. 优先查询 Neo4j；如果 Neo4j 查询失败或无结果，则回退 CSV 图谱。
5. CSV 路径使用 `search_csv_graph()`，根据 `QuestionPlan` 选择关系、公司、主题和风险记录。

图谱检索返回的核心字段：

- `company`
- `relation`
- `target`
- `evidence`
- `source`
- `page`
- `section`
- `source_tier`
- `chain_segment`

### 3.5 本地 RAG 检索

模块：`src/rag_index.py`

触发方法：`QAEngine._search_rag()`

处理逻辑：

1. 使用问题文本和 `expanded_topics` 拼接检索 query。
2. 对公司画像和风险问题添加公司过滤条件。
3. 从本地 BM25/倒排索引召回研报、年报和技术资料原文 chunk。
4. 返回带来源、页码和片段的 `RagHit`。

RAG 主要用于补充原文证据，尤其是图谱关系不够细、需要页码和上下文时。

### 3.6 投研 Claim/Dossier 检索

模块：`src/research_claims.py`

触发方法：`QAEngine._search_research()`

检索对象：

- `claims.csv`：原子投研判断。
- `evidence_spans.csv`：Claim 对应原文证据。
- `segment_dossiers.jsonl`：主题级产业链摘要。

Claim/Dossier 优先服务于专业投研问题：

- 技术为什么重要。
- 需求如何传导到产业链。
- 哪些公司是核心/直接/间接敞口。
- 哪些指标可用于验证。
- 有哪些风险和反证边界。

### 3.7 证据卡构造与排序

模块：`src/professional_qa.py`

核心对象：`EvidenceCard`

证据来源：

- `cards_from_research_hits()`：Claim/Dossier 证据。
- `cards_from_graph_records()`：图谱结构化关系证据。
- `cards_from_rag_hits()`：本地 RAG 原文片段证据。

排序与筛选：

- `rank_evidence_cards()`
- `select_cards_by_answer_type()`
- `assign_citation_ids()`

处理原则：

- 去重相似证据。
- 按答案类型分配证据预算，避免同类证据挤占全部上下文。
- 对公司比较问题保证每家公司都有证据。
- 对风险问题强制保留 `DISCLOSES_RISK` 或 risk Claim。
- 对主题研究和产业瓶颈问题优先保留 Dossier、bottleneck、mechanism、indicator、risk。
- 给最终证据卡分配 `E1/E2/...` 编号，答案引用必须对齐这些编号。

### 3.8 答案生成

模块：`src/qa_engine.py`、`src/professional_qa.py`

核心方法：

- `_generate_answer()`
- `_generate_answer_stream()`
- `build_professional_answer_prompt()`
- `fallback_professional_answer()`

生成策略：

1. 如果没有任何证据，返回“当前知识库中未找到相关证据”。
2. 如果 LLM 可用，构造 evidence pack，要求 LLM 只基于证据回答。
3. 如果 LLM 不可用或调用失败，使用确定性 fallback 模板生成答案。
4. 答案中的关键判断必须标注证据编号，例如 `[E1]`、`[E2]`。
5. 对缺少证据的栏目明确写“当前证据不足”，不补写证据外信息。

标准答案结构：

```text
核心判断
技术机理
产业传导
公司排序
领先指标
反证/边界
证据
```

### 3.9 返回前端展示

`QAEngine` 返回统一结果对象，主要字段包括：

- `question`：用户原问题。
- `contextual_question`：结合历史改写后的检索问题。
- `answer`：最终答案。
- `reasoning_content`：模型思考内容，开启 thinking 时展示。
- `answer_type`：问题类型。
- `plan`：问题规划结果。
- `cypher`、`cypher_params`、`cypher_source`：图谱查询信息。
- `graph_records`：图谱检索结果。
- `rag_hits`：RAG 命中片段。
- `research_hits`：Claim/Dossier 命中结果。
- `evidence_cards`：最终证据卡。
- `subgraph`：答案相关子图。
- `diagnostics`：耗时、命中数量、后端状态、错误信息等诊断数据。

前端展示内容：

- 答案正文。
- 问题规划。
- Cypher 查询。
- 证据卡和来源页码。
- RAG 原文片段。
- 相关图谱子图。
- 模型 thinking 内容和流式进度。

## 4. 运行时组件关系

```text
用户/前端
   ↓
FastAPI: src/api.py
   ↓
ConversationStore: src/conversation_store.py
   ↓
QAEngine: src/qa_engine.py
   ├─ 追问改写：history + contextualizer
   ├─ 问题规划：src/question_planner.py
   ├─ Cypher 生成：src/cypher_generator.py
   ├─ 图谱检索：Neo4jReadClient 或 LocalKnowledgeGraph
   ├─ RAG 检索：src/rag_index.py
   ├─ Claim/Dossier 检索：src/research_claims.py
   ├─ 证据排序：src/professional_qa.py
   └─ 答案生成：LLM 或 fallback template
   ↓
统一结果对象
   ↓
React/Streamlit 展示答案、证据、子图和诊断信息
```

## 5. 典型问题的路由方式

### 5.1 “液冷产业链有哪些上市公司？”

规划结果：

- `answer_type = topic_to_company`
- `topics = ["液冷"]`
- `relations` 包含 `USES_TECHNOLOGY`、`HAS_PRODUCT`、`BELONGS_TO_CHAIN`、`HAS_EXPOSURE`

检索重点：

- Claim/Dossier：液冷主题公司敞口。
- 图谱：公司到液冷技术、产品、产业链环节的关系。
- RAG：年报/研报中关于液冷业务、产品、客户、风险的原文片段。

答案重点：

- 按 core/direct/indirect/mentioned 分层。
- 区分液冷主业公司、服务器/电源/IDC/PCB 等间接敞口公司。
- 给出跟踪指标和风险边界。

### 5.2 “中际旭创和新易盛在光模块业务上的差异是什么？”

规划结果：

- `answer_type = company_compare`
- `companies = ["中际旭创", "新易盛"]`
- `topics = ["光模块"]`

检索重点：

- 两家公司各自的产品、技术、指标、风险 Claim。
- 图谱中两家公司相关的 `HAS_PRODUCT`、`USES_TECHNOLOGY`、`HAS_METRIC`、`DISCLOSES_RISK`。
- RAG 中年报和研报原文片段。

答案重点：

- 做差异矩阵，而不是分别罗列。
- 对比产品代际、客户/市场、技术路径、财务/指标证据和风险差异。

### 5.3 “继续说它们的主要风险”

规划结果：

- 先结合历史改写为独立问题。
- `answer_type = risk_analysis`
- 保留上一轮公司和主题。

检索重点：

- 风险 Claim。
- `DISCLOSES_RISK` 图谱关系。
- 公司年报风险章节原文。

答案重点：

- 区分业务进展证据和风险证据。
- 不根据常识外推未入库风险。

### 5.4 “AI 算力产业链当前最大的瓶颈是什么？”

规划结果：

- `answer_type = industry_bottleneck`
- `relations` 包含 `CONSTRAINS`、`DEPENDS_ON`、`HAS_INDICATOR`

检索重点：

- 技术源 Claim：GPU/AI 加速器、互联、显存/带宽、功耗、散热、数据中心交付等瓶颈。
- Segment Dossier：主题级摘要。
- 图谱中的瓶颈、供给约束和指标关系。

答案重点：

- 不只给单一瓶颈，而是解释瓶颈如何在芯片、互联、功耗、液冷、数据中心之间传导。
- 给出可跟踪领先指标和证据边界。

## 6. 关键文件索引

离线构建：

- `scripts/prepare_stage1_data.py`：下载和登记数据源。
- `scripts/parse_pdfs.py`：PDF 解析和 chunk 切分。
- `scripts/extract_knowledge.py`：LLM 抽取实体关系。
- `scripts/build_verified_graph.py`：构建 verified 图谱。
- `scripts/build_curated_graph.py`：构建 curated 图谱、Claim、EvidenceSpan、Dossier。
- `scripts/build_rag_index.py`：构建本地 RAG 索引。
- `scripts/load_neo4j.py`：导入 Neo4j。

在线问答：

- `src/api.py`：FastAPI 后端和对话接口。
- `app.py`：Streamlit 问答与图谱展示。
- `src/qa_engine.py`：问答主编排。
- `src/question_planner.py`：问题规划。
- `src/cypher_generator.py`：Cypher 生成和安全约束。
- `src/professional_qa.py`：专业检索、证据卡、fallback 答案。
- `src/rag_index.py`：本地 BM25 RAG。
- `src/research_claims.py`：Claim/Dossier 研究层。
- `src/frontend_data.py`：本地图谱读取、子图和前端数据。
- `src/conversation_store.py`：对话存储。
- `src/llm_client.py`：OpenAI 兼容 LLM 客户端。

前端：

- `web/src/App.tsx`：React 工作台。
- `web/src/api.ts`：后端 API 调用。
- `web/src/types.ts`：前端类型定义。
- `web/src/styles.css`：页面样式。

## 7. 一次完整问答的执行时序

```text
1. 用户在前端输入问题
2. FastAPI 读取 conversation_id 对应历史消息
3. QAEngine 规范化历史上下文
4. 系统判断是否需要追问改写
5. 生成 QuestionPlan：识别公司、主题、答案类型和关系类型
6. 生成展示用或执行用 Cypher
7. 查询 Neo4j；失败或不可用时回退 CSV 图谱
8. 查询本地 RAG，召回原文 chunk
9. 查询 ResearchMemory，召回 Claim 和 Dossier
10. 将 graph_records、rag_hits、research_hits 合并为 EvidenceCard
11. 按答案类型去重、排序、分配证据预算
12. 为证据卡分配 E1/E2/... citation_id
13. 构造 evidence pack，调用 LLM 生成专业答案
14. LLM 失败时使用 fallback_professional_answer
15. 生成 legacy evidence、evidence_cards、subgraph、diagnostics
16. 返回前端展示答案、证据、Cypher、子图和诊断信息
```

## 8. 答案可信度控制

当前系统通过以下机制降低幻觉风险：

- 抽取阶段要求每条关系必须包含原文证据。
- 专业技术源默认不映射到公司，除非原文明确出现核心上市公司。
- Cypher 生成限制为只读查询，并禁止写入、删除、CALL、APOC、GDS 等危险操作。
- 问答阶段优先使用 Claim/Dossier，再用图谱关系和 RAG 原文片段补充。
- 答案 prompt 要求“只能根据证据包回答”。
- 关键判断必须标注 `E1/E2/...` 证据编号。
- 没有证据时明确拒答或写“当前证据不足”。
- 返回 `unsupported_terms`、`errors`、`diagnostics` 供前端和调试使用。

## 9. 当前框架定位

当前项目已经形成“专业投研 GraphRAG”雏形：

- 图谱层负责结构化事实和关系。
- Claim 层负责投研原子判断。
- Dossier 层负责主题级产业链摘要。
- RAG 层负责回到原文证据。
- LLM 负责问题理解、证据组织和自然语言表达。

因此，系统的问答流程可以概括为：

```text
问题理解 → 检索规划 → 图谱/Claim/RAG 多路召回 → 证据包排序 → 证据约束生成 → 前端可追溯展示
```
