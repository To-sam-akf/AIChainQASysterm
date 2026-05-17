# AIChainQASysterm

AI 算力产业链知识图谱问答系统。

## 第一阶段：数据准备

初始化并下载最新可用年报、公开 AI 算力产业链研报和权威行业白皮书：

```bash
python scripts/prepare_stage1_data.py --kind all --max-research 10
```

只查看候选文件，不下载：

```bash
python scripts/prepare_stage1_data.py --kind annual --dry-run
python scripts/prepare_stage1_data.py --kind research --max-research 10 --dry-run
python scripts/prepare_stage1_data.py --kind industry --dry-run
```

输出目录：

- `data/raw_pdfs/annual/`：30 家核心上市公司的最新可用年报。
- `data/raw_pdfs/research/`：公开可直接访问的 AI 算力产业链研报。
- `data/raw_pdfs/industry/`：中国信通院等权威机构白皮书、政策和标准资料。
- `data/metadata/companies_extended.csv`：30 家核心上市公司、别名和产业链环节。
- `data/metadata/research_keywords.csv`：研报检索关键词配置。
- `data/metadata/industry_sources.csv`：权威行业知识源配置。
- `data/metadata/reports_manifest.csv`：PDF 来源、状态、SHA256、文件大小和页数。

## 第二、三阶段：知识抽取与图谱构建

配置本地环境变量文件：

```bash
cp .env.example .env
```

`.env` 默认使用 DeepSeek OpenAI 兼容接口，填入 `LLM_API_KEY` 后即可运行。当前示例配置启用 DeepSeek 思考模式：

- `LLM_MODEL=deepseek-v4-pro`
- `LLM_THINKING_ENABLED=true`
- `LLM_REASONING_EFFORT=high`
- `QA_WEB_SEARCH_ENABLED=true`

如果账号仍使用旧版推理模型，可把 `LLM_MODEL` 改为 `deepseek-reasoner`。

解析 PDF 并生成文本块：

```bash
python scripts/parse_pdfs.py --manifest data/metadata/reports_manifest.csv
```

调用 LLM 抽取实体关系。建议按报告类型分批跑：

```bash
python scripts/extract_knowledge.py --kind research --contains 算力 --limit-chunks 20 --sleep 0.3
python scripts/extract_knowledge.py --kind annual --contains 服务器 --limit-chunks 50 --resume --sleep 0.3
python scripts/extract_knowledge.py --kind industry --contains 智能算力 --limit-chunks 50 --resume --sleep 0.3
```

一次性跑完
```bash
python scripts/extract_knowledge.py --resume --sleep 0.3
```

生成可人工校验的实体和关系表：

```bash
python scripts/build_verified_graph.py
```

启动 Neo4j 并导入图谱：

```bash
docker compose up -d neo4j
python scripts/load_neo4j.py --clear
```

如果当前机器没有 Docker 权限，可以先校验 CSV 是否满足导入条件：

```bash
python scripts/load_neo4j.py --dry-run
```

生成目录：

- `data/parsed_text/`：逐页文本 JSONL 和合并 TXT。
- `data/chunks/`：面向 LLM 抽取的文本块。
- `data/extracted/`：LLM 原始抽取 JSONL 和错误记录。
- `data/verified/entities.csv`、`data/verified/relations.csv`：可人工校验后导入 Neo4j 的图谱数据。

新增行业本体节点和关系：

- 节点：`IndustryConcept`、`Policy`、`Standard`、`ValueChainSegment`。
- 关系：`UPSTREAM_OF`、`DOWNSTREAM_OF`、`ENABLES`、`CONSTRAINS`、`DEFINES`、`SUPPORTED_BY_POLICY`。
- 关系保留 `source_tier`，公司实体保留 `is_core_company`，用于区分核心上市公司和一般提及主体。

## 第四阶段：Neo4j + 本地 RAG + LLM 问答

先生成专业版 curated 图谱。该步骤会从 `data/verified/` 自动图谱中过滤非核心上市公司噪声、目录/释义页误抽取关系和低价值会计科目指标：

```bash
python scripts/build_curated_graph.py
```

构建本地 RAG 索引：

```bash
python scripts/build_rag_index.py
```

专业问答链路：

- `QuestionPlan` 先解析问题意图、公司、主题、关系、是否比较、是否只看核心上市公司。
- 图谱检索默认读取 `data/curated/`；Neo4j 可用时作为增强后端，不可用时自动降级 CSV。
- 本地 RAG 使用 `jieba + BM25` 检索原文块，带同义词扩展、来源优先级、噪声过滤和去重。
- 结构化证据会统一成 `evidence_cards`，再生成“结论、证据、研究要点、风险与边界”格式答案。
- 答案只做事实归纳和研究框架，不提供买卖建议、目标价或收益预测。

新增配置：

- `KG_DATA_DIR`：专业图谱目录，默认 `data/curated`。
- `QA_GRAPH_BACKEND`：`auto`、`csv` 或 `neo4j`，默认 `auto`。
- `QA_CORE_COMPANIES_ONLY`：公司列表类问题默认只返回核心 A 股上市公司。
- `QA_RERANK_TOP_N`：证据重排候选数量。
- `QA_EVIDENCE_TOP_N`：最终进入答案的证据卡片数量。
- `RAG_INDEX_DIR`：本地 RAG 索引目录，默认 `data/rag`。
- `RAG_TOP_K`：每次问答检索的本地文档块数量。
- `RAG_SEARCH_CACHE_SIZE`：本地 RAG 查询结果 LRU 缓存大小，默认 128。
- `QA_GRAPH_LIMIT`：Neo4j 查询结果上限。
- `QA_ENABLE_LLM_CYPHER`：是否启用 LLM 生成 Cypher；默认关闭，使用本地模板查询。
- `QA_ENABLE_LLM_PLANNER`：是否启用 LLM 问题规划；默认关闭，优先使用本地启发式规划。
- `QA_CONTEXTUALIZER_MODE`：追问改写模式，支持 `auto`、`heuristic`、`llm`，默认 `auto`。
- `QA_HISTORY_MAX_TURNS`：连续问答时传入模型的最近对话轮数，默认 3。
- `QA_HISTORY_MAX_CHARS`：连续问答历史的最大字符数，默认 4000。
- `QA_UI_RENDER_LATEST_ONLY`：前端是否只默认渲染选中轮次的证据详情，默认开启。
- `LLM_THINKING_ENABLED`：是否向 DeepSeek 请求开启思考模式，快问快答默认关闭。
- `LLM_REASONING_EFFORT`：DeepSeek 思考强度，快问快答默认 `low`。
- `QA_WEB_SEARCH_ENABLED`：是否在问答答案生成前启用应用侧联网检索；DeepSeek 配置下默认开启。
- `QA_WEB_SEARCH_TOP_K`：联网补充证据条数，默认 5。
- `QA_WEB_SEARCH_TIMEOUT`：联网检索超时时间，默认 5 秒。

运行专业问答回归评测：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_qa.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/benchmark_qa_speed.py
```

如果要让评测也调用已配置的 LLM：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_qa.py --use-llm
```

## 第五阶段：React + FastAPI 前端展示

推荐使用新的 React 工作台。后端 API 复用现有 `QAEngine`、本地图谱和 RAG 索引，并把问答历史自动保存到 `data/conversations/`，前端可以直接点击历史会话恢复并继续追问。

安装 Python 依赖后启动 API：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn src.api:app --reload --port 8000
```

安装并启动前端：

```bash
cd web
npm install
npm run dev
```

浏览器打开 Vite 输出的地址（默认 `http://localhost:5173`）。Vite 会把 `/api` 请求代理到 `http://127.0.0.1:8000`。

如果 8000 端口已被占用，可以把 API 启动到其他端口，并在启动 Vite 时指定代理目标：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn src.api:app --reload --port 8001
cd web
VITE_API_PROXY_TARGET=http://127.0.0.1:8001 npm run dev
```

React 工作台包括：

- 智能问答：主流 chatbot 式对话流，支持连续追问、发送中状态、错误提示和证据详情抽屉。
- 自动会话库：每轮问答自动落盘，侧栏可新建、恢复、重命名、删除、导出 Markdown。
- 数据概览：实体、关系、报告数量、图谱/RAG/LLM 状态和分布。
- 产业链图谱：按公司、技术、关系类型筛选子图和明细。
- 输入框模型控制：可在对话框底部切换 DeepSeek 思考模式，并循环选择 `low`、`medium`、`high` 思考强度。
- 联网补充：可按轮次开启或关闭公开网页检索；本地图谱与 RAG 仍是主证据。

生产构建：

```bash
cd web
npm run build
```

### Streamlit 旧入口

保留 Streamlit 版本作为轻量演示和回退入口：

启动 Streamlit：

```bash
streamlit run app.py
```

Streamlit 页面包括：

- 数据概览：实体、关系、报告数量和分布。
- 智能问答：支持连续多轮追问，展示问题规划、专业答案、模型思考过程、Cypher/CSV 查询意图、图谱结果、本地 RAG 命中、证据卡片、诊断状态和子图。
- 侧栏模型设置：可在前端按轮次开启或关闭 DeepSeek 思考模式，并选择 `low`、`medium`、`high` 思考强度。
- 侧栏联网检索：可开启公开网页检索作为补充证据，回答中会标注“联网补充”。
- 侧栏对话记录：保留当前会话历史，支持新建对话、保存到 `data/conversations/`、查看已保存记录、下载 Markdown 或 JSON。
- 图谱展示：支持按公司、技术、关系类型筛选子图。

可重点演示的问题：

- `液冷产业链有哪些上市公司，各自处于什么环节？`
- `中际旭创和新易盛在光模块业务上的差异是什么？`
- `英维克液冷业务进展和主要风险是什么？`
- `AI算力产业链当前最大的瓶颈是什么？`
