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

`.env` 默认使用 DeepSeek OpenAI 兼容接口，填入 `LLM_API_KEY` 后即可运行。

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
- `QA_GRAPH_LIMIT`：Neo4j 查询结果上限。
- `QA_ENABLE_LLM_CYPHER`：是否启用 LLM 生成 Cypher；关闭后使用本地启发式查询。

运行专业问答回归评测：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_qa.py
```

如果要让评测也调用已配置的 LLM：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_qa.py --use-llm
```

## 第五阶段：前端展示

启动 Streamlit：

```bash
streamlit run app.py
```

前端直接进入系统，不做营销页。页面包括：

- 数据概览：实体、关系、报告数量和分布。
- 智能问答：展示问题规划、专业答案、Cypher/CSV 查询意图、图谱结果、本地 RAG 命中、证据卡片、诊断状态和子图。
- 图谱展示：支持按公司、技术、关系类型筛选子图。

可重点演示的问题：

- `液冷产业链有哪些上市公司，各自处于什么环节？`
- `中际旭创和新易盛在光模块业务上的差异是什么？`
- `英维克液冷业务进展和主要风险是什么？`
- `AI算力产业链当前最大的瓶颈是什么？`
