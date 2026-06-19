# AIKA

基于 AI 算力产业链的投研 Agent 问答系统 —— 可嵌入第三方 Agent 的 MCP/Skill 投研插件

项目简介：面向 AI 算力产业链投研场景，构建可追溯、有证据引用的 Agent 问答系统，并以 MCP Server + Skill 插件形态嵌入 Claude Code 等第三方 Agent 工作流。覆盖 AI 服务器、芯片、光模块、液冷、数据中心等核心主题，支持产业链分析、公司对比、风险识别、证据引用和研究报告生成。

技术栈：Python + LangGraph + RAG + 知识图谱 + MCP

项目亮点：
• 投研知识库构建：整合 61 份 PDF 文档，构建 RAG 知识库、Claim / Evidence 证据体系、Segment Dossier 和实体关系图谱，形成结构化投研知识底座
• 混合问答链路设计：设计"Agent + 知识图谱 + RAG"混合问答流程，实现任务规划、多路检索、证据重排、答案生成与支撑性校验
• MCP/Skill 插件化：将投研能力封装为 MCP Server 与 Skill，支持一键安装至 Claude Code 等宿主 Agent，无需 Docker 或 Web 即可在 Agent 对话中直接调用投研工具
• RAG 检索评测与系统验证：基于 50 条测试用例完成 BM25、语义搜索及 RRF 融合检索评测，并通过 LLM 自动标注增强降低 unjudged 率；同步完成 15 个投研问题 QA smoke test 与 15 个 Agent 任务样例测试，验证端到端可用性

---

AIKA 是一个面向 AI 算力产业链的证据驱动智能问答与投研 Agent 系统。公开版主路径是本地 SQLite + MCP/Skill：用户不需要启动 Docker、PostgreSQL、Neo4j 或 Web，就可以在宿主 Agent 中调用投研工具。FastAPI + React Web 版保留为本地开发测试台，用于调试对话、证据卡、图谱、反馈和 Agent 任务视图，不作为公网多用户产品入口。


### Quick Start

用户可以直接从 PyPI 安装并运行本地 sample 流程：

```bash
pip install aika-research-mcp
aika init --sample
aika build-index
aika doctor
aika demo
```

源码开发环境可以使用 `uv run` 执行同一套公开版 CLI：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run aika init --sample
UV_CACHE_DIR=/tmp/uv-cache uv run aika build-index
UV_CACHE_DIR=/tmp/uv-cache uv run aika doctor
UV_CACHE_DIR=/tmp/uv-cache uv run aika demo
```

### Local MCP Quick Start

公开版 AIKA 可以作为本地 MCP Server + Skill 接入 Claude Code 或 Codex。用户不需要手写 `.mcp.json`、`~/.claude.json`、`~/.codex/config.toml`，也不需要手动复制 `skills/aika-research/SKILL.md`。

Claude Code 用户级安装：

```bash
aika mcp install --host claude-code --scope user --with-skill
claude
/mcp
```

Codex CLI 和 Codex IDE extension 共享 MCP 配置，用户级安装一次即可复用：

```bash
aika mcp install --host codex --scope user --with-skill
codex
/mcp
```

也可以把配置和 Skill 写入当前项目。Codex 会在该项目被信任后加载项目级 `.codex/config.toml` 和 `.codex/skills/aika-research/`：

```bash
aika mcp install --host codex --scope project --with-skill
```

安装完成后，可以在宿主 Agent 中显式使用 AIKA Skill：

```text
使用 $aika-research 分析液冷产业链，要求保留 citation id。
```

### Skill 管理

AIKA wheel/sdist 会打包 `aika-research` Skill。wheel 内部路径为 `aika/bundled_skills/aika-research/`，源码目录为 `skills/aika-research/`。

常用命令：

```bash
aika skill list
aika skill install --host codex --scope user
aika skill install --host codex --scope project
aika skill install --host claude-code --scope user
aika skill doctor --host codex --scope user
aika skill doctor --host codex --scope project
aika skill doctor --host claude-code --scope user
```

目录约定：

- Codex user scope：`${CODEX_HOME:-~/.codex}/skills/aika-research/`
- Codex project scope：当前项目 `.codex/skills/aika-research/`
- Claude Code user scope：`${CLAUDE_HOME:-~/.claude}/skills/aika-research/`

Claude Code 的 project-scope Skill 安装当前不支持；请使用 `--scope user`。Skill 安装是幂等的：内容相同时返回 up-to-date；内容不同时默认拒绝覆盖，传入 `--force` 只覆盖 `aika-research` 目录内 AIKA 管理的文件，不影响其他 skills。

### MCP 诊断与高级配置

常用诊断和高级配置命令：

```bash
aika mcp config --host claude-code
aika mcp config --host codex
aika mcp doctor --host claude-code --scope user
aika mcp doctor --host codex --scope user
aika mcp doctor --host codex --scope project
aika mcp install --host claude-code --scope user --dry-run
aika mcp install --host claude-code --scope user --force
aika mcp install --host codex --scope user --dry-run
aika mcp install --host codex --scope user --force
aika mcp install --host codex --scope project --with-skill --dry-run
```

- `aika mcp config`：打印目标宿主需要的 MCP 配置；Claude Code 输出 JSON，Codex 输出 TOML，适合高级用户手动复制。
- `aika mcp install`：注册名为 `aika` 的 MCP server；Claude Code 通过 `claude mcp` 写入，Codex user scope 通过 `codex mcp` 写入，Codex project scope 只更新当前项目 `.codex/config.toml` 的 `[mcp_servers.aika]`。
- `--with-skill`：在 MCP 注册成功后，同时安装 `aika-research` Skill。
- `--force`：只覆盖名为 `aika` 的 MCP server，不改动用户其他宿主配置。
- `aika doctor`：检查 Python、AIKA home、sample data、bundled skill、SQLite index、MCP server、tools 注册和 sample query。
- `aika mcp doctor`：检查 MCP server、SQLite index、指定宿主 MCP 配置和 Skill 安装状态，并给出修复建议。
- `aika skill doctor`：只检查 bundled skill、目标 Skill 目录、`SKILL.md` frontmatter、`agents/openai.yaml` 的 `mcp:aika` 依赖和宿主 MCP 配置。

### 轻量本地索引

公开版可以不启动 PostgreSQL/ParadeDB，直接使用单文件 SQLite FTS5 索引检索 Claim、证据片段和 Dossier。默认本地目录为 `~/.aika`，也可以通过 `AIKA_HOME` 或 `--home` 指定。

```bash
aika init --sample
aika build-index
aika doctor
aika demo
aika search-evidence "液冷产业链" --top-k 5
aika search-claims "光模块" --top-k 5
```

`aika.aika_cli` 显式使用 SQLite backend；现有 Web/API/QAEngine 默认路径仍保持 CSV、Neo4j、PostgreSQL 的原有行为。

### Knowledge Pack 与数据合规

公开发行包使用独立 sample knowledge pack，不直接打包 `data/curated`、raw PDFs、parsed text、RAG 文档或向量文件。

数据包分三档：

- `sample`：随 wheel 内置；源码位于 `data/knowledge_packs/sample/`，wheel 内映射到 `aika/aika_core/bundled_sample/`。它只包含少量公开来源的结构化记录和短证据片段，用于安装后立即试用。
- `public`：后续可发布的更大公开知识包，只使用可再分发或可合理引用的公开资料构建。
- `private/full`：本地完整数据、PostgreSQL/Neo4j 专业部署、raw PDFs、parsed text、RAG 和 semantic index，不随公开包分发。

sample pack 包含：

```text
entities.csv
relations.csv
claims.csv
evidence_spans.csv
segment_dossiers.jsonl
manifest.csv
examples.jsonl
```

`manifest.csv` 记录 `source_report_id`、`source_title`、`source_url`、`published_at`、`source_type`、`license_or_usage_note` 和 `included_fields`。sample 仅保留短摘录和结构化字段，完整上下文请回到 `source_url` 查看原始来源；它适合功能试用和开发 smoke test，不代表完整投研覆盖。

重建和验证 sample pack：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/build_sample_pack.py --output data/knowledge_packs/sample
du -sh data/curated data/knowledge_packs/sample
UV_CACHE_DIR=/tmp/uv-cache uv run python -m aika.aika_cli validate-data --path data/knowledge_packs/sample
```

安装后也可以验证本地副本：

```bash
aika init --sample
aika validate-data --path ~/.aika/knowledge/sample
aika build-index
aika demo
```

### 开发 CLI

```bash
python main.py api --host 127.0.0.1 --port 8000 --reload
python main.py web --host 127.0.0.1 --port 5173 --api http://127.0.0.1:8000
python main.py agent --type research_brief "液冷产业链" --offline --json
python main.py eval --suite qa --offline --benchmark data/eval/qa_benchmark_v1.jsonl --report-dir data/eval_runs --k 6 --json
python main.py eval --suite rag --retrievers bm25 --benchmark data/eval/rag_retrieval_v1.jsonl --report-dir data/eval_runs
python main.py eval --suite agent --offline --limit 1 --task-dir /tmp/aiqasys-agent-tasks --json
python main.py demo --offline --task-dir /tmp/aiqasys-demo-tasks
```

默认离线评测会使用本地 CSV/Claim/Dossier 证据并禁用 LLM、embedding、Neo4j。需要真实模型时，先配置 `.env`，再加 `--use-llm`；需要语义索引时加 `--use-embedding`。

## 第一阶段：数据准备

初始化并下载最新可用年报、公开 AI 算力产业链研报、权威行业白皮书和专业技术源：

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
- `data/raw_pdfs/industry/`：中国信通院等权威机构白皮书、政策、标准资料，以及 IRDS/HIR、UCIe、UALink、Ultra Ethernet、OCP/OIF、MLPerf、arXiv 论文和模型技术报告等专业技术源。
- `data/metadata/companies_extended.csv`：30 家核心上市公司、别名和产业链环节。
- `data/metadata/research_keywords.csv`：研报检索关键词配置。
- `data/metadata/industry_sources.csv`：行业知识源配置，支持 `authority_whitepaper`、`technical_roadmap`、`open_specification`、`manual_open_specification`、`benchmark_methodology`、`technical_paper`、`model_technical_report` 等类型。
- `data/metadata/reports_manifest.csv`：PDF 来源、状态、SHA256、文件大小和页数。

手工下载或暂缺直链的开放规范可以保留为 `manual_open_specification`；如果本地文件尚未补齐，manifest 会保留 `manual_reference` 状态，不阻塞已下载技术源入库。

## 第二、三阶段：知识抽取与图谱构建

配置本地环境变量文件：

```bash
cp .env.example .env
```

`.env` 默认使用 DeepSeek OpenAI 兼容接口，填入 `LLM_API_KEY` 后即可运行。当前示例配置启用 DeepSeek 思考模式：

- `LLM_MODEL=deepseek-v4-pro`
- `LLM_THINKING_ENABLED=true`
- `LLM_REASONING_EFFORT=high`

如果账号仍使用旧版推理模型，可把 `LLM_MODEL` 改为 `deepseek-reasoner`。

解析 PDF 并生成文本块：

```bash
python scripts/parse_pdfs.py \
  --manifest data/metadata/reports_manifest.csv \
  --ocr-mode auto \
  --ocr-language chi_sim+eng \
  --min-text-chars 80
```

解析器使用 PyMuPDF 的版面文本块和表格检测，将正文与表格分别清洗和分块。
表格以结构化行列及 Markdown 保存，低文本页在本机具备 Tesseract 时自动 OCR；
未安装 OCR 依赖时只记录警告。解析质量汇总写入
`data/parsed_text/parse_quality.csv`。需要全量覆盖旧解析结果时增加 `--force`。

调用 LLM 抽取实体关系。建议按报告类型分批跑：

```bash
python scripts/extract_knowledge.py --kind research --contains 算力 --limit-chunks 20 --sleep 0.3
python scripts/extract_knowledge.py --kind annual --contains 服务器 --limit-chunks 50 --resume --sleep 0.3
python scripts/extract_knowledge.py --kind industry --contains 智能算力 --limit-chunks 50 --resume --sleep 0.3
python scripts/extract_knowledge.py --kind industry --contains UCIe --limit-chunks 20 --resume --sleep 0.3
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
- 技术源抽取提示会更偏向 ontology-only：优先抽技术、瓶颈、指标、标准和产业环节；只有原文明确出现核心 A 股上市公司时才抽 `Company`。

## 第四阶段：Neo4j + 本地 RAG + LLM 问答

先生成专业版 curated 图谱。该步骤会从 `data/verified/` 自动图谱中过滤非核心上市公司噪声、目录/释义页误抽取关系和低价值会计科目指标，并同步生成投研 Claim、证据片段和主题 dossier：

```bash
python scripts/build_curated_graph.py
```

启动 ParadeDB 并执行数据库 migration：

```bash
docker compose up -d postgres
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/migrate_postgres.py
```

首次切换时，可把现有 RAG、Claim、Dossier 和审核覆盖层导入 PostgreSQL：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/bootstrap_postgres_retrieval.py
```

PDF 或清洗规则更新后的完整重建顺序：

```bash
python scripts/parse_pdfs.py --force --ocr-mode auto
python scripts/build_rag_index.py
python scripts/extract_knowledge.py --sleep 0.3
python scripts/build_verified_graph.py
python scripts/build_curated_graph.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/build_embedding_index.py --force
```

`build_rag_index.py`、`build_curated_graph.py` 和 `build_research_artifacts.py`
会事务写入 PostgreSQL，并删除数据库中已不在本次构建结果内的旧记录。知识抽取和
Embedding 会调用外部服务，执行前应确认密钥、费用和人工校验流程。

增量生成 PostgreSQL embedding：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/build_embedding_index.py
```

默认只处理 missing、stale 或 failed 记录；`--force` 会重建全部向量。
原文、Claim 和 Dossier 统一使用 2048 维向量，问答时通过 pgvector HNSW
执行 cosine 召回。可先用 `--dry-run` 查看待处理数量。

切换 API 前执行数量与检索质量门禁：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_postgres_cutover.py --run-eval
```

默认验收 6189 个原文块、568 个 Claim、9 个 Dossier、6766 条 ready 向量，
并要求 BM25、semantic、RRF 的 `recall@6` 分别不低于 0.51、0.85、0.85。

如果只想重建投研推理层，可以单独运行：

```bash
python scripts/build_research_artifacts.py
```

默认会合并两类来源：

- 从 `data/curated/relations.csv` 派生公司、主题、产业链和风险 Claim。
- 从 `data/chunks/industry_tech_*.jsonl` 选择性直抽 `mechanism`、`bottleneck`、`indicator`、`supply_chain`、`risk` 等技术 Claim，并过滤英文法律声明、版权页、目录页、参考文献和纯表格噪声。

如果需要回到只使用关系派生 Claim 的旧行为，可以运行：

```bash
python scripts/build_research_artifacts.py --no-direct-claims
```

专业问答链路：

- `QuestionPlan` 先解析问题意图、公司、主题、关系、是否比较、是否只看核心上市公司。
- 图谱检索默认读取 `data/curated/`；Neo4j 可用时作为增强后端，不可用时自动降级 CSV。
- 原文 RAG 使用 ParadeDB `pg_search` 的 Jieba BM25，保留同义词扩展、来源优先级、噪声过滤和去重。
- 结构化证据会统一成 `evidence_cards`，再生成“结论、证据、研究要点、风险与边界”格式答案。
- Claim/Dossier 规则召回和原文/Claim/Dossier 向量召回均直接读取 PostgreSQL。
- 旧 `data/rag`、Claim/Dossier 和 semantic index 文件只保留用于 bootstrap、基线对比和结果审计，不参与运行时查询。
- 答案只做事实归纳和研究框架，不提供买卖建议、目标价或收益预测。

新增配置：

- `KG_DATA_DIR`：专业图谱目录，默认 `data/curated`。
- `QA_GRAPH_BACKEND`：`auto`、`csv` 或 `neo4j`，默认 `auto`。
- `QA_CORE_COMPANIES_ONLY`：公司列表类问题默认只返回核心 A 股上市公司。
- `QA_RERANK_TOP_N`：证据重排候选数量。
- `QA_EVIDENCE_TOP_N`：最终进入答案的证据卡片数量。
- `QA_RERANK_MODE`：GraphRAG 证据精排模式，支持 `auto`、`heuristic`、`llm`，默认 `auto`；宽问题且 LLM 可用时才会调用 LLM 精排。
- `QA_DRIFT_MAX_SUBQUESTIONS`：DRIFT 宽问题最多拆解的子问题数，默认 6。
- `QA_GLOBAL_DOSSIER_TOP_K`：GraphRAG global dossier 召回数量，默认 3。
- `QA_LOCAL_CLAIM_TOP_K`：GraphRAG local claim 召回数量，默认 12。
- `QA_GRAPH_PATH_TOP_K`：GraphRAG 多跳路径数量，默认 6。
- `DATABASE_URL`：必填的 PostgreSQL 连接串。
- `QA_DISABLE_POSTGRES`：设为 `true` 时完全跳过 PostgreSQL retrieval 初始化，用于纯离线 CSV demo/eval；默认 `false`。
- `DB_POOL_MIN_SIZE`、`DB_POOL_MAX_SIZE`：共享同步连接池大小，默认 1 和 8。
- `PG_HNSW_EF_SEARCH`：每次向量查询的 HNSW 搜索宽度，默认 100。
- `RAG_TOP_K`：每次问答检索的原文块数量。
- `EMBEDDING_BASE_URL`：OpenAI-compatible embedding 服务地址；未配置时回退 `LLM_BASE_URL`。
- `EMBEDDING_API_KEY`：embedding 服务 API key；未配置时回退 `LLM_API_KEY`。
- `EMBEDDING_MODEL`：embedding 模型名；为空时关闭 embedding 语义召回。
- `EMBEDDING_BATCH_SIZE`：增量向量化时的批量大小，默认 32。
- `EMBEDDING_DIMENSIONS`：固定为 `2048`。
- `SEMANTIC_TOP_K`：Agent 每轮语义召回数量，默认 8。
- `QA_GRAPH_LIMIT`：Neo4j 查询结果上限。
- `QA_ENABLE_LLM_CYPHER`：是否启用 LLM 生成 Cypher；默认关闭，使用本地模板查询。
- `QA_ENABLE_LLM_PLANNER`：是否启用 LLM 问题规划；默认关闭，优先使用本地启发式规划。
- `QA_ENABLE_AGENT`：是否启用智能体 Runner；默认开启，可设为 `false` 回退旧 workflow。
- `QA_AGENT_RUNNER`：Agent 编排器，支持中心化并行 multi-agent 的 `langgraph` 和串行回退 `legacy`，默认 `langgraph`；同步与流式问答共用所选 runner。
- `QA_AGENT_MAX_STEPS`：Agent ReAct 循环最大步数，默认 4，实现上限 4。
- `QA_MULTI_AGENT_MAX_WORKERS`：中心化 runner 每轮查询 Agent 最大并发数，默认 5。
- `QA_MULTI_AGENT_MAX_LLM_CALLS`：中心化 runner 单次问答共享的 LLM 调用预算，默认 12；耗尽后自动使用确定性规则。
- `QA_MULTI_AGENT_TASK_TIMEOUT_SECONDS`：中心化 runner 单轮并行查询超时秒数，默认 90。
- `QA_CONTEXTUALIZER_MODE`：追问改写模式，支持 `auto`、`heuristic`、`llm`，默认 `auto`。
- `QA_ENABLE_HYDE`：是否启用 HYDE 检索扩展，默认开启；无 LLM 或生成失败时自动回退原问题检索。
- `QA_HYDE_QUERY_MODE`：HYDE 文本检索 query 模式，支持 `hybrid`（原问题 + 假想答案，默认）和 `answer_only`。
- `QA_HYDE_MAX_CHARS`：HYDE 假想答案最大字符数，默认 700；HYDE 只用于 RAG/Claim/Dossier/semantic/GraphRAG 文本召回，不作为最终证据。
- `QA_HISTORY_MAX_TURNS`：LLM 压缩后仍保留原文的最近对话轮数，默认 3。
- `QA_HISTORY_MAX_CHARS`：历史摘要与最近原文的总字符预算，默认 4000。
- `QA_HISTORY_COMPRESSION_ENABLED`：是否用 LLM 将更早对话压成短期记忆摘要，默认开启。
- `QA_HISTORY_SUMMARY_MAX_CHARS`：短期记忆摘要的最大字符数，默认 1600。
- `QA_HISTORY_COMPRESSION_CHUNK_CHARS`：长历史分段压缩时单段的最大字符数，默认 12000。
- `QA_UI_RENDER_LATEST_ONLY`：前端是否只默认渲染选中轮次的证据详情，默认开启。
- `LLM_THINKING_ENABLED`：是否向 DeepSeek 请求开启思考模式，快问快答默认关闭。
- `LLM_REASONING_EFFORT`：DeepSeek 思考强度，快问快答默认 `low`。

运行专业问答回归评测：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_frontend_data.py tests/test_research_claims.py tests/test_rag_qa.py tests/test_professional_qa.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/load_neo4j.py --dry-run
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_qa.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/benchmark_qa_speed.py
```

如果要让评测也调用已配置的 LLM：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_qa.py --use-llm
```

本次稳定版增强后的核心回归结果：

- `tests/test_frontend_data.py`、`tests/test_research_claims.py`、`tests/test_rag_qa.py`、`tests/test_professional_qa.py` 共 26 个测试通过。
- `scripts/evaluate_qa.py` 扩展为 15 个稳定版样例，当前 30/30 通过，并检查 evidence cards、citation id 和 answer subgraph。
- `scripts/load_neo4j.py --dry-run` 校验通过，当前 curated CSV 可导入。

可重点抽样的问题：

- `Ultra Ethernet 对算力网络有什么意义？`
- `DeepSeek-V3 对训练算力瓶颈有什么启示？`
- `UCIe/Chiplet 对国产算力产业链的传导是什么？`

## 第五阶段：本地 Web 开发版

Web 版只用于本机功能测试、调试、演示和开发联调，不作为公开版第一入口，也不作为公网多用户服务。普通用户优先使用上文的 Local MCP 流程；开发者需要检查 UI、证据卡、claim review、feedback 或 Agent 任务时再启动 Web。

本地无 Docker/数据库的开发启动方式：

```bash
QA_DISABLE_POSTGRES=true \
QA_GRAPH_BACKEND=csv \
EMBEDDING_MODEL= \
LLM_API_KEY= \
UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn aika.api:app --reload --host 127.0.0.1 --port 8000
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
QA_DISABLE_POSTGRES=true \
QA_GRAPH_BACKEND=csv \
EMBEDDING_MODEL= \
LLM_API_KEY= \
UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn aika.api:app --reload --host 127.0.0.1 --port 8001
cd web
VITE_API_PROXY_TARGET=http://127.0.0.1:8001 npm run dev
```

本地 Web 开发版包括：

- 智能问答：主流 chatbot 式对话流，支持连续追问、发送中状态、错误提示和证据详情抽屉。
- 自动会话库：每轮问答自动落盘，侧栏可新建、恢复、重命名、删除、导出 Markdown。
- 数据概览：实体、关系、报告数量、图谱/RAG/LLM 状态和分布。
- 产业链图谱：按公司、技术、关系类型筛选子图和明细。
- Claim review：在证据卡中修正 claim 文本、类型、状态和备注，便于本地人工校验。
- Feedback：对单轮回答提交本地反馈，默认写入 `data/feedback/feedback.jsonl`。
- Agent 任务：创建研究简报、公司画像、对比分析、证据缺口和风险审查任务。
- 输入框模型控制：可在对话框底部切换 DeepSeek 思考模式，并循环选择 `low`、`medium`、`high` 思考强度。

本地 API smoke：

```bash
curl http://127.0.0.1:8000/api/status
curl -X POST http://127.0.0.1:8000/api/feedback \
  -H 'Content-Type: application/json' \
  -d '{"question":"液冷产业链有哪些上市公司？","helpful":true,"note":"local web smoke"}'
```

浏览器 smoke：

- 打开 `http://localhost:5173` 后完成一次 query。
- 打开证据详情，确认可以查看 evidence cards。
- 如果证据卡包含 claim id，可以展开并提交一次 claim review。
- 在回答下方提交一次本地 feedback。

本地 Web 开发版当前不实现登录、API token、会话隔离、权限控制、rate limit、生产环境 CORS 和日志脱敏。不要把该开发版直接暴露到公网；如果未来作为企业私有部署或公网服务，需要单独补齐认证、授权、CORS allowlist、限流、日志脱敏和审计策略。

前端构建检查：

```bash
cd web
npm run build
```

## 第六阶段：RAG 检索评测与数据集标注增强

基于 50 条测试用例（`data/eval/rag_retrieval_v1.jsonl`）对 BM25、语义搜索和 RRF 融合进行 chunk-level 检索评测。

### 快速运行

```bash
# BM25 检索（仅需 PostgreSQL）
DATABASE_URL=postgresql://aiqasys:aiqasys@localhost:5432/aiqasys \
  uv run python main.py eval --suite rag --limit 50

# BM25 + 语义 + RRF 融合（需先构建 embedding）
DATABASE_URL=postgresql://aiqasys:aiqasys@localhost:5432/aiqasys \
  uv run python main.py eval --suite rag --use-embedding --limit 50

# 自定义参数
DATABASE_URL=... uv run python main.py eval --suite rag \
  --limit 10 --ks 1,3,6,12 --retrievers bm25,rrf \
  --candidate-k 60 --json --no-save
```

### 评估指标

| 指标 | 含义 |
|---|---|
| `recall@K` | 必需证据单元的召回率 |
| `precision@K` | 返回 K 条中相关项比例 |
| `hit_rate@K` | 至少命中一条直接证据的 case 比例 |
| `mrr@K` | 平均倒数排名（第一个相关结果的排位倒数均值） |
| `ndcg@K` | 归一化折损累计增益（排序质量） |
| `unjudged@K` | 返回结果中未被数据集标注的比例 |

### 降低 unjudged 率：LLM 自动标注增强

当 `unjudged@K` 过高（如 >80%）时，评测指标的 precision/recall 可信度低。系统提供 LLM 自动标注流水线，批量判定 review queue 中的未标注 chunk 并写回数据集。

**Step 1 — 运行评测，生成 review queue：**
评测产生的 `*_unjudged.jsonl` 文件即为待标注队列，包含每个未标注 chunk 的来源、章节和内容片段。

**Step 2 — LLM 批量自动标注：**

```bash
# 预览（不调用 LLM，查看去重统计）
python scripts/auto_judge_review.py \
  --input data/eval_runs/rag_eval_xxx_unjudged.jsonl \
  --dataset data/eval/rag_retrieval_v1.jsonl \
  --dry-run

# 正式标注（8 chunk/批次，~290 次 LLM 调用处理全部 2313 个 unjudged chunk）
python scripts/auto_judge_review.py \
  --input data/eval_runs/rag_eval_xxx_unjudged.jsonl \
  --dataset data/eval/rag_retrieval_v1.jsonl \
  --output data/eval_runs/auto_judgments.jsonl \
  --batch-size 8
```

LLM 对每个 chunk 输出 `grade`（0=不相关, 1=部分相关, 2=直接支撑）、匹配的 `unit_id`、置信度和判定理由。低置信度的项自动标记 `needs_review` 供人工抽查。

**Step 3 — 写回增强数据集：**

```bash
# 预览变更（不写入文件）
python scripts/apply_judgments.py \
  --judgments data/eval_runs/auto_judgments.jsonl \
  --dataset data/eval/rag_retrieval_v1.jsonl \
  --dry-run

# 生成增强版数据集
python scripts/apply_judgments.py \
  --judgments data/eval_runs/auto_judgments.jsonl \
  --dataset data/eval/rag_retrieval_v1.jsonl \
  --output data/eval/rag_retrieval_v2.jsonl \
  --backup

# 可选：仅应用高置信度标注，跳过 needs_review 项
python scripts/apply_judgments.py \
  --judgments data/eval_runs/auto_judgments.jsonl \
  --dataset data/eval/rag_retrieval_v1.jsonl \
  --output data/eval/rag_retrieval_v2.jsonl \
  --skip-needs-review --min-confidence 0.7
```

**Step 4 — 用增强数据集重新评测：**

```bash
DATABASE_URL=postgresql://aiqasys:aiqasys@localhost:5432/aiqasys uv run python main.py eval \
  --suite rag \
  --benchmark data/eval/rag_retrieval_v2.jsonl \
  --retrievers bm25,semantic,rrf \
  --use-embedding
```

### 检索时间测试

```bash
docker exec aiqasys-postgres psql -U aiqasys -d aiqasys -c "
EXPLAIN ANALYZE
SELECT chunk_id, company, source_title,
       pdb.score(id) AS bm25_score
FROM rag_chunks
WHERE search_text ||| '液冷 服务器 散热'
ORDER BY pdb.score(id) DESC
LIMIT 6;
"
```
