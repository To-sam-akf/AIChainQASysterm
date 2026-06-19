# AIKA

AIKA 是一个面向 AI 算力产业链的证据驱动投研 Agent 工具包。公开版以 **本地 SQLite 索引 + MCP Server + Skill** 为主路径，可以嵌入 Codex、Claude Code 等宿主 Agent，在对话工作流里完成产业链分析、公司画像、公司对比、风险识别、证据缺口审计和研究简报生成。

公开版默认不需要 Docker、PostgreSQL、Neo4j、React Web 或 LLM API key。安装后可以直接使用内置 sample knowledge pack 跑通本地 demo 和 MCP 工具调用。

## 项目定位

- **AIKA Core**：轻量投研核心库，封装 evidence、claim、graph、profile、compare、gap、brief 等结构化能力。
- **AIKA SQLite Backend**：单文件 SQLite FTS5 本地索引，用于检索 claim、evidence span、segment dossier 和产业链关系。
- **AIKA MCP Server**：向宿主 Agent 暴露稳定的投研工具 schema，返回结构化 JSON 和 citation id。
- **AIKA Skill**：为 Codex、Claude Code 等 Agent 提供调用规则、证据约束和输出边界。
- **Full Stack 开发版**：保留 FastAPI、React、PostgreSQL、Neo4j、RAG、LangGraph 等专业版路径，用于本地开发、评测和完整数据实验。

## 能力边界

AIKA 只做证据驱动的行业研究和信息整理：

- 保留 citation id，尽量让结论绑定证据。
- 证据不足时明确输出“当前证据不足”。
- 不输出买卖建议、目标价、收益预测或投资组合建议。
- 公开包只包含 sample knowledge pack，不打包 raw PDF、私有研报、完整 parsed text、向量文件或专业版数据库。

## 安装

### 从 PyPI 安装

```bash
pip install aika-research-mcp
```

要求 Python 3.11+。公开版最小依赖只包含 MCP 与 Pydantic；专业版依赖请使用源码开发环境或安装 extra。

### 从源码运行

```bash
git clone https://github.com/To-sam-akf/AIChainQASysterm.git
cd AIChainQASysterm
UV_CACHE_DIR=/tmp/uv-cache uv run aika --help
```

源码开发时推荐使用 `uv run`，避免污染系统 Python 环境。

## Quick Start

公开版本地 sample 流程：

```bash
aika init --sample
aika build-index
aika doctor
aika demo
```

源码开发环境中执行同一流程：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run aika init --sample
UV_CACHE_DIR=/tmp/uv-cache uv run aika build-index
UV_CACHE_DIR=/tmp/uv-cache uv run aika doctor
UV_CACHE_DIR=/tmp/uv-cache uv run aika demo
```

默认本地目录为 `~/.aika`：

```text
~/.aika/
  config.toml
  knowledge/sample/
  indexes/sample.sqlite
  logs/
```

也可以通过 `AIKA_HOME` 或各命令的 `--home` 指定目录。

## 本地检索 CLI

构建 SQLite FTS5 索引后，可以直接检索 sample knowledge pack：

```bash
aika search-evidence "液冷产业链" --top-k 5
aika search-claims "光模块" --top-k 5
aika search-claims "中际旭创 光模块" --company 中际旭创 --top-k 5
aika validate-data --path ~/.aika/knowledge/sample
```

`search-evidence` 和 `search-claims` 输出 JSON，结果中会尽量包含 `citation_id`、来源标题、页码、公司、主题和证据片段。

## MCP / Skill 接入

AIKA 可以作为本地 MCP Server 被宿主 Agent 调用。MCP Server 启动命令是：

```bash
aika mcp
```

一般用户不需要手写 MCP 配置，直接使用一键安装命令。

### Codex

Codex CLI 和 Codex IDE extension 共享 MCP 配置，用户级安装一次即可复用：

```bash
aika mcp install --host codex --scope user --with-skill
codex
/mcp
```

也可以把 MCP 配置和 Skill 写入当前项目：

```bash
aika mcp install --host codex --scope project --with-skill
```

项目级配置会写入 `.codex/config.toml` 和 `.codex/skills/aika-research/`。Codex 需要信任该项目后才会加载项目级 MCP 配置。

### Claude Code

```bash
aika mcp install --host claude-code --scope user --with-skill
claude
/mcp
```

Claude Code 当前只建议使用 user scope Skill 安装。

### 诊断与配置查看

```bash
aika mcp config --host codex
aika mcp config --host claude-code
aika mcp doctor --host codex --scope user
aika mcp doctor --host claude-code --scope user
aika mcp install --host codex --scope user --dry-run
aika mcp install --host claude-code --scope user --dry-run
```

常用选项：

- `--with-skill`：注册 MCP Server 后同时安装 `aika-research` Skill。
- `--force`：只覆盖名为 `aika` 的 MCP server 或 AIKA 管理的 skill 目录，不改动其他配置。
- `--dry-run`：打印将要写入的配置和命令，不修改宿主配置。

## MCP Tools

AIKA MCP Server 当前暴露以下工具：

| Tool | 用途 |
|---|---|
| `search_evidence` | 检索证据片段和 citation-ready evidence cards |
| `search_claims` | 检索结构化 claim、主题、公司和支撑证据 |
| `get_company_profile` | 生成单家公司画像、业务暴露、风险与证据缺口 |
| `compare_companies` | 对比多家公司在指定主题下的差异 |
| `query_industry_graph` | 查询产业链、技术、上下游和关系图谱 |
| `build_research_brief` | 生成确定性的主题研究简报 |
| `audit_evidence_gaps` | 审计缺失、弱证据或未编号证据 |
| `run_research_task` | 面向宽问题的一次性高阶研究任务入口 |

在宿主 Agent 中可以这样使用：

```text
使用 $aika-research 分析液冷产业链，要求保留 citation id。
```

Skill 会指导宿主 Agent 优先调用 AIKA MCP 工具，而不是凭记忆回答核心事实。

## Skill 管理

wheel/sdist 会打包 `aika-research` Skill。源码目录为 `skills/aika-research/`，安装包内路径为 `aika/bundled_skills/aika-research/`。

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

Skill 安装是幂等的。内容相同时返回 up-to-date；内容不同时默认拒绝覆盖，传入 `--force` 才会覆盖 AIKA 管理的 `aika-research` 目录。

## Knowledge Pack

公开发行包使用独立 sample knowledge pack：

```text
entities.csv
relations.csv
claims.csv
evidence_spans.csv
segment_dossiers.jsonl
manifest.csv
examples.jsonl
```

`manifest.csv` 记录 `source_report_id`、`source_title`、`source_url`、`published_at`、`source_type`、`license_or_usage_note` 和 `included_fields`。sample pack 只保留少量公开来源的结构化记录和短证据片段，用于安装后试用、开发 smoke test 和 MCP 联调，不代表完整投研覆盖。

数据分层：

- `sample`：随包内置，默认复制到 `~/.aika/knowledge/sample/`。
- `public`：后续可扩展的公开知识包，只使用可再分发或可合理引用的公开资料。
- `private/full`：完整本地数据、raw PDF、parsed text、RAG、semantic index、PostgreSQL/Neo4j 专业部署，不随公开包分发。

重建和验证 sample pack：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/build_sample_pack.py --output data/knowledge_packs/sample
UV_CACHE_DIR=/tmp/uv-cache uv run aika validate-data --path data/knowledge_packs/sample
```

## 开发命令

常用测试：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_knowledge_pack.py \
  tests/test_aika_core.py \
  tests/test_aika_sqlite_backend.py \
  tests/test_aika_mcp_tools.py \
  tests/test_aika_mcp_install.py \
  tests/test_aika_skill.py \
  tests/test_aika_skill_packaging.py \
  tests/test_offline_demo.py
```

本地离线 demo：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py demo --offline --task-dir /tmp/aiqasys-demo-tasks
```

离线评测会使用本地 CSV/Claim/Dossier 证据，并禁用 PostgreSQL、Neo4j、LLM 和 embedding。需要真实模型时，先配置 `.env`，再使用 `--use-llm`；需要语义索引时使用 `--use-embedding`。

## Full Stack 开发版

Web/API 版用于本机功能测试、证据卡调试、claim review、feedback、Agent 任务视图和专业版数据实验，不是公开版第一入口，也不建议直接暴露到公网。

无 Docker/数据库的本地 API 启动方式：

```bash
QA_DISABLE_POSTGRES=true \
QA_GRAPH_BACKEND=csv \
EMBEDDING_MODEL= \
LLM_API_KEY= \
UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn aika.api:app --reload --host 127.0.0.1 --port 8000
```

前端：

```bash
cd web
npm install
npm run dev
```

默认打开 Vite 输出的地址，通常是 `http://localhost:5173`。如需指定 API 代理：

```bash
VITE_API_PROXY_TARGET=http://127.0.0.1:8001 npm run dev
```

前端构建检查：

```bash
cd web
npm run build
```

专业版数据链路仍保留 PostgreSQL/ParadeDB、Neo4j、embedding、LLM 抽取、RAG 评测和 LangGraph Agent 编排。相关脚本位于 `scripts/`，测试位于 `tests/`。

## 专业版数据流水线概览

以下流程适合开发者在本地完整数据环境中使用：

```bash
python scripts/prepare_stage1_data.py --kind all --max-research 10
python scripts/parse_pdfs.py --manifest data/metadata/reports_manifest.csv --ocr-mode auto --ocr-language chi_sim+eng
python scripts/extract_knowledge.py --kind research --contains 算力 --limit-chunks 20 --sleep 0.3
python scripts/build_verified_graph.py
python scripts/build_curated_graph.py
```

需要专业版数据库时：

```bash
docker compose up -d postgres neo4j
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/migrate_postgres.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/bootstrap_postgres_retrieval.py
python scripts/load_neo4j.py --clear
```

PostgreSQL/embedding cutover 验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_postgres_cutover.py --run-eval
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/build_embedding_index.py
```

RAG/QA/Agent 评测：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py eval --suite qa --offline --benchmark data/eval/qa_benchmark_v1.jsonl --report-dir data/eval_runs --k 6 --json
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py eval --suite rag --retrievers bm25 --benchmark data/eval/rag_retrieval_v1.jsonl --report-dir data/eval_runs
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py eval --suite agent --offline --limit 1 --task-dir /tmp/aiqasys-agent-tasks --json
```

## 配置参考

公开版常用：

- `AIKA_HOME`：本地 AIKA 数据目录，默认 `~/.aika`。
- `UV_CACHE_DIR`：源码开发时建议设为 `/tmp/uv-cache`。

离线/专业版兼容：

- `QA_DISABLE_POSTGRES=true`：完全跳过 PostgreSQL retrieval 初始化。
- `QA_GRAPH_BACKEND=csv`：使用本地 CSV 图谱后端。
- `EMBEDDING_MODEL=`：为空时关闭 embedding 语义召回。
- `LLM_API_KEY=`：为空时关闭 LLM 调用。
- `DATABASE_URL`：专业版 PostgreSQL 连接串。
- `KG_DATA_DIR`：专业图谱目录，默认 `data/curated`。

## 当前状态

AIKA 当前处于 alpha 阶段。公开版主路径已经收敛为本地可安装、可初始化、可构建索引、可诊断、可通过 MCP/Skill 调用的投研增强工具包；Full Stack 路径继续服务完整数据、评测和本地开发。
