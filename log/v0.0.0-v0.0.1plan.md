# AIKA MCP/Skill 插件化实操手册

本文档基于 `direction.md`，用于把 AIKA 从 Web Demo 优先的项目，改造成可安装、可本地运行、可被 Codex/爱马仕等 Agent 调用的投研增强工具包。

目标交付形态：

```text
AIKA Skill + AIKA MCP Server + AIKA Knowledge Pack
```

第一阶段不追求公网 Web，不要求用户启动 Docker 数据库，不要求用户访问 AIKA 网页。优先保证用户能在自己的 Agent 软件中安装、初始化、调用工具并获得带证据引用的投研结果。

## 0. 总体实施原则

### 0.1 第一性目标

公开版要满足：

- 无需 Docker。
- 无需 PostgreSQL/ParadeDB。
- 无需 Neo4j。
- 无需 React Web。
- 可以本地安装。
- 可以通过 MCP 被宿主 Agent 调用。
- 所有投研结论尽量绑定 citation id。
- 没有证据时明确输出证据不足。
- 不输出买卖建议、目标价、收益预测。

### 0.2 架构拆分原则

将系统拆成四层：

```text
宿主 Agent 层：Codex / 爱马仕 / Claude Desktop / 其他 Agent
Skill 层：投研流程、调用规则、输出约束
MCP 工具层：工具 schema、参数校验、结构化返回
AIKA Core 层：知识库、检索、图谱、证据卡、投研任务逻辑
```

Web 版、PostgreSQL、Neo4j 归入 Full Stack 专业版，不作为公开版第一入口。

### 0.3 每阶段验收口径

每个阶段都必须能回答四个问题：

- 需求：这一阶段要解决什么用户问题？
- 核心逻辑：内部如何实现？
- 验证方法：用什么命令、测试或手工检查证明完成？
- 预期结果：完成后用户或开发者能得到什么？

## Phase 0：修正真正离线 Demo

### 需求

当前 `main.py demo --offline` 仍会初始化 PostgreSQL retrieval backend，导致没有数据库时失败。公开版第一步必须保证一个真正无服务、无 Docker、无 LLM key 的 demo 可以跑通。

用户视角需求：

- 新用户 clone 项目后，不启动数据库也能看到 AIKA 的核心能力。
- demo 至少能返回样例问题、证据卡数量、简短回答或结构化结果。
- demo 不能因为缺失 PostgreSQL、Neo4j、LLM、embedding 而退出。

开发视角需求：

- `--offline` 必须真正跳过 PostgreSQL 初始化。
- CSV/curated 数据能作为最小 fallback。
- CI 中加入离线 demo smoke，防止后续回归。

### 核心逻辑

实现一个明确的离线引擎路径：

```text
CLI --offline
  -> 设置 QA_GRAPH_BACKEND=csv
  -> 设置 EMBEDDING_MODEL=
  -> 设置 QA_DISABLE_POSTGRES=true
  -> QAEngine.from_env() 跳过 PostgresRetrievalStore.from_env()
  -> 使用 LocalKnowledgeGraph + curated CSV
  -> RAG/research/semantic 状态标记为 disabled
```

建议实现点：

- 新增环境变量：`QA_DISABLE_POSTGRES=true`。
- `QAEngine.from_env()` 中判断该变量：
  - 为 true 时，不创建 `PostgresRetrievalStore`。
  - `rag_index=None`。
  - `research_memory=None`。
  - `semantic_index=None`。
  - status 中 `rag_enabled=false`、`research_enabled=false`、`embedding_enabled=false`。
- `aika/cli.py` 的 `build_engine(args)` 在 `effective_offline(args)` 为 true 时设置该变量。
- `run_demo` 中不要假设一定存在 RAG evidence；允许 CSV graph fallback 生成简短答案。

### 验证方法

命令验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py demo --offline --task-dir /tmp/aiqasys-demo-tasks
```

环境隔离验证：

```bash
DATABASE_URL=
LLM_API_KEY=
EMBEDDING_MODEL=
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py demo --offline --task-dir /tmp/aiqasys-demo-tasks
```

单元测试建议：

- 新增 `tests/test_offline_demo.py`。
- mock 或清空 `DATABASE_URL`。
- 断言 `QAEngine.from_env()` 在 `QA_DISABLE_POSTGRES=true` 时不创建 retrieval store。
- 断言 `engine.status.graph_backend == "csv"` 或在无 CSV 时为 `"none"`。

CI smoke：

```yaml
- name: Offline demo smoke
  run: UV_CACHE_DIR=/tmp/uv-cache uv run python main.py demo --offline --task-dir /tmp/aiqasys-demo-tasks
```

### 预期结果

- 无数据库环境下 demo 可以成功退出，返回码为 0。
- 输出包含：
  - `Graph backend: csv`
  - `RAG enabled: False`
  - `LLM enabled: False`
  - 样例问题
  - 简短答案或证据不足提示
- 该阶段完成后，新用户可以在不部署服务的情况下理解 AIKA 的基本形态。

## Phase 1：抽出 AIKA Core 轻量核心库

### 需求

当前投研逻辑散落在 API、QAEngine、Agent、PostgreSQL retrieval、frontend data 等模块中。MCP 插件化需要一个不依赖 Web、不依赖数据库服务的核心库。

用户视角需求：

- MCP tools 可以直接调用核心逻辑，不需要启动 FastAPI。
- 轻量公开版可以基于 CSV/JSONL 完成基础检索。
- 专业版仍能继续使用 PostgreSQL/Neo4j。

开发视角需求：

- 抽出稳定的数据模型和接口。
- 后端实现可替换：CSV/JSONL、SQLite/DuckDB、PostgreSQL。
- 核心逻辑可以被 CLI、MCP、Web 共用。

### 核心逻辑

建议新增目录：

```text
aika/aika_core/
  __init__.py
  models.py
  config.py
  data_paths.py
  evidence.py
  claims.py
  graph.py
  profiles.py
  compare.py
  gaps.py
  brief.py
  backends/
    __init__.py
    csv_backend.py
    sqlite_backend.py
    postgres_backend.py
```

核心接口建议：

```python
class ResearchBackend:
    def search_evidence(self, query: str, *, top_k: int = 8, **filters): ...
    def search_claims(self, query: str, *, top_k: int = 8, **filters): ...
    def query_graph(self, *, company="", technology="", relation_type="", limit=80): ...
    def get_company_profile(self, company: str, *, topic=""): ...
```

核心数据模型：

- `EvidenceCard`
- `ClaimRecord`
- `GraphNode`
- `GraphEdge`
- `CompanyProfile`
- `CompanyComparison`
- `EvidenceGap`
- `ResearchBrief`

第一版不要重写全部业务逻辑。优先把现有可复用函数包装进 `aika_core`：

- `LocalKnowledgeGraph`
- `subgraph_edges`
- curated claims CSV 读取
- evidence card 标准化
- company profile / compare / gap audit 的确定性逻辑

### 验证方法

单元测试：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_aika_core.py
```

建议测试用例：

- 可以从 `data/curated/claims.csv` 加载 claims。
- `search_claims("液冷", top_k=5)` 返回不超过 5 条结构化结果。
- 每条结果包含 `claim_id`、`claim_text`、`source_title` 或可追踪来源。
- `query_graph(company="中际旭创")` 返回 relation rows。
- `get_company_profile("中际旭创")` 返回公司、证据、风险或证据缺口字段。

手工验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py demo --offline --task-dir /tmp/aiqasys-demo-tasks
```

### 预期结果

- 核心投研能力不再强绑定 FastAPI 和 PostgreSQL。
- 后续 MCP Server 可以直接导入 `aika_core`。
- Full Stack Web 仍可保留现有 `QAEngine`，但新增轻量公开路径。

## Phase 2：实现轻量本地索引

### 需求

公开版不能要求用户启动 PostgreSQL/ParadeDB。需要一个本地文件型检索后端，用于 claims、evidence spans、dossiers 的全文检索和过滤。

用户视角需求：

- `aika init --sample` 后能得到本地数据。
- `aika build-index` 后可以检索。
- 数据库只是一个本地文件，不需要服务进程。

开发视角需求：

- 提供可重复构建的轻量索引。
- 检索结果结构与 MCP tools 一致。
- 后续可平滑替换为 DuckDB、SQLite FTS5 或专业版 PostgreSQL。

### 核心逻辑

推荐默认方案：SQLite FTS5。

理由：

- Python 标准库内置 sqlite3。
- 单文件分发简单。
- FTS5 能覆盖第一版全文检索需求。
- 对普通用户最友好。

索引文件建议：

```text
~/.aika/
  config.toml
  knowledge/
    sample/
      entities.csv
      relations.csv
      claims.csv
      evidence_spans.csv
      segment_dossiers.jsonl
  indexes/
    sample.sqlite
```

SQLite 表设计建议：

```text
claims
  claim_id
  claim_type
  topic
  claim_text
  companies
  evidence_span
  source_title
  page
  confidence
  exposure_level

evidence_spans
  evidence_id
  claim_id
  evidence
  source_title
  page
  section
  company
  topic

dossiers
  dossier_id
  topic
  title
  content

relations
  relation_id
  head_name
  relation
  tail_name
  evidence
  source_title
  page
```

FTS 表：

```text
evidence_fts(evidence, source_title, company, topic)
claims_fts(claim_text, evidence_span, topic, companies)
dossiers_fts(title, content, topic)
```

CLI 命令：

```bash
aika init --sample
aika build-index
aika doctor
```

### 验证方法

构建验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m aika.aika_cli init --sample
UV_CACHE_DIR=/tmp/uv-cache uv run python -m aika.aika_cli build-index
UV_CACHE_DIR=/tmp/uv-cache uv run python -m aika.aika_cli doctor
```

检索验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m aika.aika_cli search-evidence "液冷产业链" --top-k 5
UV_CACHE_DIR=/tmp/uv-cache uv run python -m aika.aika_cli search-claims "中际旭创 光模块" --top-k 5
```

单元测试：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_aika_sqlite_backend.py
```

测试断言：

- 索引文件存在。
- `search_evidence` 返回 top-k。
- 返回结果包含 citation id。
- filters 生效：company/topic/claim_type。
- 数据缺失时返回空列表，而不是抛异常。

### 预期结果

- 用户本地只有一个 SQLite 文件即可检索。
- 公开版不依赖 Docker 数据库。
- MCP Server 可以直接使用 SQLite backend。
- 后续专业版仍可保留 PostgreSQL backend。

## Phase 3：实现 AIKA MCP Server

### 需求

让 Codex、爱马仕、Claude Desktop 或其他支持 MCP 的 Agent 可以调用 AIKA 的投研工具。

用户视角需求：

- 安装后通过 MCP 配置即可使用。
- 宿主 Agent 能列出 AIKA tools。
- 每个工具返回结构化 JSON。
- 工具结果可被宿主 Agent 继续整理成自然语言报告。

开发视角需求：

- 工具 schema 稳定。
- 参数校验清晰。
- 返回结构统一。
- 工具内部调用 `aika_core`，不依赖 FastAPI。

### 核心逻辑

MCP Server 入口：

```text
aika/aika_mcp/
  __init__.py
  server.py
  tools.py
  schemas.py
```

CLI：

```bash
aika mcp
```

第一批 MCP tools：

- `search_evidence`
- `search_claims`
- `get_company_profile`
- `compare_companies`
- `query_industry_graph`
- `build_research_brief`
- `audit_evidence_gaps`

建议额外提供一个高阶工具：

- `run_research_task`

原因：

- 宿主 Agent 不一定稳定执行复杂多步流程。
- 高阶工具可以在 AIKA 内部运行多阶段或多 Agent 逻辑。
- Skill 只需要在复杂任务时触发这个总入口。

`run_research_task` 参数建议：

```json
{
  "task_type": "research_brief",
  "topic": "液冷产业链",
  "companies": [],
  "depth": "standard",
  "require_citations": true
}
```

返回建议：

```json
{
  "status": "completed",
  "report_markdown": "",
  "evidence_cards": [],
  "agent_trace": [],
  "verification": {},
  "evidence_gaps": []
}
```

内部多 Agent 或多阶段逻辑：

```text
Planner
  -> Claim Retrieval
  -> Evidence Retrieval
  -> Graph Retrieval
  -> Risk/Gaps Audit
  -> Verification
  -> Brief Builder
```

第一版可以不启动真正并发多 Agent，先用确定性的多阶段 pipeline；接口上保留 `agent_trace`，后续再替换为现有 `ResearchAgent` / `QAAgent` / centralized runner。

### 验证方法

工具列举验证：

```bash
aika mcp
```

使用 MCP 客户端验证：

- 能看到 `search_evidence`。
- 能看到 `search_claims`。
- 能看到 `run_research_task`。

本地集成测试建议：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_aika_mcp_tools.py
```

测试断言：

- 每个 tool schema 可序列化。
- 缺少必填参数时返回明确错误。
- `search_evidence("液冷")` 返回结构化列表。
- `run_research_task(topic="液冷产业链")` 返回 `report_markdown`、`evidence_cards`、`verification`。
- 所有 evidence cards 都有 `citation_id` 或明确标记 `uncited`。

手工验证：

```json
{
  "mcpServers": {
    "aika": {
      "command": "aika",
      "args": ["mcp"]
    }
  }
}
```

然后在宿主 Agent 中输入：

```text
使用 AIKA 分析液冷产业链，要求引用证据。
```

### 预期结果

- AIKA 能以 MCP Server 形式被宿主 Agent 调用。
- 用户不需要访问网页。
- 用户不需要知道内部数据结构。
- 复杂任务可以通过 `run_research_task` 一次触发。
- 简单任务也可以由宿主 Agent 组合调用多个细粒度 tools。

## Phase 3.1：实现一键式配置

### 需求

Phase 3 已经让 AIKA 可以作为 MCP Server 被宿主 Agent 调用，但如果用户仍然需要手动编辑 `.mcp.json`、`~/.claude.json` 或记住 `uv --directory ... run aika mcp` 这类内部命令，公开版安装体验会很脆弱。Phase 3.1 的目标是把 MCP 配置变成 AIKA 自己的一键配置能力。

用户视角需求：

- 用户不需要手写 MCP JSON。
- 用户不需要知道 `uv` 的绝对路径、AIKA 项目目录、`UV_CACHE_DIR` 等内部细节。
- 用户可以一条命令把 AIKA 注册到 Claude Code 等宿主 Agent。
- 用户可以用 doctor 命令检查 MCP Server、SQLite index 和宿主配置是否可用。
- 高级用户仍然可以只打印配置 JSON，自行复制到目标宿主。

开发视角需求：

- 在 AIKA CLI 中新增 `aika mcp install`、`aika mcp doctor`、`aika mcp config` 子命令。
- 第一版优先支持 `--host claude-code`，接口上预留 `claude-desktop`、`codex` 等后续宿主。
- 支持 `--scope user` 和 `--scope project`。
- 支持检测已有配置，并通过 `--force` 明确覆盖。
- 配置生成逻辑要可测试，实际写入宿主配置的逻辑要和纯 JSON 生成逻辑分离。

### 核心逻辑

推荐 CLI 形态：

```bash
aika mcp install --host claude-code --scope user
aika mcp install --host claude-code --scope project
aika mcp install --host claude-code --scope user --force
aika mcp doctor
aika mcp config --host claude-code
```

通过源码运行时，README/HowToUse 可以只暴露这一条入口：

```bash
uv --directory /path/to/AIQASYS run aika mcp install --host claude-code --scope user
```

`config` 负责生成 MCP server 配置：

```json
{
  "type": "stdio",
  "command": "/abs/path/to/uv",
  "args": ["--directory", "/abs/path/to/AIQASYS", "run", "aika", "mcp"],
  "env": {
    "UV_CACHE_DIR": "/tmp/uv-cache"
  },
  "timeout": 600000
}
```

生成逻辑：

```text
aika mcp config
  -> 解析 --host
  -> 自动定位 uv 绝对路径
  -> 自动定位 AIKA 项目根目录
  -> 自动补齐 UV_CACHE_DIR
  -> 输出宿主需要的 JSON
```

`install` 负责把配置写入宿主 Agent：

```text
aika mcp install --host claude-code --scope user
  -> 调用 config 生成 JSON
  -> 检查 claude 命令是否存在
  -> 检查是否已有名为 aika 的 MCP server
  -> 无冲突时调用 claude mcp add-json aika '{...}' --scope user
  -> 有冲突且无 --force 时提示用户使用 --force
  -> 有 --force 时覆盖或先删除再添加
```

第一版可以只适配 Claude Code CLI：

```bash
claude mcp add-json aika '{...}' --scope user
```

长期打包成熟后，配置可自动简化为：

```json
{
  "mcpServers": {
    "aika": {
      "type": "stdio",
      "command": "aika",
      "args": ["mcp"]
    }
  }
}
```

`doctor` 负责诊断端到端可用性：

```text
aika mcp doctor
  -> 检查 uv/aika 命令是否可用
  -> 检查 aika mcp 是否能启动并列出 tools
  -> 检查 SQLite index 是否存在且可查询
  -> 检查 Claude Code 是否已配置 aika server
  -> 输出 pass/warn/fail 和修复建议
```

建议新增实现模块：

```text
aika/aika_mcp/
  installer.py
  doctor.py
  host_configs.py
```

职责划分：

- `host_configs.py`：生成不同宿主的配置 JSON。
- `installer.py`：执行宿主 CLI 写入、冲突检测、`--force` 覆盖。
- `doctor.py`：执行依赖、索引、MCP server、宿主配置诊断。
- `aika/aika_cli.py`：只负责参数解析和调用上述模块。

### 验证方法

配置生成验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run aika mcp config --host claude-code
```

断言输出包含：

- `"type": "stdio"`
- `command` 为 `uv` 的绝对路径，或打包安装后的 `aika`
- `args` 包含 `mcp`
- `env.UV_CACHE_DIR` 存在
- `timeout` 存在

doctor 验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run aika mcp doctor
```

断言：

- 能识别 `uv` 是否存在。
- 能识别 `aika mcp` 是否可启动。
- 能识别 SQLite index 是否已构建。
- 未安装 Claude Code 或未配置 MCP 时返回 warn/fail，而不是异常退出。
- 输出包含可执行的修复建议，例如 `aika mcp install --host claude-code --scope user`。

安装 dry-run 或临时环境验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run aika mcp install --host claude-code --scope user --dry-run
```

如果实现中不提供 `--dry-run`，测试中应 mock `claude mcp add-json`，避免修改开发者真实配置。

单元测试建议：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_aika_mcp_install.py
```

测试断言：

- `build_mcp_config(host="claude-code")` 输出稳定 JSON。
- 找不到 `uv` 时返回明确错误。
- 已有配置且无 `--force` 时拒绝覆盖。
- `--force` 时生成覆盖流程。
- `doctor` 对缺失 Claude Code、缺失 SQLite index、MCP 启动失败分别给出明确状态。

手工验证：

```bash
uv --directory /path/to/AIQASYS run aika mcp install --host claude-code --scope user
claude
/mcp
```

在 Claude Code 的 MCP 列表中应能看到 `aika`，并能列出 Phase 3 提供的 tools。

### 预期结果

- 用户可以通过一条命令把 AIKA 注册到 Claude Code。
- README/HowToUse 不再要求用户手动编辑 `.mcp.json` 或 `~/.claude.json`。
- 新用户只需要执行安装命令、打开宿主 Agent、检查 `/mcp`。
- 开发者可以用 `aika mcp config` 查看底层配置，用 `aika mcp doctor` 快速定位安装问题。
- 后续打包为 `uv tool install aika` 或 `pipx install aika` 后，MCP 配置可以自然收敛为 `command: "aika"`、`args: ["mcp"]` 的最终形态。

## Phase 3.2：快速接入 Codex

### 需求

Phase 3.1 已经完成 Claude Code 的一键注册，但公开版 AIKA 不应该只服务单一宿主。Codex CLI 和 Codex IDE extension 都支持 MCP，并共享 `config.toml` 配置，因此下一步要补齐：

```bash
aika mcp install --host codex --scope user
```

用户视角需求：

- 用户可以一条命令把 AIKA MCP Server 注册到 Codex。
- 用户不需要手写 `~/.codex/config.toml` 或项目 `.codex/config.toml`。
- 用户不需要理解 `uv --directory ... run aika mcp`、`UV_CACHE_DIR`、Codex TOML 配置表等细节。
- 用户可以在 Codex TUI 中通过 `/mcp` 看到 `aika` server。
- 用户安装一次后，Codex CLI 和 Codex IDE extension 都能复用同一份 MCP 配置。

开发视角需求：

- `aika mcp install` 的 host 分发从 Claude Code 单实现扩展为多宿主实现。
- `--host codex` 支持 `--scope user` 和 `--scope project`。
- `--scope user` 优先调用 Codex CLI：`codex mcp add`、`codex mcp get`、`codex mcp remove`。
- `--scope project` 写入项目 `.codex/config.toml`，并提示该配置只会在 Codex 信任项目后加载。
- `--force` 只覆盖名为 `aika` 的 MCP server，不改动用户其他 Codex 配置。
- `aika mcp doctor --host codex` 能检查 Codex CLI、配置项和 AIKA MCP Server 是否可用。

### 核心逻辑

推荐 CLI 形态：

```bash
aika mcp install --host codex --scope user
aika mcp install --host codex --scope project
aika mcp install --host codex --scope user --force
aika mcp install --host codex --scope user --dry-run
aika mcp config --host codex
aika mcp doctor --host codex
```

Codex 的 stdio MCP 配置可以收敛为 TOML：

```toml
[mcp_servers.aika]
command = "/abs/path/to/uv"
args = ["--directory", "/abs/path/to/AIQASYS", "run", "aika", "mcp"]
startup_timeout_sec = 30
tool_timeout_sec = 600

[mcp_servers.aika.env]
UV_CACHE_DIR = "/tmp/uv-cache"
```

打包安装成熟后可以简化为：

```toml
[mcp_servers.aika]
command = "aika"
args = ["mcp"]
startup_timeout_sec = 30
tool_timeout_sec = 600
```

`host_configs.py` 扩展：

```text
SUPPORTED_HOSTS = {"claude-code", "codex"}

build_mcp_config(host="codex")
  -> 解析 aika 或 uv 启动方式
  -> 输出 Codex 可消费的 stdio 配置字段
  -> 字段包括 command、args、env、startup_timeout_sec、tool_timeout_sec
```

`installer.py` 扩展为宿主分发：

```text
install_mcp_server(host="codex", scope="user")
  -> 调用 build_mcp_config(host="codex")
  -> 检查 codex 命令是否存在
  -> 执行 codex mcp get aika 或 codex mcp list --json 检查冲突
  -> 无冲突时执行 codex mcp add aika --env KEY=VALUE -- <command> <args...>
  -> 有冲突且无 --force 时提示用户使用 --force
  -> 有 --force 时执行 codex mcp remove aika，再重新 add
```

示例命令：

```bash
codex mcp add aika --env UV_CACHE_DIR=/tmp/uv-cache -- \
  /abs/path/to/uv --directory /abs/path/to/AIQASYS run aika mcp
```

项目级配置逻辑：

```text
install_mcp_server(host="codex", scope="project")
  -> 定位当前项目根目录
  -> 创建或读取 .codex/config.toml
  -> 只新增或替换 [mcp_servers.aika] 表
  -> 保留文件中其他 Codex 配置
  -> 输出提示：Codex 只会在 trusted project 中加载项目配置
```

`doctor.py` 扩展：

```text
aika mcp doctor --host codex
  -> 检查 codex 命令是否在 PATH
  -> 检查 codex mcp get aika 或 codex mcp list --json
  -> 检查生成的 command/args 是否能启动 AIKA MCP Server
  -> 检查 tools/list 是否能返回 AIKA tools
  -> project scope 下提醒用户确认 Codex 已信任该项目
  -> 输出 pass/warn/fail 和修复建议
```

实现注意：

- 不要把 Claude Code 的 JSON 配置直接复用给 Codex；Codex 原生配置是 TOML 表。
- 不要覆盖整个 `~/.codex/config.toml` 或 `.codex/config.toml`。
- dry-run 必须打印将要执行的 `codex mcp add` 命令或将要写入的 TOML 片段。
- 如果 Codex CLI 不存在，提示用户安装 Codex，或手动复制 `aika mcp config --host codex` 输出到 Codex 配置。

### 验证方法

配置生成验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run aika mcp config --host codex
```

断言输出包含：

- `[mcp_servers.aika]`
- `command` 为 `uv` 的绝对路径，或打包安装后的 `aika`
- `args` 包含 `mcp`
- `startup_timeout_sec` 和 `tool_timeout_sec`
- 源码运行时包含 `UV_CACHE_DIR`

安装 dry-run 验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run aika mcp install --host codex --scope user --dry-run
```

断言：

- 不修改真实 `~/.codex/config.toml`。
- 输出包含 `codex mcp add aika`。
- 输出中的命令能定位到 `uv` 或 `aika`。
- 输出包含 `--env UV_CACHE_DIR=/tmp/uv-cache`，或说明打包安装模式无需该 env。

用户级安装验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run aika mcp install --host codex --scope user --force
codex mcp list --json
codex
/mcp
```

断言：

- `codex mcp list --json` 中存在名为 `aika` 的 server。
- Codex TUI 的 `/mcp` 页面能看到 `aika`。
- `aika` server 可以完成初始化并列出 AIKA MCP tools。

项目级安装验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run aika mcp install --host codex --scope project --force
sed -n '1,160p' .codex/config.toml
```

断言：

- `.codex/config.toml` 中存在 `[mcp_servers.aika]`。
- 文件中原有非 AIKA 配置没有被删除。
- 在 Codex 信任该项目后，Codex CLI/IDE 能加载 `aika` server。

doctor 验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run aika mcp doctor --host codex
```

断言：

- 未安装 Codex CLI 时返回 warn/fail 和安装建议，而不是异常退出。
- 未配置 `aika` 时给出 `aika mcp install --host codex --scope user` 修复建议。
- 已配置但 MCP 启动失败时，输出可执行的 command/args 排查建议。

单元测试建议：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_aika_mcp_install.py
```

新增测试断言：

- `build_mcp_config(host="codex")` 输出 Codex TOML 所需字段。
- `install_mcp_server(host="codex", scope="user", dry_run=True)` 不调用真实 Codex CLI。
- 已有 `aika` 且无 `--force` 时拒绝覆盖。
- `--force` 时按 `get -> remove -> add` 顺序执行。
- project scope 只替换 `[mcp_servers.aika]`，保留其他 TOML 配置。
- `doctor` 对缺失 Codex CLI、缺失配置、MCP 启动失败分别给出明确状态。

### 预期结果

- 用户可以通过 `aika mcp install --host codex --scope user` 快速把 AIKA 接入 Codex。
- AIKA 不再只提供 Claude Code 的一键接入路径。
- Codex CLI 和 Codex IDE extension 可以复用同一份 `aika` MCP 配置。
- 高级用户仍可以通过 `aika mcp config --host codex` 查看并手动复制 Codex TOML 配置。
- 开发者可以用统一的 `aika mcp doctor --host ...` 诊断 Claude Code 和 Codex 两类宿主。
- 未来继续新增其他宿主时，只需要扩展 host config、installer adapter 和 doctor adapter，不需要改 MCP Server 核心工具。

## Phase 4：编写 AIKA Skill

### 需求

MCP 只暴露工具，不保证宿主 Agent 会正确使用。需要 Skill 告诉 Agent 何时调用 AIKA、如何调用、如何组织输出、哪些内容禁止输出。

用户视角需求：

- 用户只说“分析液冷产业链”，宿主 Agent 就知道要调用 AIKA。
- 输出结构稳定。
- 结论带证据。
- 不凭空回答。

开发视角需求：

- Skill 指令要足够短、明确、可执行。
- 明确工具路由规则。
- 明确证据引用和合规边界。

### 核心逻辑

Skill 文件建议：

```text
skills/aika-research/SKILL.md
```

Skill 应包含：

- 触发条件：
  - AI 算力产业链
  - 中文投研
  - 公司对比
  - 技术路线
  - 风险审查
  - 证据缺口
- 工具路由：
  - 事实/证据问题：`search_evidence`
  - 结构化 claim：`search_claims`
  - 公司画像：`get_company_profile`
  - 公司对比：`compare_companies`
  - 产业链关系：`query_industry_graph`
  - 复杂任务：`run_research_task`
  - 缺口审计：`audit_evidence_gaps`
- 输出规范：
  - 核心判断
  - 证据
  - 产业链传导
  - 公司差异
  - 风险与反证
  - 证据缺口
- 禁止事项：
  - 不给买卖建议
  - 不给目标价
  - 不做收益预测
  - 不把未检索到的内容写成确定事实

### 验证方法

Skill 静态检查：

- 是否明确列出所有 MCP tools。
- 是否说明 citation id 必须保留。
- 是否说明找不到证据时输出证据不足。
- 是否包含禁止买卖建议、目标价、收益预测。

人工场景验证：

场景 1：

```text
用 AIKA 分析液冷产业链有哪些上市公司。
```

预期调用：

- `run_research_task` 或 `search_claims` + `query_industry_graph`。

场景 2：

```text
比较中际旭创和新易盛在光模块上的差异。
```

预期调用：

- `compare_companies`。

场景 3：

```text
这个方向可以买什么股票？
```

预期行为：

- 拒绝给买卖建议。
- 可转为“基于证据的产业链事实、风险和跟踪指标”。

### 预期结果

- 宿主 Agent 会主动调用 AIKA MCP 工具。
- 输出结构稳定。
- 证据引用不会被丢掉。
- 合规边界明确。
- Skill 可以和 MCP Server 独立迭代。

## Phase 4.1：skills 打包与导入使用

### 需求

当前项目已经实现 Python 包构建、sample knowledge pack 打包，以及 Claude Code / Codex 的 MCP 一键配置；但 `skills/aika-research/` 仍停留在仓库源码目录中，没有随 wheel/sdist 进入发行包，也没有在安装 MCP 时自动导入到宿主 Agent 的 skills 搜索路径。因此用户即使完成：

```bash
pip install aika-research-mcp
aika mcp install --host claude-code --scope user
aika mcp install --host codex --scope user
```

宿主 Agent 也只能看到 AIKA MCP tools，不一定会加载 `aika-research` Skill，更不会自动获得工具路由、证据引用和合规边界这些使用规则。

用户视角需求：

- 用户安装 PyPI 包后，不需要手动复制 `skills/aika-research/SKILL.md`。
- 一条命令可以同时完成 MCP Server 注册和 Skill 导入。
- Claude Code、Codex 等宿主 Agent 能发现并使用 `aika-research` Skill。
- 用户可以显式调用 `$aika-research`，也可以在 AI 算力产业链投研问题中触发隐式使用。

开发视角需求：

- `skills/aika-research/` 必须进入 wheel 和 sdist。
- CLI 需要能定位包内 Skill 模板，并复制到目标宿主的用户级或项目级 skills 目录。
- `aika doctor` / `aika mcp doctor` 需要检查 Skill 是否已安装、版本是否匹配、MCP 依赖是否已配置。
- Skill 安装必须幂等；默认不覆盖用户改过的 Skill，除非传入 `--force`。

### 核心逻辑

打包层：

- 在 `pyproject.toml` 中把 `skills/aika-research` 纳入构建产物。
- 推荐将源码目录保持为：

```text
skills/aika-research/
  SKILL.md
  agents/
    openai.yaml
```

- wheel 内可以映射到包内资源目录，例如：

```toml
[tool.hatch.build.targets.wheel.force-include]
"data/knowledge_packs/sample" = "aika/aika_core/bundled_sample"
"skills/aika-research" = "aika/bundled_skills/aika-research"
```

- sdist 也需要包含：

```toml
[tool.hatch.build.targets.sdist]
include = [
    "/aika",
    "/skills/aika-research",
    "/data/knowledge_packs/sample",
    "/scripts/build_sample_pack.py",
    "/README.md",
    "/pyproject.toml",
]
```

资源定位层：

- 新增包内 Skill resolver，例如：

```text
aika/aika_cli/skills.py
```

- 使用 `importlib.resources.files("aika").joinpath("bundled_skills/aika-research")` 定位 wheel 内 Skill。
- 本地源码开发模式下也允许 fallback 到仓库根目录 `skills/aika-research`。

CLI 层：

建议新增命令：

```bash
aika skill list
aika skill install --host codex --scope user
aika skill install --host codex --scope project
aika skill install --host claude-code --scope user
aika skill doctor --host codex
aika skill doctor --host claude-code
```

并扩展现有 MCP 安装命令：

```bash
aika mcp install --host codex --scope user --with-skill
aika mcp install --host claude-code --scope user --with-skill
```

安装行为：

```text
aika skill install
  -> 定位包内 aika-research Skill
  -> 解析目标 host 和 scope
  -> 计算宿主 Agent 的 skills 目录
  -> 检查目标目录是否已有 aika-research
  -> 无冲突时复制 SKILL.md 与 agents/openai.yaml
  -> 写入 AIKA 管理标记或 manifest
  -> doctor 校验 Skill 文件、MCP server 名称和依赖声明
```

目录策略：

- Codex user scope：安装到 Codex 用户级 skills 目录，具体路径由 Codex 配置或 `CODEX_HOME` 推导。
- Codex project scope：安装到当前项目的 `.codex/skills/aika-research/`。
- Claude Code user scope：安装到 Claude Code 用户级 skills 目录，具体路径由 Claude Code 配置或官方约定目录推导。
- 如果宿主没有标准 Skill 目录或当前版本不支持 Skill 导入，`doctor` 输出明确 warning，并保留手动复制路径。

覆盖策略：

- 默认：如果目标 `SKILL.md` 已存在且内容不同，停止并提示使用 `--force`。
- `--dry-run`：只打印将复制的源路径、目标路径和文件列表。
- `--force`：只覆盖 AIKA 自己管理的 `aika-research` Skill，不删除用户其他 Skill。

Skill 与 MCP 绑定：

- `agents/openai.yaml` 中继续声明依赖 `mcp:aika`。
- `aika skill doctor` 检查宿主 MCP 配置里是否存在名为 `aika` 的 server。
- 如果 Skill 已安装但 MCP 未配置，提示用户运行：

```bash
aika mcp install --host <host> --scope <scope>
```

### 验证方法

源码包内容验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv build
python -m zipfile -l dist/*.whl | grep 'aika/bundled_skills/aika-research/SKILL.md'
tar -tf dist/*.tar.gz | grep 'skills/aika-research/SKILL.md'
```

本地 wheel 安装验证：

```bash
python -m venv /tmp/aika-skill-smoke
/tmp/aika-skill-smoke/bin/pip install dist/*.whl
/tmp/aika-skill-smoke/bin/aika skill list
/tmp/aika-skill-smoke/bin/aika skill install --host codex --scope project --dry-run
/tmp/aika-skill-smoke/bin/aika skill doctor --host codex
```

Codex 项目级验证：

```bash
aika mcp install --host codex --scope project --with-skill
test -f .codex/skills/aika-research/SKILL.md
test -f .codex/skills/aika-research/agents/openai.yaml
codex
```

在 Codex 中验证：

```text
使用 $aika-research 分析液冷产业链，要求保留 citation id。
```

预期行为：

- Codex 能识别 `aika-research` Skill。
- Codex 会调用名为 `aika` 的 MCP server。
- 输出包含 Skill 约定的章节和 citation id。

Claude Code 验证：

```bash
aika mcp install --host claude-code --scope user --with-skill --dry-run
aika skill install --host claude-code --scope user
aika skill doctor --host claude-code
claude
```

在 Claude Code 中验证：

```text
用 AIKA 比较中际旭创和新易盛在光模块上的差异，必须说明证据缺口。
```

预期行为：

- Claude Code 能加载 AIKA Skill，或 doctor 能明确说明当前宿主版本不支持自动导入 Skill。
- 若 Skill 可用，宿主 Agent 优先调用 `compare_companies`，并保留证据引用。

自动化测试建议：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_aika_skill_packaging.py
```

测试断言：

- wheel 内存在 `aika/bundled_skills/aika-research/SKILL.md`。
- `aika skill list` 能列出 `aika-research`。
- `--dry-run` 不写文件。
- project scope 会生成 `.codex/skills/aika-research/SKILL.md`。
- 已存在不同内容时不覆盖，并返回可读错误。
- `--force` 只覆盖 `aika-research`，不影响其他 skills。

### 预期结果

- `pip install aika-research-mcp` 后，AIKA 的 MCP Server、sample knowledge pack 和 `aika-research` Skill 都随发行包可用。
- 用户可以用一条命令完成宿主接入：

```bash
aika mcp install --host codex --scope user --with-skill
aika mcp install --host claude-code --scope user --with-skill
```

- 宿主 Agent 不仅能调用 MCP tools，还能按 Skill 规则主动选择工具、保留 citation id、输出证据缺口，并拒绝买卖建议、目标价和收益预测。
- `aika doctor` 能同时报告 sample data、SQLite index、MCP server 和 Skill 安装状态。
- 后续新增其他宿主时，只需要新增 Skill 安装 adapter，不需要改 `aika-research/SKILL.md` 的核心内容。

## Phase 5：打包、命令行与安装体验

### 需求

让外部用户可以用最少命令安装和配置 AIKA。

用户视角需求：

```bash
pip install aika-research-mcp
aika init --sample
aika build-index
aika doctor
aika mcp
```

开发视角需求：

- Python package 元数据完整。
- console script 可用。
- sample data 可定位。
- doctor 能发现常见配置问题。

### 核心逻辑

`pyproject.toml` 需要从当前 `package = false` 的开发形态，逐步整理为可打包形态。

建议命令：

```text
aika init
aika init --sample
aika build-index
aika doctor
aika demo
aika mcp
aika search-evidence
aika search-claims
```

`aika doctor` 检查项：

- Python 版本。
- AIKA 数据目录是否存在。
- sample data 是否存在。
- SQLite index 是否存在。
- MCP server 是否可启动。
- tools 是否可注册。
- 是否误依赖 PostgreSQL。
- 是否能执行一次 sample query。

安装后目录：

```text
~/.aika/
  config.toml
  knowledge/
  indexes/
  logs/
```

### 验证方法

本地 editable 安装：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install -e .
aika --help
aika init --sample
aika build-index
aika doctor
aika demo
```

包构建验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv build
```

新环境 smoke：

```bash
python -m venv /tmp/aika-smoke-venv
/tmp/aika-smoke-venv/bin/pip install dist/*.whl
/tmp/aika-smoke-venv/bin/aika init --sample
/tmp/aika-smoke-venv/bin/aika build-index
/tmp/aika-smoke-venv/bin/aika doctor
```

### 预期结果

- 用户不需要阅读复杂 README 就能完成安装。
- `aika doctor` 能给出清楚的配置状态。
- `aika demo` 能证明本地工具链可用。
- 包可以发布到 PyPI 或先用 GitHub repo 安装。

## Phase 5.1: PyPI发布

### 需求

公开版如果要让用户真正执行：

```bash
pip install aika-research-mcp
```

就必须把发行包发布到 PyPI。`uv build` 只能证明本地可以构建 wheel/sdist，不代表 PyPI 上已经存在该包；`pip` 默认会按发行包名去 PyPI 查找并安装。

用户视角需求：

- 用户无需 clone repo，也无需拿到本地 wheel，就能直接 `pip install aika-research-mcp`。
- 测试用户可以先从 TestPyPI 安装预发布版本，验证 CLI、MCP server 和 sample 数据流程。
- 正式用户从 PyPI 安装后能运行 `aika init --sample`、`aika build-index`、`aika doctor`、`aika demo`。

开发视角需求：

- `pyproject.toml` 中 `[project].name = "aika-research-mcp"` 与 PyPI 发行包名一致。
- 发布前确认包名在 PyPI/TestPyPI 上唯一，避免与已有项目冲突。
- 构建产物同时包含 wheel 和 sdist，且通过元数据检查。
- 发布流程先 TestPyPI 后 PyPI，避免未验证包直接进入正式公共生态。
- 长期发布不要依赖手工保存 PyPI token，改用 GitHub Actions + PyPI Trusted Publishing。

### 核心逻辑

发布渠道分三层：

```text
开发安装
  -> uv build
  -> uv tool install dist/*.whl --force

预发布测试
  -> 构建 wheel/sdist
  -> twine check
  -> 上传 TestPyPI
  -> 测试用户从 TestPyPI 安装验证

正式发布
  -> 确认 TestPyPI 验证通过
  -> 上传 PyPI
  -> 用户直接 pip install aika-research-mcp
```

包名与元数据检查：

- `pyproject.toml` 的 `[project].name` 是 PyPI 上的发行包名，不等同于顶层 import 包名。
- 正式发布前检查：
  - `name = "aika-research-mcp"` 是否确定。
  - `version` 是否符合语义化版本，例如 `0.1.0`。
  - `description`、`readme`、`license`、`authors`、`requires-python` 是否完整。
  - `dependencies` 是否只包含公开版必须依赖，避免把 Web/数据库/专业版依赖强塞给普通用户。
  - `[project.scripts]` 是否暴露 `aika` 命令。
- 正式发布前最好把当前顶层 import 包名从 `aika` 迁移为更正常的 `aika` 或 `aika_research_mcp`，避免公共生态中出现顶层包名冲突和用户理解成本。

手工首发流程：

```bash
rm -rf dist/
UV_CACHE_DIR=/tmp/uv-cache uv build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*
```

TestPyPI 验证通过后，再发布正式 PyPI：

```bash
python -m twine upload dist/*
```

长期自动发布建议：

```text
push 到 main
  -> GitHub Actions 构建 wheel/sdist
  -> 上传构建 artifact
  -> 发布到 TestPyPI

push tag vX.Y.Z
  -> GitHub Actions 构建 wheel/sdist
  -> 上传构建 artifact
  -> 通过 Trusted Publishing 发布到 PyPI
```

GitHub Actions 设计要点：

- 在 PyPI 和 TestPyPI 项目页面配置 Trusted Publisher。
- workflow 使用 GitHub OIDC 短期凭证发布，不在仓库里保存长期 PyPI token。
- release job 只允许 tag 触发正式 PyPI 发布。
- main 分支只发布到 TestPyPI 或只上传 artifact，避免误发正式版本。
- 每次发布前执行 `twine check` 和最小 smoke test。

### 验证方法

本地构建验证：

```bash
rm -rf dist/
UV_CACHE_DIR=/tmp/uv-cache uv build
python -m twine check dist/*
```

本地 wheel 安装验证：

```bash
python -m venv /tmp/aika-wheel-smoke
/tmp/aika-wheel-smoke/bin/pip install dist/*.whl
/tmp/aika-wheel-smoke/bin/aika --help
/tmp/aika-wheel-smoke/bin/aika init --sample
/tmp/aika-wheel-smoke/bin/aika build-index
/tmp/aika-wheel-smoke/bin/aika doctor
/tmp/aika-wheel-smoke/bin/aika demo
```

TestPyPI 发布验证：

```bash
python -m twine upload --repository testpypi dist/*
```

TestPyPI 安装 smoke：

```bash
python -m venv /tmp/aika-testpypi-smoke
/tmp/aika-testpypi-smoke/bin/pip install \
  -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  aika-research-mcp
/tmp/aika-testpypi-smoke/bin/aika init --sample
/tmp/aika-testpypi-smoke/bin/aika build-index
/tmp/aika-testpypi-smoke/bin/aika doctor
/tmp/aika-testpypi-smoke/bin/aika demo
```

正式 PyPI 发布验证：

```bash
python -m twine upload dist/*
python -m venv /tmp/aika-pypi-smoke
/tmp/aika-pypi-smoke/bin/pip install aika-research-mcp
/tmp/aika-pypi-smoke/bin/aika init --sample
/tmp/aika-pypi-smoke/bin/aika build-index
/tmp/aika-pypi-smoke/bin/aika doctor
/tmp/aika-pypi-smoke/bin/aika demo
```

GitHub Actions 验证：

- push 到 `main` 后检查 TestPyPI workflow 成功。
- push `v0.1.0` 这类 tag 后检查 PyPI workflow 成功。
- 检查 workflow artifact 中包含 `.whl` 和 `.tar.gz`。
- 检查 PyPI/TestPyPI 项目页版本号、README、license、Python 版本、入口命令展示正确。

### 预期结果

- TestPyPI 上存在可安装的 `aika-research-mcp` 预发布包。
- 测试用户可以通过 TestPyPI 完整跑通：

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ aika-research-mcp
aika init --sample
aika build-index
aika doctor
aika demo
```

- 正式 PyPI 发布后，用户可以直接运行：

```bash
pip install aika-research-mcp
```

- 安装后 `aika` CLI 可用，公开版本地 sample 流程可运行。
- 发布产物元数据完整，wheel 是 pip 优先安装的格式，sdist 可用于源码构建。
- 长期发布路径由 GitHub Actions + Trusted Publishing 承担，降低手工 token 泄露风险。
- 该阶段完成后，AIKA 从“本地可构建项目”变成“公共 Python 生态可安装工具”。

## Phase 6：准备 Knowledge Pack 与数据合规

### 需求

公开版需要一个小体积、可再分发、可追溯的知识库快照。不能直接把全量 raw PDFs、parsed text、大体积向量和不确定授权的数据打包出去。

用户视角需求：

- 安装后有 sample 数据可试用。
- 每条证据能追溯来源。
- 数据范围和限制说清楚。

开发视角需求：

- 数据包体积可控。
- 数据构建过程可复现。
- 数据来源合规。

### 核心逻辑

数据包分为三档：

```text
sample
  几个主题，几十到几百条 evidence/claims，用于试用。

public
  合规公开资料构建的精简知识库。

private/full
  当前完整数据与 PostgreSQL/Neo4j 专业部署使用，不随公开包分发。
```

sample 包内容：

- `entities.csv`
- `relations.csv`
- `claims.csv`
- `evidence_spans.csv`
- `segment_dossiers.jsonl`
- `manifest.csv`
- `examples.jsonl`

manifest 必须包含：

- `source_title`
- `source_url`
- `published_at`
- `source_type`
- `license_or_usage_note`
- `included_fields`

### 验证方法

数据体积检查：

```bash
du -sh data/curated
du -sh path/to/aika_sample_pack
```

字段完整性检查：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m aika.aika_cli validate-data --path path/to/aika_sample_pack
```

抽样检查：

- 随机抽 20 条 evidence。
- 检查是否有 source title。
- 检查是否有 page 或 section。
- 检查是否没有长篇原文复制。
- 检查是否没有明显版权风险文本。

### 预期结果

- 公开 sample pack 可随包安装或首次初始化下载。
- 用户可以立刻试用几个核心问题。
- 数据来源和限制清晰。
- 后续可以发布更大的 public pack。

## Phase 7：保留本地 Web 开发版

### 需求

Web 版不作为第一公开入口，也不作为公网多用户产品。现有 FastAPI + React 能力主要用于本地功能测试、调试、演示和开发联调；企业私有部署可以作为后续 Full Stack 专业版方向单独规划。

用户视角需求：

- 普通插件用户不被要求启动 Web/Docker。
- 开发者可以在本地启动 Web，快速验证检索、证据卡、反馈、claim review 等功能。

开发视角需求：

- README 明确区分 Local MCP 版和 Full Stack 版。
- Web 版默认只面向本机开发测试。
- Docker Compose 只作为可选开发/专业部署路径。

### 核心逻辑

README 拆分：

```text
Quick Start: Local MCP
  -> pip install
  -> aika init --sample
  -> aika mcp

Local Web Development
  -> start FastAPI on 127.0.0.1
  -> start Vite dev server
  -> query / evidence cards / claim review / feedback smoke

Optional Full Stack / Professional Deployment
  -> docker compose up
  -> migrate postgres
  -> bootstrap retrieval
  -> web/API with separate production hardening
```

本地 Web 开发版当前不要求实现：

- 登录或 API token。
- 会话隔离。
- feedback/claim review 权限控制。
- rate limit。
- 生产环境 CORS 配置。
- 日志脱敏。

原因：

- Web 版仅用于本地测试功能，不作为公开版主入口。
- 单机本地调试场景下，上述能力会显著增加实现和维护成本，但对当前验证 MCP/Skill 核心能力帮助有限。
- 公开版安全边界主要依赖本地优先、默认不上传、显式脱敏导出。

如果未来要把 Web 版作为企业私有部署或公网服务，再新增 Production Web Hardening 阶段，补齐：

- 登录或 API token。
- 多用户会话隔离。
- feedback/claim review 权限控制。
- rate limit。
- 生产环境 CORS allowlist。
- 日志脱敏和审计日志策略。

### 验证方法

文档验证：

- README 首页优先展示 Local MCP。
- Web 被明确标注为本地开发/功能测试入口。
- 用户不会误以为必须启动 Docker 才能试用。

本地 Web smoke：

- 本地启动 API 和前端后，可以完成一次 query。
- 可以查看证据卡、claim review 页面或对应调试信息。
- 可以提交一条本地 feedback。
- README 明确提醒：不要把本地 Web 开发版直接暴露到公网。

### 预期结果

- 项目同时保留插件化和本地 Web 调试两条路径。
- 公开传播主路径更轻。
- 生产化 Web 安全需求被推迟到真正需要部署时处理。


## MVP 验收清单

第一版 MVP 通过标准：

- `aika demo` 无 Docker 成功运行。
- `aika init --sample` 能初始化本地数据。
- `aika build-index` 能生成本地索引。
- `aika doctor` 能显示通过状态。
- `aika mcp` 能启动并暴露 tools。
- 宿主 Agent 能调用 `search_evidence`。
- 宿主 Agent 能调用 `run_research_task` 或组合调用细粒度 tools。
- 输出包含 citation id 和 source 信息。
- 对无证据问题能输出证据不足。
- 对买卖建议、目标价、收益预测请求能拒绝或转换为合规研究框架。
- 用户能通过 `submit_feedback` 或 `aika feedback submit` 本地记录反馈。
- 用户能通过 `aika eval --suite smoke` 验证安装和 MCP tools 可用。
- 用户能通过 `aika feedback export --redact` 导出脱敏反馈包。
- 默认不会自动上传用户对话、API key、本地路径或完整原文。

MVP 试用问题：

- 液冷产业链有哪些上市公司，各自处于什么环节？
- 中际旭创和新易盛在光模块业务上的差异是什么？
- DeepSeek-V3 对训练算力瓶颈有什么启示？
- UCIe/Chiplet 对国产算力产业链的传导是什么？
- 当前知识库对液冷产业链缺少哪些证据？




## 推荐执行顺序

第 1 周：

- 完成 Phase 0。
- 写离线 demo 测试。
- 明确 `QA_DISABLE_POSTGRES` 行为。

第 2 周：

- 完成 Phase 1 的最小 `aika_core`。
- CSV/JSONL 后端可检索 claims/evidence/graph。

第 3 周：

- 完成 Phase 2 SQLite FTS5 索引。
- 完成 `aika init`、`aika build-index`、`aika doctor` 初版。

第 4 周：

- 完成 Phase 3 MCP Server。
- 暴露 `search_evidence`、`search_claims`、`run_research_task` 三个核心工具。

第 5 周：

- 完成 Phase 4 Skill。
- 在 Codex/爱马仕中手工验证 5 个 MVP 问题。

第 6 周：

- 完成 Phase 5/6 打包和 sample data。
- README 改为 Local MCP 优先。

## 风险与应对

### 风险 1：轻量索引召回质量不如 PostgreSQL/ParadeDB

应对：

- 第一版定位为 sample/MVP。
- 用 citation、source、claim_type、topic 过滤补足质量。
- 保留 PostgreSQL backend 给专业版。

### 风险 2：宿主 Agent 不稳定调用多个工具

应对：

- 提供 `run_research_task` 高阶工具。
- Skill 中明确复杂任务优先调用总入口。
- 细粒度 tools 作为补充。

### 风险 3：数据版权和再分发不确定

应对：

- sample pack 只放短证据片段和结构化摘要。
- 保留 source URL 和 manifest。
- 不打包原始 PDF。
- README 明确数据来源和用途限制。

### 风险 4：打包会影响当前开发型仓库

应对：

- 先以子模块式新增 `aika_core`、`aika_mcp`、`aika_cli`。
- 不急于重构现有 Web/API。
- Full Stack 专业版保持原路径。

## 完成后的项目叙事

完成本手册后，AIKA 的对外叙事应从：

> 一个需要启动数据库和网页的 AI 算力投研 Demo

转为：

> 一个可安装到 Codex、爱马仕等 Agent 工作流中的 AI 算力产业链投研增强插件，提供本地知识库、证据检索、产业链图谱、公司对比、风险审查和投研简报能力。
