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
- `src/cli.py` 的 `build_engine(args)` 在 `effective_offline(args)` 为 true 时设置该变量。
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
src/aika_core/
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
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.aika_cli init --sample
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.aika_cli build-index
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.aika_cli doctor
```

检索验证：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.aika_cli search-evidence "液冷产业链" --top-k 5
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.aika_cli search-claims "中际旭创 光模块" --top-k 5
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
src/aika_mcp/
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
src/aika_mcp/
  installer.py
  doctor.py
  host_configs.py
```

职责划分：

- `host_configs.py`：生成不同宿主的配置 JSON。
- `installer.py`：执行宿主 CLI 写入、冲突检测、`--force` 覆盖。
- `doctor.py`：执行依赖、索引、MCP server、宿主配置诊断。
- `src/aika_cli.py`：只负责参数解析和调用上述模块。

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
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.aika_cli validate-data --path path/to/aika_sample_pack
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

## Phase 7：保留 Full Stack Web 专业版

### 需求

Web 版不作为第一公开入口，但现有 FastAPI + React + PostgreSQL + Neo4j 能力仍有价值，适合自用、演示、企业私有部署。

用户视角需求：

- 普通插件用户不被要求启动 Web/Docker。
- 专业部署用户仍有完整工作台。

开发视角需求：

- README 明确区分 Local MCP 版和 Full Stack 版。
- Web 版补齐基础安全。
- Docker Compose 只服务专业版路径。

### 核心逻辑

README 拆分：

```text
Quick Start: Local MCP
  -> pip install
  -> aika init --sample
  -> aika mcp

Full Stack Deployment
  -> docker compose up
  -> migrate postgres
  -> bootstrap retrieval
  -> web/API
```

Web 专业版必须补：

- 登录或 API token。
- 会话隔离。
- feedback/claim review 权限控制。
- rate limit。
- 生产环境 CORS 配置。
- 日志脱敏。

### 验证方法

文档验证：

- README 首页优先展示 Local MCP。
- Full Stack 被明确标注为专业部署。
- 用户不会误以为必须启动 Docker 才能试用。

安全 smoke：

- 未认证用户不能 delete conversation。
- 未认证用户不能 review claim。
- guest 用户只能读 demo 数据或创建自己的临时会话。

### 预期结果

- 项目同时保留插件化和完整 Web 两条路径。
- 公开传播主路径更轻。
- 专业部署能力不被丢弃。

## Phase 8：反馈与测评闭环

### 需求

MCP/Skill 安装后，用户的使用发生在自己的 Agent 软件和本地知识库中。AIKA 不能默认收集用户对话，也不能假设有公网服务接收反馈。因此反馈与测评系统要采用“本地优先、显式授权、可脱敏导出”的设计。

用户视角需求：

- 用户可以在 Agent 中直接提交一次回答的反馈。
- 用户可以本地运行 smoke/eval，判断 AIKA 是否安装正确、检索是否有效、证据引用是否可靠。
- 用户可以选择导出脱敏反馈包，用于 GitHub issue、邮件或后续远程上传。
- 默认不上传完整对话、不上传 API key、不上传本地私有路径、不上传完整研报原文。

开发视角需求：

- 反馈数据要结构化，便于后续分析。
- 评测结果要可复现，包含版本、知识包、工具调用摘要和指标。
- 出错时能快速定位是安装问题、索引问题、MCP 问题、知识包问题还是 Agent 调用问题。
- 反馈闭环要能同时服务公开版 MCP 和未来 Full Stack 专业版。

### 核心逻辑

反馈闭环分三层：

```text
本地反馈 JSONL
  -> 本地评测 eval_runs
  -> 脱敏导出 feedback bundle
```

#### 8.1 本地反馈

新增本地反馈存储：

```text
~/.aika/
  feedback/
    feedback.jsonl
  eval_runs/
    eval_*.json
  exports/
    aika_feedback_bundle_*.zip
```

MCP tool：

```text
submit_feedback
list_feedback
export_feedback_bundle
```

CLI：

```bash
aika feedback submit
aika feedback list
aika feedback export --redact
```

反馈字段建议：

```json
{
  "feedback_id": "fb_xxx",
  "created_at": "2026-06-17T17:30:00",
  "task_id": "task_xxx",
  "tool_name": "run_research_task",
  "question": "液冷产业链有哪些上市公司？",
  "answer_hash": "sha256_xxx",
  "rating": 3,
  "helpful": true,
  "evidence_supported": false,
  "wrong_citation": false,
  "missing_evidence": "缺少英维克液冷订单证据",
  "note": "结论可用，但证据不够强",
  "citation_ids": ["E1", "E2"],
  "aika_version": "0.1.0",
  "knowledge_pack": "sample-2026-06",
  "consent_to_export": false
}
```

默认策略：

- 默认只写本地。
- 默认不上传。
- 默认不保存完整 answer，只保存 `answer_hash`、citation ids 和用户主动输入的 note。
- 如果用户选择保存完整对话，需要显式参数：`--include-conversation`。

#### 8.2 本地测评

新增 CLI：

```bash
aika eval --suite smoke
aika eval --suite retrieval
aika eval --suite agent
```

MCP tool：

```text
run_eval
aika_status
aika_doctor
```

评测 suite 设计：

- `smoke`：检查 MCP tools 是否可启动、sample query 是否返回结果。
- `retrieval`：检查 `search_evidence`、`search_claims` 是否能命中 benchmark 的 citation/claim。
- `agent`：检查 `run_research_task` 是否能生成结构化结果、证据卡、缺口提示。

核心指标：

- `tool_success_rate`
- `retrieval_hit_rate`
- `citation_validity`
- `unsupported_claim_rate`
- `missing_evidence_rate`
- `empty_result_rate`
- `latency_ms_p50`
- `latency_ms_p95`
- `agent_task_completion_rate`

评测输出：

```json
{
  "run_id": "eval_20260617_xxx",
  "created_at": "2026-06-17T17:30:00",
  "suite": "smoke",
  "aika_version": "0.1.0",
  "knowledge_pack": "sample-2026-06",
  "summary": {
    "cases": 5,
    "tool_success_rate": 1.0,
    "retrieval_hit_rate": 0.8,
    "citation_validity": 1.0,
    "empty_result_rate": 0.0
  },
  "failures": []
}
```

#### 8.3 脱敏导出

新增命令：

```bash
aika feedback export --redact --since 7d --output aika_feedback_bundle.zip
aika issue-template --from aika_feedback_bundle.zip
```

脱敏包包含：

- AIKA 版本。
- Python 版本。
- OS 信息。
- knowledge pack 名称和 hash。
- MCP tool 列表。
- `aika doctor` 结果。
- eval summary。
- feedback note。
- 错误栈。
- citation ids。

脱敏包默认不包含：

- API key。
- 完整用户对话。
- 完整模型回答。
- 本地绝对路径。
- 原始 PDF。
- 长篇 evidence 原文。

如未来增加远程反馈服务，必须显式确认：

```bash
aika feedback upload --redact --confirm
```

#### 8.4 GitHub issue 模板

生成 issue 内容：

```bash
aika issue-template --latest
```

模板结构：

- 问题类型：安装 / MCP 启动 / 检索为空 / 引用错误 / 回答不稳定 / 数据缺失。
- 复现步骤。
- 期望结果。
- 实际结果。
- `aika doctor` 摘要。
- eval run id。
- feedback bundle 是否已附加。

### 验证方法

命令验证：

```bash
aika doctor
aika eval --suite smoke
aika feedback submit --rating 3 --note "引用不够强" --citation-ids E1,E2
aika feedback list
aika feedback export --redact --output /tmp/aika_feedback_bundle.zip
aika issue-template --from /tmp/aika_feedback_bundle.zip
```

MCP 验证：

- 宿主 Agent 能调用 `aika_status`。
- 宿主 Agent 能调用 `aika_doctor`。
- 宿主 Agent 能调用 `run_eval`。
- 宿主 Agent 能调用 `submit_feedback`。
- 宿主 Agent 能调用 `export_feedback_bundle`。

单元测试建议：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_aika_feedback.py tests/test_aika_eval_cli.py
```

测试断言：

- `submit_feedback` 会追加 JSONL。
- feedback id 唯一。
- 空 note、非法 rating、非法 citation ids 会返回明确错误。
- `aika eval --suite smoke` 生成 eval run 文件。
- `export --redact` 不包含 API key、本地绝对路径、完整 answer。
- issue template 能从 eval/feedback 生成 Markdown。

人工验收场景：

场景 1：用户觉得回答证据不足。

```text
这次 AIKA 回答证据不充分，帮我提交反馈：缺少英维克液冷订单证据。
```

预期：

- Agent 调用 `submit_feedback`。
- 本地 `feedback.jsonl` 增加一条记录。
- 不发生远程上传。

场景 2：用户说 AIKA 检索不到结果。

```text
帮我检查 AIKA 为什么检索不到结果。
```

预期：

- Agent 调用 `aika_doctor` 和 `run_eval`。
- 返回是索引缺失、数据包缺失、MCP 未启动还是 query 无命中。

场景 3：用户要向项目维护者反馈 bug。

```text
帮我导出一个脱敏反馈包。
```

预期：

- Agent 调用 `export_feedback_bundle`。
- 生成 zip。
- 明确提示用户自行检查后再提交。

### 预期结果

- 用户可以在不离开 Agent 的情况下提交反馈。
- 所有反馈默认保存在本地，保护用户隐私。
- 开发者可以通过脱敏反馈包复现问题。
- 本地评测可以快速判断 MCP/Skill/知识包是否正常。
- 公开版具备最小质量闭环，不依赖公网遥测。
- 未来如接入远程反馈服务，也可以沿用同一套本地 JSONL/eval/export 格式。

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
