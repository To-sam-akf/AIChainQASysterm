# AIKA 方向计划：从 Web Demo 转向 Agent 投研增强插件

## 1. 核心判断

AIKA 当前不应优先做成一个需要长期在线运维的公开 Web SaaS。更适合的方向是：

> 将 AIKA 设计成一套面向 Codex、爱马仕及其他 Agent 软件的投研增强工具包：MCP Server + Skill + 本地知识库快照。

这样可以避免公开测试阶段必须长期运行 Web 服务、FastAPI 服务、Docker 数据库和外部访问网关的问题。用户侧只需要安装、初始化、配置 MCP，即可让自己已有的 Agent 调用 AIKA 的投研检索、证据卡、图谱和报告生成能力。

## 2. 产品定位

AIKA 的公开版定位：

- 中文 AI 算力产业链投研增强插件。
- 面向 Agent 的证据驱动投研工具层，而不是独立聊天网站。
- 提供本地可运行的知识库、检索工具、证据卡、产业链图谱和投研任务模板。
- 由 Codex、爱马仕或其他 Agent 负责自然语言交互和最终生成，AIKA 负责垂直知识、证据约束和投研工作流。

一句话版本：

> AIKA 是一个可安装到 Agent 工作流中的 AI 算力产业链投研 MCP 工具包。

## 3. 为什么不优先开放 Web 测试

当前 Web 形态的问题：

- RAG 依赖 PostgreSQL/ParadeDB，公开测试需要数据库长期运行。
- FastAPI 后端需要持续在线，前端本身不能独立完成问答。
- Neo4j 虽然可选，但 PostgreSQL RAG 当前是核心依赖。
- 当前 `--offline` 路径仍会初始化 PostgreSQL，离线 demo 不是真正无服务可跑。
- 公开 Web 还需要认证、限流、会话隔离、权限控制、日志审计和数据合规。
- 运维成本会提前压过项目的核心价值验证。

因此短期内应避免把主要精力放在公网 Web 服务上。

## 4. 目标架构

推荐架构：

```text
Codex / 爱马仕 / Claude Desktop / 其他 Agent 软件
        |
        | MCP tool calls
        v
AIKA MCP Server
        |
        | 本地检索与投研逻辑
        v
SQLite / DuckDB / JSONL / CSV 知识库快照
```

配套 Skill：

```text
AIKA Skill
        |
        | 指导 Agent 如何拆解投研问题、调用 MCP 工具、约束输出格式
        v
证据驱动回答 / 公司对比 / 投研简报 / 风险审查 / 证据缺口审计
```

## 5. 分层设计

### 5.1 轻量公开版

目标：用户无需 Docker、无需网页、无需自建数据库即可试用。

使用：

- `data/curated/entities.csv`
- `data/curated/relations.csv`
- `data/curated/claims.csv`
- `data/curated/evidence_spans.csv`
- `data/curated/segment_dossiers.jsonl`
- 可选的少量示例原文 chunk

后端建议：

- SQLite FTS5：适合单文件、易安装、全文检索够用。
- DuckDB：适合结构化分析、CSV/Parquet 查询。
- JSONL/CSV：保留最简单 fallback。

公开版先不强依赖：

- PostgreSQL
- ParadeDB
- pgvector
- Neo4j
- React Web
- Docker Compose

### 5.2 完整专业版

目标：保留当前项目已有的深度能力，供自己或企业私有部署。

继续支持：

- PostgreSQL/ParadeDB BM25
- pgvector 语义检索
- Neo4j 图谱增强
- FastAPI + React 工作台
- PDF 解析、知识抽取、图谱构建、评测流水线
- Claim review、feedback、eval dashboard

专业版可以作为后续商业化或私有部署形态，不作为第一阶段公开试用主路径。

## 6. MCP 工具设计

第一阶段建议暴露以下 MCP tools：

### 6.1 证据检索

`search_evidence`

用途：从 evidence spans、claims、dossiers 或轻量全文索引中检索证据。

参数：

- `query`
- `companies`
- `topics`
- `source_types`
- `top_k`

返回：

- `citation_id`
- `evidence`
- `source_title`
- `page`
- `section`
- `company`
- `topic`
- `confidence`

### 6.2 Claim 检索

`search_claims`

用途：检索结构化投研 Claim。

参数：

- `query`
- `company`
- `topic`
- `claim_type`
- `exposure_level`
- `top_k`

返回：

- `claim_id`
- `claim_text`
- `companies`
- `topic`
- `claim_type`
- `evidence_span`
- `source`
- `confidence`

### 6.3 公司画像

`get_company_profile`

用途：返回公司在 AI 算力产业链中的位置、产品、技术、指标、风险和证据。

参数：

- `company`
- `topic`

返回：

- 公司基础信息
- 产业链环节
- 产品/技术关系
- 核心证据
- 风险证据
- 证据缺口

### 6.4 公司对比

`compare_companies`

用途：围绕某个主题对比多家公司。

参数：

- `companies`
- `topic`

返回：

- 对比表
- 共同驱动
- 差异点
- 指标证据
- 风险差异
- citation ids

### 6.5 产业链图谱查询

`query_industry_graph`

用途：查询公司、技术、产品、风险、政策、标准之间的结构化关系。

参数：

- `company`
- `technology`
- `relation_type`
- `limit`

返回：

- nodes
- edges
- relation rows
- evidence

### 6.6 投研简报生成

`build_research_brief`

用途：基于检索结果生成结构化投研简报草稿。

参数：

- `topic`
- `companies`
- `depth`

返回：

- 核心判断
- 技术机理
- 产业传导
- 公司排序
- 风险与反证
- 证据索引
- 证据缺口

注意：第一版可由工具返回结构化材料，最终自然语言成文交给宿主 Agent。

### 6.7 证据缺口审计

`audit_evidence_gaps`

用途：判断某个主题或公司当前知识库缺少哪些证据。

参数：

- `topic`
- `company`

返回：

- 缺少的公司覆盖
- 缺少的指标
- 缺少的风险证据
- 建议补充来源

## 7. Skill 设计

Skill 不负责存储和检索数据库，主要负责指导 Agent 如何使用 AIKA MCP 工具。

Skill 核心规则：

- 回答投研问题前，必须先调用 `search_evidence` 或 `search_claims`。
- 涉及公司对比时，必须调用 `compare_companies` 或组合调用 `get_company_profile`。
- 涉及产业链关系时，必须调用 `query_industry_graph`。
- 结论必须绑定 citation id。
- 找不到证据时，必须明确写“当前证据不足”。
- 禁止输出买卖建议、目标价、收益预测。
- 输出结构优先使用：
  - 核心判断
  - 证据
  - 产业链传导
  - 公司差异
  - 风险与反证
  - 证据缺口

Skill 可以命名为：

- `aika-research`
- `ai-compute-research`
- `evidence-driven-equity-research`

## 8. 包装与安装体验

目标安装路径：

```bash
pip install aika-research-mcp
aika init
aika mcp
```

用户配置 MCP：

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

可选初始化：

```bash
aika init --sample
aika build-index
aika doctor
```

命令设计：

- `aika init`：初始化本地数据目录和配置文件。
- `aika build-index`：基于 CSV/JSONL 构建 SQLite/DuckDB 索引。
- `aika mcp`：启动 MCP Server。
- `aika doctor`：检查数据、索引、MCP 配置和依赖。
- `aika demo`：无需数据库的本地 smoke demo。

## 9. 数据策略

### 9.1 公开数据包

公开包应控制体积，目标几十 MB 以内。

包含：

- 精简 curated graph。
- 精简 claims。
- 精简 evidence spans。
- 几个 segment dossiers。
- 示例问题和示例输出。
- 数据来源 manifest。

不包含：

- 原始 PDF 大文件。
- 全量 parsed text。
- 私有抽取中间文件。
- 大体积 semantic vectors。
- 不确定授权的数据源。

### 9.2 数据合规

公开前需要检查：

- PDF 和研报来源是否允许再分发。
- 是否只分发结构化摘要和短证据片段。
- 是否保留 source title、source URL、page、published_at。
- 是否有免责声明。

推荐公开版先只放：

- 公开年报来源信息。
- 权威白皮书/标准/论文来源信息。
- 少量合规证据片段。
- 可复现的数据构建脚本。

## 10. 工程改造计划

### Phase 0：修正当前 demo 阻塞

目标：当前项目至少拥有一个真正无需数据库的 demo。

任务：

- 增加 `QA_DISABLE_POSTGRES=true` 或 `--offline` 真正跳过 PostgreSQL 初始化。
- 让 `main.py demo --offline` 在无 Docker、无 Postgres、无 LLM key 时可以跑通。
- 增加 CI smoke：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py demo --offline --task-dir /tmp/aiqasys-demo-tasks
```

验收：

- 新机器 clone 后，无数据库也能看到样例问题、证据卡和简短答案。

### Phase 1：抽出轻量核心库

目标：把当前投研逻辑从 Web/API/数据库依赖中解耦。

任务：

- 新建或整理 `src/aika_core/`。
- 抽出 evidence card 数据模型。
- 抽出 claims 检索逻辑。
- 抽出 graph CSV 查询逻辑。
- 抽出 company profile / company compare / evidence gap audit 的纯函数。
- 保留 PostgreSQL 实现为可选 backend。

验收：

- 单元测试可用纯 CSV/JSONL 后端完成核心检索。
- 不启动 FastAPI、不启动 Postgres，也能运行核心投研工具。

### Phase 2：实现轻量本地索引

目标：替代公开版对 PostgreSQL/ParadeDB 的强依赖。

任务：

- 选择 SQLite FTS5 或 DuckDB 作为公开版默认索引。
- 将 `claims.csv`、`evidence_spans.csv`、`segment_dossiers.jsonl` 导入本地索引。
- 提供 `aika build-index`。
- 提供增量重建或覆盖重建。

验收：

- `search_evidence` 可在本地索引上完成 top-k 检索。
- 无 Docker 依赖。

### Phase 3：实现 MCP Server

目标：让 Codex、爱马仕等 Agent 可以调用 AIKA 工具。

任务：

- 新建 MCP server 入口。
- 暴露第一批 tools：
  - `search_evidence`
  - `search_claims`
  - `get_company_profile`
  - `compare_companies`
  - `query_industry_graph`
  - `build_research_brief`
  - `audit_evidence_gaps`
- 为每个 tool 写 schema、参数校验和返回示例。
- 增加 `aika mcp` 命令。

验收：

- 在本地 MCP 客户端中可以列出工具并成功调用。
- 工具返回结构化 JSON，且包含 citation 信息。

### Phase 4：编写 Codex/Agent Skill

目标：让宿主 Agent 知道如何正确使用 AIKA 工具。

任务：

- 编写 `SKILL.md`。
- 定义投研问题处理流程。
- 定义证据引用规则。
- 定义禁止事项：买卖建议、目标价、收益预测。
- 提供 5 个示例工作流。

验收：

- 用户安装 Skill 后，Agent 会主动调用 AIKA MCP 工具，而不是直接凭空回答。

### Phase 5：打包与发布

目标：实现命令行安装配置。

任务：

- 整理 `pyproject.toml` package 配置。
- 增加 console scripts：
  - `aika`
- 准备 sample data 包。
- 准备 README 快速开始。
- 准备 `aika doctor`。
- 准备版本号和 changelog。

验收：

```bash
pip install aika-research-mcp
aika init --sample
aika build-index
aika doctor
aika mcp
```

以上流程可以在新环境中完成。

### Phase 6：保留 Web 专业版

目标：Web 不作为公开试用第一入口，但保留为展示和私有部署形态。

任务：

- README 中明确区分：
  - Local MCP 插件版
  - Full Stack Web 专业版
- Web 版补认证、限流、会话隔离、权限控制。
- Docker Compose 只服务专业版。

验收：

- 使用者不会误以为必须启动 Web 和 Docker 才能体验 AIKA。

## 11. 优先级

最高优先级：

1. 真正离线 demo。
2. 抽出 CSV/JSONL 轻量检索核心。
3. MCP Server 第一版。
4. Skill 第一版。
5. 最小公开知识库快照。

暂缓：

- 公网 Web 登录系统。
- 多用户 SaaS。
- Neo4j 强依赖。
- pgvector 语义检索公开版。
- 大规模 PDF 自动下载和重建流程开放给普通用户。

## 12. 第一版 MVP 范围

第一版 MVP 只需要回答这些问题：

- “液冷产业链有哪些上市公司，各自处于什么环节？”
- “中际旭创和新易盛在光模块业务上的差异是什么？”
- “DeepSeek-V3 对训练算力瓶颈有什么启示？”
- “UCIe/Chiplet 对国产算力产业链的传导是什么？”
- “当前知识库对某个主题缺少哪些证据？”

MVP 成功标准：

- 用户无需 Docker 可以安装。
- 用户无需访问 AIKA 网页。
- 用户可以在 Codex/爱马仕中调用 AIKA tools。
- 每个回答有 citation id 和来源信息。
- 没有证据时不会硬编。
- 输出不包含买卖建议、目标价和收益预测。

## 13. 最终形态

长期可以形成三层产品：

1. **AIKA Skill**
   - 投研方法论、输出规范、证据约束。

2. **AIKA MCP**
   - 可被任何 Agent 调用的工具层。

3. **AIKA Knowledge Pack**
   - AI 算力产业链本地知识库快照。

专业部署另行保留：

4. **AIKA Full Stack**
   - PostgreSQL/ParadeDB + Neo4j + FastAPI + React 工作台。

这样项目的核心价值会更清晰：

> AIKA 不只是一个投研聊天 Demo，而是一套可嵌入 Agent 生态的证据驱动投研基础设施。
