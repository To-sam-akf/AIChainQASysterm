# AIQASYS Agent 化实现文档

## 1. 文档目标

本文档用于将当前 AI 算力产业链智能问答系统，从“证据增强问答/投研 GraphRAG 系统”包装并升级为一个真正的 Agent 项目。

当前系统已经具备知识图谱、RAG、Claim/Dossier、证据卡片、连续追问、前端工作台和初步 Agent Runner。下一阶段的目标不是重写系统，而是在现有能力之上补齐动态任务规划、工具执行协议、任务状态管理、证据验证、冲突识别、任务评测和工程化交付能力。

最终项目定位：

> AIQASYS 是一个面向 AI 算力产业链的证据驱动投研 Agent，基于 Claim/Dossier、知识图谱、RAG 和可追踪工具调用，自动完成产业链问答、公司对比、风险审查和投研简报生成。

## 2. 当前系统能力

### 2.1 离线知识构建

系统已经支持从年报、研报、行业白皮书、技术规范、benchmark、论文和模型技术报告中构建知识资产。

当前链路包括：

1. PDF 下载与清单管理。
2. PDF 解析、文本清洗、chunk 切分。
3. LLM 抽取实体、关系、指标、风险和技术机理。
4. 构建 verified 图谱 CSV。
5. 构建 curated 专业图谱、Claim、EvidenceSpan 和 Segment Dossier。
6. 构建本地 BM25 RAG 索引。
7. 可选构建 embedding 语义索引。
8. 可选导入 Neo4j。

当前主要产物：

- `data/curated/entities.csv`：实体表。
- `data/curated/relations.csv`：关系表。
- `data/curated/claims.csv`：投研原子判断。
- `data/curated/evidence_spans.csv`：可引用证据片段。
- `data/curated/segment_dossiers.jsonl`：主题级产业链摘要。
- `data/rag/documents.jsonl`：本地 RAG 文档块。
- `data/semantic_index/`：可选 embedding 语义索引。

当前数据规模约为：

- 6881 个实体。
- 10521 条关系。
- 11591 条 Claim。
- 11591 条证据片段。
- 9 个主题 dossier。
- 4858 个 RAG chunk。

### 2.2 在线问答能力

当前问答系统不是让大模型直接回答，而是先检索证据，再基于证据生成答案。

已具备能力：

- CSV/Neo4j 图谱检索。
- 本地 BM25 RAG 检索。
- Claim/Dossier 投研证据检索。
- 可选 embedding 语义召回。
- 问题规划和答案类型识别。
- 连续追问改写。
- 证据卡片构造。
- citation id 引用编号，例如 `E1`、`E2`。
- 回答子图生成。
- 研究报告、公司对比表、风险清单、证据缺口清单生成。
- 空证据问题降级回答。
- 禁止生成买卖建议、目标价或收益预测。

主要入口：

- `src/qa_engine.py`：问答编排核心。
- `src/professional_qa.py`：专业问答、证据卡片和答案格式。
- `src/research_agent.py`：结构化投研输出生成。
- `src/api.py`：FastAPI 后端接口。
- `web/src/App.tsx`：React 工作台。

### 2.3 前端工作台能力

React 工作台已具备：

- 智能问答。
- 流式回答。
- 对话历史保存、恢复、重命名、删除和导出。
- 证据详情抽屉。
- 投研产物页签。
- Claim 修正面板。
- 数据概览。
- 产业链图谱视图。
- Agent 任务页面。
- Agent 任务导出 Markdown/JSON。

### 2.4 当前 Agent 雏形

当前系统已有初步 Agent 能力，但仍偏规则式工作流。

已有模块：

- `src/agent_runner.py`
  - 实现固定四阶段流程：`plan -> retrieve -> supplement -> verify_answer`。
  - 支持 agent trace、tool calls、timings、diagnostics。

- `src/agent_tools.py`
  - 封装问题改写、问题规划、图谱查询、RAG 检索、Claim 检索、语义检索、证据排序和答案验证。

- `src/agents/models.py`
  - 定义 `AgentToolSpec`、`AgentStep`、`AgentTask`、`AgentTaskSummary`。

- `src/agents/store.py`
  - 使用 JSONL 保存 Agent 任务快照。

- `src/agents/tools.py`
  - 定义 Tool Registry 元数据。

- `src/agents/research_agent.py`
  - 当前只支持 `research_brief` 任务。
  - 本质是把研究目标改写成投研简报问题，再调用 QA 链路。

当前状态判断：

- 已经具备 Agent 项目的骨架。
- 已经有可追踪工具调用和任务记录。
- 但还不是完整动态 Agent。
- 当前更准确的定位是“证据约束投研 QA + 第一阶段规则 Agent”。

## 3. Agent 化目标

### 3.1 产品目标

将系统从“问答入口”升级为“任务入口”。

用户不仅可以问：

```text
液冷产业链有哪些上市公司？
```

还可以发起任务：

```text
生成一份液冷产业链投研简报，覆盖技术机理、公司排序、领先指标、风险反证和证据缺口。
```

Agent 应该能够：

1. 理解用户目标。
2. 自动拆解子任务。
3. 根据子任务选择工具。
4. 多轮检索和补证。
5. 判断证据是否足够。
6. 识别证据缺口和冲突。
7. 生成结构化研究产物。
8. 保存完整执行轨迹。
9. 支持人工审校后继续执行。

### 3.2 技术目标

升级后的 Agent 项目应具备：

- 动态任务规划。
- 标准工具协议。
- 可恢复 Agent State。
- 证据池和证据选择机制。
- 自动补证循环。
- 停止条件和预算控制。
- 事实一致性验证。
- 冲突证据识别。
- 任务级评测。
- 前端 Agent 工作台。
- CLI 和 Docker 化交付。

## 4. 需要新增的 Agent 功能

### 4.1 多任务类型

当前只支持 `research_brief`。后续建议扩展为：

| 任务类型 | 说明 | 输出 |
| --- | --- | --- |
| `qa` | 单轮或多轮证据问答 | 答案、证据卡、子图 |
| `research_brief` | 主题投研简报 | 报告、公司表、风险、缺口 |
| `company_profile` | 公司产业链画像 | 业务卡位、产品、指标、风险 |
| `company_compare` | 多家公司对比 | 差异矩阵、共同驱动、风险差异 |
| `risk_review` | 风险和反证审查 | 风险清单、反证、跟踪指标 |
| `evidence_gap_audit` | 证据缺口检查 | 缺口列表、建议数据源 |
| `monitor_topic` | 主题持续跟踪 | 指标体系、后续更新计划 |

第一阶段优先实现：

1. `company_compare`
2. `company_profile`
3. `risk_review`
4. `evidence_gap_audit`

### 4.2 动态任务规划器

新增 `TaskPlanner`，将用户目标拆成标准子任务。

示例：

```text
用户目标：生成液冷产业链投研简报

子任务：
1. 识别液冷技术定义和需求驱动。
2. 检索液冷产业链环节。
3. 检索核心/直接/间接敞口公司。
4. 检索订单、收入、产能、客户导入等领先指标。
5. 检索风险、反证和不确定性。
6. 判断证据缺口。
7. 生成报告。
8. 验证报告中的引用、公司和指标是否有证据支持。
```

规划器输出结构建议：

```json
{
  "task_type": "research_brief",
  "goal": "液冷产业链投研简报",
  "subtasks": [
    {
      "id": "s1",
      "type": "topic_mechanism",
      "query": "液冷 技术机理 需求驱动",
      "required_tools": ["search_segment_dossiers", "search_research_claims", "search_rag"],
      "success_criteria": ["has_mechanism_evidence"]
    }
  ],
  "budgets": {
    "max_steps": 8,
    "max_llm_calls": 4,
    "max_retrieval_rounds": 3
  }
}
```

### 4.3 动态工具选择

当前 AgentRunner 的工具调用顺序固定。后续需要支持按任务动态选择工具。

新增 `ToolExecutor`：

- 接收工具名和输入。
- 执行真实工具。
- 记录开始时间、结束时间、耗时、结果数量、错误。
- 支持超时。
- 支持缓存。
- 支持失败降级。

工具协议：

```python
class AgentTool:
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    timeout: float
    cost_level: str
    safety_level: str
    requires_llm: bool
    cacheable: bool

    def run(self, payload: dict) -> dict:
        ...
```

首批可执行工具：

- `contextualize_question`
- `plan_question`
- `query_graph`
- `search_rag`
- `search_research_claims`
- `search_segment_dossiers`
- `search_semantic_index`
- `rank_evidence`
- `verify_answer_support`
- `detect_evidence_gaps`
- `build_research_outputs`
- `export_report`

### 4.4 Agent State / Workspace

新增完整 `AgentState`，用于支持长任务、恢复、审校和前端展示。

建议字段：

```python
@dataclass
class AgentState:
    task_id: str
    task_type: str
    user_goal: str
    status: str
    plan: dict
    current_step: int
    subtasks: list[dict]
    tool_calls: list[dict]
    evidence_pool: list[dict]
    selected_evidence: list[dict]
    draft_findings: list[dict]
    conflicts: list[dict]
    evidence_gaps: list[dict]
    verification: dict
    final_outputs: dict
    errors: list[str]
    created_at: str
    updated_at: str
```

需要支持：

- 创建任务。
- 保存任务快照。
- 恢复任务。
- 重跑失败步骤。
- 人工修改 Claim 后继续执行。
- 导出最终报告。

### 4.5 自动补证循环

当前已有 supplement 阶段，但逻辑仍较固定。后续需要改为循环式补证。

循环步骤：

1. 根据任务计划检索第一轮证据。
2. 构建 evidence pool。
3. 检查证据覆盖：
   - 是否有技术机理。
   - 是否有公司敞口。
   - 是否有指标。
   - 是否有风险。
   - 是否有反证。
   - 是否覆盖所有目标公司。
4. 若缺证据，则生成补证 query。
5. 再次调用工具。
6. 达到证据充分、缺口不可弥补、最大步数或预算上限后停止。

停止条件：

- `evidence_sufficient`
- `evidence_gap_unrecoverable`
- `max_steps_reached`
- `max_llm_calls_reached`
- `timeout_reached`
- `user_cancelled`

### 4.6 证据验证升级

当前已有 citation、公司覆盖和 unsupported terms 检查。后续需要增强为投研级验证。

新增检查：

- 引用有效性：答案中的 `[E1]` 必须存在。
- 公司覆盖：公司对比问题中每家公司必须有证据。
- 数值校验：年份、百分比、金额、数量必须能在证据中找到。
- 指标校验：订单、收入、毛利率、产能、客户导入等指标必须有来源。
- 敞口校验：`core/direct/indirect/mentioned` 必须有依据。
- 风险校验：风险类问题必须至少包含风险或反证证据。
- 时点校验：不同年份报告不能混用成当前事实。
- 术语校验：关键技术术语不能脱离证据包生成。

验证结果结构：

```json
{
  "status": "pass|warn|fail",
  "checks": {
    "citation_validity": {},
    "company_coverage": {},
    "numeric_support": {},
    "metric_support": {},
    "risk_support": {},
    "unsupported_terms": []
  }
}
```

### 4.7 冲突证据识别

当前系统对冲突和反证只是预留字段，需要补齐。

需要识别：

- 券商乐观判断 vs 年报风险披露。
- 旧报告 vs 新报告。
- 技术路线 A vs 技术路线 B。
- 公司宣传 vs 财务兑现不足。
- 需求高增长叙事 vs 价格压力或客户集中风险。

新增字段：

- `conflict_group_id`
- `conflict_type`
- `claim_a`
- `claim_b`
- `resolution`
- `confidence`

前端展示：

- 冲突证据列表。
- 每组冲突的双方证据。
- Agent 对冲突的保守解释。
- 人工标记冲突是否成立。

### 4.8 GraphRAG / DRIFT 检索升级

当前已经有图谱、RAG、Claim/Dossier 和 embedding，但还不是真正的动态 GraphRAG。

下一步实现：

1. Query Router
   - “哪些公司/谁受益”走公司敞口排序。
   - “为什么/趋势/瓶颈”走 global dossier + claim。
   - “公司对比”走公司 claim bundle + 指标/风险矩阵。
   - “订单/业绩/指标”只走指标 Claim。
   - “继续说/它们/上述”先做历史问题改写。

2. Global Search
   - 宽问题先召回主题 Dossier 和高层 Claim。

3. Local Search
   - 围绕公司、指标、风险、产业链环节做局部检索。

4. DRIFT 流程
   - 宽问题。
   - 全局摘要。
   - 自动拆子问题。
   - 局部证据检索。
   - 综合结论。

5. 多跳路径
   - 需求驱动 -> 技术瓶颈 -> 产业链环节 -> 公司敞口 -> 指标验证 -> 风险反证。

6. 三阶段排序
   - BM25 高召回。
   - dense embedding 召回。
   - cross-encoder 或 LLM rerank 精排。

## 5. 模块改造设计

### 5.1 目录结构建议

建议逐步整理为：

```text
src/
  agents/
    base.py
    models.py
    planner.py
    executor.py
    state.py
    store.py
    tools.py
    qa_agent.py
    research_agent.py
    company_agent.py
    risk_agent.py
    verification.py
    evaluation.py
  qa_engine.py
  professional_qa.py
  research_agent.py
  research_claims.py
  rag_index.py
  semantic_index.py
  frontend_data.py
  api.py
```

说明：

- `src/agents/base.py`：Agent 基类。
- `src/agents/planner.py`：动态任务规划。
- `src/agents/executor.py`：工具执行器。
- `src/agents/state.py`：任务状态。
- `src/agents/verification.py`：事实一致性和证据验证。
- `src/agents/evaluation.py`：任务级评测。
- `src/agents/company_agent.py`：公司画像和公司对比任务。
- `src/agents/risk_agent.py`：风险审查任务。

### 5.2 BaseAgent

```python
class BaseAgent:
    def run(self, goal: str, **kwargs) -> dict:
        raise NotImplementedError

    def plan(self, state: AgentState) -> AgentState:
        raise NotImplementedError

    def step(self, state: AgentState) -> AgentState:
        raise NotImplementedError

    def should_stop(self, state: AgentState) -> bool:
        raise NotImplementedError

    def finalize(self, state: AgentState) -> AgentState:
        raise NotImplementedError
```

### 5.3 ResearchAgent

目标：

- 不再只是包装 QA。
- 能围绕主题自动拆分研究任务。
- 能多轮补证。
- 能生成完整投研报告。

输入：

```json
{
  "task_type": "research_brief",
  "goal": "液冷产业链投研简报",
  "constraints": {
    "core_companies_only": true,
    "include_risks": true,
    "include_evidence_gaps": true
  }
}
```

输出：

```json
{
  "task_id": "...",
  "status": "completed",
  "report": {},
  "company_table": {},
  "risk_checklist": [],
  "evidence_gaps": [],
  "citations": [],
  "agent_trace": []
}
```

### 5.4 CompanyAgent

支持：

- `company_profile`
- `company_compare`

公司画像输出：

- 业务卡位。
- 产品和技术。
- 产业链环节。
- 客户/订单证据。
- 财务或经营指标。
- 风险。
- 证据缺口。

公司对比输出：

- 差异矩阵。
- 共同驱动。
- 分歧点。
- 指标验证。
- 风险差异。
- 证据边界。

### 5.5 RiskAgent

支持：

- `risk_review`
- `evidence_gap_audit`

输出：

- 风险清单。
- 反证列表。
- 冲突证据。
- 风险优先级。
- 跟踪指标。
- 建议补充数据源。

## 6. API 改造

### 6.1 现有 API 保持兼容

继续保留：

- `POST /api/conversations/{conversation_id}/messages`
- `POST /api/conversations/{conversation_id}/messages/stream`
- `GET /api/status`
- `GET /api/graph/summary`
- `GET /api/graph/subgraph`
- `POST /api/research/claims/{claim_id}/review`

### 6.2 Agent API 扩展

现有：

- `GET /api/agent/tasks`
- `POST /api/agent/tasks`
- `GET /api/agent/tasks/{task_id}`
- `GET /api/agent/tasks/{task_id}/export`

建议新增：

- `POST /api/agent/tasks/{task_id}/resume`
- `POST /api/agent/tasks/{task_id}/cancel`
- `POST /api/agent/tasks/{task_id}/rerun-step`
- `GET /api/agent/tasks/{task_id}/state`
- `GET /api/agent/tasks/{task_id}/trace`
- `GET /api/agent/tasks/{task_id}/evidence`
- `POST /api/agent/tasks/{task_id}/human-review`

### 6.3 Agent 创建请求

```json
{
  "task_type": "company_compare",
  "goal": "对比中际旭创和新易盛在光模块业务上的差异、领先指标和风险",
  "thinking_enabled": true,
  "reasoning_effort": "medium",
  "constraints": {
    "core_companies_only": true,
    "max_steps": 8
  }
}
```

## 7. 前端改造

### 7.1 Agent 任务工作台

当前已有 Agent 任务页面，后续升级为完整工作台。

新增区域：

- 任务计划。
- 子任务列表。
- 当前执行步骤。
- 工具调用记录。
- 证据池。
- 已选证据。
- 冲突证据。
- 证据缺口。
- 验证结果。
- 报告草稿。
- 人工审校面板。

### 7.2 证据墙

按来源展示：

- Claim。
- Dossier。
- Graph。
- RAG。
- Semantic。

每条证据展示：

- citation id。
- 来源。
- 页码。
- 主题。
- 公司。
- Claim 类型。
- 敞口等级。
- 置信度。
- 审校状态。

### 7.3 人工反馈闭环

用户可以：

- 修正 Claim。
- 标记 Claim 为 rejected。
- 调整敞口等级。
- 标记证据噪声。
- 标记冲突证据。
- 要求 Agent 重新生成报告。

## 8. 工程化包装

### 8.1 项目元信息

需要修改 `pyproject.toml`：

- 当前 `description` 仍是占位文本。
- 建议改为：

```toml
description = "Evidence-driven AI computing value-chain research agent with GraphRAG, Claim/Dossier memory, and traceable citations."
```

### 8.2 CLI

新增命令：

```bash
aiqasys prepare-data
aiqasys parse-pdfs
aiqasys extract-knowledge
aiqasys build-graph
aiqasys build-rag
aiqasys build-embedding
aiqasys serve-api
aiqasys run-agent
aiqasys eval
```

### 8.3 Docker

当前 `docker-compose.yml` 只有 Neo4j。后续补齐：

- `api` 服务。
- `web` 服务。
- `neo4j` 服务。
- 数据 volume。
- 环境变量配置。

### 8.4 CI

建议增加 GitHub Actions：

- Python 单元测试。
- API smoke test。
- 前端 TypeScript build。
- RAG/Claim 构建 dry-run。
- Agent eval smoke test。

### 8.5 日志与观测

新增：

- `trace_id`
- `task_id`
- `tool_call_id`
- `latency_ms`
- `llm_call_count`
- `token_usage`
- `retrieval_hit_count`
- `verification_status`
- `error_type`

## 9. 评测体系

### 9.1 任务级评测集

构建 30-50 个高难投研任务，覆盖：

- 液冷产业链公司排序。
- 光模块代际变化和公司差异。
- 国产算力瓶颈。
- 训练/推理需求分化。
- AI 服务器产业链映射。
- 公司对比。
- 风险和反证判断。
- 订单、业绩和领先指标缺口。

### 9.2 评分维度

每个任务按 0-2 或 0-5 分评分：

- 任务完成度。
- 事实正确性。
- 证据支撑充分性。
- 引用有效率。
- 公司敞口排序准确性。
- 技术因果链完整性。
- 领先指标覆盖。
- 风险/反证覆盖。
- 幻觉率。
- 缺口识别质量。

### 9.3 检索指标

新增：

- `claim_recall@10`
- `direct_exposure_precision@10`
- `evidence_citation_validity`
- `source_diversity`
- `low_value_evidence_ratio`
- `unsupported_term_count`

## 10. 分阶段落地计划

### Phase 1：Agent 项目骨架整理

目标：

把现有 Agent 雏形整理成清晰、可扩展的项目结构。

任务：

- 新增 `BaseAgent`。
- 新增 `AgentState`。
- 将当前 `AgentRunner` 包装为 `QAAgent`。
- 让 `ToolRegistry` 从元数据升级为可执行工具注册表。
- 保持现有问答 API 兼容。

验收标准：

- 原有问答测试通过。
- diagnostics 中能看到标准化 agent trace。
- `QAAgent` 能完整复用当前能力。
- 前端 Agent 任务页仍可正常展示任务。

### Phase 2：多任务 ResearchAgent

目标：

支持真实任务入口，而不只是问答包装。

任务：

- 扩展 `task_type`。
- 实现 `company_compare`。
- 实现 `company_profile`。
- 实现 `risk_review`。
- 实现 `evidence_gap_audit`。
- 每类任务有独立输出 schema。

验收标准：

- 每个任务类型至少有 3 个测试样例。
- 每个任务能保存完整 `AgentTask`。
- 每个任务能导出 Markdown/JSON。

### Phase 3：动态规划与补证循环

目标：

Agent 能根据证据覆盖情况决定下一步。

任务：

- 新增 `TaskPlanner`。
- 新增 `ToolExecutor`。
- 新增 evidence coverage checker。
- 实现多轮补证。
- 增加停止条件和预算控制。

验收标准：

- 公司对比缺某家公司证据时会自动补检。
- 风险问题缺风险证据时会自动补检。
- 指标问题缺指标证据时明确拒答或输出缺口。

### Phase 4：证据验证和冲突识别

目标：

把答案从“有证据”升级为“关键判断逐条可验证”。

任务：

- 数值和年份校验。
- 指标支持校验。
- 敞口等级校验。
- 风险支持校验。
- 冲突 Claim 分组。
- 前端展示冲突证据。

验收标准：

- 答案中不存在无来源的年份、数值和公司结论。
- 冲突证据可以在任务详情中查看。
- 缺证据时 Agent 明确输出证据缺口。

### Phase 5：GraphRAG / DRIFT 升级

目标：

实现真正面向宽问题的图谱推理和多阶段检索。

任务：

- Query Router。
- Global dossier search。
- Local claim search。
- 多跳路径检索。
- LLM/cross-encoder rerank。
- DRIFT 式宽问题拆解。

验收标准：

- 宽主题问题能自动拆成子问题。
- 公司排序能同时结合主题、公司、指标和风险证据。
- 多跳路径能解释“需求 -> 技术 -> 环节 -> 公司 -> 指标 -> 风险”。

### Phase 6：工程化交付

目标：

把项目包装成可演示、可部署、可评测的正式 Agent 项目。

任务：

- 更新 README。
- 更新 `pyproject.toml`。
- 增加 CLI。
- 增加 Dockerfile。
- 补齐 docker compose。
- 增加 CI。
- 增加最小可运行 demo。

验收标准：

- 一条命令启动后端。
- 一条命令启动前端。
- 一条命令运行 Agent 任务。
- 一条命令运行评测。
- 新用户能按文档在本地跑通最小 demo。

## 11. 优先级建议

短期优先级：

1. 修改项目元信息和 README 定位。
2. 整理 Agent 模块结构。
3. 扩展 `task_type`。
4. 实现 `company_compare` 和 `risk_review`。
5. 增强 `AgentTask` 状态。
6. 前端展示任务计划、工具调用和证据池。

中期优先级：

1. 动态任务规划器。
2. 可执行 Tool Registry。
3. 多轮补证循环。
4. 数值/指标/年份校验。
5. 冲突证据识别。
6. Agent eval。

长期优先级：

1. DRIFT GraphRAG。
2. 多跳路径推理。
3. LLM rerank。
4. 财务表格和公告数据结构化。
5. 主题持续监控。
6. 完整 Docker/CI/CLI 交付。

## 12. 最小可交付版本

如果希望尽快把项目包装成“真正 Agent 项目”，建议最小版本定义如下：

必须包含：

- `QAAgent`
- `ResearchAgent`
- `CompanyCompareAgent`
- `RiskReviewAgent`
- `AgentState`
- `ToolExecutor`
- `AgentTaskStore`
- Agent trace
- evidence pool
- evidence gaps
- Markdown/JSON export
- 前端 Agent 任务工作台
- 10 个 Agent 任务评测样例

最小验收任务：

1. 生成液冷产业链投研简报。
2. 对比中际旭创和新易盛的光模块业务。
3. 分析英维克液冷业务风险。
4. 审查 AI 服务器产业链中的直接受益公司。
5. 检查国产算力瓶颈问题的证据缺口。

达到以上标准后，项目就可以从“AI 算力产业链智能问答系统”升级命名为：

```text
AIQASYS: Evidence-driven AI Computing Value-chain Research Agent
```
