# AIQASYS Agent 化改造计划

## 目标定位

将当前 AI 算力产业链智能问答系统，从“证据增强问答/投研 GraphRAG”升级为“证据驱动的产业链投研 Agent 项目”。

最终形态不是只回答单个问题，而是能够围绕用户给定的投研目标，自动拆解任务、选择工具、检索证据、识别缺口、生成结构化研究产物，并保留可追踪的执行轨迹和证据链。

## 当前基础

项目已经具备 agent 化的基础能力：

- 已有 Claim/Dossier + 图谱 + RAG 的混合证据链路。
- 已有 `src/agent_runner.py`，实现 plan -> retrieve -> supplement -> verify_answer 的固定四阶段流程。
- 已有 `src/agent_tools.py`，对问题改写、规划、图谱查询、RAG 检索、投研 Claim 检索、语义检索、证据排序和答案验证做了工具封装。
- 已有 FastAPI 会话接口、流式回答、agent 开关、diagnostics、agent trace 和前端投研结构化输出。
- 已有 Claim 审校覆盖层和 evidence gap 输出，为人工反馈闭环打下基础。

## 关键工作

### 1. 明确 Agent 产品形态

需要把系统从“问答入口”扩展为“任务入口”。

建议优先支持以下任务类型：

- `qa`: 单轮或多轮证据问答。
- `research_brief`: 围绕一个主题生成投研简报。
- `company_profile`: 生成公司产业链画像。
- `company_compare`: 对比两家或多家公司。
- `risk_review`: 汇总主题或公司的风险、反证和不确定性。
- `evidence_gap_audit`: 检查当前知识库对某个问题的证据缺口。
- `monitor_topic`: 围绕主题建立后续跟踪指标和补数清单。

每个任务都需要定义：

- 输入参数。
- 任务计划。
- 可调用工具。
- 中间状态。
- 最终产物。
- 失败和缺证据时的处理方式。
- 是否需要人工确认。

### 2. 将固定流程升级为动态任务规划

当前 `AgentRunner` 更接近规则式 ReAct 工作流，步骤基本固定。真正的 agent 应该能够根据任务动态决定下一步。

需要新增：

- 任务规划器：根据用户目标生成子任务列表。
- 动态工具选择器：根据子任务选择图谱、RAG、Claim、Dossier、语义召回、指标抽取、验证等工具。
- 证据覆盖判断：判断是否已经足够回答，还是需要继续检索。
- 循环控制：支持最多 N 轮检索/反思/补证。
- 停止条件：证据充分、证据缺口不可弥补、达到预算或达到最大步数。

建议保留当前固定流程作为 `QAAgent` 的稳定实现，再新增更通用的 `ResearchAgent`。

### 3. 建立标准 Tool Registry

当前工具集中在 `AgentTools` 类中，后续需要抽象为统一工具协议。

每个工具建议具备以下元数据：

- `name`: 工具名称。
- `description`: 工具能力说明。
- `input_schema`: 输入结构。
- `output_schema`: 输出结构。
- `timeout`: 超时时间。
- `cost_level`: 成本等级。
- `safety_level`: 安全等级。
- `requires_llm`: 是否依赖 LLM。
- `cacheable`: 是否可缓存。

首批工具建议包括：

- `contextualize_question`
- `plan_question`
- `query_graph`
- `search_rag`
- `search_research_claims`
- `search_segment_dossiers`
- `search_semantic_index`
- `rank_evidence`
- `verify_answer_support`
- `build_research_outputs`
- `detect_evidence_gaps`
- `export_report`

后续可扩展：

- 年报表格解析。
- 公告/研报新增数据接入。
- 财务指标抽取。
- 多跳图谱路径查询。
- 冲突证据识别。
- LLM rerank。

### 4. 增加 Agent State / Workspace

真正的 agent 需要有可追踪、可恢复的任务状态，而不是只返回一次问答结果。

建议新增 `AgentState`，包含：

- `task_id`
- `task_type`
- `user_goal`
- `plan`
- `current_step`
- `subtasks`
- `tool_calls`
- `evidence_pool`
- `selected_evidence`
- `draft_findings`
- `conflicts`
- `evidence_gaps`
- `verification`
- `final_outputs`
- `status`
- `created_at`
- `updated_at`

需要支持：

- 任务暂停和恢复。
- 中间结果保存。
- 前端查看执行轨迹。
- 人工编辑证据或结论后继续执行。
- 失败任务可诊断。

### 5. 强化证据验证与冲突处理

当前已经有引用编号、证据卡和 unsupported terms 检查，但还需要更严格的事实一致性检查。

需要补充：

- 引用有效性校验：答案中的 `[E1]` 必须存在于 evidence pack。
- 公司覆盖校验：公司对比问题中，每家公司都要有证据。
- 数值校验：答案中的年份、数值、比例、金额必须能在证据中找到。
- 术语校验：关键技术术语不能脱离证据包生成。
- 敞口校验：core/direct/indirect/mentioned 分级需要有依据。
- 风险校验：风险类问题必须至少包含风险或反证证据。
- 冲突识别：识别券商乐观判断 vs 年报风险披露、旧报告 vs 新报告、技术路线 A vs B。

最终目标：答案中的每个关键判断都能回溯到证据卡、Claim、Dossier、图谱关系或原文片段。

### 6. 升级为真正 GraphRAG / DRIFT 检索

当前系统已经有图谱、BM25 RAG、Claim/Dossier 和 embedding 语义召回，但仍需要更强的多阶段检索和图谱推理。

需要实现：

- Query Router：根据问题类型选择检索策略。
- Global Search：宽问题先召回主题 Dossier 和高层 Claim。
- Local Search：围绕公司、指标、风险、产业链环节做局部检索。
- DRIFT 式流程：宽问题 -> 全局摘要 -> 自动拆子问题 -> 局部证据 -> 汇总结论。
- 多跳路径检索：需求驱动 -> 技术瓶颈 -> 产业环节 -> 公司敞口 -> 指标验证 -> 风险反证。
- 三阶段排序：BM25 高召回 -> dense embedding 召回 -> cross-encoder/LLM rerank 精排。

### 7. 建立任务级评测体系

要把项目包装成 agent，不能只靠单元测试和 smoke test，需要 agent eval。

建议建立 30-50 个高难投研任务，覆盖：

- 液冷产业链公司排序。
- 光模块代际变化和公司差异。
- 国产算力瓶颈。
- 训练/推理需求分化。
- AI 服务器产业链映射。
- 公司对比。
- 风险和反证判断。
- 订单、业绩和领先指标缺口。

评分维度：

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

检索指标：

- claim recall@10
- direct exposure precision@10
- evidence citation validity
- source diversity
- low-value evidence ratio
- unsupported term count

### 8. 将前端升级为 Agent 工作台

当前前端已经展示答案、证据、子图和投研输出。下一步需要从“问答页面”升级为“研究任务工作台”。

建议新增视图：

- 任务列表。
- 任务计划。
- Agent 执行轨迹。
- 工具调用记录。
- 证据墙。
- Claim/Dossier 详情。
- 冲突证据。
- 证据缺口。
- 研究报告草稿。
- 人工审核面板。
- 报告导出入口。

前端展示重点：

- 用户能看到 agent 为什么这样回答。
- 用户能追踪每个结论来自哪里。
- 用户能修改 Claim 或标记噪声证据。
- 用户能把一次问答升级为一项持续研究任务。

### 9. 完善工程化包装

为了让项目像一个正式 agent 项目，需要补齐工程边界。

建议工作：

- 调整项目名称和 `pyproject.toml` 描述，避免占位信息。
- 增加 CLI，例如 `aiqasys run-agent`、`aiqasys eval`、`aiqasys build-index`。
- 增加 Dockerfile 和完整 docker compose，覆盖 API、前端、Neo4j。
- 增加 CI，至少运行 Python tests、前端 build、基础 lint。
- 解决 `tests/test_api.py` 在当前环境中的挂起问题。
- 增加结构化日志和 trace id。
- 增加任务超时、重试、缓存和错误降级。
- 增加配置文档和最小可运行 demo。
- 增加数据构建流水线说明。

## 建议落地路线

### Phase 1: Agent 项目骨架

目标：把现有 agent 雏形整理成清晰的项目结构。

任务：

- 新增 `src/agents/` 目录。
- 定义 `BaseAgent`、`QAAgent`、`ResearchAgent`。
- 定义 `AgentState`、`AgentStep`、`AgentTask`。
- 将当前 `AgentRunner` 迁移或包装为 `QAAgent`。
- 定义 `ToolRegistry` 和统一工具协议。
- 保持现有问答 API 兼容。

验收标准：

- 原有问答测试通过。
- diagnostics 中能看到标准化 agent trace。
- QAAgent 能完整复用当前能力。

### Phase 2: 研究任务 Agent

目标：实现第一个真正的多步投研任务。

优先实现 `research_brief`：

- 输入：主题、公司或问题。
- 输出：核心判断、技术机理、产业传导、公司排序、领先指标、风险反证、证据索引、证据缺口。
- 支持多轮补证。
- 支持中间状态保存。

验收标准：

- 能围绕“液冷产业链”“光模块”“国产算力”等主题生成结构化研究简报。
- 每个关键结论至少绑定一个证据编号。
- 缺少证据时明确输出缺口，不编造。

### Phase 3: 验证、冲突和评测

目标：让 agent 输出更可信。

任务：

- 增强 citation validator。
- 增加数值、年份、公司覆盖校验。
- 增加冲突证据识别。
- 建立 30-50 个任务级评测样例。
- 输出评测报告。

验收标准：

- 每次变更后可运行 agent eval。
- 能报告任务完成度、引用有效率、幻觉风险和失败样例。

### Phase 4: Agent 工作台

目标：让用户能操作、审阅和复用 agent 任务。

任务：

- 前端增加任务列表和任务详情页。
- 展示计划、步骤、工具调用、证据池、最终产物。
- 支持 Claim 审校和证据标记。
- 支持报告导出。

验收标准：

- 用户能从一个问题创建研究任务。
- 用户能查看 agent 的执行过程和证据来源。
- 用户能对证据或 Claim 做人工反馈。

### Phase 5: 工程化发布

目标：把项目包装成可运行、可演示、可维护的 agent 产品。

任务：

- 完善 README 和快速开始。
- 增加 Dockerfile 和完整 compose。
- 增加 CLI。
- 增加 CI。
- 修复 API 测试挂起问题。
- 增加日志、重试、缓存和任务持久化。

验收标准：

- 新环境能按文档启动 API、前端和 Neo4j。
- 一条命令可运行测试和评测。
- 一条命令可启动 demo。

## 优先级排序

P0:

- `src/agents/` 项目结构。
- `AgentState` 和标准 trace。
- `ToolRegistry`。
- `QAAgent` 兼容当前流程。
- `ResearchAgent` 的 `research_brief` 任务。

P1:

- 任务持久化。
- 动态规划和循环补证。
- 引用/数值/公司覆盖校验。
- 前端任务工作台。
- agent eval。

P2:

- 多跳 GraphRAG。
- DRIFT 检索。
- LLM/cross-encoder rerank。
- 冲突证据分组。
- 指标表格结构化抽取。
- 持续监控任务。

## 近期最小可行版本

建议先做一个 Agent MVP：

1. 保留当前问答链路。
2. 新增 `ResearchAgent`。
3. 支持 `research_brief` 任务。
4. 保存 `AgentState` 到本地 JSONL 或 SQLite。
5. 前端展示任务计划、执行轨迹、证据和研究简报。
6. 建立 10 个研究任务评测样例。

MVP 完成后，项目就可以从“智能问答系统”正式包装为“AI 算力产业链投研 Agent”。
