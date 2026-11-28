# AIKA 架构现状

本说明基于 2026-11-28 本地工作树的代码静态检查；它包含未提交的用户改动，不等同于提交 `6d7f795`。完整服务和外部连接本次未运行。

## 入口和职责

- `main.py` → `aika.cli`：本地 API、Web、独立研究 Agent、demo 和评测入口。React/Vite 在 `web/src`，通过 `/api` 访问 `aika.api` FastAPI 路由。
- API 中的 `ConversationStore` 和 `conversation_tasks` 管理会话、后台任务、角色事件和 SSE 重放；知识管理意图经 `knowledge_base`，其他问题经 `QAEngine`。任务状态和事件持久化到 `data/conversations`；当前设计为单用户、单后端进程。
- `QAEngine` 根据配置选择 Agent 或固定 workflow。Agent 默认使用 `agents/qa_agent.py` 和 `langgraph_runner.py`，后者的集中协调器 `agents/centralized.py` 调度检索工作；`agent_runner.py` 是可选 legacy 路径。关闭 Agent 的 workflow 仍有答案核验，不能当成关闭核验的消融。
- CLI 的 `agents/research_agent.py` 是另一条独立研究任务路径，组织研究结果和报告；不应与会话 QA 的角色展示视为同一执行器。
- `aika/aika_mcp/server.py` 注册 MCP 工具，`aika/aika_mcp/tools.py` 通过 `aika_core` 的 SQLite/CSV 等后端提供证据、公司和图谱研究能力；`aika/aika_cli.py` 提供安装、索引等公开 CLI。它不是 Web 请求的必经代理。Skill 在 `skills/aika-research`，打包配置包含示例知识包和 Skill。

## 问答、检索和核验

会话消息 → API 创建/恢复任务 → 角色上下文 → `QAEngine` → 规划及上下文处理 → 图谱、原文 RAG、结构化研究和语义检索工具 → 证据排序及必要时补检 → 生成答案 → `agents/verification.py` 核对答案与引用，必要时修订或收缩到证据摘录 → 保存事件并推送 SSE。LangGraph 中包含对应的检索和 `verify_answer` 节点，工具封装见 `agent_tools.py`；具体分支随配置及数据可用性变化。图谱可使用 CSV 或 Neo4j；PostgreSQL 检索实现位于 `postgres_retrieval.py`，语义检索依赖 embedding 配置。离线 CSV 演示不证明 PostgreSQL、模型或语义检索有效。

动态资料在 `knowledge_base.py` 管理接收、处理、索引、重试和删除；正文及检索块使用 PostgreSQL，关键词与语义状态分开。会话存储、资料删除和历史引用脱敏逻辑分别见 `conversation_store.py`、`knowledge_base.py`、`api.py`。数据导入/迁移由 README 中显式脚本操作，Harness 不自动执行。当前本地未提交改动将默认研究数据路径调整至 `data/curated_merged/`；该路径与所需私有资料的完整性应在集成环境另行确认。

## 约束和债务

真实角色事件对应实际执行，先保存再发送；重启遗留任务不自动重放。旧执行器及固定 workflow 仍由配置显式可达，不能仅凭名称认定废弃。`graphrag.py`、历史数据准备脚本及旧评测路径是否应退役尚未完成逐一调用审计，不应删除。实验评估、离线 smoke 和线上真实服务是不同验证层；证据质量与人工复核限制见 `QUALITY.md` 和量化评估指南。
