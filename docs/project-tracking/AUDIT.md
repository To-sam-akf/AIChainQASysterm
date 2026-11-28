# Brownfield 项目审计（2026-11-28）

## 事实

- 审计基于 `main` 的本地工作树，HEAD `6d7f795`，相对 `origin/main` 落后 1 个提交；未 fetch，因此远端状态以本地引用为准。开始时已有用户修改：`.env.example`、`.gitignore`、`README.md`、多个 `aika/`、`scripts/`、`tests/test_api.py` 和 `docker-compose.yml`；`data/curated/` 的 5 个文件显示删除，`data/curated_merged/` 和 `tests/test_curated_data_paths.py` 未跟踪。未恢复或覆盖这些差异。
- `aika/` 实现 API、Agent、检索、图谱、MCP 核心及评测；`web/` 是 React 工作台，`scripts/` 包含数据、迁移和实验脚本，`tests/` 有广泛自动化测试，`data/` 包含示例、实验和本地数据，`docs/` 保存需求及跟踪。实际请求链和存储边界见 [架构](../ARCHITECTURE.md)。
- 当前 QA 默认 LangGraph；legacy runner 与非 Agent workflow 都可通过配置到达。独立 CLI 研究 Agent 和 MCP 有各自入口；不以命名判断某路径已经废弃。
- `pyproject.toml` 使用 uv 和 Python >=3.11，CI 用 `uv sync --frozen --all-extras --dev`；前端有 npm lockfile，CI 用 Node 22。Docker Compose 提供本地服务配置；本次仅验证配置，不启动数据库或迁移。测试、smoke、CI 和已知局限见 [质量](../QUALITY.md)。
- `plan.md` 为量化评测规划；`docs/knowledge-agent-requirements.md` 包含动态知识和角色协作验收；`PROGRESS.md` 和 `DECISIONS.md` 保留过去实现、取舍与验证历史；量化指南强调探索性结果、人工审核和独立测试集的区别。新任务状态以 `HANDOFF.md`、`features.json` 为准。

## 推断与未验证

- 从仍可配置和调用的代码推断 legacy runner 是兼容路径，而非已确认可删除代码。其余历史脚本、`graphrag.py` 及旧数据路径尚未逐一证明是否可退役。
- 已有实验记录可用于开发探索，不能推断模型在独立测试集上的效果；旧版本结果不能代替本次验证。
- 未检验远端新提交内容、真实 LLM/embedding/Neo4j、真实 PostgreSQL 资料生命周期、生产部署或真实浏览器到后端链路；无法据此认定完整集成通过。清单按功能族组织，非逐个端点和需求条目的覆盖证明。
