# 质量与验证边界

基线日期 2026-11-28；验证对象为含用户未提交修改的本地工作树副本，而非干净提交。测试命令与结果见 `project-tracking/HANDOFF.md`。CI (`.github/workflows/ci.yml`) 覆盖 Python pytest、离线 demo、QA/Agent smoke、前端 build 和 Compose config；前端 lint 已有脚本，但 CI 未调用。测试目录涵盖检索、Agent、证据、API、动态知识库、MCP、评测、报告及浏览器；pytest 通过数量不是代码覆盖率，目前没有测量的覆盖率数据。

已观察问题：离线 QA 5 题 smoke 有 2 个告警；它是探索性程序评分，不是独立准确率。前端 build 报图谱 chunk 大于 500 kB。安装器 dry-run 测试使用临时项目目录、用户配置目录和 skill 来源，不应改动用户配置。没有 Git 元数据的副本会使实验快照测试失败。

真实 PostgreSQL 动态资料完整生命周期、外部 LLM/embedding、Neo4j、真实浏览器到后端端到端及生产部署本次未验证。浏览器测试依赖单独启动的 Vite、Chrome/Chromium，并拦截 API，不是完整后端集成。模型评测可能收费；`eval-full` 只运行无模型的完整 QA/Agent 集合，正式 A/B/E/F 及真实服务需要明确授权、冻结配置和数据后按量化评估指南执行。

## 命令

`init.sh setup` 明确安装锁定 Python 和前端依赖；`doctor` 只查看本机工具，不查密钥。`check` 是定向回归、lint/build 和 diff 检查；`test` 是 pytest 全量（外部集成依测试自身条件跳过）；`e2e` 要求 `AIKA_WEB_TEST_URL` 和浏览器；`eval-smoke` 为 CI 同等离线 demo/QA/Agent；`eval-full` 为不设 limit 的离线 QA/Agent；`verify-task` 串行执行 check 和 smoke，不能替代任务专属验收、独立审查或真实集成。输出写临时目录；这些命令不迁移数据库。测试和构建仍可能生成缓存/构建产物，建议在隔离副本执行。`setup` 需要网络并修改依赖环境。

## Harness 修复后的约束

`init.sh` 离线命令将 `TEST_DATABASE_URL` 与 `DATABASE_URL` 明确设为空，防止 `.env` 恢复；直接运行 pytest 时，未显式启用破坏性测试也会清空数据库地址并禁用 PostgreSQL。破坏性 PostgreSQL 测试默认跳过；仅显式传 `--allow-destructive-postgres` 且启动 pytest 前设置 `TEST_DATABASE_URL` 为本机、无 URL 参数、库名 `aika_test_*` 时才运行，并要求启动时显式设置可解析、无 URL 参数且数据库名不同的 `DATABASE_URL`；普通目标未知或库名相同时拒绝，即使主机名或端口不同。名称和本机检查不能证明库无重要数据，执行者仍必须确保库是专用临时库；实际数据库本轮未测。

QA CLI `--mode diagnostic` 检查报告完整性，不将 fail 或 warn 当作程序运行失败；`regression` 要求无 fail 且原有总体分至少 0.35，可保留 warn；`acceptance` 还要求无 warn、完整题集（无 limit）和可读取的持久报告。空、缺失、不一致或未识别结果均失败；其他套件目前仅支持诊断模式。现有 smoke/full 是诊断，退出 0 不表示质量验收。`init.sh task-gate` 验证清单的 ID、依赖、任务转换、WIP、合同和证据；`check` 和 `verify-task` 包含此检查。原独立审查报告保留，待新独立审查。


2026-11-28 Harness V1.0 收尾：CI 直接执行实际清单的 task-gate；该门禁只能检查清单自述一致性，不能证明手工填写的转换历史和证据真实，仍依赖人工复核。QA 报告校验题集身份、逐题评分和汇总；历史 QA 告警及前端包体积问题保留为技术债务。真实集成仍未验证。
