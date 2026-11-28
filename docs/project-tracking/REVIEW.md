# Brownfield Harness 独立审查

日期：2026-11-28。结论：**FAIL**。安全隔离和验收结果传播存在实质缺陷；本报告不是业务功能或生产发布验收。

## 范围与方法

审查根入口、交接与规则、功能清单、审计、历史进度、架构、质量说明及 `init.sh`，抽查 API、QA、LangGraph、dotenv 和 PostgreSQL 测试代码，核对工作树及变更归属。独立运行以下命令；没有依赖上一会话的测试结果作为本次证据。检查为抽样，不是所有代码、所有引用或全部敏感数据的完整审计。

隔离副本 `/tmp/aika-review-NBGYY1` 由本地 HEAD 克隆 Git 元数据，复制存在的已跟踪和非忽略未跟踪工作树文件，复用原 `.venv` 和 `web/node_modules`，不复制忽略的 `.env`；使用 `env -i HOME=/tmp PATH="$PATH" LANG=C.UTF-8` 运行验证。副本包含 Git 跟踪的 `.codex/config.toml`，不是完全没有项目 Codex 配置的环境。日志和离线产物位于 `/tmp`，仅作本机暂存，不应视为持久证据。

## 实际执行与结果

| 命令或检查 | 本次结果与边界 |
| --- | --- |
| `init.sh status`、`git status --short`、`git diff --stat`、完整未跟踪文件清单 | 成功；`main` 相对本地 `origin/main` 落后 1，工作树有大批既有修改及删除。未 fetch。 |
| `init.sh doctor`、`bash -n docs/project-tracking/init.sh`、`git diff --check`、Python JSON 解析及 ID/依赖/必填字段检查 | 退出 0；9 个 ID 唯一，依赖均存在，字段齐备。未验证所有可能的依赖环或完整 schema。 |
| `init.sh e2e`（未设置地址） | 退出 2，提示需启动 Vite 和设置 `AIKA_WEB_TEST_URL`；浏览器测试拦截 API。未启动浏览器服务，未验证真实端到端。 |
| `init.sh invalid-command` | 退出 2，打印用法。未对每种失败路径逐项注入故障。 |
| `init.sh test`（隔离副本） | 退出 1：524 passed、3 skipped、1 failed；失败为 `tests/test_aika_mcp_install.py::test_mcp_install_with_skill_dry_run_reports_both_steps`，预期退出 0 实得 2。副本保留了跟踪中的 `.codex/config.toml`；单独重跑该测试同样失败，尚未证明该配置是唯一原因。日志 `/tmp/aika-review-test.log`。 |
| `init.sh verify-task`（隔离副本） | 退出 0：定向测试 38 passed，前端 lint/build 完成，chunk 大于 500 kB 告警；离线 demo 完成，QA 5 题 3 pass、2 warn、0 fail，Agent 1 题 2/2。日志 `/tmp/aika-review-verify.log`。退出 0 不等于每题通过。 |
| `init.sh eval-full`（隔离副本） | 退出 0：QA 50 题 18 pass、27 warn、**5 fail**，整体分 0.6489；Agent 15 题 30/30。日志 `/tmp/aika-review-eval-full.log`；该离线程序评分不等同真实模型准确率。 |
| dotenv 与 QA 退出码的无害构造检查 | 临时文件写入 `TEST_DATABASE_URL=sentinel`，先 unset 再调用 `load_dotenv`，变量被恢复；构造 `overall_score=0.5, failed=1` 调用 `print_qa_eval_report(..., True)` 返回 0。均未连接数据库。 |

`setup` 会安装依赖，本轮未运行；`offline-api`、`web` 是持续运行的服务入口，未启动；真实 PostgreSQL、模型、embedding、Neo4j、真实浏览器到后端链路和生产部署均未测试。没有执行付费模型调用、迁移或数据库清理。

## 通过的验收项

- 根入口能找到当前交接、规则和主要架构/质量文档；抽查入口所列路径、量化指南均存在。根规则明确用户指令优先和保护工作树。产品研究 Agent 在架构中与开发用 Coding Agent 入口分开描述。
- 功能清单为合法 JSON，9 个 ID 唯一、依赖指向存在的 ID，分别记录实现和验证状态、验收标准及阻塞项；F09 标明独立审查待完成。验收文字总体偏高层，不能单凭清单证明每项功能已验收。
- 抽查代码支持架构中的默认 LangGraph、可选 legacy、固定 workflow 的核验及独立研究 Agent 说法；未发现有证据推翻这些主要路径，但没有穷举全部调用链。文档明确区分离线、模拟和真实集成，准确记录 smoke 的两个告警和构建告警。
- `set -euo pipefail`、实际全量测试失败和缺少 e2e 前提时的退出码表明相应失败没有被 shell 普遍吞掉；不代表评测逐例失败已正确传递，见下文。

## 未通过项、严重程度与建议

### 高：离线测试不能保证数据库隔离

`docs/project-tracking/init.sh:8` 使用 `env -u TEST_DATABASE_URL`，但 `aika/llm_client.py:34` 的 `load_dotenv` 使用 `os.environ.setdefault`，当变量未设置时可从仓库 `.env` 重新载入。`tests/test_postgres_retrieval.py:74` 读取该变量后执行迁移和 `TRUNCATE ... CASCADE`。如果同一 pytest 进程在该测试前加载了含此变量的 `.env`，则 `init.sh test` 的离线隔离可能失效并对所指数据库执行破坏性操作；这是一条基于代码和无害 sentinel 实验的条件风险，本次未对真实数据库复现。建议将变量显式设为空而非 unset，并在无真实服务的隔离测试中覆盖 dotenv 重载及敏感外部配置；进一步确保破坏性集成测试仅能指向明确创建的隔离测试库。

### 高：评测含失败仍能返回成功，无法作为验收门禁

本次 `init.sh eval-full` QA 出现 5 个 fail，整个命令仍退出 0。`aika/cli.py:357` 的 QA 返回值只检查总分是否至少 0.35，不检查 `failed`；构造检查也复现 fail=1 时返回 0。`init.sh verify-task` 直接串行复用评测返回值，无法保证逐题无失败。这个行为是程序评分策略，不能称为所有验收通过；当前文档虽要求如实报告告警，却没有明确提示失败题也可能返回 0。建议明确约定阈值及失败策略，并让验收入口检测逐题状态或明确标为非门禁；保留逐题输出及告警，不应降低既有标准来掩盖失败。

### 中：全量测试在包含跟踪配置的正常副本失败

上述 dry-run 测试可在保留 `.codex/config.toml` 的副本重现失败，`init.sh test` 如实返回 1。`QUALITY.md` 建议隔离时不含个人 `.codex`，但该配置实际为 Git 跟踪文件，不能简单等同可排除的个人文件；独立新 checkout 也可能遇到问题。建议让安装器测试通过临时项目目录和模拟配置自包含，或准确说明环境依赖，且保留既有用户配置。本轮未删除、改写配置，也未确认根因唯一。

### 中：任务调度不可执行，交接字段不能自动保证一致

根 `AGENTS.md` 的 WIP=1 和先写验收合同是文档规则；`features.json` 没有当前任务/活动任务或合同字段，`init.sh` 没有 WIP、依赖状态、合同或状态同步检查。当前任务和阻塞靠 `HANDOFF.md` 自由文本，清单 F09 的状态不会随审查自动变更。建议定义机器可读的当前任务、合同及阻塞，并在启动/验收入口检查 WIP、依赖和状态一致性。当前审查任务及其验收要求由用户明确给出，不把它误记为实现了自动合同检查。

### 低：恢复入口的状态提示不完整

`docs/project-tracking/AGENTS.md` 描述 `status` 检查依赖是否存在，但当前 `status` 只在 Python 存在时打印版本，不报告缺失，也不检查 Node 或前端依赖；这些由 `doctor` 单独检查。建议统一说明或恢复明确提示。历史 `PROGRESS.md` 含有被后续章节覆盖的旧状态，需依日期和当前交接阅读；未发现必然的冲突。

## 跨会话恢复演练

1. **项目**：AIKA 是中文 AI 算力产业链证据研究系统，含 FastAPI、React、研究 Agent、RAG/图谱与 MCP；可由架构和仓库代码回答。
2. **启动**：离线 `init.sh offline-api`，另开终端 `init.sh web`；完整服务及显式迁移参见 README，依赖准备见 README/`setup`。
3. **当前任务**：审查前交接指向 F09 Harness 初始化及独立审查；本轮用户指令要求独立审查，审查完成后等待人工确认。任务状态依赖自由文本而非调度器。
4. **已验证**：以本报告的独立执行记录为准；旧交接有实现会话记录，可作为历史资料但不能冒充本次结果。
5. **未验证**：真实服务与真实端到端、付费实验、生产部署；全量测试本轮未通过，e2e 未运行，完整评测有失败项。
6. **下一步**：人工审阅本报告，决定如何安排安全隔离、失败传播、全量测试和调度约束修复；修复后重新独立验证，再讨论 checkpoint。
7. **保护对象**：既有业务与数据差异、`.env`、密钥、数据库/索引、会话及本机配置；不得擅自恢复、删除、迁移或覆盖。工作树归属见下。

## 工作区与 checkpoint

依据 `AUDIT.md` 的起始记录，本次 Harness 声称新增根 `AGENTS.md`、`docs/ARCHITECTURE.md`、`docs/QUALITY.md`、`docs/project-tracking/{AUDIT.md,HANDOFF.md,features.json}`，修改 `init.sh` 并追加 `PROGRESS.md`；本轮新增本报告并更新交接。此前已有 `.env.example`、`.gitignore`、`README.md`、`aika/`、`scripts/`、`tests/test_api.py`、`docker-compose.yml` 的变更、`data/curated/` 五个删除、未跟踪 `data/curated_merged/` 与 `tests/test_curated_data_paths.py`。这些归属基于交接记录，缺少独立的起始快照，不能对每个 hunk 的历史来源作绝对证明；归属不明处保留，未覆盖或删除。业务 diff 中例如 `aika/api.py` 有大量删除，但本轮不评价或修复其业务正确性。

未跟踪的 Harness 文件不会出现在普通 `git diff --stat`，保存时必须显式逐文件审阅；`PROGRESS.md` 和 `init.sh` 是跟踪文件，仍须核对 hunk。通过明确路径和必要时分块暂存，原则上可形成不含业务变更的独立 checkpoint；现有审查 FAIL，且用户未授权，本轮不暂存、不提交、不推送。`git check-ignore` 显示 `.env`、示例会话路径与 `.sqlite` 路径被忽略；跟踪文件名检查未发现数据库文件，但未做全量内容秘密扫描，不能保证没有密钥或个人数据。隔离测试生成 `/tmp` 日志、评测目录及副本，前端构建写入副本；未清理它们，未对原工作树进行数据清理。
