# 当前交接

- 当前任务：Brownfield Harness 初始化；范围仅工程入口、审计、架构、质量、功能清单及安全离线基线，不开发业务。实现和验证已完成，独立审查尚待下一会话；未经用户授权不提交。
- 本次文件：新增根 `AGENTS.md`、`docs/ARCHITECTURE.md`、`docs/QUALITY.md`、本目录 `AUDIT.md`、`features.json`、`HANDOFF.md`；修改本目录 `init.sh`，并追加 `PROGRESS.md`。未改业务代码及用户已有差异。
- Git checkpoint：起始 HEAD `6d7f795`，`main` 相对本地 `origin/main` 落后 1；没有新提交。用户原有修改与删除/未跟踪数据详见 `AUDIT.md` 和 `git status`；保存检查点前必须按文件核对，不能包含用户改动。
- 隔离副本 `/tmp/aika-harness-UVSBqc` 含工作树快照、独立 Git 元数据和仓库 `skills`，复用依赖，排除个人 `.codex`；以下是本次运行而非历史断言。环境清空模型、embedding 和数据库配置，测试不针对外部服务。
  - `.venv/bin/python -m pytest -q`：退出 0，525 passed、3 skipped（PostgreSQL 测试缺 `TEST_DATABASE_URL`；2 个浏览器测试缺地址/浏览器）。首次无 `.git` 的副本运行退出 1，510 passed、3 skipped、15 failed，失败原因是实验快照调用 Git；补齐独立元数据后复测通过。
  - `npm --prefix web run lint && npm --prefix web run build`：退出 0；图谱 chunk 约 1.51 MB，构建提示超过 500 kB。
  - `main.py demo --offline`：退出 0，Agent completed；`main.py eval --suite qa --offline --limit 5`：退出 0，5 题 3 passed、2 warned、0 failed；`main.py eval --suite agent --offline --limit 1`：退出 0，1 题 2/2。各命令使用临时 task/report 目录，不是质量全面通过。
  - `docker compose config --quiet`：退出 0；`init.sh verify-task`：退出 0，定向测试 38 passed，lint/build 和上述 smoke 完成；`bash -n init.sh`：退出 0。`init.sh e2e`：退出 2，缺 `AIKA_WEB_TEST_URL`；浏览器测试拦截 API。
- 阻塞/未验证：真实 PostgreSQL、模型、embedding、Neo4j 和真实端到端；正式实验需冻结资料和人工复核，并在获得授权后处理模型费用。QA 两个告警和 bundle 大小不在本任务中修复。
- 下一步：另开独立审查会话审查 Harness、检查用户原有 diff 和入口安全，再决定是否授权单独 Git checkpoint；审查结束、当前任务验收后，建议针对 QA smoke 的两项告警建立有验收合同的独立任务。
- 恢复：先读根 `AGENTS.md`、本文件、`features.json`、`AUDIT.md`、`QUALITY.md`，运行 `./docs/project-tracking/init.sh status`，核对 Git 工作树和临时副本是否仍存在；按照用户指令继续。

## 2026-11-28 独立审查结果

- 当前状态：独立审查完成，结论 **FAIL**；报告见 [REVIEW.md](REVIEW.md)。按用户要求等待人工确认，本轮未修复、暂存、提交或推送。上文是审查前交接和历史验证，不代表本次重跑结论；F09 清单状态尚未更新，仍需人工决定后续修复任务。
- 本次独立隔离验证：`verify-task` 退出 0，定向 38 passed、QA smoke 3 pass/2 warn；全量 `test` 退出 1（524 passed、3 skipped、1 failed，安装器 dry-run）；`eval-full` 退出 0 但 QA 50 题中 5 fail、27 warn，Agent 30/30；`e2e` 因未设置服务地址退出 2。完整命令、环境和限制见报告。
- 优先问题：`TEST_DATABASE_URL` 在 unset 后可能被 `.env` 重新载入，而集成测试含迁移及 TRUNCATE；QA 逐题失败不一定使评测命令失败。另有跟踪的 `.codex/config.toml` 环境下安装器测试失败，以及 WIP 仅靠文档约束。不要直接在可能含数据库配置的工作树运行全量测试。
- 建议下一步：人工审核报告并决定修复范围；先安全隔离测试数据库，再明确逐题失败的验收门禁，处理测试隔离和任务状态机制，完成修复后重新独立验收。保护现有业务、数据及配置差异；checkpoint 需单独授权并核对文件。

## 修复任务 H01（进行中）

验收合同：离线入口阻止 dotenv 恢复数据库配置；有副作用的 PostgreSQL 测试必须通过 pytest 显式开关启用，并在连接前拒绝非本机、非专用测试库及与普通数据库相同的目标；以无害配置测试 dotenv 和拒绝路径，不连接真实数据库。完成后记录命令与退出码，再开始 H02。

H01 完成（待独立复审）：`TEST_DATABASE_URL='' DATABASE_URL='' .venv/bin/python -m pytest -q tests/test_database_test_safety.py tests/test_postgres_retrieval.py` 退出 0，11 passed、1 skipped；显式启用并指定远程无害 URL 的单项测试退出 1，在连接前拒绝。实际数据库未连接；本机测试库专用性仍须执行者确认。此前一次编辑命令因 `python` 不存在退出 127，已用虚拟环境 Python 修正并重跑。

## 修复任务 H02（进行中）
验收合同：区分诊断、回归和验收；验收必须有完整非空报告且无失败或告警，未执行或结果缺失均失败；维持题目和评分规则，覆盖失败、告警、空及缺失。

H02 完成（待独立复审）：`TEST_DATABASE_URL='' DATABASE_URL='' .venv/bin/python -m pytest -q tests/test_eval_quality_gate.py tests/test_eval_platform.py` 退出 0，12 passed。QA 诊断只要求结果完整，回归禁止 fail 并维持原 0.35 分数线，验收另要求无 warn、全题及保存的可读取报告；其他套件尚无约定门禁，非诊断模式明确拒绝。未更改题目或评分。

## 修复任务 H03（进行中）
验收合同：复现 dry-run 故障并确认原因；测试固定项目目录、HOME、技能来源和宿主安装状态；不修改真实配置；定向测试通过。

H03 完成（待独立复审）：修复前定向测试退出 1；根因是默认项目目录读取当前仓库 `.codex/config.toml`，已有同名 MCP 服务触发冲突并退出 2。测试改用临时 cwd、HOME、CODEX_HOME 和虚拟 skill，模拟宿主命令；`TEST_DATABASE_URL='' DATABASE_URL='' .venv/bin/python -m pytest -q tests/test_aika_mcp_install.py` 退出 0，28 passed。真实配置未改动。

## 修复任务 H04（进行中）
验收合同：在功能清单新增独立任务生命周期，校验 ID、依赖和环、WIP、转换历史、合同、完成证据及阻塞原因；将检查接入 init.sh，自动化覆盖关键拒绝路径。

H04 完成（待独立复审）：`TEST_DATABASE_URL='' DATABASE_URL='' .venv/bin/python -m pytest -q tests/test_task_gate.py` 退出 0，3 passed；`./docs/project-tracking/init.sh task-gate` 退出 0。功能状态和任务生命周期各自独立，检查任务转换历史、依赖及证据；该门禁是清单一致性检查，不是防止手动篡改的工作流引擎。

正在准备隔离副本进行最终离线回归；以上各项代码仍待新的独立复审。

## 2026-11-28 修复后最终离线验证

隔离副本 `/tmp/aika-repair-3DmJ6t/repo` 包含当前工作树、独立 Git 元数据，排除 `.env` 和个人会话目录，复用本机依赖；以 `env -i`、临时 HOME 执行。日志在 `/tmp/aika-repair-3DmJ6t/`，仅本机暂存。未使用真实数据库、付费模型或外部服务。

- `init.sh test`：最终重跑退出 0，543 passed、3 skipped（数据库 1、浏览器 2）；初次为 542 passed、3 skipped。随后增加直接 pytest 的默认隔离和 dotenv 子进程测试后重新全量验证。另以无害远程 sentinel 地址执行 `TEST_DATABASE_URL=postgresql://remote/aika_test_fake DATABASE_URL=postgresql://remote/production .venv/bin/python -m pytest -q tests/test_database_test_safety.py tests/test_postgres_retrieval.py`，退出 0，12 passed、1 skipped。
- `init.sh check`：退出 0；定向 pytest、任务门禁、前端 lint/build、diff 检查通过；前端图谱 chunk 约 1.51 MB 的告警仍在。
- `init.sh eval-smoke`：退出 0；demo 完成，QA 5 题 3 pass、2 warn；Agent 1 题 2/2。
- `init.sh eval-full`（诊断）：退出 0；QA 50 题 18 pass、27 warn、5 fail，Agent 15 题 30/30。诊断退出 0 仅表示执行及结果完整，不代表质量达标。
- `main.py eval --suite qa --offline --mode acceptance --report-dir <临时目录> --json`：退出 1；上述 5 fail、27 warn 被验收门禁拒绝。题目与评分未改，业务 Bad Case 未修复。
- `bash -n docs/project-tracking/init.sh`、`init.sh task-gate`、`git diff --check`：各退出 0。原 REVIEW.md 保持不变。

待验证：授权的隔离 PostgreSQL 集成、真实模型和外部服务、真实浏览器到后端链路及独立复审。修复并不等于独立审查通过；停止并等待新的 Evaluator。未提交、未推送。

## 2026-11-28 独立复审 REVIEW-02

- 本轮独立复审完成，Harness 初始化结论 **FAIL**：H01 FAIL（本地地址别名可绕过同库比较）、H02 FAIL（缺逐题详情的报告可通过验收函数）、H03 PASS、H04 FAIL（CI 不检查实际功能清单）。复现步骤、边界与修复建议见 [REVIEW-02.md](REVIEW-02.md)；原 [REVIEW.md](REVIEW.md) 保留。本轮不修复实现、不提交、不推送，等待人工确认。
- 本轮隔离副本 `init.sh test` 退出 0（543 passed、3 skipped）；`init.sh check` 退出 0（38 passed、前端 lint/build 成功，chunk 大小警告）；`eval-smoke` 退出 0（QA 3 pass、2 warn）；两种临时 HOME 的安装器测试各 28 passed。其他定向和负例见复审报告。未连接真实数据库或外部服务，未做真实浏览器到后端验收；历史业务评测告警不构成本轮 Harness FAIL 的理由。
- F09 已更新为部分验证及复审阻塞；H01–H04 的 done/history 保留为修复会话声明，不等于独立复审通过。未来需先依用户决定安排单独修复和独立复审；检查点须获授权并逐文件及混合文件逐 hunk 核对，保留用户原有差异。

## Harness V1.0 收尾验收合同（2026-11-28）

仅修复 REVIEW-02 的 H01、H02、H04，H03 不改。H01 对未明确或无法解析的普通数据库目标默认拒绝，别名同库在连接前拒绝；H02 核对报告必要元信息、唯一题目 ID、逐题评分和汇总，验收与实际题集核对并检查保存结果；H04 在 CI 执行实际功能清单门禁。仅运行定向回归与必要安全离线检查；不连接数据库或付费服务。修复后最多一次独立定向复审，未经人工确认不保存 Git checkpoint。

H01 收尾回归：`TEST_DATABASE_URL='' DATABASE_URL='' .venv/bin/python -m pytest -q tests/test_database_test_safety.py tests/test_postgres_retrieval.py` 退出 0（17 passed、1 skipped）；未连接数据库。普通数据库地址必须在 pytest 启动时显式提供、无 URL 选项且库名不同，不能确认时拒绝。

H02 收尾回归：`TEST_DATABASE_URL='' DATABASE_URL='' .venv/bin/python -m pytest -q tests/test_eval_quality_gate.py tests/test_eval_platform.py` 退出 0（19 passed）。报告必须含运行、题集和环境元信息、每题 ID/问题/分类/评分指标，并与独立加载的题集和重新汇总的统计一致；保存报告仍需读回一致。

H04 收尾回归：`TEST_DATABASE_URL='' DATABASE_URL='' .venv/bin/python -m pytest -q tests/test_task_gate.py` 退出 0（3 passed）；CI 中相同的 `python docs/project-tracking/task_gate.py` 对真实清单退出 0。临时复制真实清单并将 H02 ID 改成 H01 后同一程序退出 1，报告 ID 不唯一；首次负例脚本错误地断言输出含英文 duplicate，脚本退出 1，修正为检查实际输出 globally unique 后退出 0。

其他定向验证：离线 QA 2 题生成并读回报告，命令退出 0，2 pass；`git diff --check` 退出 0。未连接数据库或外部服务。H01、H02、H04 实现待一次独立定向复审；尚未获得复审通过，不能宣布 Harness V1.0 已验收或保存 checkpoint。人工确认前需逐文件及混合文件逐 hunk 核对，禁止包含用户已有差异。

## 2026-11-28 Harness V1.0 最终定向复审

- 结论：**H01 PASS、H02 PASS、H04 PASS**；H03 沿用 REVIEW-02 的 PASS，本轮未重复审查。仅限约定的三个问题，不代表真实数据库、外部服务或产品验收。未修复代码，未连接数据库或付费服务，未暂存、提交或推送。
- H01：独立临时 pytest 以 `localhost/aika_test_example` 为测试目标、分别以 `127.0.0.1` 和 `[::1]` 的同名普通库重放，两个进程均退出 1，fixture 输出 `Unsafe test database`，测试体的连接哨兵均未执行。
- H02：REVIEW-02 原始缺详情报告在 diagnostic、regression、acceptance 均返回 1；从完整对照报告分别删除 `case_id`、删除 `score`、更改汇总分数均返回 1，完整对照返回 0。定向测试覆盖重复/替换 ID、缺指标及保存状态。
- H04：CI 明确执行 `uv run python docs/project-tracking/task_gate.py`；同一程序读取其相邻实际清单。复制当前清单和门禁到临时目录，正常清单用相同命令退出 0；将 H02 ID 改为 H01 后退出 1，输出 `IDs must be nonempty and globally unique`。这是本地模拟 CI 命令，未触发远程 CI。
- `env -i`、临时 HOME、数据库变量置空执行 `pytest -q tests/test_database_test_safety.py tests/test_postgres_retrieval.py tests/test_eval_quality_gate.py tests/test_eval_platform.py tests/test_task_gate.py`：退出 0，39 passed、1 skipped；跳过的是破坏性 PostgreSQL 集成。临时复现材料位于 `/tmp/aika-final-review-Xvpadb`。
- 技术债务：任务门禁仅核验清单自述的一致性，不能证明历史或证据真实（REVIEW-02 已记录）；真实集成及既有业务评测告警仍待独立任务处理。
- 候选 checkpoint 相对本地 HEAD `6d7f795`：完整新增 `AGENTS.md`、`docs/ARCHITECTURE.md`、`docs/QUALITY.md`、`docs/project-tracking/{AUDIT.md,HANDOFF.md,REVIEW.md,REVIEW-02.md,features.json,task_gate.py}`、`tests/{conftest.py,test_database_test_safety.py,test_eval_quality_gate.py,test_task_gate.py}`；跟踪文件限 `.github/workflows/ci.yml` 新增 manifest step、`aika/cli.py` 的 QA 模式和质量门禁 diff、`docs/project-tracking/init.sh` 的 Harness 入口 diff、`docs/project-tracking/PROGRESS.md` 末尾两段、`tests/test_aika_mcp_install.py` 的 dry-run 隔离 hunk、`tests/test_postgres_retrieval.py` 的 fixture 替换及相应 import 删除。各文件按本地 HEAD 的 diff 审核，尤其混合文件逐 hunk 核实；原有业务、数据、配置及 `tests/test_curated_data_paths.py` 不纳入。缺少完整起始快照，无法仅凭现有证据证明所有混合文件 hunk 的来源；人工确认范围后再操作。

## Harness V1.0 初始化结束及 checkpoint 准备（2026-11-28）

用户已确认最终定向复审通过，Harness V1.0 初始化阶段正式结束。按已列候选范围准备选择性暂存；原有业务、数据和配置修改保留在工作树。暂存范围须经用户确认后才使用 `feat(harness): establish AIKA development harness v1.0` 提交；提交后核对分支和远程状态，另待用户确认推送。真实集成的未验证范围与已有技术债务保持不变。

暂存快照依赖核对：从 Git index 导出独立副本，复用 `.venv`，以独立临时 HOME/CODEX_HOME 和空数据库变量运行 H01/H02/H03/H04 相关测试，67 passed、1 skipped；`init.sh eval-smoke` 退出 0。首次把 HOME 设为快照目录，仓库跟踪的 `.codex/skills` 被当作用户已安装 skill，安装器测试 1 failed、66 passed、1 skipped；改为独立 HOME 后复测通过。快照未包含原有业务 diff 或实际数据库/外部服务。暂存内容和范围仍待用户确认，尚未提交或推送。
