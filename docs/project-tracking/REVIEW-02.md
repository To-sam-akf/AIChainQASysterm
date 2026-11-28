# AIKA Harness 独立复审（REVIEW-02）

日期：2026-11-28。**Harness 初始化复审 FAIL**：H01、H02、H04 各有可复现的合同缺口；H03 PASS。本报告不构成 AIKA 产品质量或真实集成验收。原 [REVIEW.md](REVIEW.md) 未修改。本轮不修复实现，不提交或推送。

## 方法与范围

从当前工作树另建 `/tmp/aika-review02-azoCkC/repo`，本地克隆独立 Git 元数据并覆盖跟踪及非忽略的未跟踪工作树文件，移除工作树中已删除的五个文件；复用 `.venv` 和 `web/node_modules`，不复制原 `.env` 和个人配置。运行时用 `env -i` 和临时 HOME；目标测试另在副本写入无害远程 `.env` sentinel 后删除。全部证据来自本轮执行；日志在 `/tmp/aika-review02-azoCkC/`，非持久档案。未连接数据库、外部模型或付费服务。未运行真实浏览器到后端联调或 PostgreSQL 集成。

## H01 — FAIL：本地别名绕过同库保护

默认隔离和部分拒绝路径有效：带无害远程 `.env` 的定向测试为 27 passed、1 skipped（包含 H01/H02/H04），且默认跳过破坏性集成测试；显式 `--allow-destructive-postgres` 配合 `TEST_DATABASE_URL=postgresql://remote/aika_test_example` 单项运行退出 1，在 fixture setup 报 `Unsafe test database`，未执行迁移或连接。`tests/conftest.py` 在 pytest 配置阶段保存启动时地址，并在未启用时清空两个变量；dotenv 授权路径亦有子进程测试。

**阻塞缺陷**：无连接调用 `validate_test_database('postgresql://localhost/aika_test_example', 'postgresql://127.0.0.1/aika_test_example')` 返回测试地址，未拒绝；相同 hostname 且默认端口不同写法则被拒绝。两个 loopback 名称可指向同一 PostgreSQL 实例和库，这违反“拒绝与普通数据库相同的目标”的合同；启用后目标测试会迁移并执行 TRUNCATE。此处只复现校验放行，未连接数据库。普通地址仅取 pytest 启动时环境快照；如果普通地址只存在于 `.env`，同库比较也无法看到它。专用库名本身不能证明库无重要数据。

建议在连接前保守规范化本地 loopback 别名及端口，对无法确定是否相同的普通地址采取拒绝策略；明确处理 `.env` 中普通目标或要求显式声明，并用模拟连接验证所有拒绝在连接之前发生。

## H02 — FAIL：不完整报告可通过验收

定向质量测试和评测平台测试通过。现有检查正确区分已识别状态：诊断允许 fail/warn，回归拒绝 fail、允许 warn，验收拒绝 fail/warn 且要求可读已保存报告；空结果、计数不一致及无保存报告被现有测试拒绝。`eval-smoke` 本轮退出 0，QA 5 题 3 pass、2 warn，属于诊断结果，不是产品验收。

**阻塞缺陷**：构造 `r={'dataset':{'cases':1}, 'summary':{'cases':1,'passed':1,'warned':0,'failed':0,'overall_score':0.8}, 'results':[{'status':'pass'}]}`，分别调用 `qa_quality_exit(r, mode, saved=True)`，diagnostic、regression、acceptance 均返回 0。这个报告缺少真实报告的题目 ID、问题、逐题分数、运行和数据集元信息等，无法核对题目结果；`saved=True` 模拟保存检查成功，不证明实际文件损坏路径已复现。现有校验只核对数量和状态，不能可靠拒绝不完整报告或重复/替代题目 ID。建议定义并验证必要报告字段和逐题唯一 ID，与独立加载的完整数据集 ID 集合核对，覆盖保存后读取的缺失、重复及替换结果。此问题独立于现存业务 Bad Case。

对照本地 HEAD：`data/eval/qa_benchmark_v1.jsonl`、`aika/eval/dataset.py`、`runner.py`、`metrics.py`、`auto_judge.py`、`answer_judge.py` 无工作树差异；未发现题目、标准答案或上述评分规则被本次工作树擅自修改。该结论限于所核对文件和本地 HEAD；`experiment_artifacts.py` 有已有 curated 数据路径变更，非题目或评分规则。

## H03 — PASS：安装器测试隔离

独立副本中的完整安装器测试分别以空 HOME、以及含预先创建的 `.codex/config.toml`（已有 `aika` 项）的另一 HOME/CODEX_HOME 运行，均退出 0、28 passed。修复的 dry-run 用例使用临时 cwd、HOME、CODEX_HOME、模拟 skill 来源和宿主命令；没有改动真实配置。两种环境均为模拟，不代表真实宿主集成已验收。

## H04 — FAIL：实际清单未接入 CI

`tests/test_task_gate.py` 本轮 3 passed，覆盖重复 ID、非法依赖、环、多个进行中任务、非法转换、缺少合同/证据/阻塞原因及未完成依赖；正常 `init.sh check` 实际调用门禁并通过。

**阻塞缺陷**：在隔离副本直接把实际 `features.json` 的 H02 ID 改为 H01，执行 CI 所运行的相关 pytest `tests/test_task_gate.py` 仍退出 0（3 passed），而 `init.sh task-gate` 对同一文件退出 1（重复 ID 等错误）。CI 的 `.github/workflows/ci.yml` 运行 pytest，却不调用 `task-gate` 或 `check`；pytest 仅对内置样例调用 `validate()`，因此非法实际清单可进入 CI。测试后已恢复副本。建议 CI 显式校验实际清单并以修改实际清单的负例验证。

另外，门禁只验证文件当前自述的一致性：在内存将任务历史重写为合法序列并填入伪造证据，`validate` 返回空错误。它不能证明转换历史和证据真实，也不能防止手工同步改写；需根据期望的可信边界，引入基于受保护基线的转换比较及可核验的证据，或明确依赖人工复核。此限制与上述 CI 未调用实际检查是两个不同问题。

## 本轮回归与边界

| 命令或检查 | 本轮结果 |
| --- | --- |
| `init.sh status` | 成功；HEAD `6d7f795`，本地 main 相对本地 origin/main 落后 1，未 fetch。 |
| `init.sh test` | 退出 0；543 passed、3 skipped（PostgreSQL 1、浏览器 2）。 |
| `init.sh check` | 退出 0；实际 task-gate、定向 38 passed、前端 lint/build 和 diff 检查通过；构建仍警告 chunk 大于 500 kB。 |
| `init.sh eval-smoke` | 退出 0；离线 demo、QA 与 Agent 完成，QA 3 pass/2 warn；诊断成功不等于质量验收。 |
| 无害 `.env` 下 H01/H02/H04 定向测试 | 退出 0；27 passed、1 skipped。显式远程数据库负例退出 1，未连接。 |
| 安装器测试，两种 HOME | 两次退出 0，各 28 passed。 |
| 非法真实清单：pytest 样例 / task-gate | 分别退出 0 / 1；见 H04。 |
| 基准、评分文件的 `git diff` | 指定文件无差异；范围见 H02。 |

未运行 `setup`、付费实验、真实 PostgreSQL、真实外部服务或完整浏览器链路；离线通过不能替代这些验收。既有业务、数据、配置及脚本改动与 Harness 文件混在同一工作树，已有来源记录见 `AUDIT.md`，但没有完整起始快照可证明每个 hunk 的归属。特别是 `aika/cli.py`、`tests/test_aika_mcp_install.py`、`tests/test_postgres_retrieval.py`、`init.sh` 及 `PROGRESS.md` 为混合或跟踪文件，未来检查点必须逐 hunk 核对；数据删除、`data/curated_merged/` 和用户已有业务改动不能仅凭本次状态整体纳入。当前 FAIL，故不提供已获验收的 checkpoint 文件清单，也不暂存、提交或推送。
