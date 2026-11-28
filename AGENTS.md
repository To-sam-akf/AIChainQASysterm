# AIKA 开发入口

先读 [当前交接](docs/project-tracking/HANDOFF.md)、[功能状态](docs/project-tracking/features.json) 和 [接手规则](docs/project-tracking/AGENTS.md)；按任务查阅 [架构](docs/ARCHITECTURE.md)、[质量基线](docs/QUALITY.md)、[历史进度](docs/project-tracking/PROGRESS.md) 与 [决策](docs/project-tracking/DECISIONS.md)。用户指令优先。

- WIP=1：先写验收合同，完成并验证当前任务后再开始下一任务；开发和独立审查分开。
- 先检查 Git 状态，保护已有改动；不得未经授权提交或推送。验收后更新功能状态、交接及必要决策；获授权时按任务单独保存 Git checkpoint，先核对范围。
- `./docs/project-tracking/init.sh status|doctor|check|test|e2e|eval-smoke|eval-full|verify-task` 为统一入口；`setup` 会安装依赖。具体边界见质量文档。
- 不自动运行付费模型、生产迁移或破坏性数据库操作；离线或模拟验证不能替代真实集成验收。对失败、跳过、告警和未验证保持如实记录。
