#!/usr/bin/env bash
# Explicit local helpers; never install packages or migrate databases automatically.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASK_ROOT"
case "${1:-status}" in
  status)
    git status --short --branch
    git log -3 --oneline
    if [[ -x .venv/bin/python ]]; then .venv/bin/python --version; else echo 'Python 环境缺失：先按 README 执行 uv sync'; fi
    if command -v node >/dev/null 2>&1; then node --version; else echo 'Node.js 未安装'; fi
    if [[ ! -d web/node_modules ]]; then echo '前端依赖缺失：在 web 目录执行 npm ci'; fi
    echo '进度：docs/project-tracking/PROGRESS.md；决策：docs/project-tracking/DECISIONS.md；接手流程：docs/project-tracking/AGENTS.md'
    ;;
  check)
    if [[ ! -x .venv/bin/python || ! -d web/node_modules ]]; then
      echo '缺少依赖，请先按 README 准备 Python 与前端环境。' >&2
      exit 1
    fi
    .venv/bin/python -m pytest -q tests/test_conversation_roles.py tests/test_conversation_store.py tests/test_api.py tests/test_centralized_multi_agent.py tests/test_dynamic_knowledge.py
    (cd web && npm run build)
    git diff --check
    ;;
  offline-api)
    [[ -x .venv/bin/python ]] || { echo '请先执行 uv sync' >&2; exit 1; }
    exec env QA_DISABLE_POSTGRES=true QA_GRAPH_BACKEND=csv LLM_API_KEY=replace-me .venv/bin/python main.py api --host 127.0.0.1 --port 8000
    ;;
  web)
    (cd web && exec npm run dev -- --host 127.0.0.1)
    ;;
  *)
    echo '用法：./docs/project-tracking/init.sh [status|check|offline-api|web]' >&2
    exit 2
    ;;
esac
