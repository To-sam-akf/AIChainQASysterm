#!/usr/bin/env bash
# Explicit helpers: no automatic database migration or paid evaluation.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASK_ROOT"
python_ready() { [[ -x .venv/bin/python ]] || { echo '缺少 Python 环境：执行 init.sh setup 或参照 README。' >&2; exit 1; }; }
frontend_ready() { [[ -d web/node_modules ]] || { echo '缺少前端依赖：执行 init.sh setup 或参照 README。' >&2; exit 1; }; }
offline() { env TEST_DATABASE_URL='' DATABASE_URL='' LLM_API_KEY='' EMBEDDING_MODEL='' QA_DISABLE_POSTGRES=true QA_GRAPH_BACKEND=csv "$@"; }
smoke() {
  python_ready
  local out
  out="$(mktemp -d "${TMPDIR:-/tmp}/aika-smoke-XXXXXX")"
  echo "离线产物：$out"
  offline .venv/bin/python main.py demo --offline --task-dir "$out/demo"
  offline .venv/bin/python main.py eval --suite qa --offline --limit 5 --report-dir "$out/eval" --json
  offline .venv/bin/python main.py eval --suite agent --offline --limit 1 --task-dir "$out/agent" --json
}
case "${1:-status}" in
  status)
    git status --short --branch
    git log -3 --oneline
    [[ ! -x .venv/bin/python ]] || .venv/bin/python --version
    echo '上下文：AGENTS.md；当前状态：docs/project-tracking/HANDOFF.md'
    ;;
  setup)
    command -v uv >/dev/null || { echo '缺少 uv' >&2; exit 1; }
    uv sync --frozen --all-extras --dev
    npm --prefix web ci
    ;;
  doctor)
    for tool in git uv node npm docker; do if command -v "$tool" >/dev/null; then echo "$tool: $(command -v "$tool")"; else echo "$tool: 缺失"; fi; done
    [[ -x .venv/bin/python ]] && .venv/bin/python --version || echo 'Python 虚拟环境缺失'
    [[ -d web/node_modules ]] && echo '前端依赖存在' || echo '前端依赖缺失'
    echo '不检查密钥或连接外部服务。'
    ;;
  task-gate) python_ready; .venv/bin/python docs/project-tracking/task_gate.py ;;
  check)
    python_ready; frontend_ready
    "$0" task-gate
    offline .venv/bin/python -m pytest -q tests/test_conversation_roles.py tests/test_conversation_store.py tests/test_api.py tests/test_centralized_multi_agent.py tests/test_dynamic_knowledge.py
    npm --prefix web run lint
    npm --prefix web run build
    git diff --check
    ;;
  test)
    python_ready
    echo '运行全部 pytest；建议在含 Git 元数据、隔离环境变量和个人数据的副本中执行。'
    offline .venv/bin/python -m pytest -q
    ;;
  e2e)
    python_ready
    [[ -n "${AIKA_WEB_TEST_URL:-}" ]] || { echo '需先启动本地 Vite 服务并设置 AIKA_WEB_TEST_URL；浏览器测试拦截 API。' >&2; exit 2; }
    command -v google-chrome >/dev/null || command -v chromium >/dev/null || { echo '缺少 Chrome/Chromium。' >&2; exit 2; }
    offline .venv/bin/python -m pytest -q tests/test_workspace_browser.py
    ;;
  eval-smoke) smoke ;;
  eval-full)
    python_ready
    out="$(mktemp -d "${TMPDIR:-/tmp}/aika-eval-full-XXXXXX")"
    echo "完整离线 QA/Agent 评测产物：$out；不包含付费 A/B/E/F 实验。"
    offline .venv/bin/python main.py eval --suite qa --offline --report-dir "$out/eval" --json
    offline .venv/bin/python main.py eval --suite agent --offline --task-dir "$out/agent" --json
    ;;
  verify-task) "$0" check; "$0" eval-smoke ;;
  offline-api)
    python_ready
    exec env QA_DISABLE_POSTGRES=true QA_GRAPH_BACKEND=csv LLM_API_KEY=replace-me .venv/bin/python main.py api --host 127.0.0.1 --port 8000
    ;;
  web) frontend_ready; (cd web && exec npm run dev -- --host 127.0.0.1) ;;
  *) echo '用法：init.sh [status|setup|doctor|check|test|e2e|eval-smoke|eval-full|verify-task|task-gate|offline-api|web]' >&2; exit 2 ;;
esac
