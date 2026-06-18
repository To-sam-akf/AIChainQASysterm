# AIKA Local MCP 使用方式

## 一键注册到 Claude Code

源码运行版推荐入口：

```bash
uv --directory /path/to/AIQASYS run aika mcp install --host claude-code --scope user
claude
/mcp
```

完成后，Claude Code 的 MCP 列表中应能看到 `aika`，并能列出 AIKA Phase 3 暴露的投研 tools。

## 常用命令

```bash
aika mcp install --host claude-code --scope user
aika mcp install --host claude-code --scope project
aika mcp install --host claude-code --scope user --force
aika mcp install --host claude-code --scope user --dry-run
aika mcp doctor
aika mcp config --host claude-code
```

- `install`：自动调用 Claude Code CLI 写入名为 `aika` 的 MCP server 配置。
- `doctor`：检查 `uv`、`aika mcp`、SQLite index 和 Claude Code MCP 配置。
- `config`：只打印底层 MCP server JSON，适合高级用户复制到目标宿主。
- `--force`：仅覆盖已存在的 `aika` server，不影响其他 MCP server。
- `--dry-run`：打印将写入的 JSON 和将执行的命令，不修改真实宿主配置。

## 诊断

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run aika mcp doctor
```

`doctor` 会输出 `[PASS]`、`[WARN]`、`[FAIL]` 和修复建议。Claude Code 未安装或尚未配置 `aika` 时会给出 warning；SQLite index 缺失、MCP server 无法启动、`uv` 不可用时会返回 failure。

## 高级 JSON 输出

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run aika mcp config --host claude-code
```

源码运行版会生成类似：

```json
{
  "type": "stdio",
  "command": "/abs/path/to/uv",
  "args": ["--directory", "/abs/path/to/AIQASYS", "run", "aika", "mcp"],
  "env": {
    "UV_CACHE_DIR": "/tmp/uv-cache"
  },
  "timeout": 600000
}
```

后续打包为 `uv tool install aika` 或 `pipx install aika` 后，配置可以自然收敛为 `command: "aika"`、`args: ["mcp"]`。
