# P0：重构报告信息架构，解决“证据 dump”

## 需求

当前报告的问题不是 evidence card 本身不好，而是证据卡放得太前、太多、太平铺，导致用户在看到答案之前先被证据淹没。对外报告应改为：

> 先回答用户想知道什么 -> 再告诉用户证据强不强 -> 最后展开证据细节。

报告拆成 4 层：

| 层级 | 名称 | 用户看到什么 | 是否默认展开 |
| --- | --- | --- | --- |
| Layer 1 | Executive Page | 1 页结论、覆盖率、关键判断、不可回答问题 | 默认展开 |
| Layer 2 | Visual Research Body | 产业链图、公司热力图、风险图、技术路线表 | 默认展开 |
| Layer 3 | Evidence Review | 每个结论对应的证据摘要、强弱评级、反证 | 半折叠 |
| Layer 4 | Evidence Appendix | 完整 evidence card、tool call、source manifest | 默认折叠 |

首页不再从“核心判断 + 证据卡片”开始，而是优先展示：

- 本次报告能回答什么。
- 本次报告不能回答什么。
- 证据覆盖评级。
- 最重要的 3 个发现。
- 当前报告适合的使用方式。

## 核心逻辑

- 将报告首页定位为 `Executive Page`，只承载高层判断和证据覆盖状态。
- 将完整 evidence card 从正文主体下沉到 `Evidence Appendix`，作为可审计底层能力保留。
- 正文中只展示证据摘要，例如证据强度、支撑证据 ID、反证状态、注意事项。
- 对证据不足的主题，不强行输出深度结论，而是明确展示“不能确认”的范围。
- 报告标题、摘要和正文语气都要和证据覆盖程度匹配。

## 验证方法

- 用“液冷产业链”样例报告验证首页结构：
  - 首页第一屏出现“一页结论”。
  - 首页包含“能回答什么 / 不能回答什么”。
  - 首页展示 Coverage、Report Type、Conclusion Usability。
  - 首页不直接堆完整 evidence card。
- 检查正文：
  - 每个核心结论附近只有证据摘要。
  - 完整 evidence card 出现在折叠附录或详情层。
  - 证据不足处明确写出“不能确认”或“当前证据不足”。
- 检查用户观感：
  - 用户无需先阅读证据卡，就能理解报告结论和局限。

## 预期效果

- 报告从“证据堆叠型 Markdown”变成“结论优先、证据可审计”的研究报告。
- 用户第一屏即可判断报告是否有用、能回答什么、不能回答什么。
- evidence card 的价值被保留，但不再干扰正文阅读。


# P1：加入报告类型自动降级机制

## 需求

如果证据覆盖不足，系统不应生成“深度研究报告”，而应自动降级为“覆盖审计报告”或“初步研究简报”。报告类型必须由证据覆盖数据驱动，避免标题给用户过高预期。

报告类型建议：

| 条件 | 标题 |
| --- | --- |
| 证据覆盖低 | 《液冷产业链证据覆盖审计报告》 |
| 证据覆盖中 | 《液冷产业链初步研究简报》 |
| 证据覆盖高 | 《液冷产业链分析报告》 |
| 证据覆盖很高 | 《液冷产业链深度研究报告》 |

## 核心逻辑

新增报告类型判断规则：

```python
def classify_report_type(coverage_score: float, company_coverage: float, direct_claim_ratio: float) -> str:
    if coverage_score < 0.3 or direct_claim_ratio < 0.25:
        return "evidence_coverage_audit"
    if coverage_score < 0.6:
        return "preliminary_research_brief"
    if coverage_score < 0.8:
        return "industry_research_report"
    return "deep_research_report"
```

核心输入：

- `coverage_score`：整体证据覆盖率。
- `company_coverage`：目标公司覆盖率。
- `direct_claim_ratio`：直接证据占比。
- `unsupported_claims`：无证据或弱证据结论数量。
- `freshness_status`：来源新鲜度。

输出影响：

- 自动选择报告标题。
- 自动选择报告模板。
- 自动控制结论措辞强度。
- 自动决定是否展示“覆盖审计”提醒。

## 验证方法

- 构造低覆盖样例：
  - `coverage_score=0.15`
  - `direct_claim_ratio=0.2`
  - 预期 `report_type=evidence_coverage_audit`
  - 标题为“证据覆盖审计报告”。
- 构造中覆盖样例：
  - `coverage_score=0.45`
  - 预期 `report_type=preliminary_research_brief`
  - 标题为“初步研究简报”。
- 构造高覆盖样例：
  - `coverage_score=0.75`
  - 预期 `report_type=industry_research_report`。
- 构造很高覆盖样例：
  - `coverage_score=0.85`
  - `direct_claim_ratio>=0.25`
  - 预期 `report_type=deep_research_report`。
- 验证低覆盖报告中不会出现“完整产业链深度报告”式标题或确定性过强的结论。

## 预期效果

- 用户不会因为标题预期过高而失望。
- AIKA 能主动承认证据边界，把“不足”变成产品可信度的一部分。
- 报告类型从 LLM 自由发挥变为可测试、可解释的规则输出。


# P2：引入 ReportSpec，不再直接生成 Markdown

## 需求

不要让 LLM 直接写最终 Markdown。工具应先生成结构化 `ReportSpec`，再由 renderer 输出 Markdown、HTML 或 PDF。

目标数据结构示例：

```json
{
  "report_type": "evidence_coverage_audit",
  "topic": "液冷产业链",
  "coverage": {
    "coverage_score": 0.15,
    "direct_claims": 3,
    "indirect_claims": 4,
    "unsupported_claims": 6,
    "covered_companies": 1,
    "target_companies": 9
  },
  "executive_summary": {
    "can_answer": [],
    "cannot_answer": [],
    "key_findings": []
  },
  "charts": {
    "flow_map": {},
    "company_coverage_heatmap": {},
    "evidence_strength_bar": {},
    "source_freshness_timeline": {}
  },
  "appendix": {
    "evidence_cards": []
  }
}
```

## 核心逻辑

建议新增结构：

```text
aika/
  report/
    spec.py
    builder.py
    templates/
      coverage_audit.html.j2
      research_brief.html.j2
    charts/
      flow_map.py
      heatmap.py
      evidence_bar.py
      freshness.py
    render_html.py
    render_markdown.py
```

实现路径：

- `spec.py` 定义 Pydantic `ReportSpec`、`CoverageSpec`、`ExecutiveSummarySpec`、`ChartSpec`、`AppendixSpec`。
- `builder.py` 将 MCP/tool results、claims、evidence cards、coverage audit 结果转换为 `ReportSpec`。
- `render_markdown.py` 负责输出 Markdown。
- `render_html.py` 负责输出 HTML 报告。
- 模板只消费结构化字段，不重新推理事实。
- LLM 只负责解释结构化数据，不负责制造图表数据和核心指标。

## 验证方法

- 新增或扩展单元测试：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_report_spec.py
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_report_builder.py
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_report_renderers.py
```

- 验证 `ReportSpec` 必填字段：
  - `report_type`
  - `topic`
  - `coverage`
  - `executive_summary`
  - `charts`
  - `appendix.evidence_cards`
- 验证 renderer：
  - Markdown renderer 能输出“一页结论”。
  - HTML renderer 能输出对应层级。
  - 低覆盖样例会选择 coverage audit 模板。
  - 图表数据为空时显示“当前证据不足”，而不是生成伪图表。

## 预期效果

- 报告生成从不可控的长文本生成，升级为结构化数据驱动。
- Markdown、HTML、PDF、Web UI 可以复用同一套 `ReportSpec`。
- 图表、证据、标题、结论强度都可被测试和回归验证。


# P3：第一阶段先做 4 张自动化图表

## 需求

第一版不要一次做 6 张图。优先实现最能改善观感、且可以由现有 AIKA 数据自动生成的 4 张：

1. 公司覆盖热力图。
2. 证据强度柱状图。
3. 产业链 flow map / Sankey。
4. Source freshness timeline。

风险雷达图和技术路线对比表放到第二阶段。

## 核心逻辑

### 公司覆盖热力图

展示当前知识库对每家公司、每个环节的证据覆盖程度，而不是展示“谁最强”。

分值规则：

| 分值 | 含义 |
| ---: | --- |
| 0 | 无证据 |
| 1 | mentioned |
| 2 | indirect |
| 3 | direct |
| 4 | direct + 多来源 |
| 5 | direct + 多来源 + 新鲜 + 有指标 |

### 证据强度柱状图

统计：

```json
{
  "direct": 3,
  "indirect": 4,
  "mentioned": 3,
  "unsupported": 6,
  "no_evidence": 9
}
```

用于展示证据质量结构，而不是包装结论。

### 产业链 flow map / Sankey

第一版命名为：

> Evidence-weighted supply chain map

节点示例：

```text
AI芯片/高功率封装
  -> AI服务器/超节点
  -> 液冷散热
  -> 冷板/CDU/工质
  -> 数据中心部署
```

重要约束：

> 线条粗细代表 AIKA 本地证据数量/强度，不代表市场规模或收入占比。

### Source freshness timeline

按年份或月份展示来源时效：

```json
[
  {"year": "2022", "count": 1, "freshness": "stale"},
  {"year": "2023", "count": 1, "freshness": "stale"},
  {"year": "2024", "count": 4, "freshness": "aging"},
  {"year": "2025", "count": 7, "freshness": "fresh"}
]
```

## 验证方法

- 为每张图新增数据构建测试：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_report_charts.py
```

- 验证公司覆盖热力图：
  - 未覆盖公司得分为 0。
  - 直接证据公司得分高于 mentioned。
  - 多来源、新鲜、有指标时得分更高。
- 验证证据强度柱状图：
  - direct、indirect、mentioned、unsupported、no_evidence 计数准确。
- 验证 Sankey：
  - links 包含 `source`、`target`、`value`、`evidence_ids`。
  - `value` 由证据数量/强度生成。
  - 图注包含“不代表市场规模或收入占比”。
- 验证 freshness timeline：
  - 不同年份来源能被正确聚合。
  - stale / aging / fresh 状态符合规则。

## 预期效果

- 报告观感从长文本升级为可视化研究报告。
- 用户能快速看出哪些环节证据充足，哪些环节证据缺失。
- AIKA 的优势从“会总结”变成“能展示研究覆盖质量”。


# P4：保留证据卡，但改为折叠式、可点击、可审计

## 需求

不要删除证据卡。证据卡应从正文主体升级为交互式/折叠式审计层。

正文只展示证据摘要：

```markdown
### 结论 1：紫光股份/H3C 有液冷产品级布局

证据强度：Direct
可用程度：可直接使用
支撑证据：E1, E2, E3
反证状态：未发现直接反证
注意事项：当前证据只能证明产品布局，不能证明收入占比或订单规模。
```

详细证据放到折叠块：

```html
<details>
  <summary>查看证据卡 E1：紫光股份 2025 年报</summary>

  来源：紫光股份 2025 年年度报告
  页码：p15
  Claim 类型：company_exposure
  证据原文：「2025年，公司又陆续推出多款液冷技术创新产品及配套解决方案」
</details>
```

## 核心逻辑

- 正文左侧展示结论。
- 正文右侧展示证据摘要卡。
- 点击证据 ID 后打开详情 drawer 或折叠块。
- drawer / appendix 中展示：
  - 原文。
  - 来源。
  - 页码或段落。
  - `source_url`。
  - `confidence`。
  - 反证状态。
  - freshness 状态。
- Markdown 版本使用 `<details>`。
- HTML 版本使用 drawer 或侧边证据面板。

## 验证方法

- Markdown 输出验证：
  - 正文每个结论有证据摘要。
  - 完整 evidence card 出现在 `<details>` 中。
  - `<summary>` 包含 evidence id 和来源名称。
- HTML 输出验证：
  - 点击 evidence id 能定位或打开对应证据详情。
  - 证据详情包含来源、日期、页码/段落、claim 类型、置信度、反证、时效。
- 数据一致性验证：
  - 正文中的 evidence id 都能在 appendix 找到。
  - appendix 中的 evidence card 能反查支撑的 conclusion id。

## 预期效果

- 证据卡从“阅读负担”变成“可信度增强层”。
- 用户可以先读报告，再按需审计证据。
- Markdown、HTML、Web UI 都能复用同一套 evidence card 数据。


# P5：第二阶段补充风险雷达图和技术路线对比表

## 需求

风险雷达图和技术路线对比表放第二阶段实现。原因是它们需要更多主观评分和行业知识，若第一阶段强行做，容易引入伪精确或幻觉。

## 核心逻辑

### 风险雷达图

评分维度：

| 维度 | 来源 |
| --- | --- |
| 技术风险 | bottleneck evidence 数量 |
| 供应链风险 | 供应商覆盖缺口 |
| 成本风险 | 是否缺 TCO/CAPEX/OPEX 数据 |
| 渗透率风险 | 是否缺部署规模/渗透率 |
| 证据风险 | unsupported/no evidence 比例 |
| 时效风险 | stale evidence 比例 |

图下注明：

> 风险分数为 AIKA 基于证据覆盖、证据时效和缺口数量生成的研究风险评分，不代表投资风险评级。

### 技术路线对比表

第一版优先用高质量表格，不强行画复杂图：

| 技术路线 | 适用场景 | 优点 | 缺点 | 关键部件 | 当前证据状态 |
| --- | --- | --- | --- | --- | --- |
| 冷板式液冷 | 高密度 AI 服务器 | 兼容性较好、落地快 | 改造复杂、仍需风液混合 | 冷板、CDU、快接头 | 待补证据 |
| 浸没式液冷 | 极高密度部署 | 散热效率高 | 运维、材料、可靠性挑战 | 工质、槽体、换热器 | 待补证据 |
| 喷淋式液冷 | 特定高热流密度场景 | 传热效率高 | 工程复杂 | 喷嘴、泵、过滤 | 待补证据 |
| 背板/门板换热 | 存量机房改造 | 改造相对温和 | 上限有限 | Rear-door HEX | 待补证据 |

## 验证方法

- 风险雷达图：
  - 每个风险分数都有可追溯的数据来源。
  - 缺少数据时标记为 `unknown` 或 `insufficient`，不生成伪精确分数。
  - 图注明确“不代表投资风险评级”。
- 技术路线表：
  - 每个技术路线的证据状态来自 evidence cards 或 coverage gaps。
  - 无证据支持的优缺点必须标为“待补证据”。
  - 表格不把行业常识包装成已验证结论。

## 预期效果

- 第二阶段在不牺牲可信度的前提下增强报告专业感。
- 风险和技术路线分析能够继承 AIKA 的证据审计能力。
- 避免图表看起来高级但实际不可验证。


# P6：工程原则和最终验收

## 需求

整个报告系统必须遵守以下原则：

- 不要让 LLM 负责“画图”，LLM 只负责解释图。
- 不要让 Sankey 代表市场规模，除非已有市场空间、收入或订单数据。
- 不要删除证据卡，而是把它变成可折叠、可点击、可审计的底层能力。
- 不要一次做 6 张图，先做自动化程度最高的 4 张。
- 不要急着做 Web SaaS，先用 Python + Plotly + HTML report 把 demo 做漂亮。

## 核心逻辑

最终方向：

> 把 AIKA 报告从“LLM 写出的长 Markdown”升级为“结构化 ReportSpec 驱动的可视化研究报告”：首页讲结论，正文讲图表，附录讲证据。

工程实现优先级：

1. `ReportSpec` 和报告类型自动降级。
2. Markdown renderer 改造。
3. HTML renderer 和 4 张图表。
4. 证据卡折叠和可追溯链接。
5. 第二阶段风险雷达图和技术路线表。

## 验证方法

核心回归命令：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_knowledge_pack.py \
  tests/test_aika_core.py \
  tests/test_aika_sqlite_backend.py \
  tests/test_aika_mcp_tools.py \
  tests/test_report_spec.py \
  tests/test_report_builder.py \
  tests/test_report_renderers.py \
  tests/test_report_charts.py
```

离线 smoke：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py demo --offline --task-dir /tmp/aiqasys-demo-tasks
UV_CACHE_DIR=/tmp/uv-cache uv run aika demo
```

人工验收：

- 用“液冷产业链”生成一份报告。
- 首页先展示结论和覆盖审计，而不是证据卡堆叠。
- 低覆盖时自动降级为“证据覆盖审计报告”。
- 图表只表达证据覆盖、证据强度、证据时效和证据流，不表达未经验证的市场规模。
- 每个结论都能追溯到 evidence card。
- 每张 evidence card 都能反查支撑了哪些结论。

## 预期效果

- AIKA 的报告体验从“长 Markdown + 引用”升级为“可视化、可追溯、可审计”的研究产品。
- 用户能快速判断结论是否可靠、证据是否充足、哪些问题仍不可回答。
- 结构化 `ReportSpec` 为后续 HTML/PDF/Web UI 复用打下基础。
- 项目差异化更清晰：AIKA 不只是产业链 RAG，而是一个以证据覆盖审计和可追溯研究为核心的 Agent。
