# AIChainQASysterm

AI 算力产业链知识图谱问答系统。

## 当前版本：V1.2 稳定版证据包增强的投研 GraphRAG

V1.2 已经从“事实型 KG + BM25 RAG”升级为“投研 Claim/Dossier + 图谱 + RAG”的混合链路，并把新增的专业技术源纳入知识图谱构建。系统现在会同时使用两类 Claim 来源：

- 从 `data/curated/relations.csv` 派生公司、主题、产业链和风险相关 Claim。
- 从 `data/chunks/industry_tech_*.jsonl` 的高信号原文 chunk 直接抽取技术机理 Claim，覆盖 technical roadmap、open specification、manual open specification、benchmark methodology、technical paper、model technical report 等行业技术源。

当前构建产物包括：

- `data/curated/claims.csv`：投研原子判断，包含主题、公司、敞口强度、机理、指标、风险、证据、置信度和时点。
- `data/curated/evidence_spans.csv`：可引用证据片段。
- `data/curated/segment_dossiers.jsonl`：按 AI 服务器、AI 芯片、光模块、液冷、数据中心、电源、PCB 等主题生成的产业链摘要。

本次构建后的 curated 层包含 6881 个实体、10521 条关系、11591 条 Claim、11591 条证据片段和 9 个主题 dossier；本地 RAG 索引包含 4858 个 chunk。

问答链路会优先使用 Claim/Dossier 组织“核心判断、技术机理、产业传导、公司排序、领先指标、反证/边界、证据”，再用图谱关系和 RAG 原文片段补充事实来源。液冷这类主题已能区分核心敞口、直接敞口、间接敞口和仅提及公司，避免把服务器、电源、IDC、PCB 等间接受益环节与液冷主业公司混在同一层级。对于 `Ultra Ethernet`、`UALink`、`UCIe/Chiplet`、`DeepSeek-V3`、`FlashAttention`、`PagedAttention`、`MLPerf`、`CPO/LPO/硅光`、`冷板/CDU` 等技术问题，检索和答案组织会优先引用 roadmap、spec、paper 和 benchmark 报告中的原文级技术 Claim。

V1.2 本轮稳定化新增：

- evidence pack 不再简单按分数截断，而是按问题类型分配 Claim/Dossier/Graph/RAG 证据预算，避免同类证据挤占全部上下文。
- 每条证据卡片带 `citation_id`，答案和前端证据抽屉可用 `E1/E2/...` 对齐关键判断。
- 回答子图由图谱记录和已选证据卡共同生成；即使答案主要来自 Claim/Dossier，React 工作台的“子图”也能展示公司、主题、风险和指标关系。
- 本地回归评测扩展到 15 个投研问题，并检查证据卡、引用编号和子图是否生成。

### V1.2 仍未完善的地方

- 原文级 Claim 抽取已经先覆盖 `industry_tech_*` 专业技术源，但年报、券商研报和普通行业白皮书仍主要依赖关系派生 Claim。后续需要把原文级抽取扩展到更多来源，并加入 LLM 审校和人工校验。
- Segment dossier 是确定性聚合摘要，不是 LLM 多文档综合后的高质量社区报告；部分机理句仍会带有原始抽取噪声，需要进一步做 LLM 审校和人工校验。
- 公司敞口分级已经可用，但仍是启发式规则。复杂场景下，例如“液冷兼容设计”“数据中心节能”“光模块上游材料”，还需要更细的敞口 taxonomy 和人工标注样本。
- 当前检索仍以 BM25、规则打分和 CSV 图谱为主，尚未接入 dense embedding、cross-encoder reranker 或真正的 GraphRAG global/local/DRIFT search。
- 反证和冲突处理只是预留字段，尚未系统识别“券商乐观判断 vs 年报风险披露”“老报告 vs 新报告”“技术路线 A vs B”的矛盾。
- 指标抽取还偏粗，不能稳定区分订单、合同负债、产能、毛利率、ASP、客户结构、资本开支、渗透率等投研指标。
- 评测集仍偏 smoke test，能保证主流程不退化，但还不能衡量“是否产生超前 insight”。需要扩展高难投研问题和分维度评分。
- 本次专业技术源并入的核心回归已通过；API 测试在当前沙箱里 `TestClient` 请求仍有挂起现象，后续需要单独定位 FastAPI/TestClient 与运行环境的兼容问题。

## 下一步：完全体升级计划

完全体目标是把系统做成“面向 AI 算力产业链上市公司的证据驱动投研分析引擎”，而不只是问答演示。核心标准是：答案能稳定给出技术因果链、产业传导路径、公司敞口排序、可验证领先指标、风险反证和证据边界。

### 1. 原文级 Claim 抽取扩展与审校

- 当前已经在 `scripts/build_research_artifacts.py` 中合并“关系派生 Claim”和“专业技术源原文直抽 Claim”。下一步要把直抽能力从 `industry_tech_*` 扩展到年报、券商研报和普通行业源。
- Claim schema 继续固定为：`claim_text`、`claim_type`、`topic`、`companies`、`exposure_level`、`mechanism`、`direction`、`horizon`、`metric/value/unit`、`source`、`evidence_span`、`confidence`、`as_of_date`。
- 抽取时强制区分：
  - 技术机理：为什么某技术重要。
  - 产业传导：需求如何从训练/推理、数据中心、芯片、互联传到上市公司。
  - 公司敞口：core/direct/indirect/mentioned。
  - 领先指标：订单、收入、毛利率、产能、客户导入、资本开支、PUE、功率密度、端口速率、渗透率。
  - 反证风险：技术路线替代、供给约束、客户集中、价格压力、政策约束、需求不及预期。
- 技术源默认不抽公司敞口，除非原文明确出现核心 A 股上市公司，避免把全球规范、论文或 benchmark 中的通用技术结论误映射到股票标的。

### 2. 主题社区报告升级

- 对每个主题生成 LLM 审校后的 `segment_dossier`，结构固定为：技术定义、需求驱动、瓶颈、产业链映射、公司敞口表、领先指标、风险反证、关键证据、证据缺口。
- 使用分层摘要：chunk claims -> report summary -> segment dossier -> cross-segment thesis。
- 对重点主题先做高质量白名单：AI 服务器、国产算力、训练/推理、光模块、CPO/LPO/硅光、液冷、电源、PCB/CCL、IDC/智算中心、算力网络。

### 3. 检索升级为真正 GraphRAG

- Query Router 明确分流：
  - “哪些公司/谁受益”走公司敞口排序。
  - “为什么/趋势/瓶颈”走 global dossier + claim 检索。
  - “公司对比”走两家公司 claim bundle + 指标/风险矩阵。
  - “订单/业绩/指标”只走指标 Claim，并在缺证据时明确拒答。
  - “继续说/它们/上述”走历史问题改写后再路由。
- 引入三阶段检索：BM25 高召回、dense embedding 语义召回、cross-encoder 或 LLM rerank 精排。
- 实现 DRIFT 式流程：宽问题先召回全局 dossier，再自动拆成子问题，最后回到局部公司/指标/风险证据。
- 图谱检索从单跳关系扩展到多跳路径：需求驱动 -> 技术瓶颈 -> 产业环节 -> 公司敞口 -> 指标验证 -> 风险反证。

### 4. 数据质量与人工校验台

- 为 `claims.csv` 增加 `review_status`、`reviewer_note`、`quality_flags`、`conflict_group_id`。
- 建立人工校验表或轻量前端：支持按主题查看 Claim、合并重复 Claim、调整敞口强度、标记噪声和冲突。
- 持续完善实体别名和技术 taxonomy。当前已初步覆盖 `UCIe`、`UALink`、`Ultra Ethernet/UET`、`Scale Up/Scale Out`、`SerDes`、`CPO/LPO/硅光`、`HBM`、`Chiplet`、`advanced packaging`、`FP8`、`FlashAttention`、`PagedAttention`、`MLPerf`、`cold plate/CDU` 等术语，后续需要继续做别名归一、主题映射和人工样本校验。
- 对年报财务表格、研报图表和白皮书指标做结构化解析，不再只依赖正文 OCR 文本。

### 5. 答案生成器升级

- 答案生成前先构造 evidence pack，每个关键结论都绑定证据编号、来源、页码、置信度和时点。
- 输出格式按问题类型变化：
  - 主题研究：核心判断、技术机理、产业传导、公司排序、领先指标、反证/边界、证据。
  - 公司对比：差异矩阵、共同驱动、分歧点、指标验证、风险差异。
  - 公司画像：业务卡位、产品代际、客户/订单证据、财务兑现、风险。
  - 空证据问题：明确缺口和建议补充的数据源。
- 加入事实一致性检查：答案中的公司、指标、年份、数值必须能在 evidence pack 中找到。

### 6. 高难评测体系

- 扩展 30-50 个高难投研问题，覆盖训练/推理分化、国产算力瓶颈、液冷渗透、光模块代际、订单缺口、公司比较、反证判断。
- 评分维度从关键词改为 2/1/0 或 5 分制：
  - 事实正确性。
  - 证据支撑。
  - 公司敞口排序。
  - 技术因果链。
  - 领先指标。
  - 反证/边界。
  - 无幻觉。
- 增加检索指标：claim recall@10、direct exposure precision@10、证据页码有效率、低价值片段占比。
- 每次构建后输出评测报告，记录失败样例和错误类型。

### 7. 工程化与前端展示

- 前端证据抽屉增加 Claim/Dossier 视图，用户能看到“这个结论来自哪个 Claim、哪个原文片段、敞口强度如何判定”。
- 数据概览页增加 Claim 数量、Dossier 数量、直接敞口公司分布、低质量证据比例。
- API status 暴露 `research_enabled` 和 research artifact 错误。
- Neo4j 导入支持新增本体和关系，并可选择导入 Claim 节点和 EvidenceSpan 节点。
- 完成 `tests/test_api.py` 在当前环境中的挂起问题定位，恢复全量 `pytest -q` 一次性通过。

## 第一阶段：数据准备

初始化并下载最新可用年报、公开 AI 算力产业链研报、权威行业白皮书和专业技术源：

```bash
python scripts/prepare_stage1_data.py --kind all --max-research 10
```

只查看候选文件，不下载：

```bash
python scripts/prepare_stage1_data.py --kind annual --dry-run
python scripts/prepare_stage1_data.py --kind research --max-research 10 --dry-run
python scripts/prepare_stage1_data.py --kind industry --dry-run
```

输出目录：

- `data/raw_pdfs/annual/`：30 家核心上市公司的最新可用年报。
- `data/raw_pdfs/research/`：公开可直接访问的 AI 算力产业链研报。
- `data/raw_pdfs/industry/`：中国信通院等权威机构白皮书、政策、标准资料，以及 IRDS/HIR、UCIe、UALink、Ultra Ethernet、OCP/OIF、MLPerf、arXiv 论文和模型技术报告等专业技术源。
- `data/metadata/companies_extended.csv`：30 家核心上市公司、别名和产业链环节。
- `data/metadata/research_keywords.csv`：研报检索关键词配置。
- `data/metadata/industry_sources.csv`：行业知识源配置，支持 `authority_whitepaper`、`technical_roadmap`、`open_specification`、`manual_open_specification`、`benchmark_methodology`、`technical_paper`、`model_technical_report` 等类型。
- `data/metadata/reports_manifest.csv`：PDF 来源、状态、SHA256、文件大小和页数。

手工下载或暂缺直链的开放规范可以保留为 `manual_open_specification`；如果本地文件尚未补齐，manifest 会保留 `manual_reference` 状态，不阻塞已下载技术源入库。

## 第二、三阶段：知识抽取与图谱构建

配置本地环境变量文件：

```bash
cp .env.example .env
```

`.env` 默认使用 DeepSeek OpenAI 兼容接口，填入 `LLM_API_KEY` 后即可运行。当前示例配置启用 DeepSeek 思考模式：

- `LLM_MODEL=deepseek-v4-pro`
- `LLM_THINKING_ENABLED=true`
- `LLM_REASONING_EFFORT=high`

如果账号仍使用旧版推理模型，可把 `LLM_MODEL` 改为 `deepseek-reasoner`。

解析 PDF 并生成文本块：

```bash
python scripts/parse_pdfs.py --manifest data/metadata/reports_manifest.csv
```

调用 LLM 抽取实体关系。建议按报告类型分批跑：

```bash
python scripts/extract_knowledge.py --kind research --contains 算力 --limit-chunks 20 --sleep 0.3
python scripts/extract_knowledge.py --kind annual --contains 服务器 --limit-chunks 50 --resume --sleep 0.3
python scripts/extract_knowledge.py --kind industry --contains 智能算力 --limit-chunks 50 --resume --sleep 0.3
python scripts/extract_knowledge.py --kind industry --contains UCIe --limit-chunks 20 --resume --sleep 0.3
```

一次性跑完
```bash
python scripts/extract_knowledge.py --resume --sleep 0.3
```

生成可人工校验的实体和关系表：

```bash
python scripts/build_verified_graph.py
```

启动 Neo4j 并导入图谱：

```bash
docker compose up -d neo4j
python scripts/load_neo4j.py --clear
```

如果当前机器没有 Docker 权限，可以先校验 CSV 是否满足导入条件：

```bash
python scripts/load_neo4j.py --dry-run
```

生成目录：

- `data/parsed_text/`：逐页文本 JSONL 和合并 TXT。
- `data/chunks/`：面向 LLM 抽取的文本块。
- `data/extracted/`：LLM 原始抽取 JSONL 和错误记录。
- `data/verified/entities.csv`、`data/verified/relations.csv`：可人工校验后导入 Neo4j 的图谱数据。

新增行业本体节点和关系：

- 节点：`IndustryConcept`、`Policy`、`Standard`、`ValueChainSegment`。
- 关系：`UPSTREAM_OF`、`DOWNSTREAM_OF`、`ENABLES`、`CONSTRAINS`、`DEFINES`、`SUPPORTED_BY_POLICY`。
- 关系保留 `source_tier`，公司实体保留 `is_core_company`，用于区分核心上市公司和一般提及主体。
- 技术源抽取提示会更偏向 ontology-only：优先抽技术、瓶颈、指标、标准和产业环节；只有原文明确出现核心 A 股上市公司时才抽 `Company`。

## 第四阶段：Neo4j + 本地 RAG + LLM 问答

先生成专业版 curated 图谱。该步骤会从 `data/verified/` 自动图谱中过滤非核心上市公司噪声、目录/释义页误抽取关系和低价值会计科目指标，并同步生成投研 Claim、证据片段和主题 dossier：

```bash
python scripts/build_curated_graph.py
```

构建本地 RAG 索引：

```bash
python scripts/build_rag_index.py
```

如果只想重建投研推理层，可以单独运行：

```bash
python scripts/build_research_artifacts.py
```

默认会合并两类来源：

- 从 `data/curated/relations.csv` 派生公司、主题、产业链和风险 Claim。
- 从 `data/chunks/industry_tech_*.jsonl` 选择性直抽 `mechanism`、`bottleneck`、`indicator`、`supply_chain`、`risk` 等技术 Claim，并过滤英文法律声明、版权页、目录页、参考文献和纯表格噪声。

如果需要回到只使用关系派生 Claim 的旧行为，可以运行：

```bash
python scripts/build_research_artifacts.py --no-direct-claims
```

专业问答链路：

- `QuestionPlan` 先解析问题意图、公司、主题、关系、是否比较、是否只看核心上市公司。
- 图谱检索默认读取 `data/curated/`；Neo4j 可用时作为增强后端，不可用时自动降级 CSV。
- 本地 RAG 使用 `jieba + BM25` 检索原文块，带同义词扩展、来源优先级、噪声过滤和去重。技术问题会提升 roadmap、spec、paper、benchmark 类证据权重；普通公司敞口问题仍优先年报和券商研报。
- 结构化证据会统一成 `evidence_cards`，再生成“结论、证据、研究要点、风险与边界”格式答案。
- 投研增强证据会额外读取 `claims.csv`、`evidence_spans.csv` 和 `segment_dossiers.jsonl`，优先组织“核心判断、技术机理、产业传导、公司排序、领先指标、反证/边界、证据”格式答案。
- 答案只做事实归纳和研究框架，不提供买卖建议、目标价或收益预测。

新增配置：

- `KG_DATA_DIR`：专业图谱目录，默认 `data/curated`。
- `QA_GRAPH_BACKEND`：`auto`、`csv` 或 `neo4j`，默认 `auto`。
- `QA_CORE_COMPANIES_ONLY`：公司列表类问题默认只返回核心 A 股上市公司。
- `QA_RERANK_TOP_N`：证据重排候选数量。
- `QA_EVIDENCE_TOP_N`：最终进入答案的证据卡片数量。
- `RAG_INDEX_DIR`：本地 RAG 索引目录，默认 `data/rag`。
- `RAG_TOP_K`：每次问答检索的本地文档块数量。
- `RAG_SEARCH_CACHE_SIZE`：本地 RAG 查询结果 LRU 缓存大小，默认 128。
- `RESEARCH_ARTIFACT_DIR`：投研 Claim/Dossier 目录，默认 `data/curated`。
- `QA_GRAPH_LIMIT`：Neo4j 查询结果上限。
- `QA_ENABLE_LLM_CYPHER`：是否启用 LLM 生成 Cypher；默认关闭，使用本地模板查询。
- `QA_ENABLE_LLM_PLANNER`：是否启用 LLM 问题规划；默认关闭，优先使用本地启发式规划。
- `QA_ENABLE_AGENT`：是否启用规则优先的最小智能体 Runner；默认开启，可设为 `false` 回退旧 workflow。
- `QA_AGENT_MAX_STEPS`：Agent ReAct 循环最大步数，默认 4，实现上限 4。
- `QA_CONTEXTUALIZER_MODE`：追问改写模式，支持 `auto`、`heuristic`、`llm`，默认 `auto`。
- `QA_HISTORY_MAX_TURNS`：连续问答时传入模型的最近对话轮数，默认 3。
- `QA_HISTORY_MAX_CHARS`：连续问答历史的最大字符数，默认 4000。
- `QA_UI_RENDER_LATEST_ONLY`：前端是否只默认渲染选中轮次的证据详情，默认开启。
- `LLM_THINKING_ENABLED`：是否向 DeepSeek 请求开启思考模式，快问快答默认关闭。
- `LLM_REASONING_EFFORT`：DeepSeek 思考强度，快问快答默认 `low`。

运行专业问答回归评测：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_frontend_data.py tests/test_research_claims.py tests/test_rag_qa.py tests/test_professional_qa.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/load_neo4j.py --dry-run
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_qa.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/benchmark_qa_speed.py
```

如果要让评测也调用已配置的 LLM：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_qa.py --use-llm
```

本次稳定版增强后的核心回归结果：

- `tests/test_frontend_data.py`、`tests/test_research_claims.py`、`tests/test_rag_qa.py`、`tests/test_professional_qa.py` 共 26 个测试通过。
- `scripts/evaluate_qa.py` 扩展为 15 个稳定版样例，当前 30/30 通过，并检查 evidence cards、citation id 和 answer subgraph。
- `scripts/load_neo4j.py --dry-run` 校验通过，当前 curated CSV 可导入。

可重点抽样的问题：

- `Ultra Ethernet 对算力网络有什么意义？`
- `DeepSeek-V3 对训练算力瓶颈有什么启示？`
- `UCIe/Chiplet 对国产算力产业链的传导是什么？`

## 第五阶段：React + FastAPI 前端展示

推荐使用新的 React 工作台。后端 API 复用现有 `QAEngine`、本地图谱和 RAG 索引，并把问答历史自动保存到 `data/conversations/`，前端可以直接点击历史会话恢复并继续追问。

安装 Python 依赖后启动 API：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn src.api:app --reload --port 8000
```

安装并启动前端：

```bash
cd web
npm install
npm run dev
```

浏览器打开 Vite 输出的地址（默认 `http://localhost:5173`）。Vite 会把 `/api` 请求代理到 `http://127.0.0.1:8000`。

如果 8000 端口已被占用，可以把 API 启动到其他端口，并在启动 Vite 时指定代理目标：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn src.api:app --reload --port 8001
cd web
VITE_API_PROXY_TARGET=http://127.0.0.1:8001 npm run dev
```

React 工作台包括：

- 智能问答：主流 chatbot 式对话流，支持连续追问、发送中状态、错误提示和证据详情抽屉。
- 自动会话库：每轮问答自动落盘，侧栏可新建、恢复、重命名、删除、导出 Markdown。
- 数据概览：实体、关系、报告数量、图谱/RAG/LLM 状态和分布。
- 产业链图谱：按公司、技术、关系类型筛选子图和明细。
- 输入框模型控制：可在对话框底部切换 DeepSeek 思考模式，并循环选择 `low`、`medium`、`high` 思考强度。

生产构建：

```bash
cd web
npm run build
```

### Streamlit 旧入口

保留 Streamlit 版本作为轻量演示和回退入口：

启动 Streamlit：

```bash
streamlit run app.py
```

Streamlit 页面包括：

- 数据概览：实体、关系、报告数量和分布。
- 智能问答：支持连续多轮追问，展示问题规划、专业答案、模型思考过程、Cypher/CSV 查询意图、图谱结果、本地 RAG 命中、证据卡片、诊断状态和子图。
- 侧栏模型设置：可在前端按轮次开启或关闭 DeepSeek 思考模式，并选择 `low`、`medium`、`high` 思考强度。
- 侧栏对话记录：保留当前会话历史，支持新建对话、保存到 `data/conversations/`、查看已保存记录、下载 Markdown 或 JSON。
- 图谱展示：支持按公司、技术、关系类型筛选子图。

可重点演示的问题：

- `液冷产业链有哪些上市公司，各自处于什么环节？`
- `中际旭创和新易盛在光模块业务上的差异是什么？`
- `英维克液冷业务进展和主要风险是什么？`
- `AI算力产业链当前最大的瓶颈是什么？`
