# AIChainQASysterm

AI 算力产业链知识图谱问答系统。

## 第一阶段：数据准备

初始化并下载最新可用年报和公开 AI 算力产业链研报：

```bash
python scripts/prepare_stage1_data.py --kind all --max-research 5
```

只查看候选文件，不下载：

```bash
python scripts/prepare_stage1_data.py --kind annual --dry-run
python scripts/prepare_stage1_data.py --kind research --max-research 5 --dry-run
```

输出目录：

- `data/raw_pdfs/annual/`：10 家目标公司的最新可用年报。
- `data/raw_pdfs/research/`：5 份公开可直接访问的 AI 算力产业链研报。
- `data/metadata/companies.csv`：目标公司清单。
- `data/metadata/reports_manifest.csv`：PDF 来源、状态、SHA256、文件大小和页数。

## 第二、三阶段：知识抽取与图谱构建

配置本地环境变量文件：

```bash
cp .env.example .env
```

`.env` 默认使用 DeepSeek OpenAI 兼容接口，填入 `LLM_API_KEY` 后即可运行。

解析 PDF 并生成文本块：

```bash
python scripts/parse_pdfs.py --manifest data/metadata/reports_manifest.csv
```

调用 LLM 抽取实体关系。当前 15 份 PDF 约 1120 个 chunk，建议分批跑：

```bash
python scripts/extract_knowledge.py --kind research --contains 算力 --limit-chunks 20 --sleep 0.3
python scripts/extract_knowledge.py --kind annual --contains 服务器 --limit-chunks 50 --resume --sleep 0.3
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

## 第五阶段：前端展示

启动 Streamlit：

```bash
streamlit run app.py
```

前端直接进入系统，不做营销页。页面包括：

- 数据概览：实体、关系、报告数量和分布。
- 智能问答：展示问题、答案、Cypher、证据链、查询结果和子图。
- 图谱展示：支持按公司、技术、关系类型筛选子图。
