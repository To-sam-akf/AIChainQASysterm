可以。新的项目流程应该改成：

> **财报/研报 → LLM 抽取知识 → Neo4j 构建图谱 → LLM 理解问题 → 查询图谱 → LLM 生成答案 → 前端展示**

也就是说，**问答必须由 LLM 参与，但答案不能让 LLM 直接瞎答，而是让 LLM 基于知识图谱检索结果回答。**

---

# 一、最终系统架构

```text
财报 PDF / 研报 PDF
        ↓
文本解析与清洗
        ↓
LLM 抽取实体和关系
        ↓
人工校验
        ↓
Neo4j 知识图谱
        ↓
用户提问
        ↓
LLM 解析问题
        ↓
生成 Cypher 查询
        ↓
查询 Neo4j
        ↓
LLM 基于查询结果生成答案
        ↓
前端展示答案、证据、图谱
```

这个流程仍然满足作业要求中的：**数据展示 + 图谱 + 问答效果 + 评测**。

---

# 二、项目最小完成版本

建议不要一开始做太大，先做一个最小可行版本。

## 数据规模

```text
10—20 家 AI 算力相关上市公司
10—20 份年报
5—10 份研报
30—50 个问答测试问题
```

## 实体类型

```text
公司 Company
技术 Technology
产品 Product
产业链环节 IndustryChain
财务指标 Metric
风险 Risk
报告 Report
```

## 关系类型

```text
公司 — 涉及技术 — 技术
公司 — 拥有产品 — 产品
公司 — 属于环节 — 产业链环节
公司 — 披露指标 — 财务指标
公司 — 面临风险 — 风险
实体 — 来源于 — 报告
```

---

# 三、完整实现流程

## 第一步：收集数据

收集 AI 算力产业链相关公司的年报和研报。

公司可以分几类选：

```text
AI 服务器：浪潮信息、中科曙光、工业富联
光模块：中际旭创、新易盛、天孚通信
液冷：英维克、申菱环境
芯片：寒武纪、海光信息
数据中心：奥飞数据、光环新网
```

文件命名建议：

```text
浪潮信息_2024年报.pdf
中科曙光_2024年报.pdf
AI算力产业链研报_2024.pdf
```

---

## 第二步：解析 PDF

把财报和研报 PDF 转成文本。

输出文件：

```text
data/parsed_text/
├── 浪潮信息_2024年报.txt
├── 中科曙光_2024年报.txt
├── AI算力产业链研报.txt
```

重点保留这些章节：

```text
公司业务概要
核心竞争力分析
管理层讨论与分析
研发投入
风险因素
财务指标
产业链分析
重点公司分析
```

这一步可以用：

```text
PyMuPDF
pdfplumber
```

---

## 第三步：用 LLM 抽取实体和关系

这一步可以半自动完成。

把每段文本交给 LLM，让它抽取结构化 JSON。

### LLM 抽取 Prompt 示例

```text
你是知识图谱构建助手。

请从以下财报或研报文本中抽取中国 AI 算力产业链相关知识。

需要抽取的实体类型：
1. Company：公司
2. Technology：技术
3. Product：产品
4. IndustryChain：产业链环节
5. Metric：财务指标
6. Risk：风险
7. Report：报告

需要抽取的关系类型：
1. USES_TECHNOLOGY：公司涉及技术
2. HAS_PRODUCT：公司拥有产品
3. BELONGS_TO_CHAIN：公司属于产业链环节
4. HAS_METRIC：公司披露财务指标
5. DISCLOSES_RISK：公司披露风险
6. MENTIONED_IN：实体来源于报告

要求：
1. 只抽取文本中明确出现的信息。
2. 不要编造。
3. 每条关系必须给出 evidence 原文证据。
4. 输出严格 JSON。

文本如下：
{{text}}
```

### LLM 输出示例

```json
{
  "entities": [
    {
      "type": "Company",
      "name": "浪潮信息"
    },
    {
      "type": "Technology",
      "name": "AI服务器"
    },
    {
      "type": "Product",
      "name": "服务器"
    },
    {
      "type": "IndustryChain",
      "name": "中游服务器制造"
    }
  ],
  "relations": [
    {
      "head": "浪潮信息",
      "head_type": "Company",
      "relation": "USES_TECHNOLOGY",
      "tail": "AI服务器",
      "tail_type": "Technology",
      "evidence": "公司布局 AI 服务器相关产品与解决方案",
      "source": "浪潮信息_2024年报"
    },
    {
      "head": "浪潮信息",
      "head_type": "Company",
      "relation": "HAS_PRODUCT",
      "tail": "服务器",
      "tail_type": "Product",
      "evidence": "公司主要产品包括服务器及相关解决方案",
      "source": "浪潮信息_2024年报"
    }
  ]
}
```

---

## 第四步：人工校验后导入 Neo4j

LLM 抽取后不要直接全导入，先导出成 Excel 检查。

关系表可以这样：

| head | relation        | tail  | evidence | source |
| ---- | --------------- | ----- | -------- | ------ |
| 浪潮信息 | USES_TECHNOLOGY | AI服务器 | 原文证据     | 2024年报 |
| 浪潮信息 | HAS_PRODUCT     | 服务器   | 原文证据     | 2024年报 |
| 中际旭创 | USES_TECHNOLOGY | 光模块   | 原文证据     | 研报     |

人工检查后导入 Neo4j。

Neo4j 中形成：

```text
(浪潮信息)-[:USES_TECHNOLOGY]->(AI服务器)
(浪潮信息)-[:HAS_PRODUCT]->(服务器)
(浪潮信息)-[:BELONGS_TO_CHAIN]->(中游服务器制造)
(中际旭创)-[:USES_TECHNOLOGY]->(光模块)
```

---

# 第五步：基于 LLM 的问答系统

这是新版流程的核心。

## 不是这样做

不要直接把问题丢给 LLM：

```text
用户问题 → LLM → 答案
```

这样容易胡编，且无法体现“基于知识图谱”。

---

## 应该这样做

```text
用户问题
 ↓
LLM 解析问题意图和实体
 ↓
LLM 生成 Cypher 查询
 ↓
Neo4j 执行查询
 ↓
返回图谱结果
 ↓
LLM 基于图谱结果生成答案
 ↓
前端展示答案和证据
```

也就是：

> **LLM 负责理解和表达，Neo4j 负责事实依据。**

---

## 5.1 用户输入问题

例如：

```text
哪些公司涉及 AI服务器？
```

---

## 5.2 LLM 解析问题

让 LLM 输出结构化 JSON。

### Prompt 示例

```text
你是知识图谱问答系统的问题解析器。

请分析用户问题，识别：
1. 问题意图
2. 涉及的实体
3. 需要查询的关系
4. 是否需要生成 Cypher

可选意图：
- tech_to_company：根据技术查询公司
- company_to_tech：查询公司涉及的技术
- company_to_product：查询公司产品
- company_to_chain：查询公司产业链环节
- company_to_metric：查询公司财务指标
- company_compare：比较公司
- multi_condition：多条件查询

用户问题：
{{question}}

输出严格 JSON。
```

### 输出示例

```json
{
  "intent": "tech_to_company",
  "entities": {
    "Technology": ["AI服务器"]
  },
  "target": "Company",
  "relation": "USES_TECHNOLOGY"
}
```

---

## 5.3 LLM 生成 Cypher 查询

给 LLM 图谱结构，让它生成 Cypher。

### 图谱 Schema

```text
节点类型：
Company(name, stock_code)
Technology(name)
Product(name)
IndustryChain(name)
Metric(name, year, value, unit)
Risk(name)
Report(title)

关系类型：
(:Company)-[:USES_TECHNOLOGY]->(:Technology)
(:Company)-[:HAS_PRODUCT]->(:Product)
(:Company)-[:BELONGS_TO_CHAIN]->(:IndustryChain)
(:Company)-[:HAS_METRIC]->(:Metric)
(:Company)-[:DISCLOSES_RISK]->(:Risk)
(:Company)-[:MENTIONED_IN]->(:Report)
```

### Cypher 生成 Prompt

```text
你是 Neo4j Cypher 查询生成器。

请根据用户问题和图谱 Schema 生成 Cypher 查询。

要求：
1. 只使用给定的节点类型和关系类型。
2. 不要生成删除、修改数据库的语句。
3. 只生成 MATCH 和 RETURN 查询。
4. 返回 Cypher，不要解释。

图谱 Schema：
{{schema}}

用户问题：
{{question}}
```

### LLM 生成 Cypher

```cypher
MATCH (c:Company)-[r:USES_TECHNOLOGY]->(t:Technology {name: "AI服务器"})
RETURN c.name AS company, r.evidence AS evidence, r.source AS source
```

---

## 5.4 后端执行 Neo4j 查询

后端拿到 Cypher 后查询 Neo4j。

返回结果：

```json
[
  {
    "company": "浪潮信息",
    "evidence": "公司布局 AI 服务器相关产品与解决方案",
    "source": "浪潮信息_2024年报"
  },
  {
    "company": "中科曙光",
    "evidence": "公司产品覆盖高性能计算、智能计算等领域",
    "source": "中科曙光_2024年报"
  }
]
```

---

## 5.5 LLM 基于查询结果生成答案

这一步再让 LLM 组织自然语言答案。

### 答案生成 Prompt

```text
你是中国 AI 算力产业链知识图谱问答助手。

请根据 Neo4j 查询结果回答用户问题。

要求：
1. 只能基于查询结果回答。
2. 如果查询结果为空，回答“当前知识图谱中未找到相关证据”。
3. 不要提供股票买卖建议。
4. 回答中要包含来源或证据。
5. 语言简洁清楚。

用户问题：
{{question}}

Neo4j 查询结果：
{{records}}
```

### LLM 最终回答

```text
当前知识图谱中，涉及 AI服务器 的公司包括：浪潮信息、中科曙光。

其中，浪潮信息的相关证据来自《浪潮信息_2024年报》，原文提到公司布局 AI 服务器相关产品与解决方案；中科曙光的相关证据来自《中科曙光_2024年报》，原文提到公司产品覆盖高性能计算、智能计算等领域。
```

---

# 六、前端展示

前端可以用 **Streamlit**，最简单。

页面做 4 个：

```text
1. 数据概览
2. 图谱展示
3. LLM 智能问答
4. 评测结果
```

## 问答页面展示内容

用户输入：

```text
哪些公司涉及 AI服务器？
```

页面返回：

```text
回答：
当前知识图谱中，涉及 AI服务器 的公司包括：浪潮信息、中科曙光。

证据：
1. 浪潮信息，来源：浪潮信息_2024年报
2. 中科曙光，来源：中科曙光_2024年报

Cypher 查询：
MATCH (c:Company)-[r:USES_TECHNOLOGY]->(t:Technology {name:"AI服务器"})
RETURN c.name AS company, r.evidence AS evidence, r.source AS source

图谱子图：
浪潮信息 → AI服务器
中科曙光 → AI服务器
```

这样答辩时非常清楚：

```text
LLM 不是直接回答，而是先查知识图谱，再根据图谱结果生成答案。
```

---

# 七、后端核心模块

项目后端可以这样分：

```text
src/
├── pdf_parser.py              # PDF 解析
├── text_cleaner.py            # 文本清洗
├── llm_extractor.py           # LLM 抽取实体关系
├── kg_loader.py               # 导入 Neo4j
├── llm_question_parser.py     # LLM 解析用户问题
├── cypher_generator.py        # LLM 生成 Cypher
├── neo4j_client.py            # 查询 Neo4j
├── answer_generator.py        # LLM 生成最终答案
└── qa_engine.py               # 串联问答流程
```

最核心的是 `qa_engine.py`。

伪代码如下：

```python
def answer_question(question):
    # 1. LLM 解析问题
    parsed = llm_parse_question(question)

    # 2. LLM 生成 Cypher
    cypher = llm_generate_cypher(question, schema)

    # 3. 安全检查，只允许 MATCH 查询
    check_cypher_safe(cypher)

    # 4. 查询 Neo4j
    records = neo4j_query(cypher)

    # 5. LLM 基于图谱结果生成答案
    answer = llm_generate_answer(question, records)

    # 6. 返回给前端
    return {
        "question": question,
        "parsed": parsed,
        "cypher": cypher,
        "records": records,
        "answer": answer
    }
```

---

# 八、一定要加 Cypher 安全检查

因为 Cypher 是 LLM 生成的，必须检查它不能修改数据库。

只允许这些开头：

```text
MATCH
OPTIONAL MATCH
RETURN
WITH
WHERE
```

禁止：

```text
CREATE
DELETE
DETACH DELETE
SET
MERGE
DROP
REMOVE
CALL dbms
```

简单规则：

```python
def check_cypher_safe(cypher):
    forbidden = ["CREATE", "DELETE", "DETACH", "SET", "MERGE", "DROP", "REMOVE", "CALL"]
    upper = cypher.upper()
    for word in forbidden:
        if word in upper:
            raise ValueError("不安全的 Cypher 查询")
    return True
```

---

# 九、评测怎么做

因为你要求问答基于 LLM，所以评测要比较三种方法。

## 对比方法

| 方法           | 说明                     |
| ------------ | ---------------------- |
| 纯 LLM 问答     | 直接问 LLM，不查图谱           |
| 关键词检索问答      | 从文本中搜关键词               |
| LLM + 知识图谱问答 | LLM 解析问题，查 Neo4j，再生成答案 |

## 预期结论

```text
纯 LLM 回答流畅，但可能编造。
关键词检索能找到文本，但不能处理复杂关系。
LLM + 知识图谱问答既能自然表达，又能基于图谱证据回答。
```

## 测试集

准备 30 个问题。

例如：

```text
1. 哪些公司涉及 AI服务器？
2. 浪潮信息涉及哪些技术？
3. 中际旭创属于哪个产业链环节？
4. 哪些公司涉及光模块？
5. 哪些公司同时涉及 AI服务器 和 液冷？
6. 比较浪潮信息和中科曙光的技术布局。
7. 某公司披露了哪些风险？
```

## 评分方式

| 得分  | 标准       |
| --- | -------- |
| 2 分 | 答案正确，有证据 |
| 1 分 | 部分正确     |
| 0 分 | 错误或无依据   |

最终得到：

```text
测试问题数量：30
LLM + 知识图谱问答正确题数：26
准确率：86.7%
```

---

# 十、最终你们要完成的东西

```text
1. 财报和研报 PDF
2. PDF 解析后的文本
3. LLM 抽取出的实体关系表
4. 人工校验后的图谱数据
5. Neo4j 知识图谱
6. LLM + 图谱问答系统
7. Streamlit 前端页面
8. 30 个问答测试题
9. 评测报告
10. 实验报告和答辩 PPT
```

---

# 十一、推荐完成顺序

```text
第 1 步：确定 10—20 家公司
第 2 步：下载年报和研报
第 3 步：PDF 转文本
第 4 步：用 LLM 抽取实体关系
第 5 步：人工校验并导入 Neo4j
第 6 步：实现 LLM 问题解析
第 7 步：实现 LLM 生成 Cypher
第 8 步：查询 Neo4j
第 9 步：LLM 根据查询结果生成答案
第 10 步：Streamlit 展示
第 11 步：准备问答评测
第 12 步：写报告和 PPT
```

---

# 十二、一句话总结新版流程

> **先用 LLM 从财报和研报中抽取“公司—技术—产品—产业链”等知识，导入 Neo4j 构建知识图谱；用户提问时，再由 LLM 解析问题并生成 Cypher 查询 Neo4j，最后让 LLM 基于图谱查询结果生成可解释答案。**

这样既符合“基于知识图谱问答”，又满足你“问答必须基于 LLM”的要求。
