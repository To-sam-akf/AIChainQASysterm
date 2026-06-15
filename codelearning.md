`src/api.py` 是项目的 **FastAPI 后端入口和业务接线层**。它本身不负责检索和推理，主要负责：

- 接收前端 HTTP 请求
- 创建并缓存 `QAEngine`
- 调用在线问答或 `ResearchAgent`
- 保存对话、任务、评测和反馈
- 把结果通过 JSON 或 SSE 流返回前端

## 在线问答，从这里看

推荐按下面顺序阅读：

1. [api.py](/home/sanmu/AIQASYS/src/api.py:598)  
   `/conversations/{id}/messages/stream` 接收问题、读取历史对话，并调用 `engine.answer_question_stream()`。

2. [qa_engine.py](/home/sanmu/AIQASYS/src/qa_engine.py:321)  
   `QAEngine.answer_question()` 是统一问答入口。默认启用 Agent，转交给 `QAAgent`。

3. [qa_agent.py](/home/sanmu/AIQASYS/src/agents/qa_agent.py:11)  
   决定使用：
   - 默认 `LangGraphAgentRunner`
   - 旧版 `AgentRunner`
   - 禁用 Agent 时的普通 QA 工作流

4. [langgraph_runner.py](/home/sanmu/AIQASYS/src/langgraph_runner.py:54)  
   在线问答状态图：

```text
plan
  → retrieve
  → coverage_check
  → supplement_round ↺
  → finalize_supplement
  → verify_answer
```

5. [agent_runner.py](/home/sanmu/AIQASYS/src/agent_runner.py:272)  
   这里有每个阶段的具体实现：

```text
_plan()
  问题改写、意图识别、检索计划

_retrieve()
  图谱 + RAG + Claim/Dossier + Embedding + GraphRAG

_supplement()
  检查公司、指标、风险和机理证据是否缺失

_verify_and_answer()
  证据排序、生成答案、事实验证、失败降级
```

6. [agent_tools.py](/home/sanmu/AIQASYS/src/agent_tools.py)  
   查看每种检索工具如何调用 `QAEngine`。

7. [qa_engine.py](/home/sanmu/AIQASYS/src/qa_engine.py:765)  
   查看真正的数据访问和回答生成：

- `_query_graph()`：CSV/Neo4j 图谱
- `_search_rag()`：原文 RAG
- `_search_research()`：Claim/Dossier
- `_generate_answer()`：LLM 或规则降级

完整在线链路是：

```text
React
 → api.py
 → QAEngine
 → QAAgent
 → LangGraphAgentRunner
 → AgentTools
 → CSV/Neo4j + RAG + Claim + Embedding
 → EvidenceCard 排序
 → LLM/模板生成
 → verify_answer_support
 → 对话存储
 → API/SSE 返回前端
```

## ResearchAgent，从这里看

建议从 [src/agents/research_agent.py](/home/sanmu/AIQASYS/src/agents/research_agent.py:35) 开始。

主要流程在 `ResearchAgent.run()`：

```text
接收 task_type + goal
 → 创建 pending 任务
 → 将目标改写成标准投研问题
 → 状态改为 running
 → QAAgent.run()
 → 提取回答、证据、验证结果、执行轨迹
 → 转换为任务专属结构
 → 状态改为 completed
 → 写入 agent_tasks.jsonl
```

调用入口有两个：

- API：[api.py](/home/sanmu/AIQASYS/src/api.py:439)
- CLI：[cli.py](/home/sanmu/AIQASYS/src/cli.py:99)

支持五种任务：

```text
research_brief      投研简报
company_compare     公司对比
company_profile     公司画像
risk_review         风险审查
evidence_gap_audit  证据缺口审查
```

任务存储看 [store.py](/home/sanmu/AIQASYS/src/agents/store.py:33)，采用追加式 JSONL：

```text
data/agent_tasks/agent_tasks.jsonl
```

## 两个 ResearchAgent 文件别混淆

### `src/agents/research_agent.py`

真正的任务 Agent：

- 管理任务状态
- 调用 QA Agent
- 保存任务
- 按任务类型组织输出

### `src/research_agent.py`

确定性的投研结果渲染器：

- 根据最终 EvidenceCard 生成报告
- 构建公司对比表
- 生成风险清单
- 标记证据缺口

它不执行检索，也不是自主 Agent。

## 最快阅读路线

只看主干时，按这个顺序：

```text
main.py
 → src/cli.py
 → src/api.py:598
 → src/qa_engine.py:321
 → src/agents/qa_agent.py
 → src/langgraph_runner.py
 → src/agent_runner.py:272
 → src/agent_tools.py
 → src/graphrag.py
 → src/agents/verification.py
 → src/research_agent.py
```

ResearchAgent 再单独看：

```text
src/api.py:439
 → src/agents/research_agent.py:48
 → src/agents/qa_agent.py
 → 在线问答主流程
 → src/research_agent.py:32
 → src/agents/store.py
```

一句话理解：**ResearchAgent 没有另一套检索引擎，它是把在线 QA Agent 包装成可持久化、可导出的专业投研任务。**