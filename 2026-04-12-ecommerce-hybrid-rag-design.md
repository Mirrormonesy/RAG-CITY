# 电商混合 RAG 系统设计文档

**日期**:2026-04-12
**项目**:电子商务垂直领域混合 RAG(Vector-RAG + GraphRAG)
**用途**:《大语言模型在电子商务系统与策略中的应用》课程期末作业
**技术栈**:LangChain + Qwen API + 本地 BGE + NetworkX + Chroma + Streamlit

---

## 1. 项目背景与目标

### 1.1 选题背景
课程选题要求「研究通用大语言模型在电子商务中的应用场景,或者开发一个基于微调技术/RAG 的电商领域垂直大模型,并且分析大模型对电子商务策略的影响」。

### 1.2 定位
构建一个面向中文电商评论与商品数据的 **混合检索增强生成(Hybrid RAG)系统**,同时使用两种检索范式:
- **Vector-RAG**:处理事实性、单品级查询
- **GraphRAG**(微软方案):处理跨商品/跨品牌的聚合与多跳推理查询

### 1.3 应用场景
- **主场景**:商品评论/市场洞察分析(运营视角)——跨品牌对比、用户抱怨趋势、品类口碑总结
- **副场景**:商品推荐与导购(消费者视角)——基于属性与评论的推荐解释

### 1.4 设计驱动的约束
作业评分:内容详实 50% / 内容深度 30% / 技术难度与代码 20%。这意味着:
- 架构需有技术亮点但不过度工程化,精力留给报告深度和实验分析
- 优先保证可演示、可分析、有定量对比(Vector-only / Graph-only / Hybrid)
- Demo 级规模即可,不追求生产级

---

## 2. 系统总体架构

### 2.1 分层结构

```
┌─────────────────────────────────────────────────────────────┐
│   应用层   Streamlit Demo(聊天界面 + 检索过程可视化)        │
├─────────────────────────────────────────────────────────────┤
│   生成层   Qwen API + Prompt 模板 + 引用标注                 │
├─────────────────────────────────────────────────────────────┤
│   检索层   LLM-Router → Vector/Graph/Hybrid → RRF → Rerank  │
├─────────────────────────────────────────────────────────────┤
│   索引层   Chroma 向量库 + NetworkX 知识图谱 + 社区摘要库   │
└─────────────────────────────────────────────────────────────┘
        ↑                                        ↑
   (离线构建:Qwen API + BGE-large-zh)      (支撑:评估模块 / 日志模块)
```

### 2.2 在线 vs 离线流程

| | 离线流程(建一次) | 在线流程(每次查询) |
|---|---|---|
| 做什么 | 把原始数据加工为可检索结构 | 用已建好的结构回答问题 |
| 耗时 | 30 分钟 ~ 2 小时 | 秒级 |
| 产物 | 向量库 / 图谱 / 社区摘要落盘 | 一条带引用的答案 |

### 2.3 项目目录结构

```
ecommerce-hybrid-rag/
├── configs/config.yaml
├── data/{raw, processed}/
├── indices/{chroma_docs, chroma_summaries, graph.pkl, communities.json}
├── src/
│   ├── data/{loader.py, cleaner.py}
│   ├── indexing/{vector_builder.py, graph_builder.py, community_builder.py}
│   ├── retrieval/{router.py, vector_retriever.py, graph_retriever.py, fusion.py, reranker.py}
│   ├── generation/{prompts.py, answerer.py}
│   ├── pipeline.py
│   └── utils/{llm_client.py, logger.py}
├── evaluation/{eval_set.jsonl, metrics.py, run_experiments.py}
├── case_studies/{01_iphone_complaints.ipynb, 02_price_segment_analysis.ipynb, 03_brand_comparison.ipynb}
├── demo/app.py
├── scripts/{build_indices.py, sample_for_review.py, run_demo.py}
├── tests/test_*.py
├── report/{figures, report.docx}
├── requirements.txt
└── README.md
```

### 2.4 核心类依赖关系

```
pipeline.HybridRAG
    ├─ retrieval.router.LLMRouter
    ├─ retrieval.vector_retriever.VectorRetriever
    ├─ retrieval.graph_retriever.GraphRetriever
    ├─ retrieval.fusion.RRFFuser
    ├─ retrieval.reranker.BgeReranker
    └─ generation.answerer.Answerer
```

每个模块一个文件、一个主类、单一职责。模块间通过 LangChain `Document` 对象交换数据,便于单元测试与 ablation 实验。

---

## 3. 数据层与离线索引构建

### 3.1 数据源

中文京东商品评论数据集(Kaggle / GitHub 公开数据)。

**规模控制**:
- 2 个类目(手机 + 美妆,互补性强)
- 每类目约 500 商品 + 5000–8000 条评论
- 总计约 1000 商品 + 15000 评论

### 3.2 数据 Schema

**products.parquet**:`product_id | title | category | brand | price | description`

**reviews.parquet**:`review_id | product_id | user_id | rating | content | timestamp`

### 3.3 清洗规则(src/data/cleaner.py)

- 过滤长度 < 10 或 > 500 的评论
- 去重(同一用户同一商品只保留一条)
- 过滤纯表情 / 纯数字评论
- 规范化品牌名(维护别名字典,如「苹果」「Apple」「APPLE」→「苹果」)

### 3.4 索引产物 1:Chroma 向量库

- 商品 chunk:`title + brand + category + description` 拼接
- 评论 chunk:评论内容(metadata 含 product_id、rating、timestamp、doc_type)
- Embedding:`BAAI/bge-large-zh-v1.5`,本地 GPU(4070 Super)批处理
- 存储:Chroma 持久化到 `indices/chroma_docs/`
- 耗时估算:15000 条 × BGE-large ≈ 3–5 分钟

### 3.5 索引产物 2:NetworkX 知识图谱

**实体 schema(预定义,避免 LLM 自由发挥)**:
- `Product` / `Brand` / `Category` / `Feature` / `Aspect` / `Sentiment`

**关系 schema**:
- `BELONGS_TO` / `MADE_BY` / `HAS_REVIEW` / `MENTIONS` / `EXPRESSES` / `ABOUT_FEATURE`

**抽取流程**:
1. 商品结构化字段直接转节点/边(免费,不调 LLM)
2. 每条评论调一次 Qwen-Turbo,输出 JSON:`{aspects, sentiment, features}`
3. 合并进同一张 NetworkX 图,序列化到 `indices/graph.pkl`

**Prompt(src/generation/prompts.py)**:
```
你是电商评论分析专家。从下面这条评论中抽取:
- 用户提到的方面(aspects,≤5 个)
- 每个方面关联的产品特性(features)
- 整体情感(sentiment: positive/negative/neutral)
只输出 JSON,无其他文字。

评论:{review_content}
```

**成本估算**:15000 × Qwen-Turbo ≈ ¥5–10

### 3.6 索引产物 3:社区检测与摘要

1. `networkx.algorithms.community.louvain_communities(G, seed=42)`
2. 过滤节点数 < 5 的小社区(噪声)
3. 对每个社区:提取核心节点 + 关键边 + 采样 10 条原始评论 → 拼 context → 调 Qwen 生成摘要
4. 摘要结构:
   ```json
   {
     "community_id": 7,
     "size": 142,
     "core_entities": ["iPhone 15", "电池续航", "差评"],
     "sample_reviews": ["review_42", "review_87"],
     "summary": "此社区聚焦 iPhone 15 系列..."
   }
   ```
5. 所有摘要的 `summary` 字段单独建 Chroma collection(`chroma_summaries/`)
6. 成本:约 40 社区 × Qwen 摘要 ≈ ¥1 以内

### 3.7 人工抽样检查(数据质量验证)

从 15000 条评论的 LLM 抽取结果中**随机采样 50 条**,人工标注:
- 实体抽取是否正确(对/部分对/错)
- 情感判断是否正确
- 是否漏抽关键信息

**产出**:数据质量报告(实体准确率、情感准确率、典型错误案例与改进措施),写入报告「数据质量分析」小节。

### 3.8 统一构建入口(scripts/build_indices.py)

- 一键命令串联:清洗 → 向量 → 图谱 → 社区
- 支持 `--skip-vector` / `--skip-graph` / `--resume`
- 增量支持:每批次结果落盘到中间 jsonl,失败后断点续传
- 打印每阶段耗时与 API 成本

---

## 4. 在线检索层

### 4.1 查询流程

```
用户问题 → LLMRouter → 分支:
  ├─ "vector" → VectorRetriever(k=20) → Reranker → Answerer
  ├─ "graph"  → GraphRetriever → Answerer
  └─ "hybrid" → Vector(k=20) ⊕ Graph → RRFFuser → Reranker(top=5) → Answerer
```

### 4.2 LLMRouter(查询路由)

```python
class LLMRouter:
    def route(self, query: str) -> RouteDecision:
        # {"route": "vector"|"graph"|"hybrid", "reason": str}
```

- 使用 Qwen-Turbo(轻量、便宜)
- Prompt 包含三类问题的 few-shot 示例
- JSON 解析失败 → 兜底为 `hybrid`(保守选择)

### 4.3 VectorRetriever

```python
class VectorRetriever:
    def retrieve(self, query: str, k: int = 20) -> List[Document]:
```

- BGE-large-zh 编码查询
- Chroma 检索 top-k,返回带 metadata 的 Document
- 支持 metadata 过滤(按类目、品牌、评分等)

### 4.4 GraphRetriever(核心模块)

三步流程:

**Step 1:社区摘要检索(语义相似)**
- 对 query 做 embedding
- 在 `chroma_summaries` 检索 top-3 社区摘要

**Step 2:实体扩展(精确匹配补召回)**
- 构建节点名的 BGE 索引
- 对 query 在节点名上做向量检索 → top-5 相似节点
- 读取每个节点的 `community_id`,加入候选社区集
- 与 Step 1 结果取并集

**Step 3:子图提取与文档化**
- 对每个候选社区:提取核心节点 + 关键边(三元组文字化)+ 采样代表性评论
- 拼成一个 Document(每社区一个),metadata 含 `community_id` 与涉及实体列表

**输出示例**:
```
[社区 7 摘要] 此社区聚焦 iPhone 15 系列电池续航负面反馈...
[核心实体] iPhone 15 Pro, iPhone 15, 电池续航, 差评
[关键关系] iPhone 15 Pro --HAS_REVIEW--> R42 --MENTIONS--> 电池续航
[代表性评论] "用了一个月,电池掉得快..." (review #42, rating=2)
```

### 4.5 RRFFuser

**公式**(Cormack et al. 2009):
```
score(doc) = Σ  1 / (k_const + rank_in_retriever(doc))
k_const = 60
```

仅依赖排名,天然归一化不同检索器的分数尺度。实现约 15 行。

### 4.6 BgeReranker

- 模型:`BAAI/bge-reranker-v2-m3`
- 实现:`sentence_transformers.CrossEncoder`,本地 GPU
- 输入 top-20 候选,输出 top-5 精排结果
- 耗时:20 条候选 < 200ms

### 4.7 HybridRAG 总装

```python
class HybridRAG:
    def query(self, question: str) -> Answer:
        decision = self.router.route(question)
        if decision.route == "vector":
            docs = self.vec_ret.retrieve(question, k=20)
        elif decision.route == "graph":
            docs = self.graph_ret.retrieve(question)
        else:
            docs = self.fuser.fuse(
                self.vec_ret.retrieve(question, k=20),
                self.graph_ret.retrieve(question)
            )
        reranked = self.reranker.rerank(question, docs, top_n=5)
        return self.answerer.answer(question, reranked, route=decision.route)
```

**可测性**:每个模块可单独替换/禁用,便于 ablation。

---

## 5. 生成层

### 5.1 Answerer 接口

```python
class Answerer:
    def answer(self, question: str, docs: List[Document], route: str) -> Answer:
        # 返回: {"text": str, "citations": Dict[str, source_info], "prompt_used": str}
```

### 5.2 三套 Prompt 模板(按 route 分派)

**模板 1:Vector 路径(事实问答)**
要求:只用参考文档、末尾标引用 `[V1][V2]`、不足时如实说"无法确定"

**模板 2:Graph 路径(聚合总结)**
要求:先结论后分点、每点配证据、涉及对比用表格、引用用 `[G1][G2]`

**模板 3:Hybrid 路径(总结 + 事实)**
要求:宏观结论优先、具体证据支撑、`[V*]` vs `[G*]` 区分来源、矛盾时明确指出

### 5.3 引用机制

**三步流水**:
1. **代码预编号**:检索到 docs 后,Answerer 为每个 doc 分配 `[V1]`/`[G1]` 等编号,记录映射 `{id → doc}`
2. **LLM 按 prompt 写引用**:答案文本中自然插入编号
3. **代码正则解析**:`\[([VG]?\d+)\]` 抽出编号,查 mapping 还原原始数据

**返回结构**:
```json
{
  "text": "苹果用户主要抱怨电池续航[G1]。用户反映一天两充[V1]。",
  "citations": {
    "G1": {"type": "graph", "community_id": 7, "summary": "..."},
    "V1": {"type": "review", "review_id": 42, "content": "...", "rating": 2}
  }
}
```

**Demo 价值**:前端可渲染为可点击气泡,点开看原文,体现 RAG 的可解释性。

### 5.4 双模型分级策略

| 场景 | 任务 | 模型 | 原因 |
|---|---|---|---|
| 图谱抽取(离线) | 抽实体关系 | qwen-turbo | 量大、输出结构化、便宜 |
| 查询路由(在线) | 3 选 1 分类 | qwen-turbo | 简单、快 |
| 最终生成(在线) | 综合证据答题 | qwen-plus | 质量优先 |

对比全用 qwen-plus,整体成本降低约 80%。

### 5.5 生成超参数(config.yaml)

```yaml
generation:
  model: qwen-plus
  temperature: 0.3      # 低随机,减少幻觉
  max_tokens: 1024
  top_p: 0.9
  timeout: 30
  max_retries: 2
```

### 5.6 流式输出(Demo 体验增强)

Qwen 客户端 `.stream()` → Streamlit `st.write_stream()`,首字节 < 1s。

---

## 6. 评估实验设计

### 6.1 评测集(evaluation/eval_set.jsonl)

规模 **30 条**,分三层:

| 类型 | 数量 | 特征 | 期望路由 |
|---|---|---|---|
| 事实型 F | 10 | 单品/事实查询 | vector |
| 聚合型 A | 10 | 跨商品/品牌聚合 | graph |
| 混合型 H | 10 | 趋势 + 具体例 | hybrid |

**构造流程**:你先写 10 种子 → Qwen 扩写 60 条 → 人工筛至 30 → 人工标注 ground truth docs 与 reference answer。

### 6.2 四组对比实验(主实验)

对 30 题跑四组配置:
- **V-only**:仅 Vector
- **G-only**:仅 Graph
- **Router**:LLM 路由三选一(无融合)
- **Full**:LLM 路由 + RRF + Reranker(完整方案)

**自动指标**:Recall@5、MRR、端到端延迟(ms)

**人工指标**(盲评):答案相关性 / 完整性 / 准确性(各 1–5 分)

### 6.3 分层对比(互补性分析)

按 F/A/H 三类问题分别看四组方法表现——论证 Vector 与 Graph 的互补性。

### 6.4 Ablation 实验

从 Full 依次拿掉:Reranker / RRF / 社区摘要 / 实体扩展(Step 2)—— 量化每个模块贡献。

### 6.5 路由器准确率分析

构造混淆矩阵(预测路由 vs 真实最佳路由),输出整体准确率。

### 6.6 电商策略 Case Study(三个 notebook)

1. **产品改进建议**:iPhone 15 近期负面反馈分类与优先级
2. **价位段选品洞察**:3000–5000 元段用户卖点关注点,品牌差异
3. **竞品口碑对比**:iPhone vs 华为 Mate,给商家选品建议

每个 case 输出:代码 + 结果图表 + 策略建议。对应选题要求「分析大模型对电商策略的影响」。

### 6.7 可视化产出(报告插图)

- 图 1:系统架构图(draw.io)
- 图 2:知识图谱可视化(NetworkX + matplotlib,一个社区的内部结构)
- 图 3:四组方法指标柱状图
- 图 4:按问题类型分组的雷达图

### 6.8 实验运行脚本

`python evaluation/run_experiments.py --eval-set evaluation/eval_set.jsonl`
- 30 题 × 4 配置 = 120 次查询
- 耗时约 30–45 分钟,成本约 ¥5
- 输出 JSON + 汇总 CSV + 自动生成表格图表

---

## 7. Demo 界面(Streamlit)

### 7.1 单页布局

- **侧边栏**:数据集选择、类目、检索策略切换器(auto/vector/graph/hybrid)、top_k、温度、索引规模统计
- **主区**:问题输入框、检索过程可视化(路由决策→向量结果→图结果→RRF 融合→重排)、流式答案输出、引用源可点击展开

### 7.2 额外 Tab:知识图谱可视化

输入实体名,展示其邻居子图(pyvis 交互式 HTML)。

### 7.3 截图资产(报告附录)

- 主界面 + 流式答案
- 检索过程四步展开
- 引用气泡点击展开原文
- 知识图谱交互可视化
- 三种策略切换对比同一问题的答案

---

## 8. 错误处理与降级

| 失败点 | 兜底策略 |
|---|---|
| Qwen API 超时 | 重试 2 次 → 友好提示 |
| 路由 JSON 解析失败 | 兜底 `hybrid` |
| Vector 检索空 | 提示未找到,走 Graph 兜底 |
| Graph 检索异常 | 降级为 V-only,日志告警 |
| Reranker GPU OOM | 跳过重排,直接用融合结果 |
| 生成超 max_tokens | 提示"回答已截断" |
| 索引文件缺失 | 启动时检查,报错并提示运行 build_indices |

日志模块分 info/warn/error 三级,写入 `logs/app.log`。报告附录可贴日志片段。

---

## 9. 测试策略

**单元测试**(pytest):
- `test_cleaner.py`:清洗规则边界
- `test_router.py`:典型问题路由正确性
- `test_fusion.py`:RRF 公式验证
- `test_graph_retriever.py`:Step 1/2/3 分别测(mock 图数据)
- `test_answerer.py`:引用正则抽取正确

**集成测试**(1 个):端到端跑 3 条示例问题,断言非空 + 含引用

**产出**:pytest 输出截图 → 报告附录(证明代码质量,对应 20% 评分)。

---

## 10. 配置管理(configs/config.yaml)

所有 API / 模型 / 路径 / 超参数集中一个 YAML 文件,支持从环境变量读取 API key。

```yaml
qwen:
  api_key: ${QWEN_API_KEY}
  router_model: qwen-turbo
  extract_model: qwen-turbo
  generate_model: qwen-plus
embedding:
  model: BAAI/bge-large-zh-v1.5
  device: cuda
reranker:
  model: BAAI/bge-reranker-v2-m3
  device: cuda
retrieval:
  vector_k: 20
  graph_k_communities: 3
  rrf_k_const: 60
  final_top_n: 5
generation:
  temperature: 0.3
  max_tokens: 1024
```

---

## 11. 项目时间线(预估 13–15 天)

| Day | 任务 | 产出 |
|---|---|---|
| 1 | 项目骨架 + 环境 + 数据清洗 | 干净 parquet |
| 2 | Vector 索引 + 抽样检查评论质量 | Chroma 索引 + 质量报告 |
| 3 | 图谱构建(实体关系抽取) | NetworkX 图 + 50 条人工检查 |
| 4 | 社区检测 + 摘要 + 图可视化 | communities.json + PNG |
| 5 | Router + Retrievers + RRF + Reranker | 检索层通 |
| 6 | Answerer + 单测 + 端到端 | 能答题 |
| 7 | 评测集构造 + 四组对比 | 主实验表 + 图 |
| 8 | Ablation + 路由准确率 | 辅助实验表 |
| 9 | 三个 case notebook | 3 个案例 |
| 10 | Streamlit demo + 截图 | demo + 图 |
| 11–12 | 报告初稿 | 初稿 |
| 13 | 精修排版 + 附录 | 终稿 |
| Buffer | 调试 / 查漏 | —— |

**最低可交付版本(9 天)**:砍掉 Ablation 与 Case 3,保留主实验和 2 个 case。

---

## 12. 交付清单

- [ ] 代码仓库(GitHub / 压缩包)
- [ ] 书面报告(Word / PDF,含封面/摘要/目录/引言/主体/结论/参考文献/附录)
- [ ] Demo 截图集
- [ ] 评测集 `eval_set.jsonl`
- [ ] 实验结果 JSON + CSV
- [ ] 3 个 case study notebook
- [ ] README(运行说明、成本说明、局限性声明)

---

## 13. 风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| Qwen 抽图质量差 | 中 | 人工抽检 + 改 prompt + 规则后处理 |
| 评论数据脏 | 中 | 迭代清洗,必要时缩减规模保质量 |
| Full 不显著优于 V-only | 中 | 如实报告(批判性反而加分),探究为何失效 |
| 时间不够 | 中 | 砍 Ablation + Case 3,保证主实验 |
| 4070S 显存不够 | 低 | 换 BGE-base(更小) |

---

## 14. 报告与评分映射

| 评分维度(权重) | 对应本设计的贡献 |
|---|---|
| 内容详实 50% | 13 天时间线保证章节完整;6 章实验 + 3 case + 附录完善 |
| 内容深度 30% | Ablation 实验 / 数据质量人工评估 / 双模型分级策略 / 路由器准确率分析 / 矛盾来源的批判性讨论 |
| 技术难度 20% | LangChain 全栈 / 自实现 GraphRAG / LLM 路由 / RRF 融合 / Cross-Encoder 重排 / 引用机制 |

---

## 附录 A:关键参考文献候选

- Edge et al. 2024, "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"(微软 GraphRAG)
- Cormack et al. 2009, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"
- Blondel et al. 2008, "Fast unfolding of communities in large networks"(Louvain)
- Xiao et al. 2023, "C-Pack: Packaged Resources To Advance General Chinese Embedding"(BGE)
- Lewis et al. 2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"(RAG 原始论文)

---

**文档状态**:已完成,待用户审阅。审阅通过后进入 writing-plans 阶段生成实施计划。
