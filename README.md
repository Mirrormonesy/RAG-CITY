# 电商混合 RAG 系统

课程《大语言模型在电子商务中的应用》期末作业。基于 LangChain 的 Hybrid RAG,同时融合 **Vector-RAG** 与 **GraphRAG** 两种检索范式,数据基于 Amazon 2018 英文商品评论(翻译为中文)。

## 1. 环境要求

- Python **3.10+**
- CUDA **12.1+**(BGE embedding / reranker 用 GPU)
- **torch >= 2.6**(sentence-transformers 新版强制,CVE-2025-32434)
- 内存 ≥ 16 GB,显存 ≥ 8 GB

## 2. 安装依赖

```bash
pip install -r requirements.txt
# 如果 requirements 里 torch 版本不够,单独升级
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

## 3. API Key

```bash
cp .env.example .env
# 编辑 .env,填入 QWEN_API_KEY
```

两种 DashScope 平台:

| 平台 | 注册地址 | base_url |
|---|---|---|
| 国内版 | https://bailian.console.aliyun.com/ | 默认 |
| 国际版(新加坡) | https://bailian.console.alibabacloud.com/ | `https://dashscope-intl.aliyuncs.com/api/v1` |

国际版的 base_url 已经写在 `configs/config.yaml` 里,不用改代码。国内版需要注释掉 `qwen.base_url` 字段。

## 4. 模型(首次运行时自动下载)

| 模型 | 大小 | 用途 |
|---|---|---|
| `BAAI/bge-large-zh-v1.5` | ~1.3 GB | 中文 embedding |
| `BAAI/bge-reranker-v2-m3` | ~2.3 GB | 重排 |
| Qwen(通过 API) | - | 路由 / 抽取 / 社区摘要 / 生成答案 |

缓存位置:`%USERPROFILE%\.cache\huggingface`(Windows) / `~/.cache/huggingface`(Linux)

## 5. 数据准备

原始数据来自 **Amazon Review 2018**(UCSD McAuley):
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

下载两个类目的 **5-core** reviews 和对应 **meta**,放入 `data/raw/`:
- `Cell_Phones_and_Accessories_5.json` + `meta_Cell_Phones_and_Accessories.json`
- `All_Beauty_5.json` + `meta_All_Beauty.json`

然后依次跑:

```bash
python scripts/prepare_amazon.py   # 采样 + 清洗(HTML、编码),产出 data/raw/products.csv, reviews.csv
python scripts/translate_data.py   # Qwen-Plus 英译中,支持断点续传
python scripts/preprocess.py       # 规范化 + 产出 data/processed/*.parquet
```

## 6. 构建索引(约 30 min,含 API 调用)

```bash
python scripts/build_indices.py
```

产出:
- `indices/chroma_docs/` — 商品 + 评论向量库
- `indices/chroma_nodes/` — 图节点向量库(用于实体扩展)
- `indices/chroma_summaries/` — 社区摘要向量库
- `indices/graph.pkl` — NetworkX 知识图谱(带社区标签)
- `indices/communities.json` — 社区 → 节点映射 + 摘要

增量控制:
```bash
python scripts/build_indices.py --skip-vector          # 只重建图 + 社区
python scripts/build_indices.py --skip-graph --skip-community   # 只重建向量
```

## 7. 交互查询

```bash
python scripts/query.py "护发产品里口碑最好的是哪些?"
python scripts/query.py              # 进入交互模式
```

输出会显示:路由决策 → Vector 召回 → Graph 召回 → 融合 → 重排 → 答案 + 引用。

## 8. 工具脚本

| 脚本 | 用途 |
|---|---|
| `scripts/view_parquet.py` | 查看 processed 下 parquet 内容 |
| `scripts/view_chroma.py` | 查看 chroma 集合样本 |
| `scripts/view_graph.py` | 可视化 top 社区子图(PNG) |
| `scripts/count_chroma.py` | 统计各 chroma 集合条数(排查重复插入) |

## 9. 目录结构

```
RAG-CITY/
├── configs/config.yaml           # 全部可调参数
├── src/
│   ├── utils/                    # 配置 / 日志 / Qwen 客户端
│   ├── indexing/                 # embedding, 向量库, 图, 社区
│   ├── retrieval/                # 路由, 向量检索, 图检索, 融合, 重排
│   ├── generation/               # Answerer + prompts
│   ├── factory.py                # 装配 HybridRAG
│   └── pipeline.py               # HybridRAG 主流程
├── scripts/                      # 数据准备 + 索引构建 + 查询入口
├── tests/                        # pytest 单测
├── data/{raw,processed}/         # (gitignore)
├── indices/                      # (gitignore)
├── demo/                         # Streamlit UI(待实现)
└── evaluation/                   # 评估脚本与人工标注(待实现)
```

## 10. 已知坑

- **torch < 2.6** 加载 BGE 会报 `CVE-2025-32434` → 升级 torch
- **DashScope 国际版 401 InvalidApiKey** → 确认 `config.yaml` 里 `qwen.base_url` 已设为 intl 端点
- **chroma 重复插入**:早期 `vector_builder.py` 无 `ids`,重跑 `build_indices.py` 会翻倍;现已用 `product::<pid>` / `review::<rid>` 作为 deterministic id(upsert)
- **PowerShell 清目录** 用 `Remove-Item -Recurse -Force indices\chroma_docs`,不要用 `rmdir /s /q`
- **Excel 锁住 products.csv** 导致写入失败 → 关掉 Excel 再重跑
