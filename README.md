# 电商混合 RAG 系统

课程《大语言模型在电商中的应用》期末作业。基于 LangChain 构建的混合 RAG 系统,同时使用 Vector-RAG 与 GraphRAG 两种检索范式。

## 运行
1. `cp .env.example .env`,填入 `QWEN_API_KEY`
2. `python scripts/preprocess.py --products-csv <path> --reviews-csv <path>`(清洗数据)
3. `python scripts/build_indices.py`(一次性离线构建索引)
4. `streamlit run demo/app.py`(启动 demo)

## 评估
`python evaluation/run_experiments.py`
