"""统计 chroma 集合里的条目数,不加载 embedding 模型。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
import pandas as pd

for name in ["chroma_docs", "chroma_summaries", "chroma_nodes"]:
    path = PROJECT_ROOT / "indices" / name
    if not path.exists():
        print(f"[skip] {name} not found")
        continue
    client = chromadb.PersistentClient(path=str(path))
    for coll in client.list_collections():
        n = coll.count()
        print(f"{name}/{coll.name}: {n} items")

# 和源数据对比
processed = PROJECT_ROOT / "data" / "processed"
p = pd.read_parquet(processed / "products.parquet")
r = pd.read_parquet(processed / "reviews.parquet")
print(f"\nExpected in chroma_docs: {len(p)} products + {len(r)} reviews = {len(p) + len(r)}")
