from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from src.utils.logger import get_logger

logger = get_logger(__name__)

def format_product_chunk(row) -> str:
    return f"[商品] {row['title']} | 品牌:{row['brand']} | 类目:{row['category']} | 描述:{row['description']}"

def format_review_chunk(row) -> str:
    return str(row["content"])

def _to_documents(products, reviews):
    docs = []
    for _, p in products.iterrows():
        docs.append(Document(
            page_content=format_product_chunk(p),
            metadata={"doc_type": "product", "product_id": p["product_id"],
                      "category": p["category"], "brand": p["brand"],
                      "price": float(p.get("price", 0))},
        ))
    for _, r in reviews.iterrows():
        docs.append(Document(
            page_content=format_review_chunk(r),
            metadata={"doc_type": "review", "review_id": r["review_id"],
                      "product_id": r["product_id"], "rating": int(r["rating"])},
        ))
    return docs

def build_vector_index(products, reviews, embedding, persist_dir: str) -> Chroma:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    docs = _to_documents(products, reviews)
    logger.info(f"Building Chroma index with {len(docs)} documents at {persist_dir}")
    db = Chroma.from_documents(docs, embedding=embedding, persist_directory=persist_dir)
    if hasattr(db, "persist"):
        db.persist()
    return db
