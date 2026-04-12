from collections import defaultdict
from langchain_core.documents import Document

def doc_key(doc: Document) -> str:
    m = doc.metadata or {}
    for k in ("review_id", "product_id", "community_id"):
        if k in m:
            return f"{k.split('_')[0]}:{m[k]}"
    return f"content:{doc.page_content[:80]}"

class RRFFuser:
    def __init__(self, k_const: int = 60):
        self.k_const = k_const

    def fuse(self, *doc_lists: list[Document]) -> list[Document]:
        scores: dict[str, float] = defaultdict(float)
        store: dict[str, Document] = {}
        for docs in doc_lists:
            for rank, d in enumerate(docs):
                key = doc_key(d)
                scores[key] += 1.0 / (self.k_const + rank + 1)
                store.setdefault(key, d)
        ranked = sorted(store.keys(), key=lambda k: scores[k], reverse=True)
        return [store[k] for k in ranked]
