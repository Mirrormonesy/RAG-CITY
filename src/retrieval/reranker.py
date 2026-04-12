from langchain_core.documents import Document

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore

class BgeReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
        if not docs:
            return []
        pairs = [[query, d.page_content] for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_n]]
