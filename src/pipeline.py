from dataclasses import dataclass
from langchain_core.documents import Document

@dataclass
class QueryResult:
    text: str
    citations: dict
    route: str
    route_reason: str
    retrieved_docs: list
    reranked_docs: list

class HybridRAG:
    def __init__(self, router, vec_retriever, graph_retriever,
                 fuser, reranker, answerer,
                 vector_k: int = 20, top_n: int = 5):
        self.router = router
        self.vec = vec_retriever
        self.graph = graph_retriever
        self.fuser = fuser
        self.reranker = reranker
        self.answerer = answerer
        self.vector_k = vector_k
        self.top_n = top_n

    def query(self, question: str) -> dict:
        decision = self.router.route(question)

        v_docs, g_docs = [], []
        if decision.route == "vector":
            v_docs = self.vec.retrieve(question, k=self.vector_k)
            docs = v_docs
        elif decision.route == "graph":
            g_docs = self.graph.retrieve(question)
            docs = g_docs
        else:  # hybrid
            v_docs = self.vec.retrieve(question, k=self.vector_k)
            g_docs = self.graph.retrieve(question)
            docs = self.fuser.fuse(v_docs, g_docs)

        reranked = self.reranker.rerank(question, docs, top_n=self.top_n)
        ans = self.answerer.answer(question, reranked, route=decision.route)

        return {
            "text": ans["text"],
            "citations": ans["citations"],
            "route": decision.route,
            "route_reason": decision.reason,
            "vector_docs": v_docs,
            "graph_docs": g_docs,
            "retrieved_docs": docs,
            "reranked_docs": reranked,
        }
