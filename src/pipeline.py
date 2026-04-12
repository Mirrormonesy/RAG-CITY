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

        if decision.route == "vector":
            docs = self.vec.retrieve(question, k=self.vector_k)
        elif decision.route == "graph":
            docs = self.graph.retrieve(question)
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
            "retrieved_docs": docs,
            "reranked_docs": reranked,
        }
