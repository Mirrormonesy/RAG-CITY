"""对图谱节点名做 embedding 索引,用于 GraphRetriever Step 2"""
from pathlib import Path
import networkx as nx
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

class NodeRetriever:
    def __init__(self, db: Chroma):
        self.db = db

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        return self.db.similarity_search(query, k=k)

    @classmethod
    def build_from_graph(cls, G: nx.Graph, embedding, persist_dir: str) -> "NodeRetriever":
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        docs = []
        for n, data in G.nodes(data=True):
            name = data.get("name") or data.get("title") or n
            if data.get("type") in {"product", "brand", "category", "aspect", "feature"}:
                docs.append(Document(
                    page_content=str(name),
                    metadata={"node_id": n, "node_type": data.get("type", "")},
                ))
        db = Chroma.from_documents(docs, embedding=embedding, persist_directory=persist_dir)
        if hasattr(db, "persist"):
            db.persist()
        return cls(db)
