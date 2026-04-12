from typing import Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

class VectorRetriever:
    def __init__(self, db: Chroma):
        self.db = db

    def retrieve(self, query: str, k: int = 20,
                 filter: Optional[dict] = None) -> list[Document]:
        return self.db.similarity_search(query, k=k, filter=filter)
