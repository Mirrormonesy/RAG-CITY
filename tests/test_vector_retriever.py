from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.retrieval.vector_retriever import VectorRetriever

def test_retrieve_delegates_to_chroma():
    fake_db = MagicMock()
    fake_db.similarity_search.return_value = [
        Document(page_content="doc1", metadata={"doc_type": "review"}),
        Document(page_content="doc2", metadata={"doc_type": "product"}),
    ]
    r = VectorRetriever(fake_db)
    out = r.retrieve("query", k=5)
    assert len(out) == 2
    fake_db.similarity_search.assert_called_once_with("query", k=5, filter=None)

def test_retrieve_with_filter():
    fake_db = MagicMock()
    fake_db.similarity_search.return_value = []
    r = VectorRetriever(fake_db)
    r.retrieve("q", k=3, filter={"brand": "苹果"})
    fake_db.similarity_search.assert_called_once_with("q", k=3, filter={"brand": "苹果"})
