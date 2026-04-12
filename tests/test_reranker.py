from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.retrieval.reranker import BgeReranker

def test_rerank_sorts_by_score():
    with patch("src.retrieval.reranker.CrossEncoder") as CE:
        model = MagicMock()
        model.predict.return_value = [0.1, 0.9, 0.5]
        CE.return_value = model
        reranker = BgeReranker("fake-model", device="cpu")
        docs = [Document(page_content=c) for c in ["low", "high", "mid"]]
        out = reranker.rerank("q", docs, top_n=3)
    assert [d.page_content for d in out] == ["high", "mid", "low"]

def test_rerank_truncates_to_top_n():
    with patch("src.retrieval.reranker.CrossEncoder") as CE:
        model = MagicMock()
        model.predict.return_value = [0.3, 0.9, 0.5, 0.1, 0.7]
        CE.return_value = model
        reranker = BgeReranker("fake-model", device="cpu")
        docs = [Document(page_content=c) for c in "ABCDE"]
        out = reranker.rerank("q", docs, top_n=2)
    assert len(out) == 2
    assert [d.page_content for d in out] == ["B", "E"]

def test_rerank_empty_returns_empty():
    with patch("src.retrieval.reranker.CrossEncoder"):
        reranker = BgeReranker("fake-model", device="cpu")
        assert reranker.rerank("q", [], top_n=5) == []
