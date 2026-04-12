from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.pipeline import HybridRAG
from src.retrieval.router import RouteDecision

def _make_rag(route="vector"):
    router = MagicMock()
    router.route.return_value = RouteDecision(route, "reason")
    vec = MagicMock(); vec.retrieve.return_value = [Document(page_content="v", metadata={"review_id":"R1"})]
    graph = MagicMock(); graph.retrieve.return_value = [Document(page_content="g", metadata={"community_id":0, "doc_type":"community"})]
    fuser = MagicMock(); fuser.fuse.return_value = [Document(page_content="mix", metadata={})]
    reranker = MagicMock(); reranker.rerank.side_effect = lambda q, d, top_n: d[:top_n]
    answerer = MagicMock(); answerer.answer.return_value = {"text": "ans", "citations": {}, "prompt_used": ""}
    rag = HybridRAG(router, vec, graph, fuser, reranker, answerer, top_n=3)
    return rag, (router, vec, graph, fuser, reranker, answerer)

def test_vector_route_skips_graph():
    rag, (_, vec, graph, fuser, _, ans) = _make_rag("vector")
    rag.query("q")
    vec.retrieve.assert_called_once()
    graph.retrieve.assert_not_called()
    fuser.fuse.assert_not_called()
    ans.answer.assert_called_once()

def test_graph_route_skips_vector():
    rag, (_, vec, graph, fuser, _, _) = _make_rag("graph")
    rag.query("q")
    graph.retrieve.assert_called_once()
    vec.retrieve.assert_not_called()
    fuser.fuse.assert_not_called()

def test_hybrid_route_calls_both_and_fuses():
    rag, (_, vec, graph, fuser, reranker, ans) = _make_rag("hybrid")
    rag.query("q")
    vec.retrieve.assert_called_once()
    graph.retrieve.assert_called_once()
    fuser.fuse.assert_called_once()
    reranker.rerank.assert_called_once()
    ans.answer.assert_called_once()
