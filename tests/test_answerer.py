from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.generation.answerer import Answerer, parse_citations, format_context

def test_parse_citations_extracts_refs():
    text = "苹果用户抱怨续航 [V1][G1]。有人反映一天两充 [V2]."
    refs = parse_citations(text)
    assert refs == ["V1", "G1", "V2"]

def test_parse_citations_dedup_preserves_order():
    text = "xx [V1] yy [V1] zz [G1]"
    assert parse_citations(text) == ["V1", "G1"]

def test_format_context_vector():
    docs = [
        Document(page_content="好手机", metadata={"doc_type": "review", "review_id": "R1"}),
        Document(page_content="旗舰机", metadata={"doc_type": "product", "product_id": "P1"}),
    ]
    ctx, mapping = format_context(docs, prefix="V")
    assert "[V1]" in ctx and "[V2]" in ctx
    assert mapping["V1"]["review_id"] == "R1"
    assert mapping["V2"]["product_id"] == "P1"

def test_answerer_vector_route():
    llm = MagicMock()
    llm.call.return_value = "答案内容 [V1]"
    a = Answerer(llm)
    docs = [Document(page_content="doc", metadata={"doc_type": "review", "review_id": "R1"})]
    ans = a.answer("q", docs, route="vector")
    assert ans["text"] == "答案内容 [V1]"
    assert "V1" in ans["citations"]

def test_answerer_hybrid_splits_graph_and_vector():
    llm = MagicMock()
    llm.call.return_value = "综合答案 [G1][V1]"
    a = Answerer(llm)
    docs = [
        Document(page_content="cmty", metadata={"doc_type": "community", "community_id": 7}),
        Document(page_content="rev", metadata={"doc_type": "review", "review_id": "R1"}),
    ]
    ans = a.answer("q", docs, route="hybrid")
    assert "G1" in ans["citations"]
    assert "V1" in ans["citations"]
    assert ans["citations"]["G1"]["community_id"] == 7
