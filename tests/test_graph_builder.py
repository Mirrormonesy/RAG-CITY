from unittest.mock import MagicMock
import pandas as pd
import networkx as nx
from src.indexing.graph_builder import build_graph, add_product_edges, add_review_facts

def test_add_product_edges():
    G = nx.Graph()
    products = pd.DataFrame([
        {"product_id": "P1", "title": "iPhone 15", "brand": "苹果", "category": "手机", "description": "x"}
    ])
    add_product_edges(G, products)
    assert G.has_node("product:P1")
    assert G.has_node("brand:苹果")
    assert G.has_node("category:手机")
    assert G.has_edge("product:P1", "brand:苹果")
    assert G.has_edge("product:P1", "category:手机")

def test_add_review_facts_creates_nodes():
    G = nx.Graph()
    G.add_node("product:P1", type="product")
    facts = {"aspects": ["续航", "屏幕"], "features": ["电池"], "sentiment": "negative"}
    add_review_facts(G, review_id="R1", product_id="P1", facts=facts)
    assert G.has_node("review:R1")
    assert G.has_node("aspect:续航")
    assert G.has_node("aspect:屏幕")
    assert G.has_edge("product:P1", "review:R1")
    assert G.has_edge("review:R1", "aspect:续航")
    assert G.nodes["review:R1"]["sentiment"] == "negative"

def test_build_graph_with_mock_llm(tmp_path):
    products = pd.DataFrame([
        {"product_id": "P1", "title": "iPhone", "brand": "苹果", "category": "手机", "description": "x"}
    ])
    reviews = pd.DataFrame([
        {"review_id": "R1", "product_id": "P1", "user_id": "U1", "rating": 2, "content": "电池烂"}
    ])
    llm = MagicMock()
    llm.call.return_value = '{"aspects":["续航"],"features":["电池"],"sentiment":"negative"}'
    resume_file = tmp_path / "partial.jsonl"
    G = build_graph(products, reviews, llm, resume_file=str(resume_file))
    assert G.has_node("review:R1")
    assert G.has_node("aspect:续航")
    assert resume_file.exists()

def test_build_graph_resumes(tmp_path):
    """已处理过的评论不再调 LLM"""
    resume_file = tmp_path / "partial.jsonl"
    resume_file.write_text(
        '{"review_id":"R1","product_id":"P1","facts":{"aspects":["续航"],"features":[],"sentiment":"negative"}}\n',
        encoding="utf-8",
    )
    products = pd.DataFrame([
        {"product_id": "P1", "title": "iPhone", "brand": "苹果", "category": "手机", "description": "x"}
    ])
    reviews = pd.DataFrame([
        {"review_id": "R1", "product_id": "P1", "user_id": "U1", "rating": 2, "content": "电池烂"}
    ])
    llm = MagicMock()
    G = build_graph(products, reviews, llm, resume_file=str(resume_file))
    llm.call.assert_not_called()
    assert G.has_node("aspect:续航")
