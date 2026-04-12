from unittest.mock import MagicMock
import networkx as nx
from langchain_core.documents import Document
from src.retrieval.graph_retriever import GraphRetriever

def _make_graph():
    G = nx.Graph()
    # 社区 0
    G.add_node("product:P1", type="product", title="iPhone 15", community_id=0)
    G.add_node("aspect:续航", type="aspect", name="续航", community_id=0)
    G.add_node("review:R1", type="review", sentiment="negative", community_id=0)
    G.add_edge("product:P1", "review:R1", relation="HAS_REVIEW")
    G.add_edge("review:R1", "aspect:续航", relation="MENTIONS")
    # 社区 1
    G.add_node("product:P2", type="product", title="小米14", community_id=1)
    G.add_node("aspect:拍照", type="aspect", name="拍照", community_id=1)
    G.add_node("review:R2", type="review", sentiment="positive", community_id=1)
    G.add_edge("product:P2", "review:R2", relation="HAS_REVIEW")
    G.add_edge("review:R2", "aspect:拍照", relation="MENTIONS")
    return G

def _make_summary_db(community_ids):
    db = MagicMock()
    def fake_search(query, k):
        return [
            Document(page_content=f"Community {cid} summary",
                     metadata={"community_id": cid, "core_entities": ""})
            for cid in community_ids[:k]
        ]
    db.similarity_search.side_effect = fake_search
    return db

def test_step1_finds_communities_by_semantic():
    G = _make_graph()
    summary_db = _make_summary_db([0, 1])
    node_retriever = MagicMock()
    node_retriever.retrieve.return_value = []  # Step 2 返回空
    reviews_map = {"R1": "电池差", "R2": "拍照好"}
    r = GraphRetriever(G, summary_db, node_retriever, reviews_map)
    docs = r.retrieve("电池怎么样", k_communities=2)
    cids = {d.metadata["community_id"] for d in docs}
    assert cids == {0, 1}

def test_step2_adds_entity_matched_communities():
    G = _make_graph()
    summary_db = _make_summary_db([0])   # Step 1 只返回社区 0
    node_retriever = MagicMock()
    node_retriever.retrieve.return_value = [
        Document(page_content="拍照", metadata={"node_id": "aspect:拍照"})
    ]
    reviews_map = {"R1": "电池差", "R2": "拍照好"}
    r = GraphRetriever(G, summary_db, node_retriever, reviews_map)
    docs = r.retrieve("拍照", k_communities=1)
    cids = {d.metadata["community_id"] for d in docs}
    assert cids == {0, 1}  # 0 来自 Step1, 1 来自 Step2

def test_docs_contain_relations_and_reviews():
    G = _make_graph()
    summary_db = _make_summary_db([0])
    node_retriever = MagicMock()
    node_retriever.retrieve.return_value = []
    reviews_map = {"R1": "电池掉得快一天两充"}
    r = GraphRetriever(G, summary_db, node_retriever, reviews_map)
    docs = r.retrieve("续航", k_communities=1)
    content = docs[0].page_content
    assert "HAS_REVIEW" in content or "MENTIONS" in content
    assert "电池掉得快" in content
