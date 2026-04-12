from unittest.mock import MagicMock
import networkx as nx
from src.indexing.community_builder import (
    detect_communities, annotate_community_ids,
    build_community_context, generate_community_summaries,
)


def _toy_graph():
    G = nx.Graph()
    for a, b in [("A", "B"), ("B", "C"), ("A", "C")]:
        G.add_edge(a, b)
    for a, b in [("X", "Y"), ("Y", "Z"), ("X", "Z")]:
        G.add_edge(a, b)
    G.add_edge("C", "X")
    for n in G.nodes:
        G.nodes[n]["type"] = "aspect"
        G.nodes[n]["name"] = n
    return G


def test_detect_communities_finds_multiple():
    G = _toy_graph()
    comms = detect_communities(G, seed=42, min_size=1)
    assert len(comms) >= 2


def test_annotate_community_ids_writes_attr():
    G = _toy_graph()
    comms = detect_communities(G, seed=42, min_size=1)
    annotate_community_ids(G, comms)
    assert all("community_id" in G.nodes[n] for n in G.nodes)


def test_build_community_context_contains_entities():
    G = _toy_graph()
    ctx = build_community_context(G, {"A", "B", "C"}, sample_reviews={})
    assert "A" in ctx["entities"] or "B" in ctx["entities"]


def test_generate_community_summaries_uses_llm():
    G = _toy_graph()
    llm = MagicMock()
    llm.call.return_value = "This is a test summary."
    comms = [{"A", "B", "C"}, {"X", "Y", "Z"}]
    summaries = generate_community_summaries(G, comms, llm, sample_reviews_map={})
    assert len(summaries) == 2
    assert summaries[0]["summary"] == "This is a test summary."
    assert summaries[0]["community_id"] == 0
