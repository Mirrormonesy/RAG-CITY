import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

_NODE_COLORS = {
    "product": "#4A90E2", "brand": "#F5A623", "category": "#7ED321",
    "aspect": "#D0021B", "feature": "#BD10E0", "review": "#9B9B9B",
    "sentiment": "#50E3C2",
}

def plot_community_subgraph(G: nx.Graph, community_id: int, out_path: str,
                             max_nodes: int = 80) -> None:
    nodes = [n for n, d in G.nodes(data=True) if d.get("community_id") == community_id][:max_nodes]
    sub = G.subgraph(nodes)
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(sub, seed=42)
    colors = [_NODE_COLORS.get(sub.nodes[n].get("type"), "#999") for n in sub.nodes]
    labels = {n: sub.nodes[n].get("name", n) for n in sub.nodes}
    nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=300, alpha=0.8)
    nx.draw_networkx_edges(sub, pos, alpha=0.3)
    nx.draw_networkx_labels(sub, pos, labels, font_size=8, font_family="SimHei")
    plt.title(f"Community #{community_id} ({len(sub.nodes)} nodes)")
    plt.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
