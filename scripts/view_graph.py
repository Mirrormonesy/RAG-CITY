"""Visualize top communities from the knowledge graph.

Usage:
  python scripts/view_graph.py           # plot top 3 biggest
  python scripts/view_graph.py 0 3 7     # plot specific community ids
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.graph_builder import load_graph
from src.indexing.graph_viz import plot_community_subgraph

G = load_graph(str(PROJECT_ROOT / "indices" / "graph.pkl"))
comms = json.load(open(PROJECT_ROOT / "indices" / "communities.json", "r", encoding="utf-8"))

if len(sys.argv) > 1:
    ids = [int(x) for x in sys.argv[1:]]
else:
    ids = [int(k) for k in sorted(comms.keys(), key=lambda k: -len(comms[k]))[:3]]

out_dir = PROJECT_ROOT / "report" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

for cid in ids:
    if str(cid) not in comms:
        print(f"[skip] community {cid} not found")
        continue
    out_path = out_dir / f"community_{cid}.png"
    plot_community_subgraph(G, cid, str(out_path))
    print(f"Community {cid}: {len(comms[str(cid)])} nodes -> {out_path}")
