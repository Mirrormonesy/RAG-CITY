import json
from pathlib import Path
import networkx as nx
import pandas as pd
from tqdm import tqdm
from src.indexing.extractor import extract_review_facts
from src.utils.logger import get_logger

logger = get_logger(__name__)

def add_product_edges(G: nx.Graph, products: pd.DataFrame) -> None:
    for _, p in products.iterrows():
        pid = f"product:{p['product_id']}"
        bid = f"brand:{p['brand']}"
        cid = f"category:{p['category']}"
        G.add_node(pid, type="product", title=p["title"], price=float(p.get("price", 0)))
        G.add_node(bid, type="brand", name=p["brand"])
        G.add_node(cid, type="category", name=p["category"])
        G.add_edge(pid, bid, relation="MADE_BY")
        G.add_edge(pid, cid, relation="BELONGS_TO")

def add_review_facts(G: nx.Graph, review_id: str, product_id: str, facts: dict) -> None:
    rid = f"review:{review_id}"
    pid = f"product:{product_id}"
    sentiment = facts.get("sentiment", "neutral")
    G.add_node(rid, type="review", sentiment=sentiment)
    G.add_edge(pid, rid, relation="HAS_REVIEW")
    for asp in facts.get("aspects", []):
        aid = f"aspect:{asp}"
        G.add_node(aid, type="aspect", name=asp)
        G.add_edge(rid, aid, relation="MENTIONS")
    for feat in facts.get("features", []):
        fid = f"feature:{feat}"
        G.add_node(fid, type="feature", name=feat)
        G.add_edge(rid, fid, relation="ABOUT_FEATURE")
    sid = f"sentiment:{sentiment}"
    G.add_node(sid, type="sentiment", name=sentiment)
    G.add_edge(rid, sid, relation="EXPRESSES")

def _load_done(resume_file: str) -> dict:
    path = Path(resume_file)
    if not path.exists():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            done[rec["review_id"]] = rec
        except json.JSONDecodeError:
            continue
    return done

def _append_done(resume_file: str, record: dict) -> None:
    Path(resume_file).parent.mkdir(parents=True, exist_ok=True)
    with open(resume_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def build_graph(products: pd.DataFrame, reviews: pd.DataFrame,
                llm, resume_file: str = "indices/graph_partial.jsonl") -> nx.Graph:
    G = nx.Graph()
    add_product_edges(G, products)
    done = _load_done(resume_file)
    logger.info(f"Resuming with {len(done)} pre-processed reviews")

    for _, r in tqdm(reviews.iterrows(), total=len(reviews), desc="Extract facts"):
        rid = r["review_id"]
        if rid in done:
            facts = done[rid]["facts"]
        else:
            facts = extract_review_facts(llm, r["content"])
            _append_done(resume_file, {
                "review_id": rid,
                "product_id": r["product_id"],
                "facts": facts,
            })
        add_review_facts(G, rid, r["product_id"], facts)
    return G

def save_graph(G: nx.Graph, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(path, "wb") as f:
        pickle.dump(G, f)

def load_graph(path: str) -> nx.Graph:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
