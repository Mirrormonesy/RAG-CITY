from pathlib import Path
import json
from typing import List, Set, Dict
import networkx as nx
from networkx.algorithms.community import louvain_communities
from tqdm import tqdm
from src.generation.prompts import COMMUNITY_SUMMARY_PROMPT
from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_communities(G: nx.Graph, seed: int = 42, min_size: int = 5) -> List[Set[str]]:
    comms = louvain_communities(G, seed=seed)
    filtered = [c for c in comms if len(c) >= min_size]
    logger.info(f"Louvain: {len(comms)} raw communities, {len(filtered)} after min_size={min_size}")
    return filtered


def annotate_community_ids(G: nx.Graph, communities: List[Set[str]]) -> None:
    mapping = {}
    for cid, nodes in enumerate(communities):
        for n in nodes:
            mapping[n] = cid
    for n in G.nodes:
        G.nodes[n]["community_id"] = mapping.get(n, -1)


def _node_label(G: nx.Graph, n: str) -> str:
    data = G.nodes[n]
    name = data.get("name") or data.get("title") or n
    return f"{data.get('type','?')}:{name}"


def build_community_context(G: nx.Graph, nodes: Set[str],
                            sample_reviews: Dict[str, str],
                            max_entities: int = 20,
                            max_relations: int = 30,
                            max_reviews: int = 10) -> dict:
    entities = [_node_label(G, n) for n in list(nodes)[:max_entities]]

    relations = []
    for u in nodes:
        for v in G.neighbors(u):
            if v in nodes and len(relations) < max_relations:
                rel = G[u][v].get("relation", "RELATED")
                relations.append(f"{_node_label(G,u)} --{rel}--> {_node_label(G,v)}")

    review_ids = [n.replace("review:", "") for n in nodes if n.startswith("review:")][:max_reviews]
    reviews_text = "\n".join(f"- {sample_reviews.get(rid, '(no content)')}" for rid in review_ids)

    return {
        "entities": "\n".join(entities),
        "relations": "\n".join(relations),
        "reviews": reviews_text,
        "core_entities": entities,
        "sample_review_ids": review_ids,
    }


def generate_community_summaries(G: nx.Graph, communities: List[Set[str]],
                                 llm, sample_reviews_map: Dict[str, str]) -> List[dict]:
    results = []
    for cid, nodes in enumerate(tqdm(communities, desc="Summarize communities")):
        ctx = build_community_context(G, nodes, sample_reviews_map)
        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            entities=ctx["entities"],
            relations=ctx["relations"],
            reviews=ctx["reviews"],
        )
        try:
            summary = llm.call(prompt).strip()
        except Exception as e:
            logger.warning(f"Community {cid} summary failed: {e}")
            summary = "(摘要生成失败)"
        results.append({
            "community_id": cid,
            "size": len(nodes),
            "core_entities": ctx["core_entities"],
            "sample_review_ids": ctx["sample_review_ids"],
            "summary": summary,
        })
    return results


def save_communities(summaries: List[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


def load_communities(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
