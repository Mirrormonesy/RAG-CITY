"""一键构建所有索引。支持 --skip-* 与断点续传。"""
import argparse
import os
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.llm_client import QwenClient
from src.indexing.embeddings import load_bge_embedding
from src.indexing.vector_builder import build_vector_index
from src.indexing.graph_builder import build_graph, save_graph, load_graph
from src.indexing.community_builder import (
    detect_communities, annotate_community_ids,
    generate_community_summaries, save_communities,
)
from src.indexing.summary_index import build_summary_index

load_dotenv()
logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--skip-community", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    api_key = os.environ[cfg["qwen"]["api_key_env"]]

    processed = Path(cfg["paths"]["processed_dir"])
    products = pd.read_parquet(processed / "products.parquet")
    reviews = pd.read_parquet(processed / "reviews.parquet")
    logger.info(f"Loaded {len(products)} products, {len(reviews)} reviews")

    embedding = load_bge_embedding(
        cfg["embedding"]["model"],
        cfg["embedding"]["device"],
        cfg["embedding"]["batch_size"],
    )

    if not args.skip_vector:
        t0 = time.time()
        build_vector_index(products, reviews, embedding, cfg["paths"]["chroma_docs"])
        logger.info(f"[1/3] Vector index built in {time.time()-t0:.1f}s")

    graph_path = cfg["paths"]["graph"]
    if not args.skip_graph:
        t0 = time.time()
        extract_llm = QwenClient(
            api_key=api_key,
            model=cfg["qwen"]["extract_model"],
            timeout=cfg["qwen"]["timeout"],
            max_retries=cfg["qwen"]["max_retries"],
        )
        G = build_graph(products, reviews, extract_llm,
                        resume_file="indices/graph_partial.jsonl")
        save_graph(G, graph_path)
        logger.info(f"[2/3] Graph built ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) in {time.time()-t0:.1f}s")
    else:
        G = load_graph(graph_path)

    if not args.skip_community:
        t0 = time.time()
        comms = detect_communities(G, seed=42, min_size=5)
        annotate_community_ids(G, comms)
        save_graph(G, graph_path)

        reviews_map = dict(zip(reviews["review_id"], reviews["content"]))
        summary_llm = QwenClient(
            api_key=api_key,
            model=cfg["qwen"]["summary_model"],
            timeout=cfg["qwen"]["timeout"],
            max_retries=cfg["qwen"]["max_retries"],
        )
        summaries = generate_community_summaries(G, comms, summary_llm, reviews_map)
        save_communities(summaries, cfg["paths"]["communities"])
        build_summary_index(summaries, embedding, cfg["paths"]["chroma_summaries"])
        logger.info(f"[3/3] {len(summaries)} communities summarized in {time.time()-t0:.1f}s")

    logger.info("All indices built successfully.")

if __name__ == "__main__":
    main()
