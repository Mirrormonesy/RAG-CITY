"""从 config 加载所有索引,构造一个 HybridRAG 实例。"""
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from src.utils.config import load_config
from src.utils.llm_client import QwenClient
from src.indexing.embeddings import load_bge_embedding
from src.indexing.graph_builder import load_graph
from src.retrieval.router import LLMRouter
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.node_retriever import NodeRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.fusion import RRFFuser
from src.retrieval.reranker import BgeReranker
from src.generation.answerer import Answerer
from src.pipeline import HybridRAG

load_dotenv()

def build_hybrid_rag(config_path: str = "configs/config.yaml") -> HybridRAG:
    cfg = load_config(config_path)
    api_key = os.environ[cfg["qwen"]["api_key_env"]]

    embedding = load_bge_embedding(
        cfg["embedding"]["model"], cfg["embedding"]["device"], cfg["embedding"]["batch_size"],
    )

    doc_db = Chroma(persist_directory=cfg["paths"]["chroma_docs"], embedding_function=embedding)
    sum_db = Chroma(persist_directory=cfg["paths"]["chroma_summaries"], embedding_function=embedding)

    G = load_graph(cfg["paths"]["graph"])
    node_retriever = NodeRetriever.build_from_graph(G, embedding, "indices/chroma_nodes")

    reviews = pd.read_parquet(f"{cfg['paths']['processed_dir']}/reviews.parquet")
    reviews_map = dict(zip(reviews["review_id"], reviews["content"]))

    router_llm = QwenClient(
        api_key=api_key, model=cfg["qwen"]["router_model"],
        timeout=cfg["qwen"]["timeout"], max_retries=cfg["qwen"]["max_retries"],
        temperature=0.0,
    )
    gen_llm = QwenClient(
        api_key=api_key, model=cfg["qwen"]["generate_model"],
        timeout=cfg["qwen"]["timeout"], max_retries=cfg["qwen"]["max_retries"],
        temperature=cfg["generation"]["temperature"],
        max_tokens=cfg["generation"]["max_tokens"],
    )

    return HybridRAG(
        router=LLMRouter(router_llm),
        vec_retriever=VectorRetriever(doc_db),
        graph_retriever=GraphRetriever(G, sum_db, node_retriever, reviews_map),
        fuser=RRFFuser(k_const=cfg["retrieval"]["rrf_k_const"]),
        reranker=BgeReranker(cfg["reranker"]["model"], cfg["reranker"]["device"]),
        answerer=Answerer(gen_llm),
        vector_k=cfg["retrieval"]["vector_k"],
        top_n=cfg["retrieval"]["final_top_n"],
    )
