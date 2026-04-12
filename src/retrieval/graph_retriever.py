from typing import Dict, List
import networkx as nx
from langchain_core.documents import Document
from src.indexing.community_builder import build_community_context

class GraphRetriever:
    """三步流程:
    Step 1 按社区摘要语义检索 top-K 社区
    Step 2 按问题实体名匹配,并取其所属社区加入候选
    Step 3 对每个候选社区提取子图 + 代表性评论,包成 Document
    """

    def __init__(self, graph: nx.Graph, summary_db,
                 node_retriever, reviews_map: Dict[str, str]):
        self.G = graph
        self.summary_db = summary_db
        self.node_retriever = node_retriever  # 用来对节点名检索
        self.reviews_map = reviews_map

    def retrieve(self, query: str, k_communities: int = 3,
                 k_entity_expand: int = 5) -> List[Document]:
        # Step 1: 语义
        sem_docs = self.summary_db.similarity_search(query, k=k_communities)
        candidate_ids = {d.metadata["community_id"] for d in sem_docs}
        summary_by_id = {d.metadata["community_id"]: d.page_content for d in sem_docs}

        # Step 2: 实体扩展
        ent_docs = self.node_retriever.retrieve(query, k=k_entity_expand)
        for d in ent_docs:
            node_id = d.metadata.get("node_id")
            if node_id and self.G.has_node(node_id):
                cid = self.G.nodes[node_id].get("community_id", -1)
                if cid >= 0 and cid not in candidate_ids:
                    candidate_ids.add(cid)
                    summary_by_id.setdefault(cid, "")

        # Step 3: 子图 → 文档
        results = []
        nodes_by_community: Dict[int, set] = {}
        for n, data in self.G.nodes(data=True):
            cid = data.get("community_id", -1)
            if cid in candidate_ids:
                nodes_by_community.setdefault(cid, set()).add(n)

        for cid in candidate_ids:
            nodes = nodes_by_community.get(cid, set())
            if not nodes:
                continue
            ctx = build_community_context(self.G, nodes, self.reviews_map)
            content = (
                f"[社区 {cid} 摘要]\n{summary_by_id.get(cid, '')}\n"
                f"[核心实体]\n{ctx['entities']}\n"
                f"[关键关系]\n{ctx['relations']}\n"
                f"[代表性评论]\n{ctx['reviews']}"
            )
            results.append(Document(
                page_content=content,
                metadata={
                    "doc_type": "community",
                    "community_id": cid,
                    "entities": ",".join(ctx["core_entities"][:10]),
                },
            ))
        return results
