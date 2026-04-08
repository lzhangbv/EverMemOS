"""
Knowledge Graph Retrieval Service

Implements query2edge + Personalized PageRank retrieval,
adapted from AutoSchemaKG's HippoRAG2 approach.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from api_specs.kg_types import KGConfig, KGEdgeType, KGNodeType
from core.observation.logger import get_logger

logger = get_logger(__name__)


class GraphRetrievalService:
    """
    Graph-based memory retrieval using edge embeddings + PPR.

    Pipeline:
    1. Load cached igraph + edge embeddings
    2. Query-to-edge cosine similarity
    3. Aggregate edge scores to nodes
    4. Personalized PageRank propagation
    5. Aggregate PPR scores to memory nodes
    6. Return ranked memory IDs with scores
    """

    def __init__(self, config: Optional[KGConfig] = None):
        self.config = config or KGConfig()

    async def search(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories via graph signals.

        Returns list of dicts compatible with RRF fusion:
        [{"id": memory_id, "score": float, "_search_source": "graph", ...}]
        """
        from agentic_layer.vectorize_service import get_vectorize_service
        from core.di.container import get_bean_by_type
        from infra_layer.adapters.out.persistence.repository.knowledge_graph_repository import (
            KnowledgeGraphRepository,
        )

        start = time.perf_counter()
        scope_key = f"group:{group_id}" if group_id else f"user:{user_id}"

        repo = get_bean_by_type(KnowledgeGraphRepository)
        cached = await repo.get_cached_graph(scope_key)

        if cached is None or cached.graph.vcount() == 0:
            return []

        g = cached.graph
        edge_emb = cached.edge_embeddings

        if edge_emb is None or edge_emb.size == 0:
            return []

        # Step 1: Embed query
        vectorize_service = get_vectorize_service()
        query_vec = np.asarray(
            await vectorize_service.get_embedding(query, is_query=True),
            dtype=np.float32,
        )
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        # Step 2: Query-to-edge matching
        scores = edge_emb @ query_vec  # cosine similarity (edge_emb already L2-normed)
        top_k_edges = min(self.config.top_k_edges, len(scores))
        top_indices = np.argsort(scores)[-top_k_edges:][::-1]

        # Step 3: Aggregate edge scores to nodes
        node_scores: Dict[int, List[float]] = {}
        for idx in top_indices:
            edge_score = float(scores[idx])
            if edge_score <= 0:
                continue
            if idx < g.ecount():
                edge = g.es[int(idx)]
                for vid in (edge.source, edge.target):
                    node_scores.setdefault(vid, []).append(edge_score)

        if not node_scores:
            return []

        # Average scores per node
        personalization = [0.0] * g.vcount()
        for vid, s_list in node_scores.items():
            personalization[vid] = sum(s_list) / len(s_list)

        # Step 4: Personalized PageRank
        ppr_scores = g.personalized_pagerank(
            vertices=None,
            directed=True,
            damping=1.0 - self.config.ppr_damping,  # igraph uses damping, not restart
            personalized=personalization,
            implementation="prpack",
        )

        # Step 5: Aggregate to memory nodes via SOURCE edges
        memory_scores: Dict[str, float] = {}
        threshold = self.config.ppr_score_threshold

        for vid in range(g.vcount()):
            if ppr_scores[vid] < threshold:
                continue
            v = g.vs[vid]
            if v["node_type"] == KGNodeType.MEMORY.value:
                mem_id = v.get("memory_id") or v.get("label", "")
                if mem_id:
                    memory_scores[mem_id] = memory_scores.get(mem_id, 0) + ppr_scores[vid]
            else:
                # Follow SOURCE edges to memory nodes
                for eid in g.incident(vid, mode="out"):
                    e = g.es[eid]
                    if e["edge_type"] == KGEdgeType.SOURCE.value:
                        target_v = g.vs[e.target]
                        if target_v["node_type"] == KGNodeType.MEMORY.value:
                            mem_id = target_v.get("memory_id") or target_v.get("label", "")
                            if mem_id:
                                memory_scores[mem_id] = (
                                    memory_scores.get(mem_id, 0) + ppr_scores[vid]
                                )

        if not memory_scores:
            return []

        # Step 6: Sort and format output
        sorted_memories = sorted(
            memory_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for mem_id, score in sorted_memories:
            results.append({
                "id": mem_id,
                "score": score,
                "_search_source": "graph",
            })

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"Graph retrieval scope={scope_key} "
            f"results={len(results)} latency={elapsed:.1f}ms"
        )
        return results
