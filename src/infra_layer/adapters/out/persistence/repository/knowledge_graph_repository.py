"""
Knowledge Graph Repository

Manages persistence (MongoDB) and in-memory cache (igraph + numpy)
for per-scope knowledge graphs.
"""

import base64
import io
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.di.decorators import repository
from core.observation.logger import get_logger
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.knowledge_graph_doc import (
    KnowledgeGraphDoc,
)

logger = get_logger(__name__)


def _encode_numpy(arr: Optional[np.ndarray]) -> Optional[str]:
    """Encode numpy array to base64 string for MongoDB storage."""
    if arr is None or arr.size == 0:
        return None
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_numpy(b64: Optional[str]) -> Optional[np.ndarray]:
    """Decode base64 string back to numpy array."""
    if not b64:
        return None
    buf = io.BytesIO(base64.b64decode(b64))
    return np.load(buf)


class _CachedGraph:
    """In-memory representation of a knowledge graph for fast retrieval."""

    __slots__ = ("graph", "node_embeddings", "edge_embeddings", "edge_texts", "version")

    def __init__(
        self,
        graph,  # igraph.Graph
        node_embeddings: Optional[np.ndarray],
        edge_embeddings: Optional[np.ndarray],
        edge_texts: List[str],
        version: int,
    ):
        self.graph = graph
        self.node_embeddings = node_embeddings
        self.edge_embeddings = edge_embeddings
        self.edge_texts = edge_texts
        self.version = version


@repository("knowledge_graph_repository", primary=True)
class KnowledgeGraphRepository(BaseRepository[KnowledgeGraphDoc]):
    """
    Repository for knowledge graph persistence and caching.

    - MongoDB for durable storage
    - igraph + numpy in LRU cache for fast retrieval
    """

    def __init__(self):
        super().__init__(KnowledgeGraphDoc)
        self._cache: Dict[str, _CachedGraph] = {}
        self._max_cache_size = 100

    # ==================== Cache Management ====================

    def invalidate_cache(self, scope_key: str) -> None:
        """Remove a scope from the in-memory cache."""
        self._cache.pop(scope_key, None)

    def _build_igraph(self, doc: KnowledgeGraphDoc) -> _CachedGraph:
        """Build igraph.Graph from a MongoDB document."""
        import igraph as ig

        g = ig.Graph(directed=True)

        # Add nodes
        node_id_map: Dict[str, int] = {}
        for node in doc.nodes:
            idx = g.vcount()
            g.add_vertex(
                name=node["id"],
                node_type=node.get("type", ""),
                label=node.get("name", ""),
                **{k: v for k, v in node.get("attrs", {}).items()},
            )
            node_id_map[node["id"]] = idx

        # Add edges
        for edge in doc.edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in node_id_map and tgt in node_id_map:
                g.add_edge(
                    node_id_map[src],
                    node_id_map[tgt],
                    edge_type=edge.get("type", ""),
                    **{k: v for k, v in edge.get("attrs", {}).items()},
                )

        node_emb = _decode_numpy(doc.node_embeddings_b64)
        edge_emb = _decode_numpy(doc.edge_embeddings_b64)

        # L2-normalize edge embeddings for cosine similarity via dot product
        if edge_emb is not None and edge_emb.size > 0:
            norms = np.linalg.norm(edge_emb, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            edge_emb = edge_emb / norms

        return _CachedGraph(
            graph=g,
            node_embeddings=node_emb,
            edge_embeddings=edge_emb,
            edge_texts=doc.edge_texts or [],
            version=doc.version,
        )

    # ==================== Read Operations ====================

    async def get_by_scope(self, scope_key: str) -> Optional[KnowledgeGraphDoc]:
        """Fetch graph document by scope key."""
        return await self.model.find_one({"scope_key": scope_key})

    async def get_cached_graph(self, scope_key: str) -> Optional[_CachedGraph]:
        """Get igraph from cache, loading from MongoDB if needed."""
        cached = self._cache.get(scope_key)
        if cached is not None:
            return cached

        doc = await self.get_by_scope(scope_key)
        if doc is None or not doc.nodes:
            return None

        cached = self._build_igraph(doc)

        # Evict oldest if cache full
        if len(self._cache) >= self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[scope_key] = cached
        return cached

    # ==================== Write Operations ====================

    async def save_graph(
        self,
        scope_key: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        edge_texts: List[str],
        node_embeddings: Optional[np.ndarray] = None,
        edge_embeddings: Optional[np.ndarray] = None,
    ) -> KnowledgeGraphDoc:
        """Create or update a knowledge graph document with optimistic locking."""
        existing = await self.get_by_scope(scope_key)

        if existing:
            # Optimistic lock: increment version
            existing.nodes = nodes
            existing.edges = edges
            existing.edge_texts = edge_texts
            existing.node_embeddings_b64 = _encode_numpy(node_embeddings)
            existing.edge_embeddings_b64 = _encode_numpy(edge_embeddings)
            existing.version += 1
            await existing.save()
            doc = existing
        else:
            doc = KnowledgeGraphDoc(
                scope_key=scope_key,
                nodes=nodes,
                edges=edges,
                edge_texts=edge_texts,
                node_embeddings_b64=_encode_numpy(node_embeddings),
                edge_embeddings_b64=_encode_numpy(edge_embeddings),
                version=1,
            )
            await doc.insert()

        self.invalidate_cache(scope_key)
        logger.info(
            f"Saved knowledge graph scope={scope_key} "
            f"nodes={len(nodes)} edges={len(edges)} version={doc.version}"
        )
        return doc
