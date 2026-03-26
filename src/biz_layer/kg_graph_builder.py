"""
Knowledge Graph Builder

Orchestrates triple extraction → entity deduplication → graph update.
Called asynchronously after memory persistence.
"""

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from api_specs.kg_types import (
    EventEntity,
    KGConfig,
    KGEdgeType,
    KGNodeType,
    Triple,
    TripleExtractionResult,
)
from core.observation.logger import get_logger
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.kg_triple_extractor import KGTripleExtractor

logger = get_logger(__name__)


def _node_id(name: str, node_type: str) -> str:
    """Deterministic node ID from name + type."""
    raw = f"{node_type}:{name.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _edge_key(source: str, target: str, edge_type: str, relation: str = "") -> str:
    """Deterministic edge dedup key."""
    raw = f"{source}|{target}|{edge_type}|{relation.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class KnowledgeGraphBuilder:
    """Builds and incrementally updates per-scope knowledge graphs."""

    def __init__(self, llm_provider: LLMProvider, config: Optional[KGConfig] = None):
        self.extractor = KGTripleExtractor(llm_provider)
        self.config = config or KGConfig()

    async def update_graph(
        self,
        atomic_facts: List[str],
        memory_id: str,
        memory_type: str,
        user_id: str,
        group_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Extract triples from atomic facts and merge into the scope's graph.

        Args:
            atomic_facts: List of atomic fact sentences from EventLog
            memory_id: The source memory's ID (for SOURCE edges)
            memory_type: e.g. "episodic_memory", "event_log"
            user_id: User ID
            group_id: Group ID (if group memory)
            timestamp: Memory timestamp
        """
        if not atomic_facts:
            return

        from agentic_layer.vectorize_service import get_vectorize_service
        from core.di.container import get_bean_by_type
        from infra_layer.adapters.out.persistence.repository.knowledge_graph_repository import (
            KnowledgeGraphRepository,
        )

        repo = get_bean_by_type(KnowledgeGraphRepository)
        vectorize_service = get_vectorize_service()
        scope_key = f"group:{group_id}" if group_id else f"user:{user_id}"
        ts = timestamp or datetime.utcnow()

        # 1. Extract triples
        extraction = await self.extractor.extract_all(atomic_facts)

        # 2. Load existing graph
        doc = await repo.get_by_scope(scope_key)
        existing_nodes: Dict[str, Dict[str, Any]] = {}
        existing_edges: Dict[str, Dict[str, Any]] = {}
        existing_edge_texts: List[str] = []

        if doc:
            for n in doc.nodes:
                existing_nodes[n["id"]] = n
            for e in doc.edges:
                key = _edge_key(e["source"], e["target"], e["type"], e.get("attrs", {}).get("relation_text", ""))
                existing_edges[key] = e
            existing_edge_texts = list(doc.edge_texts or [])

        # 3. Build new nodes and edges
        new_edge_texts: List[str] = []
        self._add_memory_node(existing_nodes, memory_id, memory_type, ts)
        self._process_entity_relations(
            extraction.entity_relations, existing_nodes, existing_edges,
            new_edge_texts, memory_id, ts,
        )
        self._process_event_entities(
            extraction.event_entities, existing_nodes, existing_edges,
            new_edge_texts, memory_id, ts,
        )
        self._process_event_relations(
            extraction.event_relations, existing_nodes, existing_edges,
            new_edge_texts, ts,
        )

        # 4. Compute embeddings for new edge texts
        all_edge_texts = existing_edge_texts + new_edge_texts
        edge_embeddings = None
        if all_edge_texts:
            try:
                emb_list = await vectorize_service.get_embeddings(all_edge_texts)
                edge_embeddings = np.array(emb_list, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Failed to compute edge embeddings: {e}")

        # 5. Persist
        nodes_list = list(existing_nodes.values())
        edges_list = list(existing_edges.values())
        await repo.save_graph(
            scope_key=scope_key,
            nodes=nodes_list,
            edges=edges_list,
            edge_texts=all_edge_texts,
            edge_embeddings=edge_embeddings,
        )
        logger.info(
            f"Graph updated scope={scope_key} "
            f"nodes={len(nodes_list)} edges={len(edges_list)} "
            f"new_edge_texts={len(new_edge_texts)}"
        )

    def _add_memory_node(
        self, nodes: Dict, memory_id: str, memory_type: str, ts: datetime
    ) -> str:
        """Add or update a memory node."""
        nid = _node_id(memory_id, KGNodeType.MEMORY.value)
        if nid not in nodes:
            nodes[nid] = {
                "id": nid,
                "type": KGNodeType.MEMORY.value,
                "name": memory_id,
                "attrs": {"memory_id": memory_id, "memory_type": memory_type, "timestamp": ts.isoformat()},
            }
        return nid

    def _get_or_create_entity(
        self, nodes: Dict, name: str, ts: datetime
    ) -> str:
        """Get existing entity node or create new one."""
        nid = _node_id(name, KGNodeType.ENTITY.value)
        if nid in nodes:
            nodes[nid]["attrs"]["last_seen"] = ts.isoformat()
        else:
            nodes[nid] = {
                "id": nid,
                "type": KGNodeType.ENTITY.value,
                "name": name.strip(),
                "attrs": {"first_seen": ts.isoformat(), "last_seen": ts.isoformat()},
            }
        return nid

    def _get_or_create_event(
        self, nodes: Dict, description: str, ts: datetime
    ) -> str:
        """Get existing event node or create new one."""
        nid = _node_id(description, KGNodeType.EVENT.value)
        if nid not in nodes:
            nodes[nid] = {
                "id": nid,
                "type": KGNodeType.EVENT.value,
                "name": description.strip(),
                "attrs": {"timestamp": ts.isoformat()},
            }
        return nid

    def _add_edge(
        self,
        edges: Dict,
        new_edge_texts: List[str],
        source: str,
        target: str,
        edge_type: str,
        relation_text: str = "",
        attrs: Optional[Dict] = None,
    ) -> None:
        """Add edge if not duplicate."""
        key = _edge_key(source, target, edge_type, relation_text)
        if key not in edges:
            edge_attrs = {"relation_text": relation_text, **(attrs or {})}
            edges[key] = {
                "source": source,
                "target": target,
                "type": edge_type,
                "attrs": edge_attrs,
            }
            # Only RELATION edges get edge text for embedding
            if edge_type == KGEdgeType.RELATION.value and relation_text:
                # Find node names for edge text
                new_edge_texts.append(relation_text)

    def _process_entity_relations(
        self, triples: List[Triple], nodes: Dict, edges: Dict,
        new_edge_texts: List[str], memory_id: str, ts: datetime,
    ) -> None:
        """Process entity-relation triples."""
        mem_nid = _node_id(memory_id, KGNodeType.MEMORY.value)
        for t in triples:
            head_nid = self._get_or_create_entity(nodes, t.head, ts)
            tail_nid = self._get_or_create_entity(nodes, t.tail, ts)
            # RELATION edge
            edge_text = f"{t.head} {t.relation} {t.tail}"
            self._add_edge(
                edges, new_edge_texts, head_nid, tail_nid,
                KGEdgeType.RELATION.value, edge_text,
                {"timestamp": ts.isoformat()},
            )
            # SOURCE edges
            self._add_edge(edges, new_edge_texts, head_nid, mem_nid, KGEdgeType.SOURCE.value)
            self._add_edge(edges, new_edge_texts, tail_nid, mem_nid, KGEdgeType.SOURCE.value)

    def _process_event_entities(
        self, event_entities: List[EventEntity], nodes: Dict, edges: Dict,
        new_edge_texts: List[str], memory_id: str, ts: datetime,
    ) -> None:
        """Process event-entity links."""
        mem_nid = _node_id(memory_id, KGNodeType.MEMORY.value)
        for ee in event_entities:
            event_nid = self._get_or_create_event(nodes, ee.event, ts)
            # SOURCE edge from event to memory
            self._add_edge(edges, new_edge_texts, event_nid, mem_nid, KGEdgeType.SOURCE.value)
            for entity_name in ee.entities:
                entity_nid = self._get_or_create_entity(nodes, entity_name, ts)
                # PARTICIPATES edge
                self._add_edge(
                    edges, new_edge_texts, entity_nid, event_nid,
                    KGEdgeType.PARTICIPATES.value,
                )
                # SOURCE edge from entity to memory
                self._add_edge(edges, new_edge_texts, entity_nid, mem_nid, KGEdgeType.SOURCE.value)

    def _process_event_relations(
        self, triples: List[Triple], nodes: Dict, edges: Dict,
        new_edge_texts: List[str], ts: datetime,
    ) -> None:
        """Process event-event temporal/causal relations."""
        causal_keywords = {"because", "as a result", "因为", "结果"}
        for t in triples:
            head_nid = self._get_or_create_event(nodes, t.head, ts)
            tail_nid = self._get_or_create_event(nodes, t.tail, ts)
            rel_lower = t.relation.strip().lower()
            if any(kw in rel_lower for kw in causal_keywords):
                edge_type = KGEdgeType.CAUSAL.value
            else:
                edge_type = KGEdgeType.TEMPORAL.value
            self._add_edge(
                edges, new_edge_texts, head_nid, tail_nid,
                edge_type, t.relation,
                {"timestamp": ts.isoformat()},
            )
