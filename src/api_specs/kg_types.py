"""
Knowledge Graph types module

Data structures for knowledge graph construction and retrieval.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class KGNodeType(str, Enum):
    """Knowledge graph node types."""

    ENTITY = "entity"
    EVENT = "event"
    MEMORY = "memory"


class KGEdgeType(str, Enum):
    """Knowledge graph edge types."""

    RELATION = "relation"          # entity → entity semantic relation
    PARTICIPATES = "participates"  # entity → event participation
    SOURCE = "source"              # entity/event → memory provenance
    TEMPORAL = "temporal"          # event → event temporal ordering
    CAUSAL = "causal"              # event → event causal link


class TemporalRelation(str, Enum):
    """Temporal relation subtypes."""

    BEFORE = "before"
    AFTER = "after"
    SIMULTANEOUS = "simultaneous"


class CausalRelation(str, Enum):
    """Causal relation subtypes."""

    CAUSES = "causes"
    RESULTS_FROM = "results_from"


@dataclass
class Triple:
    """A single extracted triple (head, relation, tail)."""

    head: str
    relation: str
    tail: str


@dataclass
class EventEntity:
    """An event with its participating entities."""

    event: str
    entities: List[str]


@dataclass
class TripleExtractionResult:
    """Result of triple extraction from atomic facts."""

    entity_relations: List[Triple] = field(default_factory=list)
    event_entities: List[EventEntity] = field(default_factory=list)
    event_relations: List[Triple] = field(default_factory=list)


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    type: KGNodeType
    name: str
    attrs: Dict[str, Any] = field(default_factory=dict)
    # entity: entity_type, first_seen, last_seen
    # event: description, timestamp
    # memory: memory_id, memory_type, timestamp


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""

    source: str  # source node id
    target: str  # target node id
    type: KGEdgeType
    attrs: Dict[str, Any] = field(default_factory=dict)
    # relation: relation_text, timestamp
    # temporal: relation (before/after/simultaneous)
    # causal: relation (causes/results_from)
    # source: memory_type


@dataclass
class KGDocument:
    """Serializable knowledge graph document for MongoDB persistence."""

    scope_key: str  # "group:{group_id}" or "user:{user_id}"
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    edge_texts: List[str] = field(default_factory=list)  # "head relation tail"
    version: int = 0
    updated_at: Optional[datetime] = None


@dataclass
class KGConfig:
    """Configuration for knowledge graph features."""

    enabled: bool = False
    # Triple extraction
    extraction_batch_size: int = 16
    # Entity deduplication
    entity_merge_threshold: float = 0.95
    # Retrieval
    top_k_edges: int = 50
    ppr_damping: float = 0.01  # equivalent to alpha=0.99
    ppr_score_threshold: float = 0.001
