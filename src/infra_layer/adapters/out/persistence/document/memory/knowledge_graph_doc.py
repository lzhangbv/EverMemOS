"""
Knowledge Graph MongoDB Document Model

Stores serialized knowledge graph per user/group scope.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from beanie import PydanticObjectId
from core.oxm.mongo.audit_base import AuditBase
from core.oxm.mongo.document_base import DocumentBase
from pydantic import ConfigDict, Field
from pymongo import ASCENDING, IndexModel


class KnowledgeGraphDoc(DocumentBase, AuditBase):
    """
    Knowledge graph document — one per user/group scope.

    Stores graph nodes, edges, and precomputed edge embeddings
    for fast query2edge retrieval.
    """

    scope_key: str = Field(
        ..., description='Tenant scope: "group:{id}" or "user:{id}"'
    )
    nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Graph nodes [{id, type, name, attrs}]",
    )
    edges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Graph edges [{source, target, type, attrs}]",
    )
    edge_texts: List[str] = Field(
        default_factory=list,
        description='Edge text representations: "head relation tail"',
    )
    # Binary fields stored as base64 strings for MongoDB compatibility
    node_embeddings_b64: Optional[str] = Field(
        default=None, description="Base64-encoded numpy array of node embeddings"
    )
    edge_embeddings_b64: Optional[str] = Field(
        default=None, description="Base64-encoded numpy array of edge embeddings"
    )
    version: int = Field(default=0, description="Optimistic lock version")

    model_config = ConfigDict(
        collection="knowledge_graphs",
        validate_assignment=True,
    )

    class Settings:
        name = "knowledge_graphs"
        indexes = [
            IndexModel([("scope_key", ASCENDING)], unique=True),
        ]
