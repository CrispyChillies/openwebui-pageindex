import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, Index, String, Text, JSON, UniqueConstraint

from open_webui.internal.db import Base


class PageIndexDocument(Base):
    __tablename__ = "pageindex_document"

    id = Column(String, primary_key=True, unique=True)
    file_id = Column(Text, ForeignKey("file.id", ondelete="CASCADE"), nullable=False)
    knowledge_id = Column(Text, ForeignKey("knowledge.id", ondelete="SET NULL"), nullable=True)
    user_id = Column(Text, nullable=False)

    doc_title = Column(Text, nullable=True)
    doc_description = Column(Text, nullable=True)
    source_type = Column(Text, nullable=True)
    source_hash = Column(Text, nullable=True)
    status = Column(Text, nullable=False, default="pending")

    tree_json = Column(JSON, nullable=True)
    generated_at = Column(BigInteger, nullable=True)
    index_options = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)

    __table_args__ = (
        Index("idx_pageindex_document_file_id", "file_id"),
        Index("idx_pageindex_document_knowledge_id", "knowledge_id"),
        Index("idx_pageindex_document_user_id", "user_id"),
        Index("idx_pageindex_document_status", "status"),
        Index("idx_pageindex_document_source_hash", "source_hash"),
        Index("idx_pageindex_document_file_status", "file_id", "status"),
    )


class PageIndexNode(Base):
    __tablename__ = "pageindex_node"

    id = Column(String, primary_key=True, unique=True)
    document_id = Column(
        Text,
        ForeignKey("pageindex_document.id", ondelete="CASCADE"),
        nullable=False,
    )

    node_id = Column(Text, nullable=False)
    parent_node_id = Column(Text, nullable=True)
    depth = Column(BigInteger, nullable=False, default=0)

    title = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    start_index = Column(BigInteger, nullable=True)
    end_index = Column(BigInteger, nullable=True)
    has_children = Column(Boolean, nullable=False, default=False)

    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)

    __table_args__ = (
        UniqueConstraint("document_id", "node_id", name="uq_pageindex_node_document_node"),
        Index("idx_pageindex_node_document_id", "document_id"),
        Index("idx_pageindex_node_node_id", "node_id"),
        Index("idx_pageindex_node_parent_node_id", "parent_node_id"),
        Index("idx_pageindex_node_document_parent", "document_id", "parent_node_id"),
        Index("idx_pageindex_node_document_depth", "document_id", "depth"),
    )


class PageIndexDocumentModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    file_id: str
    knowledge_id: Optional[str] = None
    user_id: str

    doc_title: Optional[str] = None
    doc_description: Optional[str] = None
    source_type: Optional[str] = None
    source_hash: Optional[str] = None
    status: str = "pending"

    tree_json: Optional[dict | list] = None
    generated_at: Optional[int] = None
    index_options: Optional[dict] = None
    error_message: Optional[str] = None

    created_at: int
    updated_at: int


class PageIndexNodeModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    node_id: str
    parent_node_id: Optional[str] = None
    depth: int

    title: Optional[str] = None
    summary: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    has_children: bool = False

    created_at: int
    updated_at: int


class PageIndexDocumentUpsertForm(BaseModel):
    file_id: str
    knowledge_id: Optional[str] = None
    user_id: str

    doc_title: Optional[str] = None
    doc_description: Optional[str] = None
    source_type: Optional[str] = None
    source_hash: Optional[str] = None
    status: str = "pending"

    tree_json: Optional[dict | list] = None
    generated_at: Optional[int] = None
    index_options: Optional[dict] = None
    error_message: Optional[str] = None


class PageIndexIndexResult(BaseModel):
    indexed: bool
    skipped: bool = False
    status: str
    message: Optional[str] = None
    document_id: Optional[str] = None
    file_id: str


class PageIndexStatusResponse(BaseModel):
    file_id: str
    document_id: Optional[str] = None
    status: str
    source_hash: Optional[str] = None
    error_message: Optional[str] = None
    updated_at: Optional[int] = None


class PageIndexCandidateDocument(BaseModel):
    document_id: str
    file_id: str
    knowledge_id: Optional[str] = None
    user_id: str
    doc_title: Optional[str] = None
    doc_description: Optional[str] = None
    file_name: Optional[str] = None
    score: float
    matched_nodes: int = 0
    # Additive semantic fields — None when using SQL token search fallback.
    semantic_score: Optional[float] = None
    metadata_score: Optional[float] = None
    exact_token_overlap_score: Optional[float] = None
    evidence_coverage_score: Optional[float] = None
    filename_score: Optional[float] = None
    matched_chunks: Optional[int] = None
    matched_chunk_refs: Optional[list[dict[str, Any]]] = None
    score_breakdown: Optional[dict[str, float]] = None
    query_variants_used: Optional[list[str]] = None
    confidence: Optional[str] = None


class PageIndexCandidateSearchResponse(BaseModel):
    query: str
    items: list[PageIndexCandidateDocument]
    query_variants_used: Optional[list[str]] = None
    matched_documents: Optional[int] = None
    matched_chunks: Optional[int] = None
    total_chunks: Optional[int] = None
    score_breakdown: Optional[dict[str, dict[str, float]]] = None
    confidence: Optional[str] = None


def new_pageindex_document_id() -> str:
    return str(uuid.uuid4())


def now_epoch() -> int:
    return int(time.time())
