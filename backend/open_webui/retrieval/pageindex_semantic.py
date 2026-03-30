from __future__ import annotations

import math
from typing import Callable, Optional

from open_webui.models.pageindex import PageIndexCandidateSearchResponse
from open_webui.retrieval.pageindex_selector.service import (
    CHUNK_COLLECTION_NAME as COLLECTION_NAME,
)
from open_webui.retrieval.pageindex_selector.service import PageIndexSelectorService

# Backward-compatible constants
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 400
OVERFETCH_MULTIPLIER = 20
OVERFETCH_MIN = 60


def set_embedding_function(fn: Callable) -> None:
    PageIndexSelectorService.set_embedding_function(fn)


def get_embedding_function() -> Optional[Callable]:
    return PageIndexSelectorService.get_embedding_function()


def _make_chunk_id(document_id: str, chunk_index: int) -> str:
    return PageIndexSelectorService._make_chunk_id(document_id, chunk_index)


def _clip_score(score: float) -> float:
    return PageIndexSelectorService._clip_score(score)


def _split_into_chunks(text: str, size: int, overlap: int) -> list[str]:
    return PageIndexSelectorService._split_into_chunks(text, size, overlap)


def _compute_doc_score(matched_chunk_scores: list[float], total_chunks_for_document: int) -> float:
    """Compatibility helper retained for legacy tests and callers."""
    if not matched_chunk_scores:
        return 0.0
    return sum(matched_chunk_scores) / math.sqrt(total_chunks_for_document + 1)


def _extract_text_from_tree(tree_data: dict) -> list[dict]:
    segments: list[dict] = []
    doc_description = tree_data.get("doc_description") or ""
    doc_name = tree_data.get("doc_name") or ""
    if doc_description or doc_name:
        combined = " ".join(filter(None, [doc_name, doc_description]))
        if combined.strip():
            segments.append(
                {
                    "node_id": None,
                    "section_title": doc_name or None,
                    "page_number": None,
                    "text": combined,
                }
            )

    def _walk(nodes: list[dict]) -> None:
        for node in nodes or []:
            node_id = node.get("node_id") or node.get("id")
            title = node.get("title") or node.get("node_title") or ""
            summary = node.get("summary") or node.get("node_summary") or ""
            start = node.get("start_index")
            text_parts = [p for p in [title, summary] if p and p.strip()]
            if text_parts:
                segments.append(
                    {
                        "node_id": node_id,
                        "section_title": title or None,
                        "page_number": int(start) if start is not None else None,
                        "text": " ".join(text_parts),
                    }
                )
            children = node.get("children") or node.get("nodes") or []
            if children:
                _walk(children)

    _walk(tree_data.get("structure") or [])
    return segments


def build_chunks_from_tree(
    document_id: str,
    file_id: str,
    knowledge_id: Optional[str],
    user_id: str,
    tree_data: dict,
    source_type: Optional[str] = None,
) -> list[dict]:
    segments = _extract_text_from_tree(tree_data)
    chunks: list[dict] = []
    chunk_index = 0
    for seg in segments:
        sub_texts = _split_into_chunks(seg["text"], CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
        for sub_text in sub_texts:
            chunk_id = _make_chunk_id(document_id, chunk_index)
            chunks.append(
                {
                    "id": chunk_id,
                    "text": sub_text,
                    "metadata": {
                        "record_type": "chunk",
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "file_id": file_id,
                        "knowledge_id": knowledge_id,
                        "user_id": user_id,
                        "chunk_index": chunk_index,
                        "total_chunks_for_document": None,
                        "page_number": seg["page_number"],
                        "section_title": seg["section_title"],
                        "node_id": seg["node_id"],
                        "source_type": source_type,
                    },
                }
            )
            chunk_index += 1
    return chunks


def annotate_total_chunks(chunks: list[dict]) -> list[dict]:
    total = len(chunks)
    for chunk in chunks:
        chunk["metadata"]["total_chunks_for_document"] = total
    return chunks


async def embed_chunks(chunks: list[dict], embedding_fn: Callable) -> list[dict]:
    texts = [c["text"] for c in chunks]
    try:
        vectors = await embedding_fn(texts, prefix=None)
    except TypeError:
        vectors = await embedding_fn(texts)
    if not vectors or len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding function returned {len(vectors) if vectors else 0} vectors for {len(chunks)} chunks"
        )
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector
    return chunks


def delete_document_chunks(document_id: str) -> None:
    PageIndexSelectorService.delete_selector_document(document_id)


def upsert_chunks(chunks: list[dict]) -> None:
    # Compatibility no-op; ingestion should go through index_document_chunks.
    if not chunks:
        return
    raise RuntimeError("upsert_chunks is deprecated; use index_document_chunks")


async def index_document_chunks(
    document_id: str,
    file_id: str,
    knowledge_id: Optional[str],
    user_id: str,
    tree_data: dict,
    source_type: Optional[str],
    embedding_fn: Optional[Callable] = None,
    page_list: Optional[list[tuple[str, int]]] = None,
) -> int:
    return await PageIndexSelectorService.index_document_chunks(
        document_id=document_id,
        file_id=file_id,
        knowledge_id=knowledge_id,
        user_id=user_id,
        tree_data=tree_data,
        source_type=source_type,
        embedding_fn=embedding_fn,
        page_list=page_list,
    )


async def search_candidate_documents(
    query: str,
    user_id: Optional[str],
    knowledge_id: Optional[str] = None,
    file_ids: Optional[list[str]] = None,
    limit: int = 10,
    embedding_fn: Optional[Callable] = None,
) -> PageIndexCandidateSearchResponse:
    return await PageIndexSelectorService.search_candidate_documents(
        query=query,
        user_id=user_id,
        knowledge_id=knowledge_id,
        file_ids=file_ids,
        limit=limit,
        embedding_fn=embedding_fn,
    )
