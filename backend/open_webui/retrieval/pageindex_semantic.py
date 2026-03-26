"""
pageindex_semantic.py
---------------------
Semantic chunk service for PageIndex document selection.

Responsibilities
----------------
- Build semantic chunks from raw source text (PDF pages via PyPDF2/PyMuPDF
  or Markdown section text) aligned to PageIndex tree nodes.
- Embed chunks using the repo's existing embedding stack (EMBEDDING_FUNCTION).
- Upsert chunk vectors via VECTOR_DB_CLIENT into a single deterministic collection.
- Delete all previous chunk vectors for a document before re-indexing (no stale hits).
- Search chunks for a query and aggregate chunk-level hits into ranked document scores.

Collection name
---------------
``pageindex_selector_chunks``

Milvus stores it as ``open_webui_pageindex_selector_chunks``.

Chunk ID scheme
---------------
``pageindex:{document_id}:{chunk_index}``

Document score formula
----------------------
``DocScore = sum(chunk_scores) / sqrt(total_chunks_for_document + 1)``

where ``total_chunks_for_document`` is the total number of chunks indexed for
that document (denominator is NOT the number of matched chunks).

ChunkScore normalisation
------------------------
Milvus already normalises cosine distances to [0, 1] in
``_result_to_search_result`` (``(raw + 1) / 2``), so we use the distance
value directly as a similarity score (higher = more similar).  Other backends
(Chroma, Qdrant, …) also emit distances in roughly [0, 1] when cosine metric
is used, so no additional normalisation is applied here.  If a backend emits
distances outside [0, 1], ``max(0, min(1, d))`` clipping is applied for safety.

Chunking strategy
-----------------
Primary (preferred): **Raw source text per tree node**.

  For PDF documents, ``page_list`` is a list of ``(page_text, token_count)``
  tuples produced by the pageindex library's ``get_page_tokens()``
  (PyPDF2/PyMuPDF).  Each tree node carries ``start_index`` and ``end_index``
  (1-based PDF page numbers).  We concatenate the raw page text for each
  node's page range and slide a character window over it.

  For Markdown documents, each node already carries a ``text`` field with
  the raw markdown section text extracted during indexing.  We use that
  directly.

  A doc-level header chunk (``doc_name`` + ``doc_description``) is
  prepended so the document title/description is always embedded.

Fallback: **Summary-only** (tree walk, no raw pages).
  Used when ``page_list`` is not available at index time or when the file
  format is not PDF/Markdown.

Chunking parameters
-------------------
Character-based sliding window.  ~4 chars/token.

+-----------------------------+-------+--------+
| Parameter                   | Chars | Tokens |
+=============================+=======+========+
| Chunk size                  | 2000  | ~500   |
| Overlap                     |  400  | ~100   |
+-----------------------------+-------+--------+
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Callable, Optional

from open_webui.models.pageindex import (
    PageIndexCandidateDocument,
    PageIndexCandidateSearchResponse,
)
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.retrieval.vector.main import VectorItem

log = logging.getLogger(__name__)

# ── Collection name ──────────────────────────────────────────────────────────
COLLECTION_NAME = "pageindex_selector_chunks"

# ── Chunking constants ───────────────────────────────────────────────────────
# Character-based approximation: ~4 chars/token.
# Target 500 tokens → 2000 chars; overlap 100 tokens → 400 chars.
CHUNK_SIZE_CHARS: int = 2000
CHUNK_OVERLAP_CHARS: int = 400

# ── Overfetch multiplier for search ─────────────────────────────────────────
# We fetch (limit * 20) chunk hits so the per-document aggregation has
# enough signal even when documents have many chunks.
OVERFETCH_MULTIPLIER: int = 20
OVERFETCH_MIN: int = 50


# ---------------------------------------------------------------------------
# Embedding function registry
# ---------------------------------------------------------------------------
# The embedding function lives on ``request.app.state.EMBEDDING_FUNCTION``
# which is not accessible at module import time.  Callers (router, service)
# must call ``set_embedding_function(fn)`` before indexing / searching.
# A thread-safe module-level callable holder is used.

_embedding_fn: Optional[Callable] = None


def set_embedding_function(fn: Callable) -> None:
    """Register the application-wide embedding function for PageIndex semantics."""
    global _embedding_fn
    _embedding_fn = fn


def get_embedding_function() -> Optional[Callable]:
    """Return the registered embedding function, or None if not yet set."""
    return _embedding_fn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Return a deterministic, namespace-prefixed chunk ID."""
    return f"pageindex:{document_id}:{chunk_index}"


def _clip_score(score: float) -> float:
    """Clip a similarity score to [0, 1] for safety across different backends."""
    return max(0.0, min(1.0, float(score)))


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _extract_text_from_tree(tree_data: dict) -> list[dict]:
    """
    Walk the PageIndex tree and collect (node_id, section_title, text) triples.

    Returns a list of dicts::

        {
            "node_id": str | None,
            "section_title": str | None,
            "page_number": int | None,
            "text": str,
        }

    Only nodes that have a non-empty title or summary are included.
    """
    segments: list[dict] = []

    # Document-level description
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

    structure = tree_data.get("structure") or []
    _walk(structure)
    return segments


def _split_into_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split *text* into overlapping character-windowed chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start += size - overlap
    return chunks


def _get_page_text(
    page_list: list[tuple[str, int]],
    start_index: Optional[int],
    end_index: Optional[int],
) -> str:
    """
    Concatenate raw page text for the page range [start_index, end_index]
    from *page_list* (1-based, inclusive on both ends).

    *page_list* is the list of ``(page_text, token_count)`` tuples returned
    by ``get_page_tokens()`` from the pageindex library.

    Returns an empty string if the indices are missing or out of range.
    """
    if start_index is None or end_index is None:
        return ""
    # page_list is 0-indexed; page numbers are 1-based
    lo = max(0, start_index - 1)
    hi = min(len(page_list), end_index)  # end_index is inclusive → slice to end_index
    if lo >= hi:
        return ""
    return "".join(entry[0] for entry in page_list[lo:hi])


def _walk_tree_nodes(nodes: list | dict) -> list[dict]:
    """
    Flatten all nodes in a PageIndex tree (depth-first) into a list.

    Handles both the PDF tree format (nodes have ``title``, ``start_index``,
    ``end_index``, and a ``nodes`` child list) and the Markdown tree format
    (nodes similarly structured but also carry a ``text`` field).
    """
    result: list[dict] = []
    items = nodes if isinstance(nodes, list) else [nodes]
    for node in items:
        result.append(node)
        children = node.get("nodes") or node.get("children") or []
        if children:
            result.extend(_walk_tree_nodes(children))
    return result


def build_chunks_from_raw_pages(
    document_id: str,
    file_id: str,
    knowledge_id: Optional[str],
    user_id: str,
    tree_data: dict,
    source_type: str,
    page_list: Optional[list[tuple[str, int]]] = None,
) -> list[dict]:
    """
    Build chunks using **raw source text** aligned to PageIndex tree nodes.

    Strategy
    --------
    **PDF** (``source_type == "pdf"``):
      Requires *page_list* — a list of ``(page_text, token_count)`` tuples
      from ``get_page_tokens()``.  For each node, the pages in
      ``[node.start_index, node.end_index]`` are concatenated and chunked.

    **Markdown** (``source_type == "markdown"``):
      Each node carries its raw markdown section ``text`` field (populated
      by the md indexer).  No *page_list* required.

    In both cases a doc-level header chunk is prepended from
    ``doc_name + doc_description``.

    Raises
    ------
    ``ValueError`` if source_type is ``"pdf"`` but ``page_list`` is empty/None.
    """
    if source_type == "pdf" and not page_list:
        raise ValueError(
            "page_list is required for PDF raw-text chunking but was not provided"
        )

    chunks: list[dict] = []
    chunk_index = 0

    def _emit_chunks(
        text: str,
        node_id: Optional[str],
        section_title: Optional[str],
        page_number: Optional[int],
    ) -> None:
        nonlocal chunk_index
        sub_texts = _split_into_chunks(text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
        for sub_text in sub_texts:
            chunks.append(
                {
                    "id": _make_chunk_id(document_id, chunk_index),
                    "text": sub_text,
                    "metadata": {
                        "document_id": document_id,
                        "file_id": file_id,
                        "knowledge_id": knowledge_id,
                        "user_id": user_id,
                        "chunk_index": chunk_index,
                        "total_chunks_for_document": None,  # filled later
                        "page_number": page_number,
                        "section_title": section_title,
                        "node_id": node_id,
                        "source_type": source_type,
                    },
                }
            )
            chunk_index += 1

    # ── Header chunk: doc title + description ─────────────────────────────
    doc_name = tree_data.get("doc_name") or ""
    doc_description = tree_data.get("doc_description") or ""
    header_text = " ".join(filter(None, [doc_name, doc_description])).strip()
    if header_text:
        _emit_chunks(header_text, node_id=None, section_title=doc_name or None, page_number=None)

    # ── Per-node chunks ───────────────────────────────────────────────────
    structure = tree_data.get("structure") or []
    all_nodes = _walk_tree_nodes(structure)

    for node in all_nodes:
        node_id = node.get("node_id") or node.get("id")
        title = node.get("title") or node.get("node_title") or ""
        start_idx = node.get("start_index")
        end_idx = node.get("end_index")
        page_number = int(start_idx) if start_idx is not None else None

        if source_type == "pdf":
            raw_text = _get_page_text(
                page_list,  # type: ignore[arg-type]
                start_idx,
                end_idx,
            )
        else:
            # Markdown: use the node's text field
            raw_text = node.get("text") or ""

        if not raw_text.strip():
            log.debug(
                f"pageindex_semantic: node '{title}' (id={node_id}) has no raw text; skipping"
            )
            continue

        # Prepend section title to the raw text to anchor user queries on
        # section names (title mention boosts retrieval accuracy).
        titled_text = f"{title}\n{raw_text}" if title else raw_text
        _emit_chunks(
            titled_text,
            node_id=node_id,
            section_title=title or None,
            page_number=page_number,
        )

    log.info(
        f"pageindex_semantic: built {len(chunks)} raw-text chunks "
        f"from {len(all_nodes)} nodes for document_id={document_id} "
        f"(source_type={source_type})"
    )
    return chunks


def build_chunks_for_document(
    document_id: str,
    file_id: str,
    knowledge_id: Optional[str],
    user_id: str,
    tree_data: dict,
    source_type: Optional[str],
    page_list: Optional[list[tuple[str, int]]] = None,
) -> list[dict]:
    """
    Top-level dispatcher: choose the best chunking strategy and build chunks.

    Priority
    --------
    1. **Raw-text chunking** via :func:`build_chunks_from_raw_pages` when:
       - ``source_type == "pdf"`` and ``page_list`` is provided.
       - ``source_type == "markdown"`` (page_list not needed; node.text used).
    2. **Summary-only fallback** via :func:`build_chunks_from_tree` otherwise.
    """
    use_raw = False
    if source_type == "pdf" and page_list:
        use_raw = True
        log.info(
            f"pageindex_semantic: using raw PDF page text for chunking "
            f"({len(page_list)} pages, document_id={document_id})"
        )
    elif source_type in ("markdown", "md"):
        use_raw = True
        log.info(
            f"pageindex_semantic: using Markdown node text for chunking "
            f"(document_id={document_id})"
        )
    else:
        log.info(
            f"pageindex_semantic: falling back to summary-only chunking "
            f"(source_type={source_type}, page_list={'yes' if page_list else 'no'}, "
            f"document_id={document_id})"
        )

    if use_raw:
        try:
            return build_chunks_from_raw_pages(
                document_id=document_id,
                file_id=file_id,
                knowledge_id=knowledge_id,
                user_id=user_id,
                tree_data=tree_data,
                source_type=source_type,
                page_list=page_list,
            )
        except Exception as e:
            log.warning(
                f"pageindex_semantic: raw-text chunking failed ({e}); "
                "falling back to summary-only"
            )

    # Summary-only fallback
    return build_chunks_from_tree(
        document_id=document_id,
        file_id=file_id,
        knowledge_id=knowledge_id,
        user_id=user_id,
        tree_data=tree_data,
        source_type=source_type,
    )



def build_chunks_from_tree(
    document_id: str,
    file_id: str,
    knowledge_id: Optional[str],
    user_id: str,
    tree_data: dict,
    source_type: Optional[str] = None,
) -> list[dict]:
    """
    Build a list of chunk dicts ready for embedding and upsert.

    Each chunk dict contains:
    - ``id``: deterministic chunk ID
    - ``text``: chunk text for embedding
    - ``metadata``: full metadata payload

    The ``total_chunks_for_document`` field in metadata is filled in after all
    chunks are built (via :func:`annotate_total_chunks`).
    """
    segments = _extract_text_from_tree(tree_data)
    log.info(
        f"pageindex_semantic: extracted {len(segments)} segments from tree "
        f"for document_id={document_id}"
    )

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
                        "document_id": document_id,
                        "file_id": file_id,
                        "knowledge_id": knowledge_id,
                        "user_id": user_id,
                        "chunk_index": chunk_index,
                        "total_chunks_for_document": None,  # filled by annotate_total_chunks
                        "page_number": seg["page_number"],
                        "section_title": seg["section_title"],
                        "node_id": seg["node_id"],
                        "source_type": source_type,
                    },
                }
            )
            chunk_index += 1

    log.info(
        f"pageindex_semantic: built {len(chunks)} chunks "
        f"(from {len(segments)} segments) for document_id={document_id}"
    )
    return chunks


def annotate_total_chunks(chunks: list[dict]) -> list[dict]:
    """Fill the ``total_chunks_for_document`` field in each chunk's metadata."""
    total = len(chunks)
    for chunk in chunks:
        chunk["metadata"]["total_chunks_for_document"] = total
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


async def embed_chunks(
    chunks: list[dict],
    embedding_fn: Callable,
) -> list[dict]:
    """
    Embed all chunk texts using *embedding_fn* and attach the result as
    ``"vector"`` in each chunk dict.  Returns the same list with ``"vector"``
    populated.

    Uses content prefix (None) — consistent with how the rest of the repo
    embeds document content, not queries.
    """
    texts = [c["text"] for c in chunks]
    batch_count = max(1, math.ceil(len(texts) / 32))
    log.info(
        f"pageindex_semantic: embedding {len(texts)} chunks "
        f"(approx {batch_count} batches) …"
    )

    # The embedding function accepts a list and returns list[list[float]].
    # Pass prefix=None (content, not query).
    try:
        vectors = await embedding_fn(texts, prefix=None)
    except TypeError:
        # Some callers wrap the function so it accepts positional only.
        vectors = await embedding_fn(texts)

    if not vectors or len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding function returned {len(vectors) if vectors else 0} vectors "
            f"for {len(chunks)} chunks"
        )

    log.info(f"pageindex_semantic: received {len(vectors)} embedding vectors")
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector
    return chunks


# ---------------------------------------------------------------------------
# Vector DB operations
# ---------------------------------------------------------------------------


def delete_document_chunks(document_id: str) -> None:
    """
    Delete all previously indexed chunks for *document_id* so reindexing
    does not leave stale vectors.

    Uses the ``delete(filter={"document_id": document_id})`` path.
    Falls back gracefully if the collection does not yet exist.
    """
    try:
        if not VECTOR_DB_CLIENT.has_collection(COLLECTION_NAME):
            log.debug(
                f"pageindex_semantic: collection '{COLLECTION_NAME}' does not exist; "
                "skipping delete"
            )
            return
        log.info(
            f"pageindex_semantic: deleting stale chunks for document_id={document_id}"
        )
        VECTOR_DB_CLIENT.delete(
            collection_name=COLLECTION_NAME,
            filter={"document_id": document_id},
        )
    except Exception as e:
        log.warning(
            f"pageindex_semantic: failed to delete stale chunks for "
            f"document_id={document_id}: {e}"
        )


def upsert_chunks(chunks: list[dict]) -> None:
    """
    Upsert *chunks* (each must have ``id``, ``text``, ``vector``, ``metadata``)
    into the selector collection.
    """
    if not chunks:
        log.warning("pageindex_semantic: upsert_chunks called with empty list; skipping")
        return

    items: list[VectorItem] = [
        VectorItem(
            id=c["id"],
            text=c["text"],
            vector=c["vector"],
            metadata=c["metadata"],
        )
        for c in chunks
    ]

    log.info(
        f"pageindex_semantic: upserting {len(items)} chunk vectors into '{COLLECTION_NAME}'"
    )
    VECTOR_DB_CLIENT.upsert(collection_name=COLLECTION_NAME, items=items)
    log.info(
        f"pageindex_semantic: upsert complete — {len(items)} chunks stored"
    )


# ---------------------------------------------------------------------------
# Indexing entry point
# ---------------------------------------------------------------------------


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
    """
    Full semantic indexing flow for one document.

    1. Build chunks (raw-text preferred, summary fallback).
    2. Annotate total chunk count.
    3. Embed chunks.
    4. Delete stale vectors for this document.
    5. Upsert new vectors.

    Parameters
    ----------
    page_list:
        For PDF documents: the list of ``(page_text, token_count)`` tuples
        produced by ``get_page_tokens(pdf_path)`` from the pageindex library.
        When provided, raw page text aligned to tree-node page ranges is used
        for chunking instead of node summaries.  For Markdown documents this
        is not required (node ``text`` fields are used instead).

    Returns the number of chunks indexed.
    Raises on failure so the caller can mark the document as ``failed``.
    """
    fn = embedding_fn or get_embedding_function()
    if fn is None:
        raise RuntimeError(
            "No embedding function registered. "
            "Call pageindex_semantic.set_embedding_function() at app startup."
        )

    # 1. Build chunks — raw text preferred, summary fallback
    chunks = build_chunks_for_document(
        document_id=document_id,
        file_id=file_id,
        knowledge_id=knowledge_id,
        user_id=user_id,
        tree_data=tree_data,
        source_type=source_type,
        page_list=page_list,
    )
    if not chunks:
        log.warning(
            f"pageindex_semantic: no chunks produced for document_id={document_id}; "
            "nothing will be indexed"
        )
        return 0

    # 2. Annotate total
    chunks = annotate_total_chunks(chunks)

    # 3. Embed
    chunks = await embed_chunks(chunks, fn)

    # 4. Delete stale
    delete_document_chunks(document_id)

    # 5. Upsert
    upsert_chunks(chunks)

    return len(chunks)


# ---------------------------------------------------------------------------
# Search / candidate retrieval
# ---------------------------------------------------------------------------


def _compute_doc_score(
    matched_chunk_scores: list[float],
    total_chunks_for_document: int,
) -> float:
    """
    Compute per-document relevance score.

    Formula::

        DocScore = sum(chunk_scores) / sqrt(total_chunks_for_document + 1)

    ``total_chunks_for_document`` is the total indexed chunks for the document,
    NOT the number of matched chunks.  This penalises very large documents that
    happen to match a few chunks while favouring focused matches.
    """
    if not matched_chunk_scores:
        return 0.0
    return sum(matched_chunk_scores) / math.sqrt(total_chunks_for_document + 1)


async def search_candidate_documents(
    query: str,
    user_id: Optional[str],
    knowledge_id: Optional[str] = None,
    file_ids: Optional[list[str]] = None,
    limit: int = 10,
    embedding_fn: Optional[Callable] = None,
) -> PageIndexCandidateSearchResponse:
    """
    Semantic candidate document search.

    Steps:
    1. Embed query.
    2. Overfetch chunk hits from the vector DB.
    3. Post-filter by metadata (user_id, knowledge_id, file_ids) when the
       vector DB does not natively support search-time filter expressions,
       or as a safeguard.
    4. Aggregate chunk hits per document, compute DocScore.
    5. Return top *limit* ranked documents.

    Falls back to an empty result set on any unrecoverable error so the
    calling code can use its own fallback path.
    """
    fn = embedding_fn or get_embedding_function()
    if fn is None:
        log.error(
            "pageindex_semantic: no embedding function registered; "
            "falling back to empty candidate list"
        )
        return PageIndexCandidateSearchResponse(query=query, items=[])

    if not VECTOR_DB_CLIENT.has_collection(COLLECTION_NAME):
        log.info(
            f"pageindex_semantic: collection '{COLLECTION_NAME}' does not exist; "
            "returning empty candidates"
        )
        return PageIndexCandidateSearchResponse(query=query, items=[])

    # 1. Embed query (query prefix)
    try:
        from open_webui.config import RAG_EMBEDDING_QUERY_PREFIX  # noqa: PLC0415

        query_prefix = RAG_EMBEDDING_QUERY_PREFIX
    except ImportError:
        query_prefix = None

    try:
        query_vector = await fn(query, prefix=query_prefix)
    except TypeError:
        query_vector = await fn(query)
    except Exception as e:
        log.exception(f"pageindex_semantic: failed to embed query: {e}")
        return PageIndexCandidateSearchResponse(query=query, items=[])

    # 2. Search chunk vectors — overfetch
    search_limit = max(limit * OVERFETCH_MULTIPLIER, OVERFETCH_MIN)

    # Build a native filter dict for server-side filtering where supported.
    # The VectorDBBase.search() signature accepts ``filter: Optional[dict]``
    # but most backends (Milvus, Qdrant) interpret it differently.
    # For Milvus the existing ``search`` implementation does NOT pass the
    # filter argument to the Milvus client (as of the current codebase).
    # We therefore perform server-side filtering via the ``query`` method
    # on matching IDs, and apply post-filtering always for safety.
    try:
        search_result = VECTOR_DB_CLIENT.search(
            collection_name=COLLECTION_NAME,
            vectors=[query_vector],
            limit=search_limit,
        )
    except Exception as e:
        log.exception(f"pageindex_semantic: vector search failed: {e}")
        return PageIndexCandidateSearchResponse(query=query, items=[])

    if not search_result or not search_result.ids or not search_result.ids[0]:
        log.info("pageindex_semantic: no chunk hits returned by vector search")
        return PageIndexCandidateSearchResponse(query=query, items=[])

    ids = search_result.ids[0]
    distances = search_result.distances[0] if search_result.distances else []
    metadatas = search_result.metadatas[0] if search_result.metadatas else []

    log.info(f"pageindex_semantic: search returned {len(ids)} raw chunk hits")

    # 3. Post-filter by security/scope metadata
    allowed_file_ids: Optional[set] = set(file_ids) if file_ids else None

    # Aggregate per document_id
    # doc_hits: document_id → { "scores": [...], "total": int, "doc_meta": {...} }
    doc_hits: dict[str, dict] = {}

    for i, chunk_meta in enumerate(metadatas):
        if chunk_meta is None:
            continue

        chunk_user_id = chunk_meta.get("user_id")
        chunk_knowledge_id = chunk_meta.get("knowledge_id")
        chunk_file_id = chunk_meta.get("file_id")
        chunk_document_id = chunk_meta.get("document_id")
        total_chunks = chunk_meta.get("total_chunks_for_document")

        # Security: user boundary
        if user_id and chunk_user_id != user_id:
            continue

        # Scope filter: knowledge_id
        if knowledge_id and chunk_knowledge_id != knowledge_id:
            continue

        # Scope filter: file_ids
        if allowed_file_ids and chunk_file_id not in allowed_file_ids:
            continue

        if not chunk_document_id:
            continue

        # Get similarity score (already in [0,1] from Milvus normalisation)
        score = _clip_score(distances[i]) if i < len(distances) else 0.0

        if chunk_document_id not in doc_hits:
            doc_hits[chunk_document_id] = {
                "scores": [],
                "total_chunks": int(total_chunks) if total_chunks else 1,
                "file_id": chunk_file_id,
                "knowledge_id": chunk_knowledge_id,
                "user_id": chunk_user_id,
            }

        doc_hits[chunk_document_id]["scores"].append(score)
        # Use max observed total_chunks (should be consistent across chunks)
        if total_chunks:
            doc_hits[chunk_document_id]["total_chunks"] = max(
                doc_hits[chunk_document_id]["total_chunks"],
                int(total_chunks),
            )

    log.info(
        f"pageindex_semantic: {len(doc_hits)} documents after filtering "
        f"(user_id={user_id}, knowledge_id={knowledge_id}, "
        f"file_ids={'yes' if allowed_file_ids else 'no'})"
    )

    if not doc_hits:
        return PageIndexCandidateSearchResponse(query=query, items=[])

    # 4. Compute document scores and fetch doc metadata
    from open_webui.storage.pageindex import PageIndexes  # noqa: PLC0415 (avoid circular at module level)

    ranked_items: list[PageIndexCandidateDocument] = []
    for document_id, data in doc_hits.items():
        doc_score = _compute_doc_score(
            matched_chunk_scores=data["scores"],
            total_chunks_for_document=data["total_chunks"],
        )

        # Fetch document metadata from SQL store to populate title/description
        doc_model = PageIndexes.get_document_by_id(document_id)
        if not doc_model or doc_model.status != "ready":
            continue

        ranked_items.append(
            PageIndexCandidateDocument(
                document_id=document_id,
                file_id=data["file_id"] or doc_model.file_id,
                knowledge_id=data["knowledge_id"] or doc_model.knowledge_id,
                user_id=data["user_id"] or doc_model.user_id,
                doc_title=doc_model.doc_title,
                doc_description=doc_model.doc_description,
                score=round(doc_score, 6),
                matched_nodes=len(data["scores"]),
                semantic_score=round(doc_score, 6),
                matched_chunks=len(data["scores"]),
            )
        )

    # 5. Sort descending by score and return top limit
    ranked_items.sort(key=lambda x: x.score, reverse=True)

    log.info(
        f"pageindex_semantic: returning {min(limit, len(ranked_items))} "
        f"candidate documents (total scored: {len(ranked_items)})"
    )
    return PageIndexCandidateSearchResponse(query=query, items=ranked_items[:limit])
