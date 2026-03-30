from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional

from open_webui.config import RAG_EMBEDDING_QUERY_PREFIX
from open_webui.models.files import Files
from open_webui.models.pageindex import (
    PageIndexCandidateDocument,
    PageIndexCandidateSearchResponse,
)
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.retrieval.vector.main import VectorItem
from open_webui.storage.pageindex import PageIndexes

from .query_expansion import (
    build_selector_query_variants,
    extract_query_tokens,
    extract_rare_tokens,
)
from .scoring import (
    clip01,
    combine_document_score,
    confidence_from_scores,
    evidence_coverage_score,
    filename_relevance_score,
    lexical_overlap_score,
    metadata_score,
    rare_token_overlap_score,
    rerank_adjustment,
    semantic_chunk_score,
)

log = logging.getLogger(__name__)

CHUNK_COLLECTION_NAME = "pageindex_selector_chunks"
HEADER_COLLECTION_NAME = "pageindex_selector_doc_headers"

CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 400

OVERFETCH_MULTIPLIER = 20
OVERFETCH_MIN = 60


@dataclass
class ChunkHit:
    chunk_id: str
    score: float
    document_id: str
    file_id: str
    page_number: Optional[int]
    section_title: Optional[str]
    node_id: Optional[str]
    text: str
    query_variant: str
    record_type: str


class PageIndexSelectorService:
    _embedding_fn: Optional[Callable] = None

    @classmethod
    def set_embedding_function(cls, fn: Callable) -> None:
        cls._embedding_fn = fn

    @classmethod
    def get_embedding_function(cls) -> Optional[Callable]:
        return cls._embedding_fn

    @staticmethod
    def _clip_score(value: float) -> float:
        return clip01(value)

    @staticmethod
    def _make_chunk_id(document_id: str, chunk_index: int) -> str:
        return f"pageindex:{document_id}:{chunk_index}"

    @staticmethod
    def _make_header_id(document_id: str) -> str:
        return f"pageindex:header:{document_id}"

    @staticmethod
    def _split_into_chunks(text: str, size: int, overlap: int) -> list[str]:
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + size, length)
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= length:
                break
            start += max(1, size - overlap)
        return chunks

    @staticmethod
    def _walk_tree_nodes(nodes: list | dict) -> list[dict]:
        out: list[dict] = []
        items = nodes if isinstance(nodes, list) else [nodes]
        for node in items:
            out.append(node)
            children = node.get("nodes") or node.get("children") or []
            if children:
                out.extend(PageIndexSelectorService._walk_tree_nodes(children))
        return out

    @staticmethod
    def _get_page_text(
        page_list: list[tuple[str, int]],
        start_index: Optional[int],
        end_index: Optional[int],
    ) -> str:
        if start_index is None or end_index is None:
            return ""
        lo = max(0, start_index - 1)
        hi = min(len(page_list), end_index)
        if lo >= hi:
            return ""
        return "".join(entry[0] for entry in page_list[lo:hi])

    @classmethod
    def build_chunks_for_document(
        cls,
        document_id: str,
        file_id: str,
        knowledge_id: Optional[str],
        user_id: str,
        tree_data: dict,
        source_type: Optional[str],
        file_name: Optional[str],
        page_list: Optional[list[tuple[str, int]]] = None,
    ) -> tuple[list[dict], dict]:
        doc_title = tree_data.get("doc_name") or file_name or ""
        doc_description = tree_data.get("doc_description") or ""

        header_text = "\n".join(
            [
                f"Document title: {doc_title}",
                f"File name: {file_name or ''}",
                f"Document description: {doc_description}",
                f"Source type: {source_type or ''}",
            ]
        ).strip()

        header_item = {
            "id": cls._make_header_id(document_id),
            "text": header_text,
            "metadata": {
                "record_type": "doc_header",
                "document_id": document_id,
                "file_id": file_id,
                "file_name": file_name,
                "doc_title": doc_title,
                "doc_description": doc_description,
                "source_type": source_type,
                "knowledge_id": knowledge_id,
                "user_id": user_id,
                "node_id": None,
                "page_number": None,
                "section_title": doc_title or None,
            },
        }

        chunks: list[dict] = []
        chunk_index = 0
        nodes = cls._walk_tree_nodes(tree_data.get("structure") or [])

        for node in nodes:
            node_id = node.get("node_id") or node.get("id")
            section_title = node.get("title") or node.get("node_title") or ""
            page_number = node.get("start_index")
            raw_text = ""

            if source_type == "pdf" and page_list:
                raw_text = cls._get_page_text(
                    page_list,
                    node.get("start_index"),
                    node.get("end_index"),
                )
            elif source_type in ("md", "markdown"):
                raw_text = node.get("text") or ""
            if not raw_text:
                raw_text = " ".join(
                    x
                    for x in [
                        section_title,
                        node.get("summary") or node.get("node_summary") or "",
                    ]
                    if x
                )

            if not raw_text.strip():
                continue

            chunk_text = f"{section_title}\n{raw_text}" if section_title else raw_text
            for text_piece in cls._split_into_chunks(
                chunk_text,
                CHUNK_SIZE_CHARS,
                CHUNK_OVERLAP_CHARS,
            ):
                chunk_id = cls._make_chunk_id(document_id, chunk_index)
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": text_piece,
                        "metadata": {
                            "record_type": "chunk",
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "file_id": file_id,
                            "file_name": file_name,
                            "doc_title": doc_title,
                            "doc_description": doc_description,
                            "source_type": source_type,
                            "knowledge_id": knowledge_id,
                            "user_id": user_id,
                            "node_id": node_id,
                            "page_number": int(page_number)
                            if page_number is not None
                            else None,
                            "section_title": section_title or None,
                            "chunk_index": chunk_index,
                            "total_chunks_for_document": 0,
                        },
                    }
                )
                chunk_index += 1

        total = len(chunks)
        for chunk in chunks:
            chunk["metadata"]["total_chunks_for_document"] = total

        log.info(
            "selector chunks created document_id=%s source_type=%s total_chunks=%s",
            document_id,
            source_type,
            total,
        )

        return chunks, header_item

    @classmethod
    def build_chunks_from_source_text(
        cls,
        document_id: str,
        file_id: str,
        knowledge_id: Optional[str],
        user_id: str,
        source_type: Optional[str],
        file_name: Optional[str],
        doc_title: Optional[str] = None,
        doc_description: Optional[str] = None,
        page_list: Optional[list[tuple[str, int]]] = None,
        full_text: Optional[str] = None,
    ) -> tuple[list[dict], dict]:
        resolved_title = doc_title or file_name or ""
        resolved_description = doc_description or ""

        header_text = "\n".join(
            [
                f"Document title: {resolved_title}",
                f"File name: {file_name or ''}",
                f"Document description: {resolved_description}",
                f"Source type: {source_type or ''}",
            ]
        ).strip()

        header_item = {
            "id": cls._make_header_id(document_id),
            "text": header_text,
            "metadata": {
                "record_type": "doc_header",
                "document_id": document_id,
                "file_id": file_id,
                "file_name": file_name,
                "doc_title": resolved_title,
                "doc_description": resolved_description,
                "source_type": source_type,
                "knowledge_id": knowledge_id,
                "user_id": user_id,
                "node_id": None,
                "page_number": None,
                "section_title": resolved_title or None,
            },
        }

        chunks: list[dict] = []
        chunk_index = 0

        if source_type == "pdf" and page_list:
            for page_number, (page_text, _) in enumerate(page_list, start=1):
                clean_text = (page_text or "").strip()
                if not clean_text:
                    continue
                for text_piece in cls._split_into_chunks(
                    clean_text,
                    CHUNK_SIZE_CHARS,
                    CHUNK_OVERLAP_CHARS,
                ):
                    chunk_id = cls._make_chunk_id(document_id, chunk_index)
                    chunks.append(
                        {
                            "id": chunk_id,
                            "text": text_piece,
                            "metadata": {
                                "record_type": "chunk",
                                "chunk_id": chunk_id,
                                "document_id": document_id,
                                "file_id": file_id,
                                "file_name": file_name,
                                "doc_title": resolved_title,
                                "doc_description": resolved_description,
                                "source_type": source_type,
                                "knowledge_id": knowledge_id,
                                "user_id": user_id,
                                "node_id": None,
                                "page_number": page_number,
                                "section_title": f"Page {page_number}",
                                "chunk_index": chunk_index,
                                "total_chunks_for_document": 0,
                            },
                        }
                    )
                    chunk_index += 1
        else:
            clean_text = (full_text or "").strip()
            if clean_text:
                for text_piece in cls._split_into_chunks(
                    clean_text,
                    CHUNK_SIZE_CHARS,
                    CHUNK_OVERLAP_CHARS,
                ):
                    chunk_id = cls._make_chunk_id(document_id, chunk_index)
                    chunks.append(
                        {
                            "id": chunk_id,
                            "text": text_piece,
                            "metadata": {
                                "record_type": "chunk",
                                "chunk_id": chunk_id,
                                "document_id": document_id,
                                "file_id": file_id,
                                "file_name": file_name,
                                "doc_title": resolved_title,
                                "doc_description": resolved_description,
                                "source_type": source_type,
                                "knowledge_id": knowledge_id,
                                "user_id": user_id,
                                "node_id": None,
                                "page_number": None,
                                "section_title": resolved_title or None,
                                "chunk_index": chunk_index,
                                "total_chunks_for_document": 0,
                            },
                        }
                    )
                    chunk_index += 1

        total = len(chunks)
        for chunk in chunks:
            chunk["metadata"]["total_chunks_for_document"] = total

        return chunks, header_item

    @classmethod
    async def _embed_texts(
        cls,
        texts: list[str],
        embedding_fn: Callable,
        prefix: Optional[str] = None,
    ) -> list[list[float]]:
        try:
            vectors = await embedding_fn(texts, prefix=prefix)
        except TypeError:
            vectors = await embedding_fn(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding output size mismatch: {len(vectors)} for {len(texts)}"
            )
        return vectors

    @classmethod
    def _vector_items(cls, payload: list[dict]) -> list[VectorItem]:
        return [
            VectorItem(
                id=item["id"],
                text=item["text"],
                vector=item["vector"],
                metadata=item["metadata"],
            )
            for item in payload
        ]

    @classmethod
    def delete_selector_document(cls, document_id: str) -> None:
        for collection_name in (CHUNK_COLLECTION_NAME, HEADER_COLLECTION_NAME):
            if not VECTOR_DB_CLIENT.has_collection(collection_name):
                continue
            VECTOR_DB_CLIENT.delete(
                collection_name=collection_name,
                filter={"document_id": document_id},
            )

    @classmethod
    def clear_selector_collections(cls) -> None:
        for collection_name in (CHUNK_COLLECTION_NAME, HEADER_COLLECTION_NAME):
            if VECTOR_DB_CLIENT.has_collection(collection_name):
                VECTOR_DB_CLIENT.delete_collection(collection_name)

    @classmethod
    async def index_document_chunks(
        cls,
        document_id: str,
        file_id: str,
        knowledge_id: Optional[str],
        user_id: str,
        tree_data: dict,
        source_type: Optional[str],
        embedding_fn: Optional[Callable] = None,
        page_list: Optional[list[tuple[str, int]]] = None,
    ) -> int:
        fn = embedding_fn or cls.get_embedding_function()
        if fn is None:
            raise RuntimeError("No embedding function registered for selector indexing")

        file_name = None
        file = Files.get_file_by_id(file_id)
        if file:
            file_name = file.filename

        chunks, header = cls.build_chunks_for_document(
            document_id=document_id,
            file_id=file_id,
            knowledge_id=knowledge_id,
            user_id=user_id,
            tree_data=tree_data,
            source_type=source_type,
            file_name=file_name,
            page_list=page_list,
        )
        if not chunks and not header["text"]:
            return 0

        chunk_texts = [c["text"] for c in chunks]
        if chunk_texts:
            chunk_vectors = await cls._embed_texts(chunk_texts, fn, prefix=None)
            for chunk, vector in zip(chunks, chunk_vectors):
                chunk["vector"] = vector

        header_vector = await cls._embed_texts([header["text"]], fn, prefix=None)
        header["vector"] = header_vector[0]

        log.info(
            "selector ingestion started document_id=%s chunks=%s",
            document_id,
            len(chunks),
        )
        cls.delete_selector_document(document_id)

        if chunks:
            VECTOR_DB_CLIENT.upsert(
                collection_name=CHUNK_COLLECTION_NAME,
                items=cls._vector_items(chunks),
            )
        VECTOR_DB_CLIENT.upsert(
            collection_name=HEADER_COLLECTION_NAME,
            items=cls._vector_items([header]),
        )

        log.info(
            "selector ingestion completed document_id=%s chunks=%s",
            document_id,
            len(chunks),
        )
        return len(chunks)

    @classmethod
    async def index_source_chunks(
        cls,
        document_id: str,
        file_id: str,
        knowledge_id: Optional[str],
        user_id: str,
        source_type: Optional[str],
        file_name: Optional[str],
        doc_title: Optional[str] = None,
        doc_description: Optional[str] = None,
        page_list: Optional[list[tuple[str, int]]] = None,
        full_text: Optional[str] = None,
        embedding_fn: Optional[Callable] = None,
    ) -> int:
        fn = embedding_fn or cls.get_embedding_function()
        if fn is None:
            raise RuntimeError("No embedding function registered for selector indexing")

        chunks, header = cls.build_chunks_from_source_text(
            document_id=document_id,
            file_id=file_id,
            knowledge_id=knowledge_id,
            user_id=user_id,
            source_type=source_type,
            file_name=file_name,
            doc_title=doc_title,
            doc_description=doc_description,
            page_list=page_list,
            full_text=full_text,
        )
        if not chunks and not header["text"]:
            return 0

        chunk_texts = [c["text"] for c in chunks]
        if chunk_texts:
            chunk_vectors = await cls._embed_texts(chunk_texts, fn, prefix=None)
            for chunk, vector in zip(chunks, chunk_vectors):
                chunk["vector"] = vector

        header_vector = await cls._embed_texts([header["text"]], fn, prefix=None)
        header["vector"] = header_vector[0]

        cls.delete_selector_document(document_id)
        if chunks:
            VECTOR_DB_CLIENT.upsert(
                collection_name=CHUNK_COLLECTION_NAME,
                items=cls._vector_items(chunks),
            )
        VECTOR_DB_CLIENT.upsert(
            collection_name=HEADER_COLLECTION_NAME,
            items=cls._vector_items([header]),
        )

        return len(chunks)

    @classmethod
    async def replace_selector_chunks(
        cls,
        document_id: str,
        user_id: str,
        chunk_texts: list[str],
        embedding_fn: Optional[Callable] = None,
    ) -> int:
        doc = PageIndexes.get_document_by_id(document_id)
        if not doc:
            raise RuntimeError(f"Document not found: {document_id}")
        if doc.user_id != user_id:
            raise RuntimeError("Document access denied")

        fn = embedding_fn or cls.get_embedding_function()
        if fn is None:
            raise RuntimeError("No embedding function registered for selector indexing")

        if not chunk_texts:
            cls.delete_selector_document(document_id)
            return 0

        clean_texts = [" ".join((text or "").split()) for text in chunk_texts]
        clean_texts = [text for text in clean_texts if text]
        vectors = await cls._embed_texts(clean_texts, fn, prefix=None)

        payload: list[dict] = []
        for index, (text, vector) in enumerate(zip(clean_texts, vectors)):
            chunk_id = cls._make_chunk_id(document_id, index)
            payload.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "vector": vector,
                    "metadata": {
                        "record_type": "chunk",
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "file_id": doc.file_id,
                        "file_name": None,
                        "doc_title": doc.doc_title,
                        "doc_description": doc.doc_description,
                        "source_type": doc.source_type,
                        "knowledge_id": doc.knowledge_id,
                        "user_id": doc.user_id,
                        "node_id": None,
                        "page_number": None,
                        "section_title": None,
                        "chunk_index": index,
                        "total_chunks_for_document": len(clean_texts),
                    },
                }
            )

        cls.delete_selector_document(document_id)
        VECTOR_DB_CLIENT.upsert(
            collection_name=CHUNK_COLLECTION_NAME,
            items=cls._vector_items(payload),
        )

        header_text = "\n".join(
            [
                f"Document title: {doc.doc_title or ''}",
                f"Document description: {doc.doc_description or ''}",
                f"Source type: {doc.source_type or ''}",
            ]
        )
        header_vector = await cls._embed_texts([header_text], fn, prefix=None)
        header_payload = {
            "id": cls._make_header_id(document_id),
            "text": header_text,
            "vector": header_vector[0],
            "metadata": {
                "record_type": "doc_header",
                "document_id": document_id,
                "file_id": doc.file_id,
                "doc_title": doc.doc_title,
                "doc_description": doc.doc_description,
                "source_type": doc.source_type,
                "knowledge_id": doc.knowledge_id,
                "user_id": doc.user_id,
            },
        }
        VECTOR_DB_CLIENT.upsert(
            collection_name=HEADER_COLLECTION_NAME,
            items=cls._vector_items([header_payload]),
        )
        return len(payload)

    @classmethod
    def _query_collection(
        cls,
        collection_name: str,
        filter_payload: dict,
        limit: Optional[int] = None,
    ) -> list[dict]:
        if not VECTOR_DB_CLIENT.has_collection(collection_name):
            return []

        result = VECTOR_DB_CLIENT.query(
            collection_name=collection_name,
            filter=filter_payload,
            limit=limit,
        )
        if result and result.ids and result.ids[0]:
            docs = result.documents[0] if result.documents else []
            metas = result.metadatas[0] if result.metadatas else []
            out: list[dict] = []
            for idx, item_id in enumerate(result.ids[0]):
                out.append(
                    {
                        "id": item_id,
                        "text": docs[idx] if idx < len(docs) else "",
                        "metadata": metas[idx] if idx < len(metas) else {},
                    }
                )
            return out

        raw = VECTOR_DB_CLIENT.get(collection_name=collection_name)
        if not raw or not raw.ids or not raw.ids[0]:
            return []

        docs = raw.documents[0] if raw.documents else []
        metas = raw.metadatas[0] if raw.metadatas else []
        out = []
        for idx, item_id in enumerate(raw.ids[0]):
            metadata = metas[idx] if idx < len(metas) else {}
            if all(metadata.get(k) == v for k, v in filter_payload.items()):
                out.append(
                    {
                        "id": item_id,
                        "text": docs[idx] if idx < len(docs) else "",
                        "metadata": metadata,
                    }
                )
        if limit is not None:
            return out[:limit]
        return out

    @classmethod
    def list_selector_chunks(
        cls,
        document_id: str,
        limit: int = 200,
    ) -> list[dict]:
        rows = cls._query_collection(
            CHUNK_COLLECTION_NAME,
            {"document_id": document_id},
            limit=limit,
        )
        rows.sort(key=lambda row: row.get("metadata", {}).get("chunk_index", 0))
        return rows

    @classmethod
    def get_selector_chunk(cls, chunk_id: str) -> Optional[dict]:
        # Fast path when the backend supports metadata filtering.
        rows = cls._query_collection(CHUNK_COLLECTION_NAME, {"chunk_id": chunk_id}, limit=5)
        for row in rows:
            if row["id"] == chunk_id or row.get("metadata", {}).get("chunk_id") == chunk_id:
                return row

        # Fallback by scanning IDs.
        raw = VECTOR_DB_CLIENT.get(collection_name=CHUNK_COLLECTION_NAME)
        if not raw or not raw.ids or not raw.ids[0]:
            return None
        docs = raw.documents[0] if raw.documents else []
        metas = raw.metadatas[0] if raw.metadatas else []
        for idx, item_id in enumerate(raw.ids[0]):
            if item_id == chunk_id:
                return {
                    "id": item_id,
                    "text": docs[idx] if idx < len(docs) else "",
                    "metadata": metas[idx] if idx < len(metas) else {},
                }
        return None

    @classmethod
    def delete_selector_chunk(cls, chunk_id: str) -> bool:
        if not VECTOR_DB_CLIENT.has_collection(CHUNK_COLLECTION_NAME):
            return False
        VECTOR_DB_CLIENT.delete(
            collection_name=CHUNK_COLLECTION_NAME,
            ids=[chunk_id],
        )
        return True

    @classmethod
    async def search_selector_chunks(
        cls,
        query: str,
        user_id: Optional[str],
        knowledge_id: Optional[str] = None,
        file_ids: Optional[list[str]] = None,
        limit: int = 20,
        embedding_fn: Optional[Callable] = None,
    ) -> list[dict]:
        fn = embedding_fn or cls.get_embedding_function()
        if fn is None:
            return []
        if not VECTOR_DB_CLIENT.has_collection(CHUNK_COLLECTION_NAME):
            return []

        variants = build_selector_query_variants(query)
        if not variants:
            return []

        allowed_files = set(file_ids) if file_ids else None
        merged: dict[str, dict] = {}
        per_query_limit = max(10, min(limit * 3, 100))

        for variant in variants:
            try:
                vec = await fn(variant, prefix=RAG_EMBEDDING_QUERY_PREFIX)
            except TypeError:
                vec = await fn(variant)

            result = VECTOR_DB_CLIENT.search(
                collection_name=CHUNK_COLLECTION_NAME,
                vectors=[vec],
                limit=per_query_limit,
            )
            if not result or not result.ids or not result.ids[0]:
                continue

            ids = result.ids[0]
            distances = result.distances[0] if result.distances else []
            docs = result.documents[0] if result.documents else []
            metas = result.metadatas[0] if result.metadatas else []

            for i, item_id in enumerate(ids):
                metadata = metas[i] if i < len(metas) and metas[i] else {}
                if metadata.get("record_type") != "chunk":
                    continue
                if user_id and metadata.get("user_id") != user_id:
                    continue
                if knowledge_id and metadata.get("knowledge_id") != knowledge_id:
                    continue
                if allowed_files and metadata.get("file_id") not in allowed_files:
                    continue

                score = cls._clip_score(distances[i] if i < len(distances) else 0.0)
                existing = merged.get(item_id)
                payload = {
                    "chunk_id": item_id,
                    "score": score,
                    "document_id": metadata.get("document_id"),
                    "file_id": metadata.get("file_id"),
                    "page_number": metadata.get("page_number"),
                    "section_title": metadata.get("section_title"),
                    "node_id": metadata.get("node_id"),
                    "query_variant": variant,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metadata,
                }
                if existing is None or payload["score"] > existing["score"]:
                    merged[item_id] = payload

        return sorted(merged.values(), key=lambda row: row["score"], reverse=True)[:limit]

    @classmethod
    async def search_candidate_documents(
        cls,
        query: str,
        user_id: Optional[str],
        knowledge_id: Optional[str] = None,
        file_ids: Optional[list[str]] = None,
        limit: int = 10,
        embedding_fn: Optional[Callable] = None,
    ) -> PageIndexCandidateSearchResponse:
        fn = embedding_fn or cls.get_embedding_function()
        if fn is None:
            return PageIndexCandidateSearchResponse(query=query, items=[])

        if not VECTOR_DB_CLIENT.has_collection(CHUNK_COLLECTION_NAME):
            return PageIndexCandidateSearchResponse(query=query, items=[])

        variants = build_selector_query_variants(query)
        if not variants:
            return PageIndexCandidateSearchResponse(query=query, items=[])
        log.info("selector query variants generated count=%s variants=%s", len(variants), variants)

        query_tokens = extract_query_tokens(query)
        rare_tokens = extract_rare_tokens(query)
        allowed_files = set(file_ids) if file_ids else None

        candidate_chunks: dict[str, ChunkHit] = {}
        candidate_headers: dict[str, dict] = {}
        per_query_limit = max(OVERFETCH_MIN, limit * OVERFETCH_MULTIPLIER)

        for variant in variants:
            try:
                qv = await fn(variant, prefix=RAG_EMBEDDING_QUERY_PREFIX)
            except TypeError:
                qv = await fn(variant)
            except Exception as err:
                log.warning("selector variant embedding failed: %s", err)
                continue

            chunk_result = VECTOR_DB_CLIENT.search(
                collection_name=CHUNK_COLLECTION_NAME,
                vectors=[qv],
                limit=per_query_limit,
            )
            header_result = None
            if VECTOR_DB_CLIENT.has_collection(HEADER_COLLECTION_NAME):
                header_result = VECTOR_DB_CLIENT.search(
                    collection_name=HEADER_COLLECTION_NAME,
                    vectors=[qv],
                    limit=max(limit * 3, 20),
                )

            if chunk_result and chunk_result.ids and chunk_result.ids[0]:
                ids = chunk_result.ids[0]
                distances = chunk_result.distances[0] if chunk_result.distances else []
                docs = chunk_result.documents[0] if chunk_result.documents else []
                metas = chunk_result.metadatas[0] if chunk_result.metadatas else []
                for i, item_id in enumerate(ids):
                    metadata = metas[i] if i < len(metas) and metas[i] else {}
                    if metadata.get("record_type") != "chunk":
                        continue
                    if user_id and metadata.get("user_id") != user_id:
                        continue
                    if knowledge_id and metadata.get("knowledge_id") != knowledge_id:
                        continue
                    if allowed_files and metadata.get("file_id") not in allowed_files:
                        continue

                    doc_id = metadata.get("document_id")
                    if not doc_id:
                        continue
                    hit = ChunkHit(
                        chunk_id=item_id,
                        score=cls._clip_score(distances[i] if i < len(distances) else 0.0),
                        document_id=doc_id,
                        file_id=str(metadata.get("file_id") or ""),
                        page_number=metadata.get("page_number"),
                        section_title=metadata.get("section_title"),
                        node_id=metadata.get("node_id"),
                        text=docs[i] if i < len(docs) else "",
                        query_variant=variant,
                        record_type="chunk",
                    )
                    existing = candidate_chunks.get(item_id)
                    if existing is None or hit.score > existing.score:
                        candidate_chunks[item_id] = hit

            if header_result and header_result.ids and header_result.ids[0]:
                ids = header_result.ids[0]
                distances = header_result.distances[0] if header_result.distances else []
                docs = header_result.documents[0] if header_result.documents else []
                metas = header_result.metadatas[0] if header_result.metadatas else []
                for i, item_id in enumerate(ids):
                    metadata = metas[i] if i < len(metas) and metas[i] else {}
                    if metadata.get("record_type") != "doc_header":
                        continue
                    if user_id and metadata.get("user_id") != user_id:
                        continue
                    if knowledge_id and metadata.get("knowledge_id") != knowledge_id:
                        continue
                    if allowed_files and metadata.get("file_id") not in allowed_files:
                        continue
                    doc_id = metadata.get("document_id")
                    if not doc_id:
                        continue
                    payload = {
                        "header_id": item_id,
                        "document_id": doc_id,
                        "score": cls._clip_score(distances[i] if i < len(distances) else 0.0),
                        "text": docs[i] if i < len(docs) else "",
                        "metadata": metadata,
                    }
                    existing = candidate_headers.get(item_id)
                    if existing is None or payload["score"] > existing["score"]:
                        candidate_headers[item_id] = payload

        log.info(
            "selector raw candidate hits chunks=%s headers=%s",
            len(candidate_chunks),
            len(candidate_headers),
        )

        if not candidate_chunks and not candidate_headers:
            return PageIndexCandidateSearchResponse(
                query=query,
                items=[],
                query_variants_used=variants,
                matched_documents=0,
                matched_chunks=0,
                total_chunks=0,
                confidence="low",
            )

        per_doc: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "chunk_hits": [],
                "header_hits": [],
                "file_id": None,
            }
        )

        for chunk_hit in candidate_chunks.values():
            doc_entry = per_doc[chunk_hit.document_id]
            doc_entry["chunk_hits"].append(chunk_hit)
            doc_entry["file_id"] = doc_entry["file_id"] or chunk_hit.file_id

        for header_hit in candidate_headers.values():
            doc_id = header_hit["document_id"]
            doc_entry = per_doc[doc_id]
            doc_entry["header_hits"].append(header_hit)
            doc_entry["file_id"] = doc_entry["file_id"] or header_hit.get("metadata", {}).get("file_id")

        items: list[PageIndexCandidateDocument] = []
        global_breakdown: dict[str, dict[str, float]] = {}

        for document_id, hit_data in per_doc.items():
            document = PageIndexes.get_document_by_id(document_id)
            if not document or document.status != "ready":
                continue
            if user_id and document.user_id != user_id:
                continue

            file_name = None
            file_obj = Files.get_file_by_id(document.file_id)
            if file_obj:
                file_name = file_obj.filename

            chunk_hits = sorted(hit_data["chunk_hits"], key=lambda h: h.score, reverse=True)
            header_hits = sorted(
                hit_data["header_hits"],
                key=lambda h: h["score"],
                reverse=True,
            )

            chunk_scores = [hit.score for hit in chunk_hits]
            header_scores = [hit["score"] for hit in header_hits]
            node_ids = [hit.node_id for hit in chunk_hits if hit.node_id]
            page_numbers = [hit.page_number for hit in chunk_hits if isinstance(hit.page_number, int)]

            title_desc = "\n".join(
                [
                    document.doc_title or "",
                    document.doc_description or "",
                    file_name or "",
                ]
            )
            title_desc_overlap = lexical_overlap_score(query_tokens, title_desc)
            exact_overlap = rare_token_overlap_score(
                rare_tokens,
                [title_desc] + [hit.text for hit in chunk_hits[:8]],
            )

            sem_score = semantic_chunk_score(chunk_scores)
            meta_score = metadata_score(header_scores, title_desc_overlap)
            cover_score = evidence_coverage_score(node_ids, page_numbers, len(chunk_hits))
            fname_score = filename_relevance_score(query_tokens, file_name or "")

            final_score, breakdown = combine_document_score(
                semantic_score=sem_score,
                metadata_score_value=meta_score,
                exact_overlap_score=exact_overlap,
                coverage_score=cover_score,
                filename_score=fname_score,
            )
            strict_score = rerank_adjustment(
                current_score=final_score,
                exact_overlap_score=exact_overlap,
                metadata_score_value=meta_score,
                best_chunk_score=max(chunk_scores, default=0.0),
            )

            matched_refs = []
            for hit in chunk_hits[:8]:
                matched_refs.append(
                    {
                        "chunk_id": hit.chunk_id,
                        "score": round(hit.score, 6),
                        "page_number": hit.page_number,
                        "section_title": hit.section_title,
                        "node_id": hit.node_id,
                        "query_variant": hit.query_variant,
                    }
                )

            global_breakdown[document_id] = {
                **{k: round(v, 6) for k, v in breakdown.items()},
                "rerank_score": round(strict_score, 6),
            }

            items.append(
                PageIndexCandidateDocument(
                    document_id=document_id,
                    file_id=document.file_id or hit_data.get("file_id") or "",
                    knowledge_id=document.knowledge_id,
                    user_id=document.user_id,
                    doc_title=document.doc_title,
                    doc_description=document.doc_description,
                    file_name=file_name,
                    score=round(strict_score, 6),
                    matched_nodes=len(set(node_ids)),
                    semantic_score=round(sem_score, 6),
                    metadata_score=round(meta_score, 6),
                    exact_token_overlap_score=round(exact_overlap, 6),
                    evidence_coverage_score=round(cover_score, 6),
                    filename_score=round(fname_score, 6),
                    matched_chunks=len(chunk_hits),
                    matched_chunk_refs=matched_refs,
                    score_breakdown={
                        **{k: round(v, 6) for k, v in breakdown.items()},
                        "rerank_score": round(strict_score, 6),
                    },
                    query_variants_used=variants,
                )
            )

        items.sort(key=lambda item: item.score, reverse=True)
        items = items[:limit]
        log.info(
            "selector reranked candidates count=%s top_scores=%s",
            len(items),
            [item.score for item in items[:5]],
        )

        confidence = confidence_from_scores([item.score for item in items])
        for item in items:
            item.confidence = confidence

        if items:
            top = items[0]
            log.info(
                "selector final selected document_id=%s file_id=%s score=%s confidence=%s",
                top.document_id,
                top.file_id,
                top.score,
                confidence,
            )
        else:
            log.info("selector final selection empty confidence=%s", confidence)

        return PageIndexCandidateSearchResponse(
            query=query,
            items=items,
            query_variants_used=variants,
            matched_documents=len(items),
            matched_chunks=len(candidate_chunks),
            total_chunks=len(candidate_chunks),
            score_breakdown=global_breakdown,
            confidence=confidence,
        )
