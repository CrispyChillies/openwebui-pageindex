"""
Tests for PageIndex semantic chunk service.

Covered:
- Chunk building from tree data
- total_chunks annotation
- Document score formula
- Ranking correctness (higher semantic relevance → higher score)
- User / file / knowledge_id filtering (post-filter logic)
- Reindex stale-vector replacement (delete before upsert)
- Empty / no-hit fallback (returns empty list, not an error)
- Score clipping for out-of-range distance values
"""

from __future__ import annotations

import asyncio
import math
import unittest
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without real Open WebUI infra
# ---------------------------------------------------------------------------


# Stub VectorItem
class _VectorItem:
    def __init__(self, *, id, text, vector, metadata):
        self.id = id
        self.text = text
        self.vector = vector
        self.metadata = metadata


# Stub SearchResult
class _SearchResult:
    def __init__(self, ids, distances, documents, metadatas):
        self.ids = ids
        self.distances = distances
        self.documents = documents
        self.metadatas = metadatas


# Patch all heavy dependencies before importing the module under test
import sys
import types

# Provide a minimal open_webui.retrieval.vector.main stub
vector_main = types.ModuleType("open_webui.retrieval.vector.main")
vector_main.VectorItem = _VectorItem
sys.modules["open_webui.retrieval.vector.main"] = vector_main

# Provide a mock VECTOR_DB_CLIENT via the factory module
mock_vector_client = MagicMock()
factory_mod = types.ModuleType("open_webui.retrieval.vector.factory")
factory_mod.VECTOR_DB_CLIENT = mock_vector_client
sys.modules["open_webui.retrieval.vector.factory"] = factory_mod

# Provide a minimal config stub
config_mod = types.ModuleType("open_webui.config")
config_mod.RAG_EMBEDDING_QUERY_PREFIX = None
sys.modules["open_webui.config"] = config_mod

# Provide storage stub (used in search path)
pageindex_candidate_doc_cls = MagicMock()

storage_mod = types.ModuleType("open_webui.storage.pageindex")
mock_pageindexes = MagicMock()
mock_pageindexes.get_document_by_id = MagicMock(return_value=None)
storage_mod.PageIndexes = mock_pageindexes
sys.modules["open_webui.storage.pageindex"] = storage_mod

# Provide models stub
from open_webui.retrieval import pageindex_semantic  # noqa: E402  (must be after stubs)
# If direct import of models fails, patch inline
try:
    from open_webui.models.pageindex import PageIndexCandidateDocument, PageIndexCandidateSearchResponse
except Exception:
    # Minimal stubs
    class PageIndexCandidateDocument:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class PageIndexCandidateSearchResponse:  # type: ignore
        def __init__(self, *, query, items):
            self.query = query
            self.items = items


# Re-import with fresh module after stubs are in place
import importlib

importlib.reload(pageindex_semantic)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tree(
    doc_name: str = "Test Doc",
    doc_description: str = "A description",
    nodes: Optional[list] = None,
) -> dict:
    return {
        "doc_name": doc_name,
        "doc_description": doc_description,
        "structure": nodes or [],
    }


def _make_node(
    node_id: str,
    title: str,
    summary: str,
    children: Optional[list] = None,
    start_index: Optional[int] = None,
) -> dict:
    n = {
        "node_id": node_id,
        "title": title,
        "summary": summary,
    }
    if start_index is not None:
        n["start_index"] = start_index
    if children is not None:
        n["children"] = children
    return n


async def _dummy_embedding_fn(texts, prefix=None):
    """Return unit vectors (all 1.0) of length 4 for every text."""
    if isinstance(texts, str):
        return [1.0, 1.0, 1.0, 1.0]
    return [[1.0, 1.0, 1.0, 1.0] for _ in texts]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestChunkBuilding(unittest.TestCase):
    def test_empty_tree_produces_no_chunks(self):
        chunks = pageindex_semantic.build_chunks_from_tree(
            document_id="doc-1",
            file_id="file-1",
            knowledge_id=None,
            user_id="user-1",
            tree_data=_make_tree(nodes=[]),
        )
        # Only the doc-level description segment should exist
        # (if description is non-empty)
        self.assertGreater(len(chunks), 0)

    def test_tree_with_nodes_yields_chunks(self):
        nodes = [
            _make_node("1", "Introduction", "Intro summary", start_index=1),
            _make_node("2", "Chapter 1", "Chapter one summary", start_index=5),
        ]
        chunks = pageindex_semantic.build_chunks_from_tree(
            document_id="doc-1",
            file_id="file-1",
            knowledge_id="kb-1",
            user_id="user-1",
            tree_data=_make_tree(nodes=nodes),
        )
        self.assertGreater(len(chunks), 0)
        # All chunks must have required fields
        for c in chunks:
            self.assertIn("id", c)
            self.assertIn("text", c)
            self.assertIn("metadata", c)
            meta = c["metadata"]
            self.assertEqual(meta["document_id"], "doc-1")
            self.assertEqual(meta["file_id"], "file-1")
            self.assertEqual(meta["knowledge_id"], "kb-1")
            self.assertEqual(meta["user_id"], "user-1")

    def test_chunk_ids_are_deterministic_and_unique(self):
        nodes = [_make_node("n1", "Section", "Summary text " * 100)]
        chunks = pageindex_semantic.build_chunks_from_tree(
            document_id="doc-X",
            file_id="f",
            knowledge_id=None,
            user_id="u",
            tree_data=_make_tree(nodes=nodes),
        )
        ids = [c["id"] for c in chunks]
        # All IDs must be unique
        self.assertEqual(len(ids), len(set(ids)))
        # IDs must start with the namespace prefix
        for chunk_id in ids:
            self.assertTrue(chunk_id.startswith("pageindex:doc-X:"))

    def test_total_chunks_annotation(self):
        nodes = [_make_node("n1", "Section A", "Text")]
        chunks = pageindex_semantic.build_chunks_from_tree(
            document_id="doc-2",
            file_id="f",
            knowledge_id=None,
            user_id="u",
            tree_data=_make_tree(nodes=nodes),
        )
        chunks = pageindex_semantic.annotate_total_chunks(chunks)
        total = len(chunks)
        for c in chunks:
            self.assertEqual(c["metadata"]["total_chunks_for_document"], total)

    def test_nested_children_are_walked(self):
        child = _make_node("child-1", "Child Section", "Child summary")
        parent = _make_node("parent-1", "Parent Section", "Parent summary", children=[child])
        chunks = pageindex_semantic.build_chunks_from_tree(
            document_id="doc-3",
            file_id="f",
            knowledge_id=None,
            user_id="u",
            tree_data=_make_tree(nodes=[parent]),
        )
        node_ids = [c["metadata"].get("node_id") for c in chunks]
        self.assertIn("child-1", node_ids)
        self.assertIn("parent-1", node_ids)

    def test_section_title_and_page_number_captured(self):
        nodes = [_make_node("n1", "My Section", "Summary", start_index=42)]
        chunks = pageindex_semantic.build_chunks_from_tree(
            document_id="d",
            file_id="f",
            knowledge_id=None,
            user_id="u",
            tree_data=_make_tree(nodes=nodes),
        )
        # Find chunks from this node
        node_chunks = [c for c in chunks if c["metadata"].get("node_id") == "n1"]
        self.assertGreater(len(node_chunks), 0)
        c = node_chunks[0]
        self.assertEqual(c["metadata"]["section_title"], "My Section")
        self.assertEqual(c["metadata"]["page_number"], 42)


class TestDocumentScoring(unittest.TestCase):
    def test_score_formula_basic(self):
        # sum(scores) / sqrt(total + 1)
        scores = [0.9, 0.8, 0.7]
        total = 10
        expected = sum(scores) / math.sqrt(total + 1)
        result = pageindex_semantic._compute_doc_score(scores, total)
        self.assertAlmostEqual(result, expected, places=6)

    def test_score_uses_total_not_matched_count(self):
        # Two documents: same chunk scores, but doc B has more total chunks.
        # Doc A should score higher.
        scores = [0.9, 0.8]
        score_a = pageindex_semantic._compute_doc_score(scores, total_chunks_for_document=5)
        score_b = pageindex_semantic._compute_doc_score(scores, total_chunks_for_document=50)
        self.assertGreater(score_a, score_b)

    def test_empty_scores_returns_zero(self):
        result = pageindex_semantic._compute_doc_score([], total_chunks_for_document=10)
        self.assertEqual(result, 0.0)

    def test_score_clip(self):
        self.assertEqual(pageindex_semantic._clip_score(1.5), 1.0)
        self.assertEqual(pageindex_semantic._clip_score(-0.1), 0.0)
        self.assertAlmostEqual(pageindex_semantic._clip_score(0.7), 0.7)

    def test_ranking_correctness(self):
        """A document with higher-similarity chunk hits should rank first."""
        doc_high = pageindex_semantic._compute_doc_score([0.95, 0.90], 5)
        doc_low = pageindex_semantic._compute_doc_score([0.40, 0.30], 5)
        self.assertGreater(doc_high, doc_low)


class TestChunkSplitting(unittest.TestCase):
    def test_long_text_is_split(self):
        long_text = "word " * 600  # well above CHUNK_SIZE_CHARS
        chunks = pageindex_semantic._split_into_chunks(
            long_text,
            pageindex_semantic.CHUNK_SIZE_CHARS,
            pageindex_semantic.CHUNK_OVERLAP_CHARS,
        )
        self.assertGreater(len(chunks), 1)

    def test_short_text_is_single_chunk(self):
        short_text = "A very short description."
        chunks = pageindex_semantic._split_into_chunks(
            short_text,
            pageindex_semantic.CHUNK_SIZE_CHARS,
            pageindex_semantic.CHUNK_OVERLAP_CHARS,
        )
        self.assertEqual(len(chunks), 1)

    def test_empty_text_returns_empty(self):
        result = pageindex_semantic._split_into_chunks("", 2000, 400)
        self.assertEqual(result, [])


class TestEmbedChunks(unittest.TestCase):
    def test_embed_attaches_vectors(self):
        chunks = [
            {"id": "pageindex:d:0", "text": "hello world", "metadata": {}},
            {"id": "pageindex:d:1", "text": "foo bar", "metadata": {}},
        ]
        result = asyncio.run(pageindex_semantic.embed_chunks(chunks, _dummy_embedding_fn))
        for c in result:
            self.assertIn("vector", c)
            self.assertEqual(len(c["vector"]), 4)

    def test_embed_raises_on_mismatch(self):
        async def bad_fn(texts, prefix=None):
            return [[1.0]]  # only one vector for two chunks

        chunks = [
            {"id": "a", "text": "text1", "metadata": {}},
            {"id": "b", "text": "text2", "metadata": {}},
        ]
        with self.assertRaises(RuntimeError):
            asyncio.run(pageindex_semantic.embed_chunks(chunks, bad_fn))


class TestDeleteDocumentChunks(unittest.TestCase):
    def setUp(self):
        mock_vector_client.reset_mock()

    def test_delete_called_when_collection_exists(self):
        mock_vector_client.has_collection.return_value = True
        pageindex_semantic.delete_document_chunks("doc-delete-1")
        mock_vector_client.delete.assert_called_once_with(
            collection_name=pageindex_semantic.COLLECTION_NAME,
            filter={"document_id": "doc-delete-1"},
        )

    def test_delete_skipped_when_collection_absent(self):
        mock_vector_client.has_collection.return_value = False
        pageindex_semantic.delete_document_chunks("doc-delete-2")
        mock_vector_client.delete.assert_not_called()

    def test_delete_error_does_not_raise(self):
        mock_vector_client.has_collection.return_value = True
        mock_vector_client.delete.side_effect = Exception("Milvus error")
        # Should not raise
        pageindex_semantic.delete_document_chunks("doc-delete-fail")
        mock_vector_client.delete.side_effect = None  # reset


class TestUpsertChunks(unittest.TestCase):
    def setUp(self):
        mock_vector_client.reset_mock()

    def test_upsert_called_with_correct_items(self):
        chunks = [
            {
                "id": "pageindex:d:0",
                "text": "chunk 0",
                "vector": [0.1, 0.2],
                "metadata": {"document_id": "d"},
            }
        ]
        pageindex_semantic.upsert_chunks(chunks)
        mock_vector_client.upsert.assert_called_once()
        call_args = mock_vector_client.upsert.call_args
        self.assertEqual(
            call_args.kwargs.get("collection_name") or call_args[1].get("collection_name")
            or call_args[0][0],
            pageindex_semantic.COLLECTION_NAME,
        )

    def test_upsert_empty_does_not_call_client(self):
        mock_vector_client.reset_mock()
        pageindex_semantic.upsert_chunks([])
        mock_vector_client.upsert.assert_not_called()


class TestIndexDocumentChunks(unittest.TestCase):
    def setUp(self):
        mock_vector_client.reset_mock()
        mock_vector_client.has_collection.return_value = True

    def test_raises_when_no_embedding_function(self):
        pageindex_semantic.set_embedding_function(None)
        with self.assertRaises(RuntimeError):
            asyncio.run(
                pageindex_semantic.index_document_chunks(
                    document_id="d",
                    file_id="f",
                    knowledge_id=None,
                    user_id="u",
                    tree_data=_make_tree(nodes=[_make_node("n1", "T", "S")]),
                    source_type="pdf",
                )
            )

    def test_full_flow_calls_delete_then_upsert(self):
        pageindex_semantic.set_embedding_function(_dummy_embedding_fn)
        mock_vector_client.has_collection.return_value = True
        mock_vector_client.delete.reset_mock()
        mock_vector_client.upsert.reset_mock()

        count = asyncio.run(
            pageindex_semantic.index_document_chunks(
                document_id="doc-flow",
                file_id="file-flow",
                knowledge_id="kb-flow",
                user_id="user-flow",
                tree_data=_make_tree(
                    nodes=[_make_node("n1", "Section", "Summary text for the section")]
                ),
                source_type="pdf",
            )
        )
        self.assertGreater(count, 0)
        mock_vector_client.delete.assert_called_once()
        mock_vector_client.upsert.assert_called_once()

    def test_returns_zero_for_empty_tree(self):
        pageindex_semantic.set_embedding_function(_dummy_embedding_fn)
        # Pure-empty tree with no description and no nodes
        count = asyncio.run(
            pageindex_semantic.index_document_chunks(
                document_id="doc-empty",
                file_id="f",
                knowledge_id=None,
                user_id="u",
                tree_data={"structure": []},  # no description either
                source_type="pdf",
            )
        )
        self.assertEqual(count, 0)


class TestSearchCandidateDocuments(unittest.TestCase):
    """
    Tests for the search_candidate_documents async function.

    We mock the VECTOR_DB_CLIENT calls to simulate hit scenarios.
    """

    def setUp(self):
        mock_vector_client.reset_mock()
        pageindex_semantic.set_embedding_function(_dummy_embedding_fn)

    def _make_search_result(self, hits: list[dict]) -> _SearchResult:
        """Build a search result from a list of hit metadata dicts."""
        ids = [h["id"] for h in hits]
        distances = [h.get("distance", 0.8) for h in hits]
        metadatas = [h["metadata"] for h in hits]
        return _SearchResult(
            ids=[ids],
            distances=[distances],
            documents=[["" for _ in hits]],
            metadatas=[metadatas],
        )

    def test_returns_empty_when_collection_absent(self):
        mock_vector_client.has_collection.return_value = False
        result = asyncio.run(
            pageindex_semantic.search_candidate_documents(
                query="test query",
                user_id="u1",
            )
        )
        self.assertEqual(result.items, [])

    def test_returns_empty_when_no_hits(self):
        mock_vector_client.has_collection.return_value = True
        mock_vector_client.search.return_value = _SearchResult(
            ids=[[]], distances=[[]], documents=[[]], metadatas=[[]]
        )
        result = asyncio.run(
            pageindex_semantic.search_candidate_documents(
                query="no hits query",
                user_id="u1",
            )
        )
        self.assertEqual(result.items, [])

    def test_user_boundary_filtering(self):
        """Chunks belonging to a different user must be excluded."""
        mock_vector_client.has_collection.return_value = True
        mock_vector_client.search.return_value = self._make_search_result(
            [
                {
                    "id": "pageindex:doc-A:0",
                    "distance": 0.9,
                    "metadata": {
                        "document_id": "doc-A",
                        "file_id": "file-A",
                        "knowledge_id": None,
                        "user_id": "user-OTHER",
                        "total_chunks_for_document": 5,
                    },
                }
            ]
        )
        # Request as user-MINE → foreign chunk should be filtered out
        result = asyncio.run(
            pageindex_semantic.search_candidate_documents(
                query="q",
                user_id="user-MINE",
            )
        )
        self.assertEqual(result.items, [])

    def test_knowledge_id_filtering(self):
        """Chunks with wrong knowledge_id must be excluded."""
        mock_vector_client.has_collection.return_value = True
        mock_vector_client.search.return_value = self._make_search_result(
            [
                {
                    "id": "pageindex:doc-B:0",
                    "distance": 0.85,
                    "metadata": {
                        "document_id": "doc-B",
                        "file_id": "file-B",
                        "knowledge_id": "kb-OTHER",
                        "user_id": "user-1",
                        "total_chunks_for_document": 3,
                    },
                }
            ]
        )
        result = asyncio.run(
            pageindex_semantic.search_candidate_documents(
                query="q",
                user_id="user-1",
                knowledge_id="kb-MINE",
            )
        )
        self.assertEqual(result.items, [])

    def test_file_ids_filtering(self):
        """Chunks whose file_id is not in the allowed set must be excluded."""
        mock_vector_client.has_collection.return_value = True
        mock_vector_client.search.return_value = self._make_search_result(
            [
                {
                    "id": "pageindex:doc-C:0",
                    "distance": 0.8,
                    "metadata": {
                        "document_id": "doc-C",
                        "file_id": "file-EXCLUDED",
                        "knowledge_id": None,
                        "user_id": "user-1",
                        "total_chunks_for_document": 4,
                    },
                }
            ]
        )
        result = asyncio.run(
            pageindex_semantic.search_candidate_documents(
                query="q",
                user_id="user-1",
                file_ids=["file-ALLOWED"],
            )
        )
        self.assertEqual(result.items, [])

    def test_matching_chunks_aggregate_into_document_score(self):
        """Two chunks from the same document should aggregate (sum/sqrt)."""
        mock_vector_client.has_collection.return_value = True

        # Provide a real document record so the candidate is returned
        mock_doc = MagicMock()
        mock_doc.status = "ready"
        mock_doc.file_id = "file-D"
        mock_doc.knowledge_id = None
        mock_doc.user_id = "user-1"
        mock_doc.doc_title = "Doc D"
        mock_doc.doc_description = "Desc"
        mock_pageindexes.get_document_by_id.return_value = mock_doc

        mock_vector_client.search.return_value = self._make_search_result(
            [
                {
                    "id": "pageindex:doc-D:0",
                    "distance": 0.9,
                    "metadata": {
                        "document_id": "doc-D",
                        "file_id": "file-D",
                        "knowledge_id": None,
                        "user_id": "user-1",
                        "total_chunks_for_document": 10,
                    },
                },
                {
                    "id": "pageindex:doc-D:1",
                    "distance": 0.8,
                    "metadata": {
                        "document_id": "doc-D",
                        "file_id": "file-D",
                        "knowledge_id": None,
                        "user_id": "user-1",
                        "total_chunks_for_document": 10,
                    },
                },
            ]
        )
        result = asyncio.run(
            pageindex_semantic.search_candidate_documents(
                query="q",
                user_id="user-1",
            )
        )
        self.assertEqual(len(result.items), 1)
        item = result.items[0]
        expected_score = (0.9 + 0.8) / math.sqrt(10 + 1)
        self.assertAlmostEqual(item.score, expected_score, places=4)
        self.assertEqual(item.matched_chunks, 2)

    def test_documents_sorted_by_score_descending(self):
        """When multiple documents match, higher score must come first."""
        mock_vector_client.has_collection.return_value = True

        def _doc_mock(title, file_id):
            m = MagicMock()
            m.status = "ready"
            m.file_id = file_id
            m.knowledge_id = None
            m.user_id = "user-1"
            m.doc_title = title
            m.doc_description = ""
            return m

        doc_e = _doc_mock("Doc E", "file-E")
        doc_f = _doc_mock("Doc F", "file-F")

        def side_effect(document_id):
            return doc_e if document_id == "doc-E" else doc_f

        mock_pageindexes.get_document_by_id.side_effect = side_effect

        # Doc E: two strong hits, 5 total chunks
        # Doc F: one weak hit, 5 total chunks
        mock_vector_client.search.return_value = self._make_search_result(
            [
                {
                    "id": "pageindex:doc-E:0",
                    "distance": 0.95,
                    "metadata": {
                        "document_id": "doc-E",
                        "file_id": "file-E",
                        "knowledge_id": None,
                        "user_id": "user-1",
                        "total_chunks_for_document": 5,
                    },
                },
                {
                    "id": "pageindex:doc-E:1",
                    "distance": 0.90,
                    "metadata": {
                        "document_id": "doc-E",
                        "file_id": "file-E",
                        "knowledge_id": None,
                        "user_id": "user-1",
                        "total_chunks_for_document": 5,
                    },
                },
                {
                    "id": "pageindex:doc-F:0",
                    "distance": 0.50,
                    "metadata": {
                        "document_id": "doc-F",
                        "file_id": "file-F",
                        "knowledge_id": None,
                        "user_id": "user-1",
                        "total_chunks_for_document": 5,
                    },
                },
            ]
        )
        result = asyncio.run(
            pageindex_semantic.search_candidate_documents(
                query="q",
                user_id="user-1",
            )
        )
        self.assertEqual(len(result.items), 2)
        self.assertGreater(result.items[0].score, result.items[1].score)
        self.assertEqual(result.items[0].doc_title, "Doc E")

    def test_no_embedding_function_returns_empty(self):
        pageindex_semantic.set_embedding_function(None)
        mock_vector_client.has_collection.return_value = True
        result = asyncio.run(
            pageindex_semantic.search_candidate_documents(
                query="q",
                user_id="u",
            )
        )
        self.assertEqual(result.items, [])

    def test_reindex_replacement_deletes_before_upsert(self):
        """Verify delete is called before upsert during index_document_chunks."""
        pageindex_semantic.set_embedding_function(_dummy_embedding_fn)
        mock_vector_client.has_collection.return_value = True

        call_order = []
        mock_vector_client.delete.side_effect = lambda **kw: call_order.append("delete")
        mock_vector_client.upsert.side_effect = lambda **kw: call_order.append("upsert")

        asyncio.run(
            pageindex_semantic.index_document_chunks(
                document_id="doc-reindex",
                file_id="f",
                knowledge_id=None,
                user_id="u",
                tree_data=_make_tree(nodes=[_make_node("n", "Title", "Summary")]),
                source_type="pdf",
            )
        )
        self.assertIn("delete", call_order)
        self.assertIn("upsert", call_order)
        self.assertLess(call_order.index("delete"), call_order.index("upsert"))

        # Cleanup
        mock_vector_client.delete.side_effect = None
        mock_vector_client.upsert.side_effect = None


if __name__ == "__main__":
    unittest.main()
