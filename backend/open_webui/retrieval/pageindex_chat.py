import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from open_webui.models.access_grants import AccessGrants
from open_webui.models.files import Files
from open_webui.models.knowledge import Knowledges
from open_webui.models.users import UserModel
from open_webui.retrieval.pageindex_service import PageIndexing
from open_webui.storage.pageindex import PageIndexes
from open_webui.storage.provider import Storage
from open_webui.utils.access_control.files import has_access_to_file


log = logging.getLogger(__name__)


class PageIndexChatService:
    MAX_AUTO_INDEX_KB_FILES = 5

    @staticmethod
    def _ensure_pageindex_import_path() -> None:
        backend_dir = Path(__file__).resolve().parents[2]
        pageindex_root = str(backend_dir / "pageindex")
        if pageindex_root not in sys.path:
            sys.path.append(pageindex_root)

    @staticmethod
    def _get_qa_model(preferred_model: Optional[str] = None) -> str:
        return preferred_model or os.getenv("PAGEINDEX_QA_MODEL", "gpt-4o-mini")

    @staticmethod
    def _can_read_knowledge(user: UserModel, knowledge_id: str) -> bool:
        knowledge = Knowledges.get_knowledge_by_id(knowledge_id)
        if not knowledge:
            return False

        if user.role == "admin" or knowledge.user_id == user.id:
            return True

        return AccessGrants.has_access(
            user_id=user.id,
            resource_type="knowledge",
            resource_id=knowledge_id,
            permission="read",
        )

    @staticmethod
    def _load_tree_data_for_document(document_id: str) -> Optional[dict]:
        tree_data = PageIndexes.load_tree_data_by_document_id(document_id=document_id)
        if not tree_data:
            return None
        if isinstance(tree_data, list):
            return {"structure": tree_data}
        if isinstance(tree_data, dict):
            return tree_data
        return None

    @staticmethod
    def _select_best_result(results: list[dict]) -> Optional[dict]:
        if not results:
            return None

        def sort_key(item: dict) -> tuple[int, int, int]:
            return (
                1 if item.get("evidence_sufficient") == "yes" else 0,
                len(item.get("citations", [])),
                1 if item.get("used_full_text") else 0,
            )

        ranked = sorted(results, key=sort_key, reverse=True)
        return ranked[0]

    @staticmethod
    def _run_agentic_qa_for_document(
        query: str,
        document_id: str,
        qa_model: Optional[str] = None,
        user: Optional[UserModel] = None,
    ) -> Optional[dict]:
        doc = PageIndexes.get_document_by_id(document_id=document_id)
        if not doc or doc.status != "ready":
            return None

        if user and doc.user_id != user.id and user.role != "admin":
            if not has_access_to_file(doc.file_id, "read", user):
                return None

        tree_data = PageIndexChatService._load_tree_data_for_document(document_id)
        if not tree_data:
            return None

        file = Files.get_file_by_id(doc.file_id)
        source_path = None
        if file and file.path:
            try:
                source_path = Storage.get_file(file.path)
            except Exception as e:
                log.debug(f"Unable to resolve source path for file {doc.file_id}: {e}")

        if source_path and isinstance(tree_data, dict):
            tree_data.setdefault("source_path", source_path)
            tree_data.setdefault("source_type", doc.source_type)

        PageIndexChatService._ensure_pageindex_import_path()
        from pageindex.agentic_qa import AgenticPageIndexQA  # type: ignore[reportMissingImports]

        client = OpenAI()
        qa = AgenticPageIndexQA(
            tree_data=tree_data,
            client=client,
            model=PageIndexChatService._get_qa_model(qa_model),
            source_path=source_path,
        )
        qa_result = qa.answer(query=query)

        selected_document = {
            "document_id": doc.id,
            "file_id": doc.file_id,
            "knowledge_id": doc.knowledge_id,
            "doc_title": doc.doc_title,
            "doc_description": doc.doc_description,
            "status": doc.status,
        }

        citations = []
        for citation in qa_result.get("citations", []) or []:
            if not isinstance(citation, dict):
                continue
            citations.append(
                {
                    "document_id": doc.id,
                    "file_id": doc.file_id,
                    "doc_title": doc.doc_title,
                    "node_id": citation.get("node_id"),
                    "title": citation.get("title"),
                    "start_index": citation.get("start_index"),
                    "end_index": citation.get("end_index"),
                }
            )

        return {
            "answer": qa_result.get("answer"),
            "citations": citations,
            "retrieved_node_ids": qa_result.get("retrieved_node_ids", []),
            "evidence_sufficient": qa_result.get("evidence_sufficient", "no"),
            "insufficient_reason": qa_result.get("insufficient_reason", ""),
            "used_full_text": bool(qa_result.get("used_full_text", False)),
            "summary_enough": qa_result.get("summary_enough", "no"),
            "full_text_unavailable_reason": qa_result.get("full_text_unavailable_reason", ""),
            "selected_document": selected_document,
        }

    @staticmethod
    def query_single_file(
        query: str,
        file_id: str,
        user: UserModel,
        auto_index: bool = True,
        force_reindex: bool = False,
        qa_model: Optional[str] = None,
        knowledge_id: Optional[str] = None,
    ) -> dict:
        if not has_access_to_file(file_id=file_id, access_type="read", user=user):
            return {
                "available": False,
                "mode": "single_file",
                "query": query,
                "error": "Access denied to requested file",
            }

        status = PageIndexing.get_indexing_status(file_id=file_id, user_id=user.id)
        if (not status or status.status != "ready") and auto_index:
            PageIndexing.index_file_by_id(
                file_id=file_id,
                knowledge_id=knowledge_id,
                force_reindex=force_reindex,
            )
            status = PageIndexing.get_indexing_status(file_id=file_id, user_id=user.id)

        if not status or status.status != "ready" or not status.document_id:
            return {
                "available": False,
                "mode": "single_file",
                "query": query,
                "error": "PageIndex is not ready for this file",
                "status": status.model_dump() if status else None,
            }

        result = PageIndexChatService._run_agentic_qa_for_document(
            query=query,
            document_id=status.document_id,
            qa_model=qa_model,
            user=user,
        )
        if not result:
            return {
                "available": False,
                "mode": "single_file",
                "query": query,
                "error": "Failed to run PageIndex QA for file",
            }

        return {
            "available": True,
            "mode": "single_file",
            "query": query,
            **result,
            "selected_documents": [result["selected_document"]],
            "document_results": [result],
        }

    @staticmethod
    def query_multi_files(
        query: str,
        file_ids: list[str],
        user: UserModel,
        auto_index: bool = True,
        force_reindex: bool = False,
        candidate_limit: int = 5,
        max_documents: int = 3,
        qa_model: Optional[str] = None,
    ) -> dict:
        allowed_file_ids = [
            file_id
            for file_id in file_ids
            if has_access_to_file(file_id=file_id, access_type="read", user=user)
        ]

        if not allowed_file_ids:
            return {
                "available": False,
                "mode": "multi_file",
                "query": query,
                "error": "No accessible files were provided",
            }

        if auto_index:
            for file_id in allowed_file_ids:
                status = PageIndexing.get_indexing_status(file_id=file_id, user_id=user.id)
                if not status or status.status != "ready":
                    PageIndexing.index_file_by_id(
                        file_id=file_id,
                        force_reindex=force_reindex,
                    )

        candidates = PageIndexing.search_candidate_documents(
            query=query,
            user_id=user.id,
            limit=max(candidate_limit * 5, 20),
        )

        filtered_candidates = [
            candidate
            for candidate in candidates.items
            if candidate.file_id in set(allowed_file_ids)
        ]

        if not filtered_candidates:
            ready_docs = []
            for file_id in allowed_file_ids:
                status = PageIndexing.get_indexing_status(file_id=file_id, user_id=user.id)
                if status and status.status == "ready" and status.document_id:
                    ready_docs.append(status.document_id)
            candidate_doc_ids = ready_docs[:max_documents]
        else:
            candidate_doc_ids = [
                candidate.document_id for candidate in filtered_candidates[:max_documents]
            ]

        results = []
        for document_id in candidate_doc_ids:
            result = PageIndexChatService._run_agentic_qa_for_document(
                query=query,
                document_id=document_id,
                qa_model=qa_model,
                user=user,
            )
            if result:
                results.append(result)

        best = PageIndexChatService._select_best_result(results)
        if not best:
            return {
                "available": False,
                "mode": "multi_file",
                "query": query,
                "error": "No indexed PageIndex documents available for the requested files",
            }

        return {
            "available": True,
            "mode": "multi_file",
            "query": query,
            "answer": best.get("answer"),
            "citations": best.get("citations", []),
            "retrieved_node_ids": best.get("retrieved_node_ids", []),
            "evidence_sufficient": best.get("evidence_sufficient", "no"),
            "insufficient_reason": best.get("insufficient_reason", ""),
            "used_full_text": best.get("used_full_text", False),
            "summary_enough": best.get("summary_enough", "no"),
            "full_text_unavailable_reason": best.get("full_text_unavailable_reason", ""),
            "selected_document": best.get("selected_document"),
            "selected_documents": [result.get("selected_document") for result in results],
            "document_results": results,
        }

    @staticmethod
    def query_knowledge(
        query: str,
        knowledge_id: str,
        user: UserModel,
        auto_index: bool = True,
        force_reindex: bool = False,
        candidate_limit: int = 5,
        max_documents: int = 3,
        qa_model: Optional[str] = None,
    ) -> dict:
        if not PageIndexChatService._can_read_knowledge(user=user, knowledge_id=knowledge_id):
            return {
                "available": False,
                "mode": "knowledge",
                "query": query,
                "error": "Access denied to requested knowledge base",
            }

        candidates = PageIndexing.search_candidate_documents(
            query=query,
            user_id=user.id,
            knowledge_id=knowledge_id,
            limit=max(candidate_limit, max_documents),
        )

        if not candidates.items and auto_index:
            files = Knowledges.get_files_by_id(knowledge_id)
            for file in files[: PageIndexChatService.MAX_AUTO_INDEX_KB_FILES]:
                if has_access_to_file(file.id, "read", user):
                    PageIndexing.index_file_by_id(
                        file_id=file.id,
                        knowledge_id=knowledge_id,
                        force_reindex=force_reindex,
                    )

            candidates = PageIndexing.search_candidate_documents(
                query=query,
                user_id=user.id,
                knowledge_id=knowledge_id,
                limit=max(candidate_limit, max_documents),
            )

        candidate_doc_ids = [
            candidate.document_id for candidate in candidates.items[:max_documents]
        ]

        results = []
        for document_id in candidate_doc_ids:
            result = PageIndexChatService._run_agentic_qa_for_document(
                query=query,
                document_id=document_id,
                qa_model=qa_model,
                user=user,
            )
            if result:
                results.append(result)

        best = PageIndexChatService._select_best_result(results)
        if not best:
            return {
                "available": False,
                "mode": "knowledge",
                "query": query,
                "error": "No indexed PageIndex documents available for this knowledge base",
            }

        return {
            "available": True,
            "mode": "knowledge",
            "knowledge_id": knowledge_id,
            "query": query,
            "answer": best.get("answer"),
            "citations": best.get("citations", []),
            "retrieved_node_ids": best.get("retrieved_node_ids", []),
            "evidence_sufficient": best.get("evidence_sufficient", "no"),
            "insufficient_reason": best.get("insufficient_reason", ""),
            "used_full_text": best.get("used_full_text", False),
            "summary_enough": best.get("summary_enough", "no"),
            "full_text_unavailable_reason": best.get("full_text_unavailable_reason", ""),
            "selected_document": best.get("selected_document"),
            "selected_documents": [result.get("selected_document") for result in results],
            "document_results": results,
        }

    @staticmethod
    def query_for_chat_items(
        query: str,
        items: list[dict],
        user: UserModel,
        auto_index: bool = True,
        force_reindex: bool = False,
        candidate_limit: int = 5,
        max_documents: int = 3,
        qa_model: Optional[str] = None,
    ) -> Optional[dict]:
        file_ids: list[str] = []
        knowledge_ids: list[str] = []

        for item in items or []:
            item_type = item.get("type")
            if item_type == "file" and item.get("id"):
                file_ids.append(item["id"])
            elif item_type == "collection" and item.get("id"):
                knowledge_ids.append(item["id"])

        file_ids = list(dict.fromkeys(file_ids))
        knowledge_ids = list(dict.fromkeys(knowledge_ids))

        if file_ids:
            if len(file_ids) == 1:
                return PageIndexChatService.query_single_file(
                    query=query,
                    file_id=file_ids[0],
                    user=user,
                    auto_index=auto_index,
                    force_reindex=force_reindex,
                    qa_model=qa_model,
                )
            return PageIndexChatService.query_multi_files(
                query=query,
                file_ids=file_ids,
                user=user,
                auto_index=auto_index,
                force_reindex=force_reindex,
                candidate_limit=candidate_limit,
                max_documents=max_documents,
                qa_model=qa_model,
            )

        if knowledge_ids:
            return PageIndexChatService.query_knowledge(
                query=query,
                knowledge_id=knowledge_ids[0],
                user=user,
                auto_index=auto_index,
                force_reindex=force_reindex,
                candidate_limit=candidate_limit,
                max_documents=max_documents,
                qa_model=qa_model,
            )

        return None

    @staticmethod
    def to_chat_source(result: dict) -> Optional[dict]:
        if not result or not result.get("available"):
            return None

        selected_document = result.get("selected_document") or {}
        document_id = selected_document.get("document_id") or "pageindex"
        name = selected_document.get("doc_title") or "PageIndex"

        return {
            "source": {
                "type": "pageindex",
                "id": f"pageindex:{document_id}",
                "name": name,
                "document_id": selected_document.get("document_id"),
                "file_id": selected_document.get("file_id"),
                "knowledge_id": selected_document.get("knowledge_id"),
            },
            "document": [result.get("answer", "")],
            "metadata": [
                {
                    "source": f"pageindex:{document_id}",
                    "name": name,
                    "pageindex": {
                        "citations": result.get("citations", []),
                        "retrieved_node_ids": result.get("retrieved_node_ids", []),
                        "evidence_sufficient": result.get("evidence_sufficient", "no"),
                        "used_full_text": result.get("used_full_text", False),
                        "summary_enough": result.get("summary_enough", "no"),
                    },
                }
            ],
        }


PageIndexChat = PageIndexChatService()
