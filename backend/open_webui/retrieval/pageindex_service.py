import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from open_webui.models.files import Files
from open_webui.models.pageindex import (
    PageIndexCandidateSearchResponse,
    PageIndexDocumentUpsertForm,
    PageIndexIndexResult,
    PageIndexStatusResponse,
    now_epoch,
)
from open_webui.storage.pageindex import PageIndexes
from open_webui.storage.provider import Storage
from open_webui.utils.misc import calculate_sha256
from open_webui.retrieval.pageindex_selector.service import (
    HEADER_COLLECTION_NAME,
    PageIndexSelectorService,
)


log = logging.getLogger(__name__)


class PageIndexService:
    @staticmethod
    def _ensure_pageindex_import_path() -> None:
        backend_dir = Path(__file__).resolve().parents[2]
        pageindex_root = str(backend_dir / "pageindex")
        if pageindex_root not in sys.path:
            sys.path.append(pageindex_root)

    @staticmethod
    def _detect_source_type(file_path: str, filename: str) -> str:
        lower_name = (filename or "").lower()
        lower_path = (file_path or "").lower()
        if lower_name.endswith(".pdf") or lower_path.endswith(".pdf"):
            return "pdf"
        if lower_name.endswith(".md") or lower_name.endswith(".markdown"):
            return "markdown"
        raise ValueError(f"Unsupported source type for PageIndex: {filename}")

    @staticmethod
    def _build_index_options(index_options: Optional[dict], source_type: str) -> dict:
        options = dict(index_options or {})

        if source_type == "pdf":
            options.setdefault("if_add_node_id", "yes")
            options.setdefault("if_add_node_summary", "yes")
            options.setdefault("if_add_doc_description", "yes")
            # PDF: raw text extracted separately via get_page_tokens(); no need
            # to store it in the tree JSON (keeps tree compact).
            options.setdefault("if_add_node_text", "no")
            options.setdefault("toc_check_page_num", 20)
            options.setdefault("max_page_num_each_node", 10)
            options.setdefault("max_token_num_each_node", 20000)
        else:
            options.setdefault("if_add_node_id", "yes")
            options.setdefault("if_add_node_summary", "yes")
            options.setdefault("if_add_doc_description", "yes")
            # Markdown: preserve raw node text so the semantic indexer can
            # chunk it directly from node.text without re-reading the file.
            options.setdefault("if_add_node_text", "yes")
            options.setdefault("if_thinning", False)
            options.setdefault("min_token_threshold", 5000)
            options.setdefault("summary_token_threshold", 200)

        return options


    @staticmethod
    def _extract_page_list(
        local_file_path: str,
        source_type: str,
    ) -> Optional[list]:
        """
        Extract the raw page list from a PDF using the pageindex library's
        ``get_page_tokens()`` (PyPDF2/PyMuPDF) so raw page text is available
        for semantic chunking.

        Returns ``None`` when:
        - source_type is not PDF.
        - The pageindex library is not importable.
        - Extraction fails for any reason (logged, not raised).

        When successful, returns a list of ``(page_text, token_count)`` tuples.
        """
        if source_type != "pdf":
            return None
        try:
            PageIndexService._ensure_pageindex_import_path()
            from pageindex.utils import get_page_tokens  # type: ignore[reportMissingImports]

            page_list = get_page_tokens(local_file_path)
            log.info(
                f"pageindex_service: extracted {len(page_list)} raw pages "
                f"from PDF for semantic chunking"
            )
            return page_list
        except Exception as e:
            log.warning(
                f"pageindex_service: failed to extract raw page list from PDF ({e}); "
                "semantic chunking will fall back to summary-only"
            )
            return None

    @staticmethod
    def _run_pdf_indexing(local_file_path: str, index_options: dict) -> dict:
        PageIndexService._ensure_pageindex_import_path()
        from pageindex.page_index import page_index  # type: ignore[reportMissingImports]

        user_opt = {
            "model": index_options.get("model"),
            "toc_check_page_num": index_options.get("toc_check_page_num"),
            "max_page_num_each_node": index_options.get("max_page_num_each_node"),
            "max_token_num_each_node": index_options.get("max_token_num_each_node"),
            "if_add_node_id": index_options.get("if_add_node_id"),
            "if_add_node_summary": index_options.get("if_add_node_summary"),
            "if_add_doc_description": index_options.get("if_add_doc_description"),
            "if_add_node_text": index_options.get("if_add_node_text"),
        }
        user_opt = {k: v for k, v in user_opt.items() if v is not None}

        tree_data = page_index(local_file_path, **user_opt)
        if not isinstance(tree_data, dict):
            raise RuntimeError("PageIndex PDF indexing returned invalid tree data")
        return tree_data


    @staticmethod
    def _run_markdown_indexing(local_file_path: str, index_options: dict) -> dict:
        PageIndexService._ensure_pageindex_import_path()
        from pageindex.page_index_md import md_to_tree  # type: ignore[reportMissingImports]

        async def run_md() -> dict:
            return await md_to_tree(
                md_path=local_file_path,
                if_thinning=bool(index_options.get("if_thinning", False)),
                min_token_threshold=index_options.get("min_token_threshold", 5000),
                if_add_node_summary=index_options.get("if_add_node_summary", "yes"),
                summary_token_threshold=index_options.get("summary_token_threshold", 200),
                model=index_options.get("model"),
                if_add_doc_description=index_options.get("if_add_doc_description", "yes"),
                if_add_node_text=index_options.get("if_add_node_text", "no"),
                if_add_node_id=index_options.get("if_add_node_id", "yes"),
            )

        return asyncio.run(run_md())

    @staticmethod
    async def _run_semantic_indexing(
        document_id: str,
        file_id: str,
        knowledge_id: Optional[str],
        user_id: str,
        tree_data: dict,
        source_type: Optional[str],
        page_list: Optional[list] = None,
    ) -> int:
        """
        Embed and upsert semantic chunks for this document.

        *page_list* — for PDF files, the raw page-text list from
        ``get_page_tokens()``.  When provided, raw page text aligned to
        tree-node page ranges is used for chunking (higher-quality embeddings).

        Returns number of chunks indexed.  Raises on failure.
        """
        from open_webui.retrieval import pageindex_semantic  # noqa: PLC0415

        return await pageindex_semantic.index_document_chunks(
            document_id=document_id,
            file_id=file_id,
            knowledge_id=knowledge_id,
            user_id=user_id,
            tree_data=tree_data,
            source_type=source_type,
            page_list=page_list,
        )

    @staticmethod
    def _get_file_and_existing_doc(file_id: str):
        file = Files.get_file_by_id(file_id)
        if not file:
            return None, None
        user_id = file.user_id
        existing_doc = PageIndexes.get_document_by_file_id(file_id=file.id, user_id=user_id)
        return file, existing_doc

    @staticmethod
    def _selector_vectors_exist(document_id: str) -> bool:
        chunk_rows = PageIndexSelectorService.list_selector_chunks(document_id=document_id, limit=1)
        if chunk_rows:
            return True

        header_rows = PageIndexSelectorService._query_collection(  # noqa: SLF001
            HEADER_COLLECTION_NAME,
            {"document_id": document_id},
            limit=1,
        )
        return bool(header_rows)

    def build_tree_index_by_file_id(
        self,
        file_id: str,
        knowledge_id: Optional[str] = None,
        force_reindex: bool = False,
        index_options: Optional[dict] = None,
    ) -> PageIndexIndexResult:
        file, existing_doc = self._get_file_and_existing_doc(file_id)
        if not file:
            return PageIndexIndexResult(
                indexed=False,
                status="failed",
                message=f"File not found: {file_id}",
                file_id=file_id,
            )

        if not file.path:
            return PageIndexIndexResult(
                indexed=False,
                status="failed",
                message=f"File path missing for file: {file_id}",
                file_id=file_id,
            )

        user_id = file.user_id

        try:
            local_file_path = Storage.get_file(file.path)
            source_hash = file.hash or calculate_sha256(local_file_path, chunk_size=1024 * 1024)
            source_type = self._detect_source_type(local_file_path, file.filename)
            resolved_knowledge_id = knowledge_id if knowledge_id is not None else (
                existing_doc.knowledge_id if existing_doc else None
            )

            if (
                existing_doc
                and existing_doc.status == "ready"
                and existing_doc.source_hash == source_hash
                and not force_reindex
            ):
                return PageIndexIndexResult(
                    indexed=False,
                    skipped=True,
                    status="ready",
                    message="Skipped tree reindex: source hash unchanged",
                    document_id=existing_doc.id,
                    file_id=file.id,
                )

            transition_form = PageIndexDocumentUpsertForm(
                file_id=file.id,
                knowledge_id=resolved_knowledge_id,
                user_id=user_id,
                doc_title=(
                    existing_doc.doc_title
                    if existing_doc and existing_doc.doc_title
                    else file.filename
                ),
                doc_description=(existing_doc.doc_description if existing_doc else None),
                source_type=source_type,
                source_hash=source_hash,
                status="processing",
                tree_json=(existing_doc.tree_json if existing_doc else None),
                generated_at=(existing_doc.generated_at if existing_doc else None),
                index_options=(
                    index_options
                    if index_options is not None
                    else (existing_doc.index_options if existing_doc else None)
                ),
                error_message=None,
            )
            processing_doc = PageIndexes.upsert_document(transition_form)
            if not processing_doc:
                raise RuntimeError("Failed to transition PageIndex status to processing")

            effective_options = self._build_index_options(index_options, source_type=source_type)
            if source_type == "pdf":
                tree_data = self._run_pdf_indexing(local_file_path, effective_options)
            else:
                tree_data = self._run_markdown_indexing(local_file_path, effective_options)

            if not isinstance(tree_data, dict):
                raise RuntimeError("PageIndex indexing returned non-dict tree data")

            save_form = PageIndexDocumentUpsertForm(
                file_id=file.id,
                knowledge_id=resolved_knowledge_id,
                user_id=user_id,
                doc_title=tree_data.get("doc_name") or file.filename,
                doc_description=tree_data.get("doc_description"),
                source_type=source_type,
                source_hash=source_hash,
                status="ready",
                tree_json=tree_data,
                generated_at=now_epoch(),
                index_options=effective_options,
                error_message=None,
            )
            saved_doc = PageIndexes.save_document_with_flattened_nodes(form_data=save_form)
            if not saved_doc:
                raise RuntimeError("Failed to persist PageIndex tree and nodes")

            return PageIndexIndexResult(
                indexed=True,
                status="ready",
                message="PageIndex tree indexing completed",
                document_id=saved_doc.id,
                file_id=file.id,
            )
        except Exception as e:
            log.exception(f"PageIndex tree indexing failed for file_id={file_id}: {e}")
            PageIndexes.update_document_status(
                file_id=file.id,
                user_id=user_id,
                status="failed",
                knowledge_id=knowledge_id,
                error_message=str(e),
            )
            return PageIndexIndexResult(
                indexed=False,
                status="failed",
                message=str(e),
                document_id=(existing_doc.id if existing_doc else None),
                file_id=file.id,
            )

    def ingest_selector_by_file_id(
        self,
        file_id: str,
        knowledge_id: Optional[str] = None,
        force_reindex: bool = False,
        index_options: Optional[dict] = None,
    ) -> PageIndexIndexResult:
        file, existing_doc = self._get_file_and_existing_doc(file_id)
        if not file:
            return PageIndexIndexResult(
                indexed=False,
                status="failed",
                message=f"File not found: {file_id}",
                file_id=file_id,
            )

        if not existing_doc:
            return PageIndexIndexResult(
                indexed=False,
                status="failed",
                message="No PageIndex tree found for file. Run /pageindex/index first.",
                file_id=file.id,
            )

        if existing_doc.status != "ready":
            return PageIndexIndexResult(
                indexed=False,
                status="failed",
                message=f"PageIndex document is not ready: {existing_doc.status}",
                document_id=existing_doc.id,
                file_id=file.id,
            )

        if not force_reindex and self._selector_vectors_exist(existing_doc.id):
            return PageIndexIndexResult(
                indexed=False,
                skipped=True,
                status="ready",
                message="Skipped selector vector ingest: vectors already exist",
                document_id=existing_doc.id,
                file_id=file.id,
            )

        try:
            local_file_path = Storage.get_file(file.path)
            source_type = existing_doc.source_type or self._detect_source_type(
                local_file_path, file.filename
            )
            page_list = self._extract_page_list(local_file_path, source_type)
            full_text: Optional[str] = None
            if source_type in ("md", "markdown"):
                with open(local_file_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
            elif source_type == "pdf" and not page_list:
                raise RuntimeError("Failed to extract PDF text for selector vector ingest")

            resolved_knowledge_id = (
                knowledge_id if knowledge_id is not None else existing_doc.knowledge_id
            )

            if resolved_knowledge_id != existing_doc.knowledge_id or index_options is not None:
                PageIndexes.upsert_document(
                    PageIndexDocumentUpsertForm(
                        file_id=file.id,
                        knowledge_id=resolved_knowledge_id,
                        user_id=existing_doc.user_id,
                        doc_title=existing_doc.doc_title,
                        doc_description=existing_doc.doc_description,
                        source_type=source_type,
                        source_hash=existing_doc.source_hash,
                        status=existing_doc.status,
                        tree_json=existing_doc.tree_json,
                        generated_at=existing_doc.generated_at,
                        index_options=(
                            index_options
                            if index_options is not None
                            else existing_doc.index_options
                        ),
                        error_message=existing_doc.error_message,
                    )
                )

            chunk_count = asyncio.run(
                PageIndexSelectorService.index_source_chunks(
                    document_id=existing_doc.id,
                    file_id=file.id,
                    knowledge_id=resolved_knowledge_id,
                    user_id=existing_doc.user_id,
                    source_type=source_type,
                    file_name=file.filename,
                    doc_title=existing_doc.doc_title,
                    doc_description=existing_doc.doc_description,
                    page_list=page_list,
                    full_text=full_text,
                )
            )

            return PageIndexIndexResult(
                indexed=True,
                status="ready",
                message=f"Selector vector indexing completed ({chunk_count} chunks)",
                document_id=existing_doc.id,
                file_id=file.id,
            )
        except Exception as e:
            log.exception(f"Selector vector indexing failed for file_id={file_id}: {e}")
            return PageIndexIndexResult(
                indexed=False,
                status="failed",
                message=str(e),
                document_id=existing_doc.id,
                file_id=file.id,
            )

    def index_file_by_id(
        self,
        file_id: str,
        knowledge_id: Optional[str] = None,
        force_reindex: bool = False,
        index_options: Optional[dict] = None,
    ) -> PageIndexIndexResult:
        tree_result = self.build_tree_index_by_file_id(
            file_id=file_id,
            knowledge_id=knowledge_id,
            force_reindex=force_reindex,
            index_options=index_options,
        )
        if tree_result.status != "ready":
            return tree_result

        selector_result = self.ingest_selector_by_file_id(
            file_id=file_id,
            knowledge_id=knowledge_id,
            force_reindex=force_reindex,
            index_options=index_options,
        )
        if selector_result.status != "ready":
            return selector_result

        return PageIndexIndexResult(
            indexed=bool(tree_result.indexed or selector_result.indexed),
            skipped=bool(tree_result.skipped and selector_result.skipped),
            status="ready",
            message="PageIndex tree and selector vector indexing completed",
            document_id=selector_result.document_id or tree_result.document_id,
            file_id=file_id,
        )

    def get_indexing_status(self, file_id: str, user_id: Optional[str] = None) -> Optional[PageIndexStatusResponse]:
        return PageIndexes.get_indexing_status_by_file_id(file_id=file_id, user_id=user_id)

    async def search_candidate_documents_semantic(
        self,
        query: str,
        user_id: Optional[str] = None,
        knowledge_id: Optional[str] = None,
        file_ids: Optional[list[str]] = None,
        limit: int = 10,
    ) -> PageIndexCandidateSearchResponse:
        """
        Search candidate documents using vector similarity over semantic chunks.

        Delegates to :mod:`open_webui.retrieval.pageindex_semantic`.
        Falls back to SQL token search if the semantic search fails or returns
        no results (e.g., embedding function not yet registered, collection empty).
        """
        from open_webui.retrieval import pageindex_semantic  # noqa: PLC0415

        try:
            result = await pageindex_semantic.search_candidate_documents(
                query=query,
                user_id=user_id,
                knowledge_id=knowledge_id,
                file_ids=file_ids,
                limit=limit,
            )
            if result.items:
                log.debug(
                    f"pageindex_service: semantic search returned {len(result.items)} candidates"
                )
                return result
            log.info(
                "pageindex_service: semantic search returned no candidates; "
                "falling back to SQL token search"
            )
        except Exception as e:
            log.warning(
                f"pageindex_service: semantic search error ({e}); "
                "falling back to SQL token search"
            )

        # Fallback: SQL token match search
        return PageIndexes.search_candidate_documents(
            query=query,
            user_id=user_id,
            knowledge_id=knowledge_id,
            limit=limit,
        )

    def search_candidate_documents(
        self,
        query: str,
        user_id: Optional[str] = None,
        knowledge_id: Optional[str] = None,
        limit: int = 10,
    ) -> PageIndexCandidateSearchResponse:
        """
        Synchronous wrapper that calls the async semantic search path.

        When called from a thread (e.g., via ``run_in_threadpool``), we need
        to run the async coroutine manually.  Uses ``asyncio.run()`` when there
        is no running event loop, otherwise schedules via the loop.
        """
        file_ids: Optional[list[str]] = None  # no file-id scoping at this level
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures  # noqa: PLC0415

                future = asyncio.ensure_future(
                    self.search_candidate_documents_semantic(
                        query=query,
                        user_id=user_id,
                        knowledge_id=knowledge_id,
                        file_ids=file_ids,
                        limit=limit,
                    )
                )
                # We are inside a thread already; use a separate event loop.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(
                        asyncio.run,
                        self.search_candidate_documents_semantic(
                            query=query,
                            user_id=user_id,
                            knowledge_id=knowledge_id,
                            file_ids=file_ids,
                            limit=limit,
                        ),
                    ).result()
                return result
            else:
                return loop.run_until_complete(
                    self.search_candidate_documents_semantic(
                        query=query,
                        user_id=user_id,
                        knowledge_id=knowledge_id,
                        file_ids=file_ids,
                        limit=limit,
                    )
                )
        except RuntimeError:
            # No event loop available; create one.
            return asyncio.run(
                self.search_candidate_documents_semantic(
                    query=query,
                    user_id=user_id,
                    knowledge_id=knowledge_id,
                    file_ids=file_ids,
                    limit=limit,
                )
            )

    def load_tree_data_by_document_id(self, document_id: str) -> Optional[dict]:
        return PageIndexes.load_tree_data_by_document_id(document_id=document_id)

    def load_tree_data_by_file_id(self, file_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        return PageIndexes.load_tree_data_by_file_id(file_id=file_id, user_id=user_id)


PageIndexing = PageIndexService()
