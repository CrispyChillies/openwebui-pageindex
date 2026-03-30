import logging
import re
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from open_webui.internal.db import get_db_context
from open_webui.models.files import File
from open_webui.models.knowledge import Knowledge
from open_webui.models.pageindex import (
    PageIndexCandidateDocument,
    PageIndexCandidateSearchResponse,
    PageIndexDocument,
    PageIndexDocumentModel,
    PageIndexDocumentUpsertForm,
    PageIndexStatusResponse,
    PageIndexNode,
    PageIndexNodeModel,
    new_pageindex_document_id,
    now_epoch,
)
from open_webui.utils.pageindex_tree import flatten_pageindex_tree, reconstruct_pageindex_tree


log = logging.getLogger(__name__)

# Ensure FK target tables are registered in SQLAlchemy metadata.
_ = (File, Knowledge)


class PageIndexStore:
    @staticmethod
    def _tokenize_query(query: str) -> list[str]:
        tokens = [token.strip().lower() for token in re.findall(r"[a-zA-Z0-9]+", query)]
        deduped: list[str] = []
        for token in tokens:
            if token and token not in deduped:
                deduped.append(token)
        return deduped[:8]

    def upsert_document(
        self,
        form_data: PageIndexDocumentUpsertForm,
        db: Optional[Session] = None,
    ) -> Optional[PageIndexDocumentModel]:
        with get_db_context(db) as db:
            try:
                now = now_epoch()
                existing = (
                    db.query(PageIndexDocument)
                    .filter_by(file_id=form_data.file_id, user_id=form_data.user_id)
                    .order_by(PageIndexDocument.updated_at.desc())
                    .first()
                )

                if existing:
                    doc = existing
                    update_data = {
                        "knowledge_id": form_data.knowledge_id,
                        "doc_title": form_data.doc_title,
                        "doc_description": form_data.doc_description,
                        "source_type": form_data.source_type,
                        "source_hash": form_data.source_hash,
                        "status": form_data.status,
                        "tree_json": form_data.tree_json,
                        "generated_at": form_data.generated_at,
                        "index_options": form_data.index_options,
                        "error_message": form_data.error_message,
                        "updated_at": now,
                    }
                    for key, value in update_data.items():
                        setattr(doc, key, value)
                else:
                    doc = PageIndexDocument(
                        id=new_pageindex_document_id(),
                        file_id=form_data.file_id,
                        knowledge_id=form_data.knowledge_id,
                        user_id=form_data.user_id,
                        doc_title=form_data.doc_title,
                        doc_description=form_data.doc_description,
                        source_type=form_data.source_type,
                        source_hash=form_data.source_hash,
                        status=form_data.status,
                        tree_json=form_data.tree_json,
                        generated_at=form_data.generated_at,
                        index_options=form_data.index_options,
                        error_message=form_data.error_message,
                        created_at=now,
                        updated_at=now,
                    )
                    db.add(doc)

                db.commit()
                db.refresh(doc)
                return PageIndexDocumentModel.model_validate(doc)
            except Exception as e:
                log.exception(f"Failed to upsert pageindex document: {e}")
                db.rollback()
                return None

    def update_document_status(
        self,
        file_id: str,
        user_id: str,
        status: str,
        knowledge_id: Optional[str] = None,
        source_hash: Optional[str] = None,
        index_options: Optional[dict] = None,
        error_message: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Optional[PageIndexDocumentModel]:
        existing = self.get_document_by_file_id(file_id=file_id, user_id=user_id, db=db)

        form_data = PageIndexDocumentUpsertForm(
            file_id=file_id,
            knowledge_id=(knowledge_id if knowledge_id is not None else (existing.knowledge_id if existing else None)),
            user_id=user_id,
            doc_title=(existing.doc_title if existing else None),
            doc_description=(existing.doc_description if existing else None),
            source_type=(existing.source_type if existing else None),
            source_hash=(source_hash if source_hash is not None else (existing.source_hash if existing else None)),
            status=status,
            tree_json=(existing.tree_json if existing else None),
            generated_at=(existing.generated_at if existing else None),
            index_options=(index_options if index_options is not None else (existing.index_options if existing else None)),
            error_message=error_message,
        )
        return self.upsert_document(form_data=form_data, db=db)

    def save_document_with_flattened_nodes(
        self,
        form_data: PageIndexDocumentUpsertForm,
        db: Optional[Session] = None,
    ) -> Optional[PageIndexDocumentModel]:
        with get_db_context(db) as db:
            try:
                document_model = self.upsert_document(form_data=form_data, db=db)
                if document_model is None:
                    return None

                db.query(PageIndexNode).filter_by(document_id=document_model.id).delete()

                now = now_epoch()
                flat_nodes = flatten_pageindex_tree(form_data.tree_json)
                for node in flat_nodes:
                    db.add(
                        PageIndexNode(
                            id=new_pageindex_document_id(),
                            document_id=document_model.id,
                            node_id=node["node_id"],
                            parent_node_id=node.get("parent_node_id"),
                            depth=node.get("depth", 0),
                            title=node.get("title"),
                            summary=node.get("summary"),
                            start_index=node.get("start_index"),
                            end_index=node.get("end_index"),
                            has_children=bool(node.get("has_children", False)),
                            created_at=now,
                            updated_at=now,
                        )
                    )

                db.commit()
                return self.get_document_by_id(document_model.id, db=db)
            except Exception as e:
                log.exception(f"Failed to persist pageindex document and nodes: {e}")
                db.rollback()
                return None

    def get_document_by_id(
        self, document_id: str, db: Optional[Session] = None
    ) -> Optional[PageIndexDocumentModel]:
        with get_db_context(db) as db:
            doc = db.get(PageIndexDocument, document_id)
            if doc:
                return PageIndexDocumentModel.model_validate(doc)
            return None

    def get_document_by_file_id(
        self,
        file_id: str,
        user_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Optional[PageIndexDocumentModel]:
        with get_db_context(db) as db:
            query = db.query(PageIndexDocument).filter_by(file_id=file_id)
            if user_id:
                query = query.filter_by(user_id=user_id)

            doc = query.order_by(PageIndexDocument.updated_at.desc()).first()
            if doc:
                return PageIndexDocumentModel.model_validate(doc)
            return None

    def list_documents(
        self,
        user_id: Optional[str] = None,
        knowledge_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        db: Optional[Session] = None,
    ) -> list[PageIndexDocumentModel]:
        with get_db_context(db) as db:
            query = db.query(PageIndexDocument)
            if user_id:
                query = query.filter(PageIndexDocument.user_id == user_id)
            if knowledge_id:
                query = query.filter(PageIndexDocument.knowledge_id == knowledge_id)
            if status:
                query = query.filter(PageIndexDocument.status == status)

            rows = (
                query.order_by(PageIndexDocument.updated_at.desc())
                .offset(max(0, int(offset)))
                .limit(max(1, min(int(limit), 500)))
                .all()
            )
            return [PageIndexDocumentModel.model_validate(row) for row in rows]

    def update_document_metadata(
        self,
        document_id: str,
        user_id: Optional[str] = None,
        doc_title: Optional[str] = None,
        doc_description: Optional[str] = None,
        source_type: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Optional[PageIndexDocumentModel]:
        with get_db_context(db) as db:
            row = db.get(PageIndexDocument, document_id)
            if not row:
                return None
            row_user_id = str(getattr(row, "user_id", "") or "")
            if user_id and row_user_id != user_id:
                return None

            if doc_title is not None:
                setattr(row, "doc_title", doc_title)
            if doc_description is not None:
                setattr(row, "doc_description", doc_description)
            if source_type is not None:
                setattr(row, "source_type", source_type)
            setattr(row, "updated_at", now_epoch())

            db.commit()
            db.refresh(row)
            return PageIndexDocumentModel.model_validate(row)

    def delete_document(
        self,
        document_id: str,
        user_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> bool:
        with get_db_context(db) as db:
            row = db.get(PageIndexDocument, document_id)
            if not row:
                return False
            row_user_id = str(getattr(row, "user_id", "") or "")
            if user_id and row_user_id != user_id:
                return False

            db.delete(row)
            db.commit()
            return True

    def get_indexing_status_by_file_id(
        self,
        file_id: str,
        user_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Optional[PageIndexStatusResponse]:
        doc = self.get_document_by_file_id(file_id=file_id, user_id=user_id, db=db)
        if not doc:
            return None

        return PageIndexStatusResponse(
            file_id=doc.file_id,
            document_id=doc.id,
            status=doc.status,
            source_hash=doc.source_hash,
            error_message=doc.error_message,
            updated_at=doc.updated_at,
        )

    def get_nodes_by_document_id(
        self,
        document_id: str,
        db: Optional[Session] = None,
    ) -> list[PageIndexNodeModel]:
        with get_db_context(db) as db:
            nodes = (
                db.query(PageIndexNode)
                .filter_by(document_id=document_id)
                .order_by(PageIndexNode.depth.asc(), PageIndexNode.node_id.asc())
                .all()
            )
            return [PageIndexNodeModel.model_validate(node) for node in nodes]

    def load_document_tree(
        self,
        document_id: str,
        db: Optional[Session] = None,
    ) -> list[dict]:
        nodes = self.get_nodes_by_document_id(document_id=document_id, db=db)
        return reconstruct_pageindex_tree([node.model_dump() for node in nodes])

    def load_tree_data_by_document_id(
        self,
        document_id: str,
        db: Optional[Session] = None,
    ) -> Optional[dict]:
        doc = self.get_document_by_id(document_id=document_id, db=db)
        if not doc:
            return None

        if doc.tree_json:
            if isinstance(doc.tree_json, dict):
                return doc.tree_json
            if isinstance(doc.tree_json, list):
                return {
                    "doc_name": doc.doc_title,
                    "doc_description": doc.doc_description,
                    "structure": doc.tree_json,
                }

        return {
            "doc_name": doc.doc_title,
            "doc_description": doc.doc_description,
            "structure": self.load_document_tree(document_id=document_id, db=db),
        }

    def load_tree_data_by_file_id(
        self,
        file_id: str,
        user_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Optional[dict]:
        doc = self.get_document_by_file_id(file_id=file_id, user_id=user_id, db=db)
        if not doc:
            return None
        return self.load_tree_data_by_document_id(document_id=doc.id, db=db)

    def search_candidate_documents(
        self,
        query: str,
        user_id: Optional[str] = None,
        knowledge_id: Optional[str] = None,
        limit: int = 10,
        db: Optional[Session] = None,
    ) -> PageIndexCandidateSearchResponse:
        tokens = self._tokenize_query(query)
        if not tokens:
            return PageIndexCandidateSearchResponse(query=query, items=[])

        with get_db_context(db) as db:
            base_filters = [PageIndexDocument.status == "ready"]
            if user_id:
                base_filters.append(PageIndexDocument.user_id == user_id)
            if knowledge_id:
                base_filters.append(PageIndexDocument.knowledge_id == knowledge_id)

            doc_text_match_clauses = []
            node_text_match_clauses = []
            for token in tokens:
                like_pattern = f"%{token}%"
                doc_text_match_clauses.extend(
                    [
                        PageIndexDocument.doc_title.ilike(like_pattern),
                        PageIndexDocument.doc_description.ilike(like_pattern),
                    ]
                )
                node_text_match_clauses.extend(
                    [
                        PageIndexNode.title.ilike(like_pattern),
                        PageIndexNode.summary.ilike(like_pattern),
                    ]
                )

            doc_rows = (
                db.query(PageIndexDocument)
                .filter(and_(*base_filters))
                .filter(or_(*doc_text_match_clauses))
                .order_by(PageIndexDocument.updated_at.desc())
                .limit(max(limit * 20, 100))
                .all()
            )

            node_hit_rows = (
                db.query(
                    PageIndexNode.document_id.label("document_id"),
                    func.count(PageIndexNode.id).label("match_count"),
                )
                .join(PageIndexDocument, PageIndexDocument.id == PageIndexNode.document_id)
                .filter(and_(*base_filters))
                .filter(or_(*node_text_match_clauses))
                .group_by(PageIndexNode.document_id)
                .order_by(func.count(PageIndexNode.id).desc())
                .limit(max(limit * 40, 300))
                .all()
            )

            node_hit_map = {
                row.document_id: int(row.match_count or 0)
                for row in node_hit_rows
            }

            if node_hit_map:
                docs_from_nodes = (
                    db.query(PageIndexDocument)
                    .filter(and_(*base_filters))
                    .filter(PageIndexDocument.id.in_(list(node_hit_map.keys())))
                    .all()
                )
            else:
                docs_from_nodes = []

            docs_by_id: dict[str, PageIndexDocument] = {
                str(getattr(doc, "id")): doc for doc in doc_rows
            }
            for doc in docs_from_nodes:
                docs_by_id[str(getattr(doc, "id"))] = doc

            def score_doc(doc: PageIndexDocument) -> tuple[float, int]:
                title = (doc.doc_title or "").lower()
                description = (doc.doc_description or "").lower()

                score = 0.0
                for token in tokens:
                    if token in title:
                        score += 6.0
                    if token in description:
                        score += 2.5

                matched_nodes = node_hit_map.get(doc.id, 0)
                score += min(10.0, matched_nodes * 0.8)

                return score, matched_nodes

            ranked_items: list[PageIndexCandidateDocument] = []
            for doc in docs_by_id.values():
                score, matched_nodes = score_doc(doc)
                if score <= 0:
                    continue
                document_id = str(getattr(doc, "id"))
                file_id = str(getattr(doc, "file_id"))
                doc_knowledge_id = getattr(doc, "knowledge_id")
                doc_user_id = str(getattr(doc, "user_id"))
                doc_title = getattr(doc, "doc_title")
                doc_description = getattr(doc, "doc_description")
                ranked_items.append(
                    PageIndexCandidateDocument(
                        document_id=document_id,
                        file_id=file_id,
                        knowledge_id=(str(doc_knowledge_id) if doc_knowledge_id else None),
                        user_id=doc_user_id,
                        doc_title=(str(doc_title) if doc_title else None),
                        doc_description=(str(doc_description) if doc_description else None),
                        score=round(score, 4),
                        matched_nodes=matched_nodes,
                    )
                )

            ranked_items.sort(
                key=lambda item: (item.score, item.matched_nodes),
                reverse=True,
            )
            return PageIndexCandidateSearchResponse(query=query, items=ranked_items[:limit])


PageIndexes = PageIndexStore()
