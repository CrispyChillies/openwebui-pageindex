"""Add pageindex persistence tables

Revision ID: d6c4b8a9f102
Revises: b2c3d4e5f6a7
Create Date: 2026-03-24 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from open_webui.migrations.util import get_existing_tables


revision: str = "d6c4b8a9f102"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    existing_tables = set(get_existing_tables())

    if "pageindex_document" not in existing_tables:
        op.create_table(
            "pageindex_document",
            sa.Column("id", sa.String(), primary_key=True, nullable=False),
            sa.Column(
                "file_id",
                sa.Text(),
                sa.ForeignKey("file.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "knowledge_id",
                sa.Text(),
                sa.ForeignKey("knowledge.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("user_id", sa.Text(), nullable=False),
            sa.Column("doc_title", sa.Text(), nullable=True),
            sa.Column("doc_description", sa.Text(), nullable=True),
            sa.Column("source_type", sa.Text(), nullable=True),
            sa.Column("source_hash", sa.Text(), nullable=True),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("tree_json", sa.JSON(), nullable=True),
            sa.Column("generated_at", sa.BigInteger(), nullable=True),
            sa.Column("index_options", sa.JSON(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
        )

        op.create_index(
            "idx_pageindex_document_file_id", "pageindex_document", ["file_id"]
        )
        op.create_index(
            "idx_pageindex_document_knowledge_id",
            "pageindex_document",
            ["knowledge_id"],
        )
        op.create_index(
            "idx_pageindex_document_user_id", "pageindex_document", ["user_id"]
        )
        op.create_index(
            "idx_pageindex_document_status", "pageindex_document", ["status"]
        )
        op.create_index(
            "idx_pageindex_document_source_hash",
            "pageindex_document",
            ["source_hash"],
        )
        op.create_index(
            "idx_pageindex_document_file_status",
            "pageindex_document",
            ["file_id", "status"],
        )

    if "pageindex_node" not in existing_tables:
        op.create_table(
            "pageindex_node",
            sa.Column("id", sa.String(), primary_key=True, nullable=False),
            sa.Column(
                "document_id",
                sa.Text(),
                sa.ForeignKey("pageindex_document.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("node_id", sa.Text(), nullable=False),
            sa.Column("parent_node_id", sa.Text(), nullable=True),
            sa.Column("depth", sa.BigInteger(), nullable=False),
            sa.Column("title", sa.Text(), nullable=True),
            sa.Column("summary", sa.Text(), nullable=True),
            sa.Column("start_index", sa.BigInteger(), nullable=True),
            sa.Column("end_index", sa.BigInteger(), nullable=True),
            sa.Column("has_children", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.UniqueConstraint(
                "document_id",
                "node_id",
                name="uq_pageindex_node_document_node",
            ),
        )

        op.create_index(
            "idx_pageindex_node_document_id", "pageindex_node", ["document_id"]
        )
        op.create_index("idx_pageindex_node_node_id", "pageindex_node", ["node_id"])
        op.create_index(
            "idx_pageindex_node_parent_node_id", "pageindex_node", ["parent_node_id"]
        )
        op.create_index(
            "idx_pageindex_node_document_parent",
            "pageindex_node",
            ["document_id", "parent_node_id"],
        )
        op.create_index(
            "idx_pageindex_node_document_depth",
            "pageindex_node",
            ["document_id", "depth"],
        )


def downgrade() -> None:
    op.drop_index("idx_pageindex_node_document_depth", table_name="pageindex_node")
    op.drop_index("idx_pageindex_node_document_parent", table_name="pageindex_node")
    op.drop_index("idx_pageindex_node_parent_node_id", table_name="pageindex_node")
    op.drop_index("idx_pageindex_node_node_id", table_name="pageindex_node")
    op.drop_index("idx_pageindex_node_document_id", table_name="pageindex_node")
    op.drop_table("pageindex_node")

    op.drop_index("idx_pageindex_document_file_status", table_name="pageindex_document")
    op.drop_index("idx_pageindex_document_source_hash", table_name="pageindex_document")
    op.drop_index("idx_pageindex_document_status", table_name="pageindex_document")
    op.drop_index("idx_pageindex_document_user_id", table_name="pageindex_document")
    op.drop_index("idx_pageindex_document_knowledge_id", table_name="pageindex_document")
    op.drop_index("idx_pageindex_document_file_id", table_name="pageindex_document")
    op.drop_table("pageindex_document")
