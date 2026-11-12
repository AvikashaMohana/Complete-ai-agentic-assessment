"""create users, tokens and threads

Revision ID: 7d1c1e648c03
Revises:
Create Date: 2026-11-10
"""

from alembic import op
import sqlalchemy as sa

revision = "7d1c1e648c03"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("username", sa.String(), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("created", sa.Float(), nullable=False),
    )

    op.create_table(
        "tokens",
        sa.Column("token", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("created", sa.Float(), nullable=False),
    )

    op.create_table(
        "threads",
        sa.Column("thread_id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("threads")
    op.drop_table("tokens")
    op.drop_table("users")