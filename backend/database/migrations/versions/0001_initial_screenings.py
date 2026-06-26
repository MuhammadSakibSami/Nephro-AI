"""initial screenings table

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-27
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# JSON everywhere; JSONB (indexable binary) on PostgreSQL.
JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "screenings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("input_features", JSONType, nullable=False),
        sa.Column("n_missing_features", sa.Integer(), nullable=True),
        sa.Column("risk_class", sa.String(length=16), nullable=True),
        sa.Column("risk_class_display", sa.String(length=64), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("probabilities", JSONType, nullable=True),
        sa.Column("prediction", JSONType, nullable=True),
        sa.Column("guidance", JSONType, nullable=True),
        sa.Column("safety_status", sa.String(length=16), nullable=True),
        sa.Column("llm_mode", sa.String(length=32), nullable=True),
        sa.Column("model_spec", sa.String(length=128), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_screenings_created_at", "screenings", ["created_at"])
    op.create_index("ix_screenings_source", "screenings", ["source"])
    op.create_index("ix_screenings_risk_class", "screenings", ["risk_class"])


def downgrade() -> None:
    op.drop_index("ix_screenings_risk_class", table_name="screenings")
    op.drop_index("ix_screenings_source", table_name="screenings")
    op.drop_index("ix_screenings_created_at", table_name="screenings")
    op.drop_table("screenings")
