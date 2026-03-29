"""
DbSource Schema — SQLModel table for dynamic database collection sources.

Each row represents a database connection that will be polled on a schedule
to extract rows for MLOps fine-tuning datasets.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class DbSourceType(str, Enum):
    """Supported database engine types."""
    POSTGRESQL = "postgresql"
    MYSQL      = "mysql"
    SQLITE     = "sqlite"
    MONGODB    = "mongodb"


class DbSourceStatus(str, Enum):
    """Result of the last collection run."""
    SUCCESS  = "success"
    ERROR    = "error"
    NO_DATA  = "no_data"
    PENDING  = "pending"


# ---------------------------------------------------------------------------
# Table model
# ---------------------------------------------------------------------------

class DbSource(SQLModel, table=True):
    """Persisted configuration for a database collection source."""

    __tablename__ = "db_source"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
    )
    user_id: uuid.UUID = Field(index=True)

    # --- Identity ---
    name: str = Field(index=True, description="Human-readable label, e.g. 'SCADA_Sensors'")
    description: Optional[str] = Field(default=None)

    # --- Connection ---
    db_type: DbSourceType = Field(description="Database engine type")
    connection_string_enc: str = Field(
        description="Fernet-encrypted connection URI. Never returned in API responses."
    )

    # --- Query ---
    query: str = Field(
        description=(
            "For SQL: SELECT statement. "
            "For MongoDB: JSON string with keys 'collection', 'filter' (optional), 'limit' (optional)."
        )
    )

    # --- Scheduling ---
    cron_expression: str = Field(
        default="0 0 * * *",
        description="Standard 5-field cron expression for APScheduler (e.g. '0 0 * * *' = daily).",
    )
    is_enabled: bool = Field(default=True)

    # --- MLOps Context ---
    tenant_id: str = Field(default="aura_tenant_01")
    sector: str = Field(default="Industrial", description="Used to enrich ShareGPT dataset context tag")
    domain: str = Field(default="General", description="Used to enrich ShareGPT dataset context tag")

    # --- Run Metadata ---
    last_run_at: Optional[datetime] = Field(default=None)
    last_run_status: Optional[DbSourceStatus] = Field(default=None)
    last_run_rows: Optional[int] = Field(default=None, description="Number of rows fetched in last run")
    last_error_detail: Optional[str] = Field(default=None)

    # --- MLOps Auto-Trigger ---
    training_threshold_rows: Optional[int] = Field(
        default=None, 
        description="Trigger training when accumulated rows exceed this threshold"
    )
    accumulated_rows: int = Field(
        default=0, 
        description="Rows collected since last training trigger"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )


# ---------------------------------------------------------------------------
# API models (Pydantic / no table=True)
# ---------------------------------------------------------------------------

class DbSourceCreate(SQLModel):
    name: str
    description: Optional[str] = None
    db_type: DbSourceType
    connection_string: str  # Plain text — encrypted before storage
    query: str
    cron_expression: str = "0 0 * * *"
    is_enabled: bool = True
    tenant_id: str = "aura_tenant_01"
    sector: str = "Industrial"
    domain: str = "General"
    training_threshold_rows: Optional[int] = None
    accumulated_rows: int = 0


class DbSourceUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    db_type: Optional[DbSourceType] = None
    connection_string: Optional[str] = None  # Plain — re-encrypted on update
    query: Optional[str] = None
    cron_expression: Optional[str] = None
    is_enabled: Optional[bool] = None
    tenant_id: Optional[str] = None
    sector: Optional[str] = None
    domain: Optional[str] = None
    training_threshold_rows: Optional[int] = None
    accumulated_rows: Optional[int] = None


class DbSourceRead(SQLModel):
    """Public response — connection_string_enc is intentionally excluded."""
    id: uuid.UUID
    name: str
    description: Optional[str]
    db_type: DbSourceType
    query: str
    cron_expression: str
    is_enabled: bool
    tenant_id: str
    sector: str
    domain: str
    last_run_at: Optional[datetime]
    last_run_status: Optional[DbSourceStatus]
    last_run_rows: Optional[int]
    last_error_detail: Optional[str]
    training_threshold_rows: Optional[int]
    accumulated_rows: int
    created_at: datetime
