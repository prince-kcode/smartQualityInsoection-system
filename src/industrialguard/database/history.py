"""SQLite persistence for local inspection metadata."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATABASE = PROJECT_ROOT / "artifacts" / "industrialguard.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    product_category TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    model_name TEXT NOT NULL,
    anomaly_score REAL NOT NULL,
    inspection_status TEXT NOT NULL CHECK (inspection_status IN ('NORMAL', 'DEFECTIVE')),
    processing_time REAL NOT NULL,
    input_image_path TEXT NOT NULL,
    heatmap_path TEXT NOT NULL,
    overlay_path TEXT NOT NULL,
    report_path TEXT NOT NULL
)
"""


def _connect(database_path: Path | None = None) -> sqlite3.Connection:
    path = Path(database_path) if database_path else DEFAULT_DATABASE
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: Path | None = None) -> None:
    """Create the inspection table if it does not exist."""

    with _connect(database_path) as connection:
        connection.execute(SCHEMA)


def add_inspection(
    *,
    inspection_id: str,
    created_at: str | None = None,
    product_category: str,
    original_filename: str,
    anomaly_score: float,
    inspection_status: str,
    processing_time: float,
    input_image_path: str,
    heatmap_path: str,
    overlay_path: str,
    report_path: str,
    model_name: str = "PatchCore (Wide ResNet-50)",
    database_path: Path | None = None,
) -> str:
    """Save one completed inspection and return its UTC timestamp."""

    timestamp = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO inspections (
                inspection_id, created_at, product_category, original_filename,
                model_name, anomaly_score, inspection_status, processing_time,
                input_image_path, heatmap_path, overlay_path, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inspection_id,
                timestamp,
                product_category,
                Path(original_filename).name,
                model_name,
                float(anomaly_score),
                inspection_status,
                float(processing_time),
                input_image_path,
                heatmap_path,
                overlay_path,
                report_path,
            ),
        )
    return timestamp


def list_inspections(
    limit: int = 100,
    database_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the newest inspection metadata rows first."""

    if limit < 1:
        raise ValueError("limit must be at least 1")
    initialize_database(database_path)
    with _connect(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM inspections ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_inspection(
    inspection_id: str,
    database_path: Path | None = None,
) -> dict[str, Any] | None:
    """Look up one inspection by its ID."""

    initialize_database(database_path)
    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM inspections WHERE inspection_id = ?",
            (inspection_id,),
        ).fetchone()
    return dict(row) if row else None


def inspection_counts(database_path: Path | None = None) -> dict[str, int]:
    """Return total, normal, and defective counts over the full history."""

    initialize_database(database_path)
    with _connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN inspection_status = 'NORMAL' THEN 1 ELSE 0 END) AS normal,
                SUM(CASE WHEN inspection_status = 'DEFECTIVE' THEN 1 ELSE 0 END) AS defective
            FROM inspections
            """,
        ).fetchone()
    return {key: int(row[key] or 0) for key in ("total", "normal", "defective")}
