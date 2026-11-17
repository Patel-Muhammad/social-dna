from __future__ import annotations

from pathlib import Path

from database.connection import get_connection
from database.schema import apply_migrations


def initialise_database(db_path: Path | str | None = None) -> int:
    """Create the database file and tables if needed. Returns the schema version."""
    conn = get_connection(db_path)
    try:
        return apply_migrations(conn)
    finally:
        conn.close()
