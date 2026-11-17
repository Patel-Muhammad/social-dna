"""Create the SQLite database (safe to run repeatedly)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config  # noqa: E402
from database.initialisation import initialise_database  # noqa: E402

if __name__ == "__main__":
    version = initialise_database(config.DB_PATH)
    print(f"Database ready at {config.DB_PATH} (schema version {version})")
