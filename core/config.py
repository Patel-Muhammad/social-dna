"""Central configuration: paths, dataset constants and environment settings.

Everything that might differ between machines (paths, API key, model name)
is resolved here so the rest of the code never builds paths by hand.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env if present (never committed). Real environment variables win.
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR = Path(os.getenv("SOCIAL_DNA_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "evaluation"
DB_PATH = Path(os.getenv("SOCIAL_DNA_DB_PATH", PROCESSED_DIR / "social_dna.sqlite"))

# --- Dataset -----------------------------------------------------------------
DATASET_NAME = "LastFM Asia Social Network"
DATASET_VERSION = "SNAP feather-lastfm-social (collected March 2020)"
DATASET_PAGE = "https://snap.stanford.edu/data/feather-lastfm-social.html"
DATASET_URLS = [
    # Primary: official SNAP archive.
    "https://snap.stanford.edu/data/lastfm_asia.zip",
    # Fallback: UCI repository copy of the same data.
    "https://archive.ics.uci.edu/static/public/599/lastfm+asia+social+network.zip",
]
DATASET_FILES = {
    "edges": "lastfm_asia_edges.csv",
    "features": "lastfm_asia_features.json",
    "target": "lastfm_asia_target.csv",
}
# Numbers documented by SNAP; used only as a sanity check, never as results.
DOCUMENTED_USERS = 7624
DOCUMENTED_EDGES = 27806

# --- Analysis defaults ---------------------------------------------------------
DEFAULT_SEED = int(os.getenv("SOCIAL_DNA_SEED", "42"))
DEFAULT_RESOLUTION = 1.0
BETWEENNESS_SAMPLE_K = int(os.getenv("SOCIAL_DNA_BETWEENNESS_K", "1000"))
TOP_FEATURES_PER_COMMUNITY = 30
STABILITY_SEEDS = 10

# --- Gemini ----------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def get_gemini_api_key() -> str | None:
    """Return the Gemini key from the environment, or None if unset/placeholder."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key.lower().startswith("your_") or key == "changeme":
        return None
    return key


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, EVALUATION_DIR):
        directory.mkdir(parents=True, exist_ok=True)
