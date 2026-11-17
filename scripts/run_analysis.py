"""Run (or re-run) the full analysis from the command line."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config  # noqa: E402
from core.cache_manager import ensure_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    p.add_argument("--resolution", type=float, default=config.DEFAULT_RESOLUTION)
    p.add_argument("--force", action="store_true", help="recompute even if results exist")
    a = p.parse_args()
    state = ensure_pipeline(seed=a.seed, resolution=a.resolution, force=a.force,
                            progress=lambda m, f: print(f"[{f:4.0%}] {m}", flush=True))
    print(f"Done. Louvain run id = {state.run_id} (database: {state.db_path})")


if __name__ == "__main__":
    main()
