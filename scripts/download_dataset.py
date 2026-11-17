"""Download the LastFM Asia dataset (skips the download if it is already present)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config, dataset_loader  # noqa: E402


def main() -> int:
    try:
        files = dataset_loader.ensure_raw_dataset(config.RAW_DIR)
    except dataset_loader.DatasetDownloadError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Dataset ready in {config.RAW_DIR}")
    for f in (files.edges, files.features, files.target):
        print(f"  - {f.name if f else '(missing optional target file)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
