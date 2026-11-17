"""Validate the dataset and write the processed files (cleaned edges, feature matrix)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config, dataset_loader  # noqa: E402
from core.data_validator import DatasetValidationError  # noqa: E402


def main() -> int:
    try:
        files = dataset_loader.ensure_raw_dataset(config.RAW_DIR)
        dataset = dataset_loader.process_dataset(files, config.PROCESSED_DIR)
    except (dataset_loader.DatasetDownloadError, DatasetValidationError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(dataset.report.to_dict(), indent=2))
    print(f"Processed files written to {config.PROCESSED_DIR}")
    return 0 if dataset.report.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
