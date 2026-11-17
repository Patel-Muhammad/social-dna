"""Dataset acquisition, parsing and processing for LastFM Asia.

Workflow: locate raw files -> (download if missing) -> parse -> validate ->
save processed artefacts (cleaned edges, sparse feature matrix, countries,
validation report). Processed files are reused on later launches.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import sparse

from core import config
from core.data_validator import DatasetValidationError, ValidationReport, validate_dataset


class DatasetDownloadError(RuntimeError):
    """Raised when the dataset cannot be downloaded; message explains manual setup."""


MANUAL_HELP = (
    "Download 'lastfm_asia.zip' from {page} (or the UCI mirror) in a browser, then either "
    "put the zip file or its three files ({files}) into the folder:\n  {raw}\n"
    "and start the application again."
)


def manual_instructions(raw_dir: Path) -> str:
    return MANUAL_HELP.format(
        page=config.DATASET_PAGE, files=", ".join(config.DATASET_FILES.values()), raw=raw_dir
    )


@dataclass
class RawFiles:
    edges: Path
    features: Path
    target: Path | None

    def sha256(self) -> str:
        """Combined hash of the raw files; identifies the dataset version."""
        digest = hashlib.sha256()
        for path in (self.edges, self.features, self.target):
            if path is not None and path.exists():
                digest.update(path.read_bytes())
        return digest.hexdigest()


@dataclass
class ProcessedDataset:
    node_ids: np.ndarray            # sorted original user IDs; row i of the matrix = node_ids[i]
    feature_ids: np.ndarray         # original artist-feature IDs; column j = feature_ids[j]
    features: sparse.csr_matrix     # binary users x features
    edges: pd.DataFrame             # cleaned undirected edges (node_1 < node_2)
    countries: pd.Series            # user_id -> anonymised country code (descriptive only)
    report: ValidationReport
    raw_hash: str


# --- locating / downloading ---------------------------------------------------------
def locate_raw_files(raw_dir: Path) -> RawFiles | None:
    """Find the three expected files anywhere under raw_dir (zip or extracted)."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return None
    found: dict[str, Path] = {}
    for key, name in config.DATASET_FILES.items():
        matches = sorted(raw_dir.rglob(name))
        if matches:
            found[key] = matches[0]
    if "edges" in found and "features" in found:
        return RawFiles(found["edges"], found["features"], found.get("target"))
    return None


def _extract_zip(content: bytes, raw_dir: Path) -> None:
    """Extract only the expected files (by basename), ignoring any path in the zip."""
    wanted = set(config.DATASET_FILES.values())
    extracted = 0
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if name in wanted:
                (raw_dir / name).write_bytes(archive.read(member))
                extracted += 1
    if extracted < 2:
        raise DatasetDownloadError("Downloaded archive did not contain the expected dataset files.")


def download_dataset(raw_dir: Path, urls: list[str] | None = None, timeout: int = 60) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls or config.DATASET_URLS:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            _extract_zip(response.content, raw_dir)
            return
        except (requests.RequestException, zipfile.BadZipFile, DatasetDownloadError) as exc:
            errors.append(f"{url}: {exc}")
    raise DatasetDownloadError(
        "Could not download the LastFM Asia dataset.\n"
        + "\n".join(f"  - {e}" for e in errors)
        + "\n\n"
        + manual_instructions(raw_dir)
    )


def ensure_raw_dataset(raw_dir: Path | None = None, urls: list[str] | None = None) -> RawFiles:
    """Return raw files, downloading them first only if they are missing."""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Support a manually supplied zip.
    files = locate_raw_files(raw_dir)
    if files is None:
        for zip_path in sorted(raw_dir.glob("*.zip")):
            try:
                _extract_zip(zip_path.read_bytes(), raw_dir)
            except (zipfile.BadZipFile, DatasetDownloadError):
                continue
        files = locate_raw_files(raw_dir)
    if files is None:
        download_dataset(raw_dir, urls)
        files = locate_raw_files(raw_dir)
    if files is None:
        raise DatasetDownloadError("Dataset files not found after download.\n" + manual_instructions(raw_dir))
    return files


# --- parsing ------------------------------------------------------------------------
def load_raw_tables(files: RawFiles) -> tuple[pd.DataFrame, dict[int, list[int]], pd.DataFrame | None]:
    try:
        edges = pd.read_csv(files.edges)
        with open(files.features, encoding="utf-8") as handle:
            raw_features = json.load(handle)
        features = {int(user): [int(f) for f in feats] for user, feats in raw_features.items()}
        targets = pd.read_csv(files.target) if files.target is not None else None
    except (ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise DatasetValidationError(f"Dataset files are malformed: {exc}") from exc
    return edges, features, targets


def build_feature_matrix(
    features: dict[int, list[int]], node_ids: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Binary user x feature matrix using the dataset's own IDs (nothing invented)."""
    if not features:
        raise DatasetValidationError("No feature records available.")
    feature_ids = np.array(sorted({f for feats in features.values() for f in feats}), dtype=np.int64)
    column = {f: j for j, f in enumerate(feature_ids)}
    rows, cols = [], []
    for i, user in enumerate(node_ids):
        for f in set(features.get(int(user), [])):
            rows.append(i)
            cols.append(column[f])
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(node_ids), len(feature_ids)),
    )
    return matrix, feature_ids


# --- processing / persistence ---------------------------------------------------------
PROCESSED_NAMES = {
    "edges": "edges_clean.csv",
    "features": "features.npz",
    "countries": "countries.csv",
    "report": "validation_report.json",
    "meta": "processed_meta.json",
}


def process_dataset(files: RawFiles, processed_dir: Path | None = None) -> ProcessedDataset:
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    edges_raw, features, targets = load_raw_tables(files)
    clean, report = validate_dataset(edges_raw, features, targets)
    if not report.is_valid:
        raise DatasetValidationError("; ".join(report.errors))

    node_ids = np.array(sorted(features.keys()), dtype=np.int64)
    matrix, feature_ids = build_feature_matrix(features, node_ids)
    if targets is not None:
        countries = targets.set_index("id")["target"].astype(int).reindex(node_ids)
    else:
        countries = pd.Series(np.nan, index=node_ids, name="target")
    countries.index.name = "user_id"

    raw_hash = files.sha256()
    clean.to_csv(processed_dir / PROCESSED_NAMES["edges"], index=False)
    sparse.save_npz(processed_dir / PROCESSED_NAMES["features"], matrix)
    np.savez(processed_dir / "index_arrays.npz", node_ids=node_ids, feature_ids=feature_ids)
    countries.rename("country_code").to_csv(processed_dir / PROCESSED_NAMES["countries"])
    (processed_dir / PROCESSED_NAMES["report"]).write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    (processed_dir / PROCESSED_NAMES["meta"]).write_text(
        json.dumps({"raw_hash": raw_hash}), encoding="utf-8"
    )
    return ProcessedDataset(node_ids, feature_ids, matrix, clean, countries, report, raw_hash)


def processed_is_current(processed_dir: Path | None = None, raw_hash: str | None = None) -> bool:
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    needed = [processed_dir / n for n in PROCESSED_NAMES.values()] + [processed_dir / "index_arrays.npz"]
    if not all(p.exists() for p in needed):
        return False
    if raw_hash is None:
        return True
    try:
        meta = json.loads((processed_dir / PROCESSED_NAMES["meta"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("raw_hash") == raw_hash


def load_processed(processed_dir: Path | None = None) -> ProcessedDataset:
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    edges = pd.read_csv(processed_dir / PROCESSED_NAMES["edges"])
    matrix = sparse.load_npz(processed_dir / PROCESSED_NAMES["features"]).tocsr()
    arrays = np.load(processed_dir / "index_arrays.npz")
    countries = pd.read_csv(processed_dir / PROCESSED_NAMES["countries"], index_col="user_id")["country_code"]
    report_dict = json.loads((processed_dir / PROCESSED_NAMES["report"]).read_text(encoding="utf-8"))
    report_dict.pop("is_valid", None)
    report = ValidationReport(**report_dict)
    meta = json.loads((processed_dir / PROCESSED_NAMES["meta"]).read_text(encoding="utf-8"))
    return ProcessedDataset(
        arrays["node_ids"], arrays["feature_ids"], matrix, edges, countries, report, meta["raw_hash"]
    )
