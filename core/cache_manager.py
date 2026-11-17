"""First-run / later-run orchestration: dataset -> database -> graph -> analysis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from analysis.pipeline import run_full_analysis
from core import config, dataset_loader
from core.graph_builder import NetworkData, load_network
from database.initialisation import initialise_database
from database.repositories import AnalysisRepository, DatasetRepository

Progress = Callable[[str, float], None]


@dataclass
class PipelineState:
    network: NetworkData
    dataset_id: int
    run_id: int
    db_path: Path
    seed: int
    resolution: float
    validation: dict


def prepare_dataset(raw_dir=None, processed_dir=None, db_path=None, progress: Progress | None = None):
    """Steps 1-5: ensure data exists, validate, persist metadata, build graph."""
    say = progress or (lambda m, f: None)
    raw_dir = Path(raw_dir or config.RAW_DIR)
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    config.ensure_directories()
    initialise_database(db_path)
    say("Checking for the dataset", 0.0)
    files = dataset_loader.ensure_raw_dataset(raw_dir)
    raw_hash = files.sha256()
    if dataset_loader.processed_is_current(processed_dir, raw_hash):
        dataset = dataset_loader.load_processed(processed_dir)
    else:
        say("Validating and processing the dataset", 0.02)
        dataset = dataset_loader.process_dataset(files, processed_dir)
    dataset_id = DatasetRepository(db_path).upsert(
        name=config.DATASET_NAME, source=config.DATASET_PAGE, version=config.DATASET_VERSION,
        users=dataset.report.total_users, edges=dataset.report.total_edges_clean,
        feature_count=dataset.report.feature_count, raw_sha256=raw_hash,
        processed_path=processed_dir, validation=dataset.report.to_dict())
    network = load_network(dataset, processed_dir)
    return network, dataset_id, dataset.report.to_dict()


def ensure_pipeline(*, seed: int = config.DEFAULT_SEED, resolution: float = config.DEFAULT_RESOLUTION,
                    force: bool = False, raw_dir=None, processed_dir=None, db_path=None,
                    progress: Progress | None = None) -> PipelineState:
    """Reuse stored results when they exist; otherwise compute and save them."""
    db_path = Path(db_path or config.DB_PATH)
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    network, dataset_id, validation = prepare_dataset(raw_dir, processed_dir, db_path, progress)
    repo = AnalysisRepository(db_path)
    run = None if force else repo.find_run(dataset_id, "louvain", seed, resolution)
    if run is None:
        run_id = run_full_analysis(network, repo, dataset_id, seed=seed, resolution=resolution,
                                   processed_dir=processed_dir, progress=progress or (lambda m, f: None))
    else:
        run_id = run["id"]
    return PipelineState(network, dataset_id, run_id, db_path, seed, resolution, validation)
