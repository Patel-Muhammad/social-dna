"""Validation and cleaning of the raw LastFM Asia files.

Nothing here invents data: problems are counted and reported, edges that
cannot be used (self-loops, duplicates, unknown users) are dropped, and
every drop is recorded in the validation report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from core import config


class DatasetValidationError(ValueError):
    """Raised when the dataset is unusable (e.g. empty or malformed)."""


@dataclass
class ValidationReport:
    total_users: int = 0
    total_edges_raw: int = 0
    total_edges_clean: int = 0
    duplicate_edges: int = 0
    self_loops: int = 0
    edges_with_unknown_users: int = 0
    isolated_nodes: int = 0
    users_without_features: int = 0
    users_missing_from_features: int = 0
    feature_count: int = 0
    min_features_per_user: int = 0
    max_features_per_user: int = 0
    mean_features_per_user: float = 0.0
    connected_components: int = 0
    largest_component_size: int = 0
    target_rows: int = 0
    target_ids_consistent: bool = True
    matches_documented_counts: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        data = asdict(self)
        data["is_valid"] = self.is_valid
        return data


def clean_edges(edges: pd.DataFrame, valid_users: set[int]) -> tuple[pd.DataFrame, dict]:
    """Return undirected, de-duplicated edges between known users only."""
    if not {"node_1", "node_2"}.issubset(edges.columns):
        raise DatasetValidationError("Edge file must have columns node_1,node_2.")
    work = edges[["node_1", "node_2"]].copy()
    work = work.dropna().astype("int64")
    raw_count = len(work)

    self_loop_mask = work["node_1"] == work["node_2"]
    self_loops = int(self_loop_mask.sum())
    work = work[~self_loop_mask]

    # Canonical orientation (small id first) so (a,b) and (b,a) collapse.
    low = work[["node_1", "node_2"]].min(axis=1)
    high = work[["node_1", "node_2"]].max(axis=1)
    work = pd.DataFrame({"node_1": low, "node_2": high})
    before = len(work)
    work = work.drop_duplicates()
    duplicates = before - len(work)

    known = work["node_1"].isin(valid_users) & work["node_2"].isin(valid_users)
    unknown = int((~known).sum())
    work = work[known].sort_values(["node_1", "node_2"]).reset_index(drop=True)

    stats = {
        "raw": raw_count,
        "self_loops": self_loops,
        "duplicates": duplicates,
        "unknown_users": unknown,
    }
    return work, stats


def validate_dataset(
    edges: pd.DataFrame,
    features: dict[int, list[int]],
    targets: pd.DataFrame | None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate raw tables and return (clean_edges, report)."""
    report = ValidationReport()
    if len(features) == 0:
        raise DatasetValidationError("The feature file contains no users (empty dataset).")
    if len(edges) == 0:
        raise DatasetValidationError("The edge file contains no edges (empty dataset).")

    user_ids = set(features.keys())
    report.total_users = len(user_ids)
    clean, stats = clean_edges(edges, user_ids)
    report.total_edges_raw = stats["raw"]
    report.self_loops = stats["self_loops"]
    report.duplicate_edges = stats["duplicates"]
    report.edges_with_unknown_users = stats["unknown_users"]
    report.total_edges_clean = len(clean)

    # Users that appear in edges but have no feature record would be "missing".
    edge_users = set(edges["node_1"].astype(int)) | set(edges["node_2"].astype(int))
    report.users_missing_from_features = len(edge_users - user_ids)

    # Feature statistics.
    lengths = np.array([len(v) for v in features.values()])
    report.users_without_features = int((lengths == 0).sum())
    report.min_features_per_user = int(lengths.min())
    report.max_features_per_user = int(lengths.max())
    report.mean_features_per_user = float(lengths.mean())
    report.feature_count = len({f for v in features.values() for f in v})

    # Connectivity (computed on a sparse adjacency matrix; no graph needed yet).
    order = np.array(sorted(user_ids))
    index = {u: i for i, u in enumerate(order)}
    n = len(order)
    rows = clean["node_1"].map(index).to_numpy()
    cols = clean["node_2"].map(index).to_numpy()
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adjacency = adjacency + adjacency.T
    n_components, labels = connected_components(adjacency, directed=False)
    report.connected_components = int(n_components)
    report.largest_component_size = int(np.bincount(labels).max())
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    report.isolated_nodes = int((degrees == 0).sum())

    # Target (country) file consistency; used for description only.
    if targets is not None:
        report.target_rows = len(targets)
        report.target_ids_consistent = set(targets["id"].astype(int)) == user_ids
        if not report.target_ids_consistent:
            report.warnings.append("Target file user IDs differ from feature file user IDs.")

    report.matches_documented_counts = (
        report.total_users == config.DOCUMENTED_USERS
        and report.total_edges_clean == config.DOCUMENTED_EDGES
    )
    if not report.matches_documented_counts:
        report.warnings.append(
            f"Counts ({report.total_users} users, {report.total_edges_clean} edges) differ from "
            f"the documented SNAP numbers ({config.DOCUMENTED_USERS}, {config.DOCUMENTED_EDGES})."
        )
    if report.users_without_features:
        report.warnings.append(f"{report.users_without_features} users have no liked-artist features.")
    if report.users_missing_from_features:
        report.warnings.append(
            f"{report.users_missing_from_features} edge endpoints have no feature record; "
            "edges touching them were dropped."
        )
    if report.total_edges_clean == 0:
        report.errors.append("No valid edges remain after cleaning.")
    return clean, report
