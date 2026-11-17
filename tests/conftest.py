"""Shared fixtures. Synthetic data is used ONLY here, for unit tests."""
import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from analysis.pipeline import run_full_analysis
from core.graph_builder import NetworkData
from database.initialisation import initialise_database
from database.repositories import AnalysisRepository, DatasetRepository


def two_cliques() -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from((i, j) for i in range(5) for j in range(i + 1, 5))
    g.add_edges_from((i, j) for i in range(5, 10) for j in range(i + 1, 10))
    g.add_edge(4, 5)  # single bridge
    return g


@pytest.fixture
def cliques():
    return two_cliques()


def planted_network(seed: int = 1) -> NetworkData:
    """60 users in 3 planted groups with group-specific features (feature IDs start at 100)."""
    rng = np.random.default_rng(seed)
    g = nx.Graph()
    g.add_nodes_from(range(60))
    for grp in range(3):
        members = list(range(grp * 20, grp * 20 + 20))
        for i in members:
            for j in members:
                if i < j and rng.random() < 0.4:
                    g.add_edge(i, j)
    g.add_edges_from([(0, 20), (21, 40), (41, 1)])
    rows, cols = [], []
    for u in range(60):
        grp = u // 20
        for f in range(grp * 10, grp * 10 + 8):
            rows.append(u); cols.append(f)
        rows.append(u); cols.append(29)  # a universally liked feature
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(60, 30))
    feature_ids = np.arange(30) + 100
    node_ids = np.arange(60)
    countries = pd.Series(rng.integers(0, 3, 60), index=node_ids)
    return NetworkData(g, X, node_ids, feature_ids, countries, raw_hash="synthetic")


@pytest.fixture
def network():
    return planted_network()


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.sqlite"
    initialise_database(path)
    return path


@pytest.fixture
def analysed(network, db_path, tmp_path):
    """Run the full pipeline on the synthetic network into a temp DB."""
    ds = DatasetRepository(db_path).upsert(name="synthetic", source="test", version="0", users=60, edges=network.graph.number_of_edges(),
                                           feature_count=30, raw_sha256="synthetic", processed_path=tmp_path, validation={})
    repo = AnalysisRepository(db_path)
    run_id = run_full_analysis(network, repo, ds, seed=7, resolution=1.0, processed_dir=tmp_path)
    return repo, run_id, network


def write_raw(directory, edges, features, targets=None):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(edges, columns=["node_1", "node_2"]).to_csv(directory / "lastfm_asia_edges.csv", index=False)
    (directory / "lastfm_asia_features.json").write_text(json.dumps({str(k): v for k, v in features.items()}))
    if targets is not None:
        pd.DataFrame({"id": list(targets), "target": list(targets.values())}).to_csv(
            directory / "lastfm_asia_target.csv", index=False)
    return directory
