"""Graph construction, validation, caching and lookups (NetworkX)."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from core import config
from core.dataset_loader import ProcessedDataset


class UnknownUserError(KeyError):
    pass


@dataclass
class NetworkData:
    """Everything the analysis needs, all keyed by the ORIGINAL dataset user IDs."""

    graph: nx.Graph
    features: sparse.csr_matrix      # row i <-> node_ids[i]
    node_ids: np.ndarray
    feature_ids: np.ndarray
    countries: pd.Series
    raw_hash: str

    def __post_init__(self) -> None:
        self.row_of = {int(u): i for i, u in enumerate(self.node_ids)}

    # --- lookups -----------------------------------------------------------------
    def has_user(self, user_id: int) -> bool:
        return int(user_id) in self.graph

    def user_features(self, user_id: int) -> list[int]:
        """Original artist-feature IDs liked by the user."""
        if not self.has_user(user_id):
            raise UnknownUserError(f"User {user_id} is not in the dataset.")
        row = self.features[self.row_of[int(user_id)]]
        return [int(self.feature_ids[j]) for j in row.indices]

    def neighbours(self, user_id: int) -> list[int]:
        if not self.has_user(user_id):
            raise UnknownUserError(f"User {user_id} is not in the dataset.")
        return sorted(int(n) for n in self.graph.neighbors(int(user_id)))

    def community_of(self, user_id: int, partition: dict[int, int]) -> int:
        if int(user_id) not in partition:
            raise UnknownUserError(f"User {user_id} has no community assignment.")
        return partition[int(user_id)]

    def members_of(self, community: int, partition: dict[int, int]) -> list[int]:
        return sorted(u for u, c in partition.items() if c == community)


def build_graph(dataset: ProcessedDataset) -> nx.Graph:
    """Undirected graph: node = user, edge = friendship; features attached to nodes."""
    graph = nx.Graph(name="lastfm_asia")
    graph.add_nodes_from(int(u) for u in dataset.node_ids)
    graph.add_edges_from(
        zip(dataset.edges["node_1"].astype(int), dataset.edges["node_2"].astype(int))
    )
    counts = np.asarray(dataset.features.sum(axis=1)).ravel().astype(int)
    nx.set_node_attributes(graph, {int(u): int(c) for u, c in zip(dataset.node_ids, counts)}, "n_features")
    country = dataset.countries
    nx.set_node_attributes(
        graph,
        {int(u): (None if pd.isna(country.get(u)) else int(country.get(u))) for u in dataset.node_ids},
        "country_code",
    )
    return graph


def validate_graph(graph: nx.Graph) -> dict:
    """Independent structural check performed on the built graph object."""
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "self_loops": nx.number_of_selfloops(graph),
        "isolated_nodes": sum(1 for _ in nx.isolates(graph)),
        "components": nx.number_connected_components(graph),
        "is_directed": graph.is_directed(),
    }


def _cache_path(processed_dir: Path) -> Path:
    return Path(processed_dir) / "graph_cache.pkl"


def load_network(dataset: ProcessedDataset, processed_dir: Path | None = None) -> NetworkData:
    """Build (or load from the pickle cache) the graph for this exact raw dataset."""
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    cache = _cache_path(processed_dir)
    graph = None
    if cache.exists():
        try:
            with open(cache, "rb") as handle:
                cached = pickle.load(handle)
            if cached.get("raw_hash") == dataset.raw_hash:
                graph = cached["graph"]
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, KeyError):
            graph = None  # stale or unreadable cache: rebuild below
    if graph is None:
        graph = build_graph(dataset)
        try:
            with open(cache, "wb") as handle:
                pickle.dump({"raw_hash": dataset.raw_hash, "graph": graph}, handle)
        except OSError:
            pass  # caching is an optimisation only
    return NetworkData(
        graph=graph,
        features=dataset.features,
        node_ids=dataset.node_ids,
        feature_ids=dataset.feature_ids,
        countries=dataset.countries,
        raw_hash=dataset.raw_hash,
    )
