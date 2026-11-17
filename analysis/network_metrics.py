"""Global and per-user network metrics (NetworkX)."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from core import config


def global_metrics(graph: nx.Graph) -> dict[str, float]:
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    components = list(nx.connected_components(graph))
    degrees = np.array([d for _, d in graph.degree()])
    return {
        "users": float(n),
        "edges": float(m),
        "density": float(nx.density(graph)),
        "average_degree": float(2 * m / n) if n else 0.0,
        "max_degree": float(degrees.max()) if n else 0.0,
        "median_degree": float(np.median(degrees)) if n else 0.0,
        "average_clustering": float(nx.average_clustering(graph)),
        "transitivity": float(nx.transitivity(graph)),
        "connected_components": float(len(components)),
        "largest_component_size": float(max((len(c) for c in components), default=0)),
        "isolated_nodes": float(sum(1 for _ in nx.isolates(graph))),
    }


def centrality_table(graph: nx.Graph, seed: int = config.DEFAULT_SEED,
                     betweenness_k: int | None = config.BETWEENNESS_SAMPLE_K) -> tuple[pd.DataFrame, dict[str, str]]:
    """Per-user centralities. Betweenness is a sampled approximation when k < n
    (exact is O(n*m)); everything else is exact. Returns (table, notes)."""
    n = graph.number_of_nodes()
    k = None if betweenness_k is None or betweenness_k >= n else betweenness_k
    betweenness = nx.betweenness_centrality(graph, k=k, seed=seed, normalized=True)
    table = pd.DataFrame({
        "degree": dict(graph.degree()),
        "degree_centrality": nx.degree_centrality(graph),
        "betweenness_centrality": betweenness,
        "closeness_centrality": nx.closeness_centrality(graph),
        "pagerank": nx.pagerank(graph),
    })
    table.index.name = "user_id"
    notes = {
        "betweenness_centrality": (
            f"Approximation from {k} sampled source nodes (seed {seed})." if k else "Exact."
        ),
        "closeness_centrality": "Exact (Wasserman-Faust normalisation for disconnected graphs).",
        "pagerank": "Exact power iteration, damping 0.85.",
    }
    return table, notes
