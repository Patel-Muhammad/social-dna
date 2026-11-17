"""Greedy modularity (Clauset-Newman-Moore) as the comparison algorithm. Deterministic."""
from __future__ import annotations

import networkx as nx

from analysis.louvain import CommunityResult, partition_modularity, renumber_partition


def run_greedy_modularity(graph: nx.Graph, resolution: float = 1.0) -> CommunityResult:
    if graph.number_of_edges() == 0:
        raise ValueError("Graph has no edges; community detection is undefined.")
    groups = nx.community.greedy_modularity_communities(graph, resolution=resolution)
    raw = {int(n): i for i, members in enumerate(groups) for n in members}
    partition = renumber_partition(raw)
    return CommunityResult("greedy_modularity", partition, partition_modularity(graph, partition), None, resolution)
