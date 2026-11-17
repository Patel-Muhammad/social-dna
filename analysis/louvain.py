"""Louvain community detection (python-louvain) with deterministic numbering."""
from __future__ import annotations

from dataclasses import dataclass

import community as community_louvain  # package name of python-louvain
import networkx as nx


@dataclass
class CommunityResult:
    algorithm: str
    partition: dict[int, int]       # original user id -> community number (1 = largest)
    modularity: float
    seed: int | None = None
    resolution: float | None = None

    @property
    def community_count(self) -> int:
        return len(set(self.partition.values()))

    def sizes(self) -> dict[int, int]:
        return community_sizes(self.partition)


def community_sizes(partition: dict[int, int]) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for c in partition.values():
        sizes[c] = sizes.get(c, 0) + 1
    return dict(sorted(sizes.items()))


def renumber_partition(partition: dict[int, int]) -> dict[int, int]:
    """Relabel communities 1..k by size (largest = 1); ties broken by smallest member id,
    so the same grouping always gets the same numbers."""
    groups: dict[int, list[int]] = {}
    for node, c in partition.items():
        groups.setdefault(c, []).append(node)
    ordered = sorted(groups.values(), key=lambda members: (-len(members), min(members)))
    return {node: i + 1 for i, members in enumerate(ordered) for node in members}


def partition_modularity(graph: nx.Graph, partition: dict[int, int]) -> float:
    groups: dict[int, set[int]] = {}
    for node, c in partition.items():
        groups.setdefault(c, set()).add(node)
    return float(nx.community.modularity(graph, list(groups.values())))


def run_louvain(graph: nx.Graph, seed: int = 42, resolution: float = 1.0) -> CommunityResult:
    """Louvain greedy modularity optimisation. `seed` makes the run reproducible."""
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot detect communities in an empty graph.")
    if graph.number_of_edges() == 0:
        raise ValueError("Graph has no edges; community detection is undefined.")
    raw = community_louvain.best_partition(graph, resolution=resolution, random_state=seed)
    partition = renumber_partition({int(n): int(c) for n, c in raw.items()})
    return CommunityResult("louvain", partition, partition_modularity(graph, partition), seed, resolution)
