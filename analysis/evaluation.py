"""Community-detection evaluation: stability across seeds and agreement between algorithms.

NMI/ARI here compare two PARTITIONS with each other (agreement/stability). They are not
accuracy against ground truth: this dataset has no ground-truth communities.
"""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from analysis.louvain import run_louvain


def labels_for(partition_a: dict[int, int], partition_b: dict[int, int]) -> tuple[list[int], list[int]]:
    nodes = sorted(partition_a)
    if set(nodes) != set(partition_b):
        raise ValueError("Partitions cover different users; cannot compare.")
    return [partition_a[n] for n in nodes], [partition_b[n] for n in nodes]


def partition_agreement(a: dict[int, int], b: dict[int, int]) -> dict[str, float]:
    la, lb = labels_for(a, b)
    return {"nmi": float(normalized_mutual_info_score(la, lb)), "ari": float(adjusted_rand_score(la, lb))}


def louvain_stability(graph: nx.Graph, seeds: list[int], resolution: float = 1.0) -> dict:
    """Run Louvain with several seeds and compare every pair of results."""
    runs = [run_louvain(graph, seed=s, resolution=resolution) for s in seeds]
    pair_nmi, pair_ari = [], []
    for a, b in itertools.combinations(runs, 2):
        ag = partition_agreement(a.partition, b.partition)
        pair_nmi.append(ag["nmi"])
        pair_ari.append(ag["ari"])
    mods = [r.modularity for r in runs]
    counts = [r.community_count for r in runs]
    return {
        "seeds": list(seeds),
        "pairwise_nmi_mean": float(np.mean(pair_nmi)), "pairwise_nmi_min": float(np.min(pair_nmi)),
        "pairwise_ari_mean": float(np.mean(pair_ari)), "pairwise_ari_min": float(np.min(pair_ari)),
        "modularity_mean": float(np.mean(mods)), "modularity_std": float(np.std(mods)),
        "modularity_min": float(np.min(mods)), "modularity_max": float(np.max(mods)),
        "community_count_min": int(min(counts)), "community_count_max": int(max(counts)),
    }
