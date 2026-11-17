"""Structural metrics per community and between communities."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def community_structure(graph: nx.Graph, partition: dict[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-community table, cross-community edge table with a<b)."""
    internal: dict[int, int] = {}
    external: dict[int, int] = {}
    cross: dict[tuple[int, int], int] = {}
    for u, v in graph.edges():
        cu, cv = partition[u], partition[v]
        if cu == cv:
            internal[cu] = internal.get(cu, 0) + 1
        else:
            external[cu] = external.get(cu, 0) + 1
            external[cv] = external.get(cv, 0) + 1
            key = (min(cu, cv), max(cu, cv))
            cross[key] = cross.get(key, 0) + 1
    sizes: dict[int, int] = {}
    for c in partition.values():
        sizes[c] = sizes.get(c, 0) + 1
    rows = []
    for c, size in sorted(sizes.items()):
        i, e = internal.get(c, 0), external.get(c, 0)
        possible = size * (size - 1) / 2
        rows.append({
            "community": c, "member_count": size, "internal_edges": i, "external_edges": e,
            "internal_density": i / possible if possible else 0.0,
            "avg_internal_degree": 2 * i / size,
            "avg_degree": (2 * i + e) / size,
            # conductance: share of edge endpoints leaving the community (lower = tighter)
            "conductance": e / (2 * i + e) if (2 * i + e) else 0.0,
        })
    cross_df = pd.DataFrame([(a, b, n) for (a, b), n in sorted(cross.items())],
                            columns=["community_a", "community_b", "cross_edges"])
    return pd.DataFrame(rows), cross_df
