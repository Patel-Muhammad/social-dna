"""Hub vs bridge analysis.

Hub    = many connections (high degree / PageRank), possibly all inside one community.
Bridge = connections spread over several communities (high participation coefficient)
         and sitting on shortest paths between groups (betweenness).
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def bridge_metrics(graph: nx.Graph, partition: dict[int, int]) -> pd.DataFrame:
    """external_degree, number of distinct neighbour communities, participation coefficient
    P_i = 1 - sum_c (k_ic / k_i)^2   (0 = all neighbours in one community)."""
    rows = {}
    for node in graph.nodes():
        neighbours = list(graph.neighbors(node))
        k = len(neighbours)
        counts: dict[int, int] = {}
        for nb in neighbours:
            counts[partition[nb]] = counts.get(partition[nb], 0) + 1
        own = partition[node]
        rows[node] = {
            "external_degree": k - counts.get(own, 0),
            "neighbour_communities": len(counts),
            "participation_coefficient": (1.0 - sum((c / k) ** 2 for c in counts.values())) if k else 0.0,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "user_id"
    return out


def top_users(metrics: pd.DataFrame, column: str, n: int = 10) -> pd.DataFrame:
    return metrics.sort_values([column, "degree"], ascending=[False, False]).head(n)


def select_bridge_users(metrics: pd.DataFrame, n: int = 10, min_degree: int = 5,
                        min_communities: int = 3) -> pd.DataFrame:
    """Bridge candidates: enough links to be meaningful, reaching >= min_communities groups,
    ranked by (approximate) betweenness centrality."""
    eligible = metrics[(metrics["degree"] >= min_degree) & (metrics["neighbour_communities"] >= min_communities)]
    return eligible.sort_values(["betweenness_centrality", "participation_coefficient"],
                                ascending=False).head(n)


def explain_user(user_id: int, row: pd.Series, own_community: int, rank_degree: int,
                 rank_betweenness: int, rank_pagerank: int, n_users: int) -> str:
    """Plain-language explanation built only from measured values."""
    share_ext = row["external_degree"] / row["degree"] if row["degree"] else 0.0
    return (
        f"User {user_id} (community {own_community}) has {int(row['degree'])} friends, "
        f"{share_ext:.0%} of them outside their own community, spread over "
        f"{int(row['neighbour_communities'])} communities. Rank by connections: {rank_degree}/{n_users}; "
        f"by betweenness: {rank_betweenness}/{n_users}; by PageRank: {rank_pagerank}/{n_users}."
    )
