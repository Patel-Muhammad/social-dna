"""Orchestrates a full analysis run and persists the results to SQLite."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from analysis import community_metrics, evaluation, influence_analysis, interest_analysis, network_metrics, similarity
from analysis.greedy_modularity import run_greedy_modularity
from analysis.louvain import CommunityResult, run_louvain
from core import config
from core.graph_builder import NetworkData
from database.repositories import AnalysisRepository

Progress = Callable[[str, float], None]


def _noop(_msg: str, _frac: float) -> None:
    return None


def load_or_compute_centralities(net: NetworkData, processed_dir: Path, seed: int,
                                 progress: Progress = _noop) -> tuple[pd.DataFrame, dict[str, str]]:
    """Centralities depend only on the graph, so they are cached on disk (keyed by dataset hash)."""
    path = Path(processed_dir) / "centrality.csv"
    meta_path = Path(processed_dir) / "centrality_meta.json"
    key = {"raw_hash": net.raw_hash, "seed": seed, "k": config.BETWEENNESS_SAMPLE_K}
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key:
            return pd.read_csv(path, index_col="user_id"), meta["notes"]
    progress("Computing centralities (betweenness is sampled; this takes about a minute)", 0.3)
    table, notes = network_metrics.centrality_table(net.graph, seed=seed)
    table.to_csv(path)
    meta_path.write_text(json.dumps({"key": key, "notes": notes}), encoding="utf-8")
    return table, notes


def _labels(net: NetworkData, partition: dict[int, int]) -> np.ndarray:
    return np.array([partition[int(u)] for u in net.node_ids])


def run_full_analysis(net: NetworkData, repo: AnalysisRepository, dataset_id: int, *, seed: int,
                      resolution: float, processed_dir: Path | None = None,
                      progress: Progress = _noop) -> int:
    """Louvain + profiles + metrics + comparison with Greedy Modularity. Returns the run id."""
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    graph = net.graph
    n_users = graph.number_of_nodes()
    run_id = repo.create_run(dataset_id, "louvain", seed, resolution)
    try:
        progress("Detecting communities with Louvain", 0.05)
        result = run_louvain(graph, seed=seed, resolution=resolution)
        ids = repo.save_partition(run_id, result.partition, n_users, result.modularity)
        labels = _labels(net, result.partition)
        communities = sorted(ids)

        progress("Computing network and user metrics", 0.15)
        glob = network_metrics.global_metrics(graph)
        cent, cent_notes = load_or_compute_centralities(net, processed_dir, seed, progress)
        bridge = influence_analysis.bridge_metrics(graph, result.partition)
        users = cent.join(bridge)
        user_rows = [
            (run_id, int(u), int(r.degree), float(r.degree_centrality), float(r.betweenness_centrality),
             float(r.closeness_centrality), float(r.pagerank), ids[result.partition[int(u)]],
             int(r.external_degree), int(r.neighbour_communities), float(r.participation_coefficient))
            for u, r in users.iterrows()
        ]
        repo.save_user_metrics(user_rows)

        progress("Aggregating common and distinctive interests", 0.55)
        interests = interest_analysis.compute_interest_table(net.features, labels, net.feature_ids)
        repo.save_interests([
            (run_id, ids[int(r.community)], int(r.feature_id), r.kind, int(r.rank), int(r.frequency),
             float(r.prevalence), float(r.network_prevalence), float(r.lift), float(r.distinctiveness_score))
            for r in interests.itertuples()
        ])

        progress("Computing similarities between communities", 0.7)
        struct, cross = community_metrics.community_structure(graph, result.partition)
        U = similarity.tfidf_rows(net.features)
        cents = similarity.community_centroids(U, labels, communities)
        cos = similarity.cosine_matrix(cents)
        network_centroid = np.asarray(U.mean(axis=0)).ravel()
        top_sets = {c: set(interests[(interests.community == c) & (interests.kind == "common")].feature_id)
                    for c in communities}
        cross_lookup = {(r.community_a, r.community_b): int(r.cross_edges) for r in cross.itertuples()}
        sim_rows = []
        for i, a in enumerate(communities):
            for j in range(i + 1, len(communities)):
                b = communities[j]
                sim_rows.append((run_id, ids[a], ids[b], float(cos[i, j]),
                                 similarity.jaccard_similarity(top_sets[a], top_sets[b]),
                                 cross_lookup.get((a, b), 0)))
        repo.save_similarity(sim_rows)

        comm_rows = []
        feat_counts = np.asarray(net.features.sum(axis=1)).ravel()
        for i, c in enumerate(communities):
            s = struct[struct.community == c].iloc[0]
            member_rows = np.where(labels == c)[0]
            comm_rows.append((
                ids[c], run_id, int(s.internal_edges), int(s.external_edges), float(s.internal_density),
                float(s.avg_internal_degree), float(s.avg_degree), float(s.conductance),
                similarity.within_group_cosine(U, labels, c),
                similarity.mean_pairwise_jaccard(net.features, member_rows),
                similarity.cosine_between(cents[i], network_centroid),
                float(feat_counts[member_rows].mean()),
            ))
        repo.save_community_metrics(comm_rows)

        progress("Comparing with Greedy Modularity and testing stability", 0.85)
        metrics: dict[str, tuple[float | None, str | None]] = {
            k: (v, None) for k, v in glob.items()
        }
        for name in ("betweenness_centrality", "closeness_centrality", "pagerank"):
            metrics[f"note_{name}"] = (None, cent_notes.get(name))
        metrics["modularity"] = (result.modularity, "Louvain modularity of this run")
        sizes = np.array(list(result.sizes().values()))
        metrics["largest_community_size"] = (float(sizes.max()), None)
        metrics["smallest_community_size"] = (float(sizes.min()), None)
        metrics["average_community_size"] = (float(sizes.mean()), None)
        metrics["inter_community_edge_share"] = (float(cross.cross_edges.sum() / graph.number_of_edges()), None)

        greedy_run = repo.find_run(dataset_id, "greedy_modularity")
        greedy = run_greedy_modularity(graph, resolution=resolution) if greedy_run is None else None
        if greedy is not None:
            g_id = repo.create_run(dataset_id, "greedy_modularity", None, resolution)
            repo.save_partition(g_id, greedy.partition, n_users, greedy.modularity)
            repo.complete_run(g_id)
            g_partition = greedy.partition
        else:
            g_members = repo.members(greedy_run["id"])
            g_partition = dict(zip(g_members.user_id.astype(int), g_members.community.astype(int)))
        g_sizes = np.array(list(pd.Series(list(g_partition.values())).value_counts()))
        g_mod = greedy.modularity if greedy else greedy_run["modularity"]
        agreement = evaluation.partition_agreement(result.partition, g_partition)
        metrics["greedy_community_count"] = (float(len(g_sizes)), "Greedy Modularity (deterministic)")
        metrics["greedy_modularity"] = (float(g_mod), None)
        metrics["louvain_vs_greedy_nmi"] = (agreement["nmi"], "Agreement between partitions, not accuracy")
        metrics["louvain_vs_greedy_ari"] = (agreement["ari"], "Agreement between partitions, not accuracy")

        stab = evaluation.louvain_stability(graph, list(range(config.STABILITY_SEEDS)), resolution)
        for key, value in stab.items():
            if isinstance(value, (int, float)):
                metrics[f"stability_{key}"] = (float(value), f"Louvain over seeds 0-{config.STABILITY_SEEDS - 1}")

        # Descriptive only: how strongly communities line up with the anonymised country code.
        known = net.countries.notna().to_numpy()
        if known.any():
            from sklearn.metrics import normalized_mutual_info_score
            nmi_country = normalized_mutual_info_score(net.countries.to_numpy()[known], labels[known])
            metrics["country_association_nmi"] = (
                float(nmi_country),
                "Descriptive association with country code. NOT ground truth and NOT accuracy.")
        repo.save_network_metrics(run_id, metrics)
        repo.complete_run(run_id)
        progress("Analysis complete", 1.0)
        return run_id
    except Exception as exc:
        repo.mark_failed(run_id, f"{type(exc).__name__}: {exc}")
        raise
