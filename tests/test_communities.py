import networkx as nx
import pytest

from analysis import community_metrics, evaluation
from analysis.greedy_modularity import run_greedy_modularity
from analysis.louvain import partition_modularity, renumber_partition, run_louvain


def test_louvain_finds_two_cliques(cliques):
    r = run_louvain(cliques, seed=1)
    assert r.community_count == 2
    assert len({r.partition[i] for i in range(5)}) == 1 and len({r.partition[i] for i in range(5, 10)}) == 1
    assert r.partition[0] != r.partition[9]


def test_modularity_matches_hand_calculation(cliques):
    # m=21 edges; each clique: 10 internal edges, degree sum 21 (one clique has the bridge twice counted per side).
    # Q = sum_c [ L_c/m - (d_c/(2m))^2 ] = 2 * (10/21 - (21/42)^2) = 2*(0.47619-0.25) = 0.45238
    part = {i: (1 if i < 5 else 2) for i in range(10)}
    assert partition_modularity(cliques, part) == pytest.approx(0.452381, abs=1e-5)


def test_louvain_reproducible_and_numbering_deterministic(network):
    a, b = run_louvain(network.graph, seed=3), run_louvain(network.graph, seed=3)
    assert a.partition == b.partition
    assert set(a.partition.values()) == set(range(1, a.community_count + 1))
    sizes = list(a.sizes().values())
    assert sizes == sorted(sizes, reverse=True) or a.sizes()[1] == max(sizes)  # community 1 is the largest


def test_every_user_assigned(network):
    r = run_louvain(network.graph, seed=0)
    assert set(r.partition) == set(network.graph.nodes)


def test_empty_and_edgeless_graphs_rejected():
    with pytest.raises(ValueError):
        run_louvain(nx.Graph())
    g = nx.Graph(); g.add_nodes_from([1, 2])
    with pytest.raises(ValueError):
        run_louvain(g)
    with pytest.raises(ValueError):
        run_greedy_modularity(g)


def test_greedy_modularity_and_agreement(cliques):
    g = run_greedy_modularity(cliques)
    l = run_louvain(cliques, seed=0)
    assert g.community_count == 2
    ag = evaluation.partition_agreement(l.partition, g.partition)
    assert ag["nmi"] == pytest.approx(1.0) and ag["ari"] == pytest.approx(1.0)


def test_agreement_requires_same_users():
    with pytest.raises(ValueError):
        evaluation.partition_agreement({1: 1, 2: 2}, {1: 1})


def test_agreement_detects_different_partitions():
    a = {i: (1 if i < 4 else 2) for i in range(8)}
    b = {i: (i % 2) for i in range(8)}
    assert evaluation.partition_agreement(a, b)["nmi"] < 0.2


def test_stability_on_clear_structure(network):
    s = evaluation.louvain_stability(network.graph, [0, 1, 2, 3])
    assert s["pairwise_nmi_mean"] > 0.9 and s["modularity_std"] < 0.05


def test_internal_external_connections(cliques):
    part = {i: (1 if i < 5 else 2) for i in range(10)}
    struct, cross = community_metrics.community_structure(cliques, part)
    row = struct[struct.community == 1].iloc[0]
    assert row.internal_edges == 10 and row.external_edges == 1
    assert row.internal_density == pytest.approx(1.0)
    assert row.conductance == pytest.approx(1 / 21)
    assert cross.iloc[0].cross_edges == 1


def test_renumber_ties_are_deterministic():
    assert renumber_partition({5: 9, 6: 9, 1: 4, 2: 4, 3: 7}) == {1: 1, 2: 1, 5: 2, 6: 2, 3: 3}
