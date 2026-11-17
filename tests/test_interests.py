import numpy as np
import pytest
from scipy import sparse

from analysis import influence_analysis, interest_analysis, network_metrics, similarity


def X_small():
    #        f0 f1 f2
    return sparse.csr_matrix(np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1]], dtype=float))


def test_jaccard():
    assert similarity.jaccard_similarity({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert similarity.jaccard_similarity(set(), set()) == 0.0
    assert similarity.jaccard_similarity({1}, {2}) == 0.0
    assert similarity.jaccard_similarity({1, 2}, {1, 2}) == 1.0


def test_cosine():
    assert similarity.cosine_between(np.array([1.0, 0]), np.array([1.0, 0])) == pytest.approx(1.0)
    assert similarity.cosine_between(np.array([1.0, 0]), np.array([0, 1.0])) == 0.0
    assert similarity.cosine_between(np.zeros(2), np.array([1.0, 1])) == 0.0
    cm = similarity.cosine_matrix(np.array([[1.0, 1], [1, 0]]))
    assert cm[0, 1] == pytest.approx(1 / np.sqrt(2))


def test_feature_aggregation_by_hand():
    X = X_small()
    labels = np.array([1, 1, 2, 2])
    counts = interest_analysis.community_feature_counts(X, labels, [1, 2])
    assert counts.tolist() == [[2, 1, 0], [0, 1, 2]]


def test_interest_table_values():
    X = X_small()
    labels = np.array([1, 1, 2, 2])
    t = interest_analysis.compute_interest_table(X, labels, np.array([100, 101, 102]), top_n=3)
    top1 = t[(t.community == 1) & (t.kind == "common")].iloc[0]
    assert top1.feature_id == 100 and top1.frequency == 2 and top1.prevalence == 1.0
    assert top1.network_prevalence == 0.5 and top1.lift == pytest.approx(2.0)


def test_distinctive_features_planted(analysed):
    repo, run_id, net = analysed
    inter = repo.interests(run_id, "distinctive")
    members = repo.members(run_id)
    for c in members.community.unique():
        group_users = members[members.community == c].user_id
        planted = int(np.bincount(group_users // 20).argmax())
        expected = set(range(100 + planted * 10, 100 + planted * 10 + 8))
        top = set(inter[(inter.community == c) & (inter["rank"] <= 8)].feature_id)
        assert top <= expected, (c, top, expected)       # distinctive = the planted group features
    assert 129 not in set(inter.feature_id)               # the universally-liked feature is never distinctive


def test_within_cosine_matches_bruteforce():
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix((rng.random((12, 8)) > 0.5).astype(float))
    U = similarity.tfidf_rows(X)
    labels = np.array([1] * 6 + [2] * 6)
    dense = U.toarray()
    brute = np.mean([dense[i] @ dense[j] for i in range(6) for j in range(6) if i != j])
    assert similarity.within_group_cosine(U, labels, 1) == pytest.approx(brute)
    cross = np.mean([dense[i] @ dense[j] for i in range(6) for j in range(6, 12)])
    assert similarity.between_group_cosine(U, labels, 1, 2) == pytest.approx(cross)


def test_mean_pairwise_jaccard_matches_bruteforce():
    X = X_small()
    rows = np.arange(4)
    sets = [set(X[i].indices) for i in range(4)]
    brute = np.mean([similarity.jaccard_similarity(sets[i], sets[j]) for i in range(4) for j in range(4) if i != j])
    assert similarity.mean_pairwise_jaccard(X, rows) == pytest.approx(brute)


def test_missing_features_rejected():
    with pytest.raises(ValueError):
        interest_analysis.compute_interest_table(sparse.csr_matrix((3, 0)), np.array([1, 1, 2]), np.array([]))


def test_users_without_features_do_not_break_similarity():
    X = sparse.csr_matrix(np.array([[1, 0], [0, 0], [0, 1]], dtype=float))
    U = similarity.tfidf_rows(X)
    assert np.isfinite(U.toarray()).all()


def test_global_and_user_metrics(cliques):
    g = network_metrics.global_metrics(cliques)
    assert g["users"] == 10 and g["edges"] == 21 and g["connected_components"] == 1
    assert g["density"] == pytest.approx(21 / 45) and g["average_degree"] == pytest.approx(4.2)
    table, notes = network_metrics.centrality_table(cliques, betweenness_k=None)
    assert table.loc[4, "betweenness_centrality"] > table.loc[0, "betweenness_centrality"]   # bridge node
    assert table.loc[0, "degree"] == 4 and table["pagerank"].sum() == pytest.approx(1.0)
    assert notes["betweenness_centrality"] == "Exact."


def test_bridge_metrics(cliques):
    part = {i: (1 if i < 5 else 2) for i in range(10)}
    b = influence_analysis.bridge_metrics(cliques, part)
    assert b.loc[4, "external_degree"] == 1 and b.loc[4, "neighbour_communities"] == 2
    assert b.loc[0, "external_degree"] == 0 and b.loc[0, "participation_coefficient"] == 0.0
    assert b.loc[4, "participation_coefficient"] == pytest.approx(1 - (0.8 ** 2 + 0.2 ** 2))
