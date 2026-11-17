"""Jaccard and cosine similarity between users and communities (sparse operations)."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


def jaccard_similarity(a: set, b: set) -> float:
    """|A ∩ B| / |A ∪ B|; two empty sets are defined as 0 (nothing shared)."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def cosine_between(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu and nv else 0.0


def idf_weights(X: sparse.csr_matrix) -> np.ndarray:
    """Smoothed inverse document frequency: rare artists weigh more than ubiquitous ones."""
    n = X.shape[0]
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    return np.log((1 + n) / (1 + df)) + 1.0


def tfidf_rows(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """L2-normalised TF-IDF user vectors (binary TF)."""
    weighted = X.multiply(idf_weights(X)).tocsr()
    return normalize(weighted, norm="l2", axis=1)


def community_centroids(U: sparse.csr_matrix, labels: np.ndarray, communities: list[int]) -> np.ndarray:
    """Mean TF-IDF vector for each community -> dense (k x features)."""
    cents = np.zeros((len(communities), U.shape[1]))
    for i, c in enumerate(communities):
        cents[i] = np.asarray(U[labels == c].mean(axis=0)).ravel()
    return cents


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit = vectors / norms
    return unit @ unit.T


def within_group_cosine(U: sparse.csr_matrix, labels: np.ndarray, community: int) -> float:
    """Exact mean pairwise cosine among members of one community.
    With unit vectors: sum_{i!=j} u_i.u_j = |sum u|^2 - n."""
    rows = U[labels == community]
    n = rows.shape[0]
    if n < 2:
        return 0.0
    total = np.asarray(rows.sum(axis=0)).ravel()
    return float((total @ total - n) / (n * (n - 1)))


def between_group_cosine(U: sparse.csr_matrix, labels: np.ndarray, a: int, b: int) -> float:
    """Exact mean cosine between all user pairs (one from each community)."""
    ra, rb = U[labels == a], U[labels == b]
    if ra.shape[0] == 0 or rb.shape[0] == 0:
        return 0.0
    sa, sb = np.asarray(ra.sum(axis=0)).ravel(), np.asarray(rb.sum(axis=0)).ravel()
    return float(sa @ sb / (ra.shape[0] * rb.shape[0]))


def mean_pairwise_jaccard(X: sparse.csr_matrix, rows: np.ndarray, max_users: int = 1500,
                          seed: int = 0) -> float:
    """Mean Jaccard over all user pairs in `rows` (sampled if the group is huge)."""
    if len(rows) > max_users:
        rows = np.random.default_rng(seed).choice(rows, max_users, replace=False)
    B = (X[rows] > 0).astype(np.float32)
    n = B.shape[0]
    if n < 2:
        return 0.0
    inter = (B @ B.T).toarray()
    sizes = np.asarray(B.sum(axis=1)).ravel()
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
    return float((jac.sum() - np.trace(jac)) / (n * (n - 1)))
