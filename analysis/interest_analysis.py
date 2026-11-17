"""Common and distinctive interests (artist features) per community.

Definitions (all from real counts):
  frequency          = members of the community who like the artist feature
  prevalence         = frequency / community size
  network_prevalence = share of ALL users who like it
  lift               = prevalence / network_prevalence
  distinctiveness    = prevalence * log2(lift)   (>0 only when over-represented;
                       the prevalence factor stops rare one-off features from winning)
Common = highest frequency; Distinctive = highest distinctiveness with minimum support.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import sparse

from core import config


def community_feature_counts(X: sparse.csr_matrix, labels: np.ndarray, communities: list[int]) -> np.ndarray:
    """(k x features) matrix of member counts, via one sparse product."""
    rows = np.arange(len(labels))
    cols = np.searchsorted(communities, labels)
    indicator = sparse.csr_matrix((np.ones(len(labels)), (cols, rows)), shape=(len(communities), len(labels)))
    return np.asarray((indicator @ (X > 0).astype(np.float32)).todense())


def compute_interest_table(X: sparse.csr_matrix, labels: np.ndarray, feature_ids: np.ndarray,
                           top_n: int = config.TOP_FEATURES_PER_COMMUNITY) -> pd.DataFrame:
    """Long table: community, kind, rank, feature_id, frequency, prevalence, lift, distinctiveness."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Feature matrix is empty; interest analysis needs user features.")
    communities = sorted(int(c) for c in np.unique(labels))
    counts = community_feature_counts(X, labels, communities)
    sizes = np.array([(labels == c).sum() for c in communities])
    network_prev = np.asarray((X > 0).sum(axis=0)).ravel() / X.shape[0]

    records = []
    for i, c in enumerate(communities):
        freq = counts[i]
        prevalence = freq / sizes[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(network_prev > 0, prevalence / network_prev, 0.0)
            distinct = np.where(prevalence > 0, prevalence * np.log2(np.where(lift > 0, lift, 1.0)), 0.0)
        min_support = max(3, math.ceil(0.02 * sizes[i]))
        eligible = freq >= min_support

        # Common: sort by frequency desc, then feature id asc (deterministic).
        order_common = np.lexsort((feature_ids, -freq))
        order_common = [j for j in order_common if freq[j] > 0][:top_n]
        # Distinctive: over-represented (lift > 1) features with enough support.
        cand = np.where(eligible & (lift > 1.0))[0]
        order_dist = cand[np.lexsort((feature_ids[cand], -distinct[cand]))][:top_n]

        for kind, order in (("common", order_common), ("distinctive", order_dist)):
            for rank, j in enumerate(order, start=1):
                records.append((c, kind, rank, int(feature_ids[j]), int(freq[j]), float(prevalence[j]),
                                float(network_prev[j]), float(lift[j]), float(distinct[j])))
    return pd.DataFrame(records, columns=["community", "kind", "rank", "feature_id", "frequency",
                                          "prevalence", "network_prevalence", "lift", "distinctiveness_score"])
