# Algorithms

## Graph
Undirected `networkx.Graph`; node = LastFM user (original dataset ID), edge = mutual follow. Cleaning removes self-loops, duplicate and unknown-user edges (the real data had none; counts are reported).

## Louvain (primary)
`python-louvain` (`community.best_partition`) with a configurable `random_state` and `resolution`. Communities are renumbered 1..k by size (ties: smallest member ID) so the same grouping always gets the same numbers. Modularity is computed with `networkx.community.modularity` for both algorithms so the numbers are comparable.

## Greedy Modularity (comparison)
`networkx.community.greedy_modularity_communities` (Clauset-Newman-Moore); deterministic.

## Comparison and stability
Louvain is re-run with seeds 0-9; every pair of partitions is compared with NMI and ARI. Louvain vs Greedy agreement uses NMI/ARI. These measure **agreement/stability, not accuracy**: the dataset has no ground-truth communities. Country codes are not treated as ground truth (a descriptive NMI is shown, labelled as such).

## Interest analysis
For community c and artist feature f: `frequency` = members liking f; `prevalence` = frequency / |c|; `network_prevalence` = share of all users liking f; `lift` = prevalence / network_prevalence; `distinctiveness = prevalence * log2(lift)` for lift > 1 with minimum support max(3, 2% of members). Common = highest frequency. Counts come from one sparse matrix product.

## Similarity
- Users: binary vectors, TF-IDF weighting `idf = ln((1+n)/(1+df)) + 1`, L2-normalised.
- Community profile = mean of member TF-IDF vectors; **cosine** between profiles.
- Exact mean pairwise cosine inside a community uses `(|Σu|² − n) / (n(n−1))` (verified against brute force in tests).
- **Jaccard** between communities = |A∩B| / |A∪B| over each community's top-30 common features; within-community Jaccard = mean over user pairs (sampled to 1,500 users for very large communities).

## Network metrics
Degree centrality, closeness (exact), PageRank (exact), betweenness (**sampled approximation**, k=1000 sources, seeded; labelled in the UI), density, average degree, average clustering, transitivity, connected components, internal/external edges, conductance.

## Hub vs bridge
Participation coefficient `P = 1 − Σ_c (k_ic / k_i)²`. Bridge candidates: degree ≥ 5 and friends in ≥ 3 communities, ranked by betweenness.
