# Database (SQLite)

File: `data/processed/social_dna.sqlite` (created automatically; git-ignored). Foreign keys are enforced (`PRAGMA foreign_keys=ON`); every multi-row save is one transaction. Versioning: `schema_version` table + ordered migrations in `database/schema.py`.

| Table | Purpose |
|---|---|
| `dataset_metadata` | Dataset name/source/version, counts, SHA-256 of raw files, validation JSON |
| `analysis_runs` | One row per algorithm run: algorithm, seed, resolution, modularity, community count, status |
| `communities` | Community number, size, % of network, cached AI description + model |
| `community_members` | Run, community and original user ID |
| `community_interests` | Top common/distinctive features per community (`kind`, `rank`, `frequency`, `prevalence`, `network_prevalence`, `lift`, `distinctiveness_score`) |
| `community_metrics` | Internal/external edges, density, conductance, within-community cosine/Jaccard |
| `community_similarity` | Cosine, Jaccard and cross-edge count for every community pair (a < b) |
| `network_metrics` | Global metrics, stability and algorithm-comparison results (with notes) |
| `user_metrics` | Degree, centralities, PageRank, community, external degree, participation coefficient |
| `chat_sessions`, `chat_messages` | AI analyst conversations (message + optional JSON meta) |

Large artefacts (cleaned edges, sparse feature matrix, graph cache, centrality cache) are files in `data/processed/`, not rows. Only the top-30 features per community are stored, not the full matrix.
