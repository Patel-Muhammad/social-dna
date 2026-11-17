# Data folder

| Folder | Content | In Git? |
|---|---|---|
| `raw/` | Original dataset files (`lastfm_asia_edges.csv`, `lastfm_asia_features.json`, `lastfm_asia_target.csv`) | No - downloaded automatically |
| `processed/` | Cleaned edges, sparse feature matrix, graph cache, centrality cache, **SQLite database** | No - re-created automatically |
| `evaluation/` | `results.json` written by `scripts/run_evaluation.py` | Yes (small) |

## Dataset: LastFM Asia Social Network

- Official page: https://snap.stanford.edu/data/feather-lastfm-social.html (file `lastfm_asia.zip`)
- Mirror: https://archive.ics.uci.edu/dataset/599/lastfm+asia+social+network
- 7,624 users, 27,806 mutual-follower edges; users are LastFM users from Asian countries, collected in March 2020 via the public API.
- Files: edges (`node_1,node_2`), features (JSON: user id -> liked-artist feature IDs; **no artist names**), target (anonymised country code per user).
- Terms: distributed by Stanford SNAP for research use. The data contains only anonymised numeric IDs. Check the SNAP page for current terms before redistributing the raw files (they are intentionally not committed here).
- Citation:

```
@inproceedings{rozemberczki2020characteristic,
  title     = {Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models},
  author    = {Rozemberczki, Benedek and Sarkar, Rik},
  booktitle = {Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20)},
  year      = {2020}
}
```

## Manual download

If the automatic download fails (firewall, offline machine): download `lastfm_asia.zip` in a browser, place the zip (or the three files) in `data/raw/`, and start the app again.
