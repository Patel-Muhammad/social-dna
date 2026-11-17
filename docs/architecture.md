# Architecture

```
LastFM Asia files ──► core/dataset_loader ──► core/data_validator ──► processed files (data/processed)
                                                         │                   │
                                                         ▼                   ▼
                                                 SQLite (dataset_metadata)   core/graph_builder (NetworkX graph + sparse features)
                                                                             │
                                  analysis/pipeline ◄────────────────────────┘
   louvain · greedy_modularity · network_metrics · community_metrics · interest_analysis · similarity · influence_analysis · evaluation
                                  │
                                  ▼
                        SQLite (runs, communities, members, interests, metrics, similarity)
                                  │
                    analysis/queries.AnalysisService  (verified backend functions)
                          │                         │
                  Streamlit pages            llm/query_router ─► llm/gemini_client (google-genai)
                  (pages/, components/)       routes, validates, explains verified JSON
```

## Layers and responsibilities

| Layer | Folder | Rule |
|---|---|---|
| Data acquisition | `core/` | Downloads once, validates, caches processed files. Never invents data. |
| Analysis | `analysis/` | Pure functions on NetworkX graphs and scipy sparse matrices; no UI, no SQL. |
| Persistence | `database/` | All SQL lives in `repositories.py`; schema changes are versioned migrations. |
| Verified API | `analysis/queries.py` | Returns JSON-serialisable dicts from stored results; validates IDs. |
| LLM | `llm/` | Gemini explains verified results; it never calculates or chooses data. |
| UI | `pages/`, `components/` | Streamlit; reads through the service/repository layer only. |

## First launch vs later launches

1. Locate raw files (use zip/files in `data/raw/`, else download). 2. Validate, clean, save processed files. 3. Create/migrate SQLite, store dataset metadata. 4. Build graph (pickle-cached). 5. If no completed Louvain run exists for the chosen seed/resolution: run the full analysis and save it. Later launches only load.

## Deviations from the suggested structure

- `app.py` uses `st.navigation` (so pages get clean titles); page files stay in `pages/`.
- Added `analysis/community_metrics.py`, `analysis/pipeline.py`, `analysis/evaluation.py`, `analysis/queries.py`, `llm/evaluation.py`, `components/theme.py`, `components/context.py`.
- Extra tables `community_metrics` and `community_similarity`; extra columns (`kind`, `rank`, `lift`, `external_degree`, ...) on the suggested tables. Documented in `docs/database.md`.
- Dependencies are version ranges with an additional `requirements-lock.txt` of tested versions.
