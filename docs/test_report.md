# Test report

Executed 2026-11-17 17:21 UTC with `python -m pytest -q` (Python 3.12.10, darwin).

**76 tests: 76 passed, 0 failed, 0 errors, 0 skipped.**

| Test file | Tests | Failed |
|---|---|---|
| `test_communities.py` | 11 | 0 |
| `test_database.py` | 8 | 0 |
| `test_dataset.py` | 7 | 0 |
| `test_evaluation.py` | 10 | 0 |
| `test_graph.py` | 3 | 0 |
| `test_interests.py` | 11 | 0 |
| `test_llm.py` | 26 | 0 |

Covers: dataset loading/validation/download failure, graph construction and caching, Louvain, Greedy Modularity, modularity (hand-checked), stability/agreement, interest aggregation, Jaccard and cosine (vs brute force), user and community metrics, SQLite persistence/constraints/transactions, query routing, Gemini context generation, invalid community IDs, missing API key, API failures/cool-down, empty datasets, missing features, and the evaluation scorer itself (it must fail wrong answers).

Unit tests use small synthetic graphs only; the application never uses synthetic data for its results. The Gemini client's error handling is unit-tested with a stubbed SDK object; the live API is exercised by `scripts/run_evaluation.py` (see `evaluation.md`).

```
........................................................................ [ 94%]
....                                                                     [100%]
76 passed in 2.13s
```
