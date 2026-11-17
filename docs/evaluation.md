# Evaluation report

Generated 2026-11-17 17:20 UTC on Louvain run 1 (seed 42, resolution 1). All numbers below were produced by `scripts/run_evaluation.py` and the Louvain/Greedy analysis stored in SQLite. Nothing in this file is hand-written except the explanations.

## 1. Why there is no single "accuracy" for community detection

Louvain is unsupervised and the dataset has **no ground-truth communities**. The country code is not a community label (people from one country need not be friends), so NMI/ARI against it would be misleading and is **not** used as accuracy. The 93% target is therefore applied only where a meaningful per-item correctness exists: the AI analyst (section 3) and the unit-tested calculations (section 2).

## 2. Community detection and interest analysis

| Measure | Value | Note |
|---|---|---|
| Louvain modularity | 0.8155 | higher = more internal links than chance |
| Louvain communities | 31 | |
| Stability, 10 seeds: mean pairwise NMI | 0.900 | min 0.864 |
| Stability: mean pairwise ARI | 0.877 | min 0.820 |
| Modularity across seeds | 0.8132 – 0.8162 | std 0.0009 |
| Communities across seeds | 26 – 30 | |
| Greedy Modularity: communities / modularity | 43 / 0.7960 | deterministic |
| Louvain vs Greedy agreement (NMI / ARI) | 0.819 / 0.727 | agreement, not accuracy |
| Communities vs country code (NMI) | 0.624 | descriptive association only |

Both algorithms reach high modularity. Louvain scores slightly higher, but modularity is the objective both optimise, so this is **not** evidence that either is more accurate.

Interest-analysis correctness (feature counts, prevalence, lift, Jaccard, cosine including exact within-community cosine vs brute force, planted-feature recovery) is verified by the automated tests in `tests/test_interests.py`; see `docs/test_report.md` for the executed counts.

## 3. AI analyst evaluation

Scoring procedure:

- **Routing accuracy** – correct backend function and parameters for supported questions.
- **Numerical correctness** – share of required key numbers (taken from the independently recomputed verified result) found in the answer.
- **Unsupported-claim rate** – numbers in the answer that appear nowhere in the verified result (small structural numbers 1,2,3,4,5,10,30 are ignored; this is a known limitation of the check).
- **Missing-information handling** – unsupported questions are refused and invalid IDs are corrected without inventing data.
- **Factual consistency** – per answer 0.5 x numerical correctness + 0.5 x (1 - unsupported-claim share); 1/0 for missing-information questions; mean over all questions.

Question set: 29 fixed questions (23 supported, 6 unsupported/invalid).

| Metric | Target | deterministic (no LLM) | gemini (live API) |
|---|---|---|---|
| Routing accuracy | >= 93% | 100.0% (met) | 95.7% (met) |
| Numerical correctness | >= 93% | 100.0% (met) | 96.7% (met) |
| Unsupported-claim rate | <= 7% | 0.0% (met) | 0.1% (met) |
| Missing-information handling | >= 93% | 100.0% (met) | 100.0% (met) |
| Factual consistency score | >= 93% | 100.0% (met) | 95.1% (met) |

- **deterministic (no LLM)**: 0 of 29 answers were written by Gemini; 0 of 513 numbers were unsupported.
- **gemini (live API)**: 22 of 29 answers were written by Gemini; 1 of 769 numbers were unsupported.

### Scorer revision (disclosure)

The first live Gemini run was scored with an earlier scorer: routing 95.7%, numerical correctness 88.8%, unsupported-claim rate 0.3%, missing-information handling 100.0%, factual consistency 90.2% (stored in `data/evaluation/results_v1_scorer.json`). Reading the answers showed three scorer problems, not model errors: (1) a correct answer of 0.8155 was rejected because the scorer demanded the pre-rounded 0.815; (2) questions such as "How many communities are there?" were penalised for omitting facts they did not ask for; (3) "anonymised" was rejected because the check looked for "anonymous". The scorer was corrected (precision-aware matching, per-question key facts, spelling tolerance, regression tests in `tests/test_evaluation.py`) and the evaluation was **re-run live**; the table above is that re-run. Genuine model misses (e.g. a question routed to the wrong function) are still counted against Gemini.

### Questions below full score (deterministic (no LLM))

None.

### Questions below full score (gemini (live API))

- *Is community 9 tightly connected?* – routing correct: True, numerical: 0.75, unsupported numbers: [], score 0.88
- *What do members of community 7 like?* – routing correct: False, numerical: 1.0, unsupported numbers: [], score 0.00
- *Who are the most influential users?* – routing correct: True, numerical: 0.5, unsupported numbers: [], score 0.75
- *Tell me about user 7237.* – routing correct: True, numerical: 1.0, unsupported numbers: [203.0], score 0.96

## 4. Interpretation and limits

- The deterministic mode shows the correctness of routing and of the verified numbers. It does not test Gemini's wording.
- Only the Gemini column measures whether the LLM stays faithful to the verified context.
- 29 questions is a small benchmark; percentages have coarse resolution (one question = 3.4 points).
- Numeric faithfulness does not prove the qualitative interpretation is right.
