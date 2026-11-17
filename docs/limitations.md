# Limitations

1. **Anonymous artists.** The SNAP feature file contains artist IDs without names, so interests are shown as "artist feature #N". No genres or names are inferred.
2. **No ground truth.** There are no reference communities, so NMI/ARI against labels and "accuracy" of Louvain are undefined. Only modularity, stability and algorithm agreement are reported.
3. **Country is not community.** The target file holds anonymised country codes; they are not used as labels.
4. **Approximate betweenness.** Computed from 1,000 sampled sources; the UI labels this. Rankings of lower-ranked users may shift slightly with another seed.
5. **Louvain is heuristic.** Different seeds give similar but not identical partitions (quantified in `evaluation.md`).
6. **Snapshot data.** March 2020, Asian LastFM users only. Findings do not generalise.
7. **LLM.** Gemini only rephrases verified numbers, but wording can still be imperfect. Availability and model names depend on Google; the app falls back to verified plain summaries when Gemini cannot be reached.
8. **Evaluation size.** The analyst benchmark has 29 questions, so each point of the score has coarse resolution.
9. **Platforms.** Developed and tested on macOS; Windows/Linux are covered by the CI matrix in `.github/workflows/tests.yml`, which runs when the repository is pushed to GitHub.
