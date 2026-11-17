"""Evaluate the AI analyst on a fixed question set and write data/evaluation/results.json
and docs/evaluation.md. The Gemini section only runs if a live Gemini call succeeds first;
otherwise it is reported as NOT RUN together with the real error (never simulated)."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.queries import AnalysisService  # noqa: E402
from core import config  # noqa: E402
from core.cache_manager import ensure_pipeline  # noqa: E402
from database.repositories import AnalysisRepository  # noqa: E402
from llm.evaluation import run_evaluation  # noqa: E402
from llm.gemini_client import GeminiClient, GeminiUnavailableError  # noqa: E402
from llm.query_router import Analyst  # noqa: E402

TARGET = 0.93


class NoKey(GeminiClient):
    def __init__(self):
        super().__init__(api_key="")


def pct(x):
    return "n/a" if x != x else f"{x:.1%}"


def summary_rows(runs: dict) -> list[dict]:
    names = [("routing_accuracy", "Routing accuracy", True), ("numerical_correctness", "Numerical correctness", True),
             ("unsupported_claim_rate", "Unsupported-claim rate", False),
             ("missing_info_handling", "Missing-information handling", True),
             ("factual_consistency", "Factual consistency score", True)]
    rows = []
    for key, label, higher_better in names:
        row = {"Metric": label, "Target": (f">= {TARGET:.0%}" if higher_better else f"<= {1 - TARGET:.0%}")}
        for mode, data in runs.items():
            if data is None:
                row[mode] = "not run"
                continue
            v = data["summary"][key]
            ok = v >= TARGET if higher_better else v <= 1 - TARGET
            row[mode] = f"{pct(v)} ({'met' if ok else 'NOT met'})"
        rows.append(row)
    return rows


def write_markdown(path: Path, svc, runs: dict, gemini_error: str | None, ts: str) -> None:
    m = svc.metrics
    L = [f"# Evaluation report\n\nGenerated {ts} on Louvain run {svc.run_id} (seed {svc.run['random_seed']}, resolution "
         f"{svc.run['resolution']:g}). All numbers below were produced by `scripts/run_evaluation.py` and the Louvain/Greedy "
         "analysis stored in SQLite. Nothing in this file is hand-written except the explanations.\n",
         "## 1. Why there is no single \"accuracy\" for community detection\n",
         "Louvain is unsupervised and the dataset has **no ground-truth communities**. The country code is not a community label "
         "(people from one country need not be friends), so NMI/ARI against it would be misleading and is **not** used as accuracy. "
         "The 93% target is therefore applied only where a meaningful per-item correctness exists: the AI analyst (section 3) and "
         "the unit-tested calculations (section 2).\n",
         "## 2. Community detection and interest analysis\n",
         "| Measure | Value | Note |\n|---|---|---|",
         f"| Louvain modularity | {svc.run['modularity']:.4f} | higher = more internal links than chance |",
         f"| Louvain communities | {svc.run['community_count']} | |",
         f"| Stability, {config.STABILITY_SEEDS} seeds: mean pairwise NMI | {m['stability_pairwise_nmi_mean']:.3f} | min {m['stability_pairwise_nmi_min']:.3f} |",
         f"| Stability: mean pairwise ARI | {m['stability_pairwise_ari_mean']:.3f} | min {m['stability_pairwise_ari_min']:.3f} |",
         f"| Modularity across seeds | {m['stability_modularity_min']:.4f} – {m['stability_modularity_max']:.4f} | std {m['stability_modularity_std']:.4f} |",
         f"| Communities across seeds | {int(m['stability_community_count_min'])} – {int(m['stability_community_count_max'])} | |",
         f"| Greedy Modularity: communities / modularity | {int(m['greedy_community_count'])} / {m['greedy_modularity']:.4f} | deterministic |",
         f"| Louvain vs Greedy agreement (NMI / ARI) | {m['louvain_vs_greedy_nmi']:.3f} / {m['louvain_vs_greedy_ari']:.3f} | agreement, not accuracy |",
         f"| Communities vs country code (NMI) | {m.get('country_association_nmi', float('nan')):.3f} | descriptive association only |",
         "\nBoth algorithms reach high modularity. Louvain scores slightly higher, but modularity is the objective both optimise, "
         "so this is **not** evidence that either is more accurate.\n",
         "Interest-analysis correctness (feature counts, prevalence, lift, Jaccard, cosine including exact within-community "
         "cosine vs brute force, planted-feature recovery) is verified by the automated tests in `tests/test_interests.py`; "
         "see `docs/test_report.md` for the executed counts.\n",
         "## 3. AI analyst evaluation\n",
         "Scoring procedure:\n",
         "- **Routing accuracy** – correct backend function and parameters for supported questions.",
         "- **Numerical correctness** – share of required key numbers (taken from the independently recomputed verified result) found in the answer.",
         "- **Unsupported-claim rate** – numbers in the answer that appear nowhere in the verified result (small structural numbers 1,2,3,4,5,10,30 are ignored; this is a known limitation of the check).",
         "- **Missing-information handling** – unsupported questions are refused and invalid IDs are corrected without inventing data.",
         "- **Factual consistency** – per answer 0.5 x numerical correctness + 0.5 x (1 - unsupported-claim share); 1/0 for missing-information questions; mean over all questions.\n"]
    first = next(d for d in runs.values() if d)
    L.append(f"Question set: {first['summary']['questions']} fixed questions "
             f"({first['summary']['supported_questions']} supported, {first['summary']['missing_info_questions']} unsupported/invalid).\n")
    cols = list(runs)
    L.append("| Metric | Target | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
    for r in summary_rows(runs):
        L.append(f"| {r['Metric']} | {r['Target']} | " + " | ".join(r[c] for c in cols) + " |")
    L.append("")
    for mode, data in runs.items():
        if data:
            s = data["summary"]
            L.append(f"- **{mode}**: {s['answers_written_by_gemini']} of {s['questions']} answers were written by Gemini; "
                     f"{s['unsupported_claims']} of {s['total_claims']} numbers were unsupported.")
    v1 = config.EVALUATION_DIR / "results_v1_scorer.json"
    if v1.exists():
        g1 = json.loads(v1.read_text(encoding="utf-8"))["runs"].get("gemini (live API)")
        if g1:
            s1 = g1["summary"]
            L.append("\n### Scorer revision (disclosure)\n")
            L.append(f"The first live Gemini run was scored with an earlier scorer: routing {pct(s1['routing_accuracy'])}, numerical "
                     f"correctness {pct(s1['numerical_correctness'])}, unsupported-claim rate {pct(s1['unsupported_claim_rate'])}, "
                     f"missing-information handling {pct(s1['missing_info_handling'])}, factual consistency {pct(s1['factual_consistency'])} "
                     "(stored in `data/evaluation/results_v1_scorer.json`). Reading the answers showed three scorer problems, not model "
                     "errors: (1) a correct answer of 0.8155 was rejected because the scorer demanded the pre-rounded 0.815; "
                     "(2) questions such as \"How many communities are there?\" were penalised for omitting facts they did not ask for; "
                     "(3) \"anonymised\" was rejected because the check looked for \"anonymous\". The scorer was corrected "
                     "(precision-aware matching, per-question key facts, spelling tolerance, regression tests in `tests/test_evaluation.py`) "
                     "and the evaluation was **re-run live**; the table above is that re-run. Genuine model misses (e.g. a question "
                     "routed to the wrong function) are still counted against Gemini.")
    if gemini_error:
        L.append(f"\n> **Gemini live evaluation was NOT run.** Reason: {gemini_error}\n> Re-run `python scripts/run_evaluation.py` "
                 "when the API is reachable; the table will then include the real Gemini results.")
    for mode, data in runs.items():
        if not data:
            continue
        fails = [r for r in data["rows"] if r["factual_consistency"] < 1.0 or not r["routing_correct"]]
        L.append(f"\n### Questions below full score ({mode})\n")
        if not fails:
            L.append("None.")
        for r in fails:
            L.append(f"- *{r['question']}* – routing correct: {r['routing_correct']}, numerical: "
                     f"{r.get('numerical_correctness')}, unsupported numbers: {r.get('unsupported_values')}, "
                     f"score {r['factual_consistency']:.2f}")
    L.append("\n## 4. Interpretation and limits\n")
    L.append("- The deterministic mode shows the correctness of routing and of the verified numbers. It does not test Gemini's wording.")
    L.append("- Only the Gemini column measures whether the LLM stays faithful to the verified context.")
    L.append("- 29 questions is a small benchmark; percentages have coarse resolution (one question = 3.4 points).")
    L.append("- Numeric faithfulness does not prove the qualitative interpretation is right.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gemini", action="store_true")
    a = ap.parse_args()
    state = ensure_pipeline(progress=lambda m, f: print(f"[{f:4.0%}] {m}", flush=True))
    svc = AnalysisService(AnalysisRepository(state.db_path), state.run_id, state.network)
    print("Running deterministic evaluation...")
    runs: dict = {"deterministic (no LLM)": run_evaluation(svc, Analyst(svc, NoKey()))}
    gemini_error = None
    if a.skip_gemini:
        gemini_error = "skipped with --skip-gemini"
    else:
        client = GeminiClient(cooldown_s=0)   # no cool-down: every question really tries Gemini
        if not client.is_configured:
            gemini_error = "GEMINI_API_KEY is not set"
        else:
            try:
                client.generate("Reply with the single word: ok")     # real call; no simulation
                print(f"Gemini reachable (model {client.last_model_used}). Running live evaluation...")
                runs["gemini (live API)"] = run_evaluation(
                    svc, Analyst(svc, client), on_row=lambda r: print("  scored:", r["question"][:60], flush=True))
            except GeminiUnavailableError as exc:
                gemini_error = str(exc)[:300]
    if "gemini (live API)" not in runs:
        runs["gemini (live API)"] = None
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": ts, "mode": ", ".join(k for k, v in runs.items() if v),
               "summary_table": summary_rows(runs), "gemini_error": gemini_error,
               "scoring_note": "Target 93%. See docs/evaluation.md for the scoring procedure and limits.",
               "runs": runs}
    (config.EVALUATION_DIR / "results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_markdown(config.PROJECT_ROOT / "docs" / "evaluation.md", svc, runs, gemini_error, ts)
    for r in summary_rows(runs):
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
