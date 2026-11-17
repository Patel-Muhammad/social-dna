import json

import pandas as pd
import streamlit as st

from components.context import get_context
from components.sidebar import render_sidebar
from components.theme import callout, page_header
from core import config

ctx = get_context()
render_sidebar(ctx)
svc, m = ctx.service, ctx.service.metrics
page_header("Methodology & Dataset", "How Social DNA works, explained simply enough to present in a viva.")

st.subheader("1. Dataset")
st.markdown(f"""
**{config.DATASET_NAME}** — {config.DATASET_VERSION}. Source: [{config.DATASET_PAGE}]({config.DATASET_PAGE})
(alternative: UCI repository #599). Nodes are LastFM users from Asian countries; an edge means two users follow each
other; each user has a list of artists they like, encoded as anonymous feature IDs (no artist names are provided).
A target file stores an anonymised country code per user; we do **not** treat it as community ground truth.

*Licence and attribution:* distributed by SNAP for research. Cite Rozemberczki & Sarkar, *Characteristic Functions on Graphs:
Birds of a Feather, from Statistical Descriptors to Parametric Models*, CIKM 2020 (arXiv:2005.07959).
""")
v = ctx.state.validation
st.dataframe(pd.DataFrame(
    [(k.replace("_", " "), str(val)) for k, val in v.items() if k not in ("warnings", "errors")],
    columns=["Validation check", "Result"]), hide_index=True, use_container_width=True)

st.subheader("2. Graph representation")
st.markdown("Each user is a **node**, each friendship an undirected **edge** (NetworkX `Graph`). Original dataset user IDs are "
            "used everywhere. Duplicate edges and self-loops would be removed; the counts above show how many there were.")

st.subheader("3. Louvain community detection and modularity")
st.markdown(f"""
**Modularity** compares the fraction of friendships inside groups with what we would expect if friendships were random with the
same degrees. 0 means no better than random; values around 0.3–0.7 are typically considered strong.

**Louvain** greedily moves users between communities to raise modularity, merges communities into super-nodes, and repeats.
It is unsupervised: no labels are used. A random **seed** controls the order users are visited, so runs are reproducible.

This run: seed **{svc.run['random_seed']}**, resolution **{svc.run['resolution']:g}**, **{svc.run['community_count']}** communities,
modularity **{svc.run['modularity']:.4f}**.
""")

st.subheader("4. Interest aggregation and similarity")
st.markdown("""
For each community we count how many members like each artist feature: **frequency**, **prevalence** (frequency ÷ members) and
**lift** (prevalence in the community ÷ prevalence in the whole network). *Common* features have the highest frequency;
*distinctive* features have the highest `prevalence × log2(lift)` with a minimum-support rule, so rare accidents do not win.

**Cosine similarity** compares TF-IDF weighted community profiles (rare artists count more). **Jaccard similarity** is
|A∩B| / |A∪B| over each community's top-30 common features. Both range 0–1.
""")

st.subheader("5. Gemini integration")
st.markdown("""
Gemini never computes anything. The backend computes verified results; a router (Gemini function calling, or keyword rules when
Gemini is unavailable) chooses which function to call and validates its parameters; Gemini then **explains** the returned JSON.
Questions the data cannot answer (artist names, real identities) are refused. If Gemini is unreachable the dashboard keeps working.
""")

st.subheader("6. Evaluation")
st.markdown("Louvain has **no ground truth** here, so classification accuracy does not apply. We report evidence that can be measured:")
eval_rows = [
    ("Louvain modularity", f"{svc.run['modularity']:.4f}"),
    (f"Stability over {config.STABILITY_SEEDS} seeds: mean pairwise NMI / ARI",
     f"{m['stability_pairwise_nmi_mean']:.3f} / {m['stability_pairwise_ari_mean']:.3f}"),
    ("Modularity range over seeds", f"{m['stability_modularity_min']:.4f} – {m['stability_modularity_max']:.4f}"),
    ("Greedy Modularity: communities / modularity", f"{int(m['greedy_community_count'])} / {m['greedy_modularity']:.4f}"),
    ("Louvain vs Greedy agreement: NMI / ARI", f"{m['louvain_vs_greedy_nmi']:.3f} / {m['louvain_vs_greedy_ari']:.3f}"),
    ("Association with country code (descriptive, not accuracy)", f"NMI {m.get('country_association_nmi', float('nan')):.3f}"),
]
st.dataframe(pd.DataFrame(eval_rows, columns=["Measure", "Value"]), hide_index=True, use_container_width=True)
path = config.EVALUATION_DIR / "results.json"
if path.exists():
    ev = json.loads(path.read_text(encoding="utf-8"))
    st.markdown("**Analyst evaluation (fixed question set)** — "
                f"generated {ev.get('generated_at', '?')}, mode: {ev.get('mode', '?')}")
    st.dataframe(pd.DataFrame(ev["summary_table"]), hide_index=True, use_container_width=True)
    st.caption(ev.get("scoring_note", ""))
else:
    st.info("No analyst evaluation saved yet. Run `python scripts/run_evaluation.py` and reload this page.")

st.subheader("7. Limitations")
st.markdown("""
- Artist features are anonymous IDs; we cannot name artists or genres.
- Betweenness is a sampled approximation (labelled in the UI); other centralities are exact.
- Louvain is heuristic: different seeds give slightly different partitions (quantified above).
- Communities reflect friendship structure; they are not proof of shared taste, and shared taste is not proof of friendship.
- Data is a March 2020 snapshot of Asian LastFM users; results do not generalise to other networks.
- Gemini explanations are only as good as the supplied context and can still be phrased imperfectly; always check the numbers shown.
""")
