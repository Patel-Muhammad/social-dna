"""Build the IEEE-format paper: figures (matplotlib) + docs/paper/social_dna_paper.tex (filled from the
real database / evaluation results) and, if `tectonic` is installed, the PDF.
Needs: pip install matplotlib   (and optionally `tectonic` to compile LaTeX)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from analysis.queries import AnalysisService  # noqa: E402
from core import config  # noqa: E402
from database.repositories import AnalysisRepository, DatasetRepository  # noqa: E402

OUT = ROOT / "docs" / "paper"
FIG = OUT / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": .6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
BLUE, TEAL, SLATE, AMBER = "#1D4ED8", "#0F766E", "#475569", "#B45309"


def pct(x, d=1):
    return f"{x * 100:.{d}f}\\%"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    repo = AnalysisRepository()
    ds = DatasetRepository().latest()
    run = repo.find_run(ds["id"], "louvain", config.DEFAULT_SEED, config.DEFAULT_RESOLUTION)
    svc = AnalysisService(repo, run["id"])
    m, c, u, sim = svc.metrics, svc.communities, svc.users, svc.similarity
    val = json.loads(ds["validation_json"])
    ids = svc.community_ids()

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.bar(range(1, len(ids) + 1), c.member_count, color=BLUE, width=.8)
    ax.set_yscale("log"); ax.set_xlabel("Community (ranked by size)"); ax.set_ylabel("Members (log)")
    ax.set_xlim(.4, len(ids) + .6); fig.tight_layout(pad=.3); fig.savefig(FIG / "sizes.pdf"); plt.close(fig)

    pos = {x: i for i, x in enumerate(ids)}
    cos = np.eye(len(ids))
    for r in sim.itertuples():
        cos[pos[r.community_a], pos[r.community_b]] = cos[pos[r.community_b], pos[r.community_a]] = r.cosine_similarity
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    im = ax.imshow(cos, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Community"); ax.set_ylabel("Community")
    ax.set_xticks(range(0, len(ids), 5)); ax.set_xticklabels(ids[::5]); ax.set_yticks(range(0, len(ids), 5)); ax.set_yticklabels(ids[::5])
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03); cb.set_label("Cosine similarity")
    for s in ax.spines.values():
        s.set_visible(True)
    fig.tight_layout(pad=.3); fig.savefig(FIG / "heatmap.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    sc = ax.scatter(u.degree, u.betweenness_centrality, c=u.participation_coefficient, cmap="viridis", s=5, alpha=.65, linewidths=0)
    ax.set_xscale("log"); ax.set_xlabel("Degree (friends)"); ax.set_ylabel("Betweenness (sampled)")
    fig.colorbar(sc, ax=ax, pad=.02).set_label("Participation coefficient")
    hub_id = int(u.degree.idxmax()); br = svc.get_bridge_users(1)["users"][0]
    ax.scatter([u.degree[hub_id]], [u.betweenness_centrality[hub_id]], marker="D", s=26, facecolors="none", edgecolors="#B91C1C", linewidths=1.1,
               label=f"hub (user {hub_id})")
    ax.scatter([u.degree[br["user_id"]]], [u.betweenness_centrality[br["user_id"]]], marker="*", s=70, c="k",
               label=f"bridge (user {br['user_id']})")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, handletextpad=.3)
    fig.tight_layout(pad=.3); fig.savefig(FIG / "hubbridge.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 1.55)); ax.axis("off")
    boxes = [("LastFM Asia\nfiles", "download, validate,\nclean, cache"), ("Graph and\nfeatures", "NetworkX,\nSciPy CSR"),
             ("Analysis", "Louvain, Greedy,\ninterests, centralities"), ("SQLite", "runs, communities,\nprofiles, metrics"),
             ("Query\nservice", "13 validated\nfunctions"), ("Analyst and\ndashboard", "Gemini explains\nverified JSON")]
    w, g = 1.0, .22
    for i, (t, s) in enumerate(boxes):
        x = i * (w + g)
        ax.add_patch(FancyBboxPatch((x, .35), w, .6, boxstyle="round,pad=0.02,rounding_size=0.05", fc="#EFF6FF" if i != 4 else "#ECFDF5",
                                    ec=BLUE if i != 4 else TEAL, lw=.8))
        ax.text(x + w / 2, .75, t, ha="center", va="center", fontsize=7, fontweight="bold")
        ax.text(x + w / 2, .17, s, ha="center", va="center", fontsize=6.2, color=SLATE)
        if i < len(boxes) - 1:
            ax.annotate("", (x + w + g - .02, .65), (x + w + .02, .65), arrowprops=dict(arrowstyle="->", lw=.8, color=SLATE))
    ax.set_xlim(-.05, len(boxes) * (w + g) - g + .05); ax.set_ylim(0, 1.05)
    fig.savefig(FIG / "architecture.pdf", bbox_inches="tight", pad_inches=.02); plt.close(fig)

    # ---------------- numbers ----------------
    c1 = svc.get_common_interests(1)
    feat_rows = "\n".join(f"{f['feature_id']} & {pct(f['prevalence'], 0)} & {pct(f['network_prevalence'], 0)} & {f['lift']:.1f}$\\times$ \\\\"
                          for f in c1["distinctive_features"][:5])
    hub = svc.get_user_details(hub_id)
    rk = lambda col, uid: int(u[col].rank(ascending=False, method="min")[uid])  # noqa: E731
    top_sim = sim.sort_values("cosine_similarity", ascending=False).iloc[0]
    low_sim = sim.sort_values("cosine_similarity").iloc[0]
    n_bridge = int(((u.degree >= 5) & (u.neighbour_communities >= 3)).sum())
    ev = json.loads((config.EVALUATION_DIR / "results.json").read_text(encoding="utf-8"))
    v1 = json.loads((config.EVALUATION_DIR / "results_v1_scorer.json").read_text(encoding="utf-8"))
    det, gem, g1 = (ev["runs"]["deterministic (no LLM)"]["summary"], ev["runs"]["gemini (live API)"]["summary"],
                    v1["runs"]["gemini (live API)"]["summary"])
    fallback = [r["question"] for r in ev["runs"]["gemini (live API)"]["rows"] if r["kind"] == "supported" and not r["used_llm"]]
    tr = (ROOT / "docs" / "test_report.md").read_text(encoding="utf-8")
    import re
    mt = re.search(r"\*\*(\d+) tests: (\d+) passed", tr)

    def ev_row(label, key, lower=False):
        f = lambda s: pct(s[key])  # noqa: E731
        return f"{label} & {pct(det[key])} & {pct(g1[key])} & {pct(gem[key])} \\\\"

    T = {
        "N_USERS": f"{val['total_users']:,}".replace(",", "{,}"), "N_EDGES": f"{val['total_edges_clean']:,}".replace(",", "{,}"),
        "N_FEATURES": f"{val['feature_count']:,}".replace(",", "{,}"), "AVG_FEATURES": f"{val['mean_features_per_user']:.0f}",
        "NO_FEAT": str(val["users_without_features"]), "DENSITY": f"{m['density']:.4f}", "AVG_DEG": f"{m['average_degree']:.2f}",
        "MED_DEG": f"{m['median_degree']:.0f}", "MAX_DEG": f"{m['max_degree']:.0f}", "CLUST": f"{m['average_clustering']:.3f}",
        "TRANS": f"{m['transitivity']:.3f}", "NCOMM": str(run["community_count"]), "Q": f"{run['modularity']:.4f}",
        "LARGEST": f"{m['largest_community_size']:.0f}".replace(",", ""), "SMALLEST": f"{m['smallest_community_size']:.0f}",
        "MEAN_SIZE": f"{m['average_community_size']:.0f}", "TOP3": pct(c.member_count.sort_values(ascending=False).head(3).sum() / m["users"]),
        "CROSS": pct(m["inter_community_edge_share"]), "G_COMM": f"{m['greedy_community_count']:.0f}", "G_Q": f"{m['greedy_modularity']:.4f}",
        "G_NMI": f"{m['louvain_vs_greedy_nmi']:.3f}", "G_ARI": f"{m['louvain_vs_greedy_ari']:.3f}",
        "ST_NMI": f"{m['stability_pairwise_nmi_mean']:.3f}", "ST_NMI_MIN": f"{m['stability_pairwise_nmi_min']:.3f}",
        "ST_ARI": f"{m['stability_pairwise_ari_mean']:.3f}", "ST_ARI_MIN": f"{m['stability_pairwise_ari_min']:.3f}",
        "ST_QMIN": f"{m['stability_modularity_min']:.4f}", "ST_QMAX": f"{m['stability_modularity_max']:.4f}",
        "ST_QSTD": f"{m['stability_modularity_std']:.4f}", "ST_CMIN": f"{m['stability_community_count_min']:.0f}",
        "ST_CMAX": f"{m['stability_community_count_max']:.0f}", "COUNTRY_NMI": f"{m['country_association_nmi']:.3f}",
        "WITHIN_COS": f"{c.within_cosine.mean():.3f}", "WITHIN_COS_MIN": f"{c.within_cosine.min():.3f}", "WITHIN_COS_MAX": f"{c.within_cosine.max():.3f}",
        "WITHIN_JAC": f"{c.within_jaccard.mean():.3f}", "COS_NET": f"{c.cosine_to_network.mean():.3f}",
        "COS_MEAN": f"{sim.cosine_similarity.mean():.3f}", "COS_MAX": f"{top_sim.cosine_similarity:.3f}", "COS_MIN": f"{low_sim.cosine_similarity:.3f}",
        "TOP_A": str(int(top_sim.community_a)), "TOP_B": str(int(top_sim.community_b)), "TOP_JAC": f"{top_sim.jaccard_similarity:.3f}",
        "TOP_CROSS": str(int(top_sim.cross_edges)), "PAIRS": str(len(sim)), "JAC_MEAN": f"{sim.jaccard_similarity.mean():.3f}",
        "FEAT_ROWS": feat_rows, "C1_SIZE": str(c1["member_count"]),
        "HUB_ID": str(hub_id), "HUB_DEG": str(hub["degree"]), "HUB_EXT": str(hub["external_degree"]), "HUB_NC": str(hub["neighbour_communities"]),
        "HUB_P": f"{hub['participation_coefficient']:.3f}", "HUB_B": f"{hub['betweenness_centrality']:.4f}", "HUB_RB": str(rk("betweenness_centrality", hub_id)),
        "BR_ID": str(br["user_id"]), "BR_DEG": str(br["degree"]), "BR_EXT": str(br["external_degree"]), "BR_NC": str(br["neighbour_communities"]),
        "BR_P": f"{br['participation_coefficient']:.3f}", "BR_B": f"{br['betweenness_centrality']:.4f}", "BR_RB": str(rk("betweenness_centrality", br["user_id"])),
        "BR_RD": str(rk("degree", br["user_id"])), "HUB_RD": str(rk("degree", hub_id)), "N_BRIDGE": str(n_bridge),
        "EV_N": str(gem["questions"]), "EV_SUP": str(gem["supported_questions"]), "EV_MISS": str(gem["missing_info_questions"]),
        "EV_LLM": str(gem["answers_written_by_gemini"]), "EV_CLAIMS": str(gem["total_claims"]), "EV_BAD": str(gem["unsupported_claims"]),
        "EV_DET_CLAIMS": str(det["total_claims"]), "EV_FALLBACK": str(len(fallback)),
        "EV_ROWS": "\n".join([ev_row("Routing accuracy", "routing_accuracy"), ev_row("Numerical correctness", "numerical_correctness"),
                              ev_row("Unsupported-claim rate", "unsupported_claim_rate"), ev_row("Missing-information handling", "missing_info_handling"),
                              ev_row("Factual consistency", "factual_consistency")]),
        "G_ROUTING": pct(gem["routing_accuracy"]), "G_NUM": pct(gem["numerical_correctness"]),
        "G_UNSUP": pct(gem["unsupported_claim_rate"]), "G_FACT": pct(gem["factual_consistency"]),
        "G1_NUM": pct(g1["numerical_correctness"]), "G1_FACT": pct(g1["factual_consistency"]),
        "TESTS": mt.group(1) if mt else "n/a", "MODEL": config.GEMINI_MODEL.replace("_", "\\_"), "BETW_K": str(config.BETWEENNESS_SAMPLE_K),
        "SEEDS": str(config.STABILITY_SEEDS), "SEED": str(config.DEFAULT_SEED),
    }
    tex = (ROOT / "scripts" / "paper_template.tex").read_text(encoding="utf-8")
    for k, v in T.items():
        tex = tex.replace(f"@@{k}@@", v)
    import re as _re
    left = sorted(set(_re.findall(r"@@\w+@@", tex)))
    if left:
        raise SystemExit(f"Unfilled tokens: {left}")
    (OUT / "social_dna_paper.tex").write_text(tex, encoding="utf-8")
    print("wrote", OUT / "social_dna_paper.tex")
    if shutil.which("tectonic"):
        r = subprocess.run(["tectonic", "-X", "compile", "social_dna_paper.tex"] if False else ["tectonic", "social_dna_paper.tex"],
                           cwd=OUT, capture_output=True, text=True)
        print(r.stdout[-600:], r.stderr[-1500:])
        print("PDF:", (OUT / "social_dna_paper.pdf").exists())
    else:
        print("tectonic not found: compile social_dna_paper.tex with pdflatex (IEEEtran)")


if __name__ == "__main__":
    main()
