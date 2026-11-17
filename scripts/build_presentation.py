"""Build docs/Social_DNA_Presentation.html, a single self-contained HTML presentation.
Every number is read from the SQLite database, data/evaluation/results.json and docs/test_report.md
at build time. Open the file in any modern browser (arrow keys, drag, S = notes, O = overview, F = fullscreen)."""
import base64
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from analysis.queries import AnalysisService  # noqa: E402
from core import config  # noqa: E402
from database.repositories import AnalysisRepository, DatasetRepository  # noqa: E402

SHOTS = [("01_overview.png", "Overview"), ("02_explorer.png", "Explorer"), ("04_comparison.png", "Comparison"),
         ("05_influence.png", "Influence"), ("06_ai_analyst.png", "AI analyst")]


def shot_tag(path: Path, alt: str) -> str:
    img = Image.open(path).convert("RGB")
    w = 1500
    img = img.resize((w, int(img.height * w / img.width)), Image.LANCZOS)
    img = img.crop((0, 0, w, min(img.height, int(w * 0.625))))           # top 16:10
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80, optimize=True)
    return f'<img alt="{alt}" src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def main() -> None:
    repo = AnalysisRepository()
    ds = DatasetRepository().latest()
    run = repo.find_run(ds["id"], "louvain", config.DEFAULT_SEED, config.DEFAULT_RESOLUTION)
    svc = AnalysisService(repo, run["id"])
    m, o = svc.metrics, svc.get_network_overview()
    val = json.loads(ds["validation_json"])
    comms = svc.communities
    ids = svc.community_ids()

    # cosine matrix between communities (diagonal 1)
    pos = {c: i for i, c in enumerate(ids)}
    cos = np.eye(len(ids))
    for r in svc.similarity.itertuples():
        cos[pos[r.community_a], pos[r.community_b]] = cos[pos[r.community_b], pos[r.community_a]] = r.cosine_similarity
    c1 = svc.get_common_interests(1)
    features = [{"id": f["feature_id"], "prev": f["prevalence"], "net": f["network_prevalence"], "lift": f["lift"]}
                for f in c1["distinctive_features"][:5]]
    hub_id = int(svc.users["degree"].idxmax())
    hub, br = svc.get_user_details(hub_id), svc.get_bridge_users(1)["users"][0]
    sim = svc.similarity.sort_values("cosine_similarity", ascending=False).iloc[0]
    top3 = comms.member_count.sort_values(ascending=False).head(3).sum() / o["users"]

    # evaluation rows (real results; "not run" stays "not run")
    ev_path = config.EVALUATION_DIR / "results.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None
    names = [("routing_accuracy", "Routing accuracy", False), ("numerical_correctness", "Numerical correctness", False),
             ("unsupported_claim_rate", "Unsupported-claim rate", True), ("missing_info_handling", "Missing-info handling", False),
             ("factual_consistency", "Factual consistency", False)]
    rows, nq = [], 0

    def cell(run_summary, key, lower):
        if not run_summary:
            return None
        v = run_summary[key]
        return {"pct": v * 100, "ok": (v <= 0.07) if lower else (v >= 0.93)}

    if ev:
        det = ev["runs"].get("deterministic (no LLM)")
        gem = ev["runs"].get("gemini (live API)")
        ds_ = det["summary"] if det else None
        gs_ = gem["summary"] if gem else None
        nq = (ds_ or gs_)["questions"]
        rows = [{"name": n, "lower": lw, "det": cell(ds_, k, lw), "gem": cell(gs_, k, lw)} for k, n, lw in names]
    tr = ROOT / "docs" / "test_report.md"
    test_line = ""
    if tr.exists():
        mt = re.search(r"\*\*(\d+) tests: (\d+) passed, (\d+) failed", tr.read_text(encoding="utf-8"))
        if mt:
            test_line = f"{mt.group(2)} of {mt.group(1)} automated tests passed ({mt.group(3)} failed)."

    data = {"communities": [{"id": int(c), "n": int(n)} for c, n in zip(comms.community_number, comms.member_count)],
            "ids": ids, "cos": [[round(float(v), 4) for v in row] for row in cos], "features": features,
            "hub": {"nc": hub["neighbour_communities"]}, "br": {"nc": br["neighbour_communities"]},
            "eval": {"rows": rows}}
    f = lambda x: f"{x:,}"  # noqa: E731
    tokens = {
        "users": f(val["total_users"]), "edges": f(val["total_edges_clean"]), "ncomm": str(o["communities"]),
        "users_raw": val["total_users"], "edges_raw": val["total_edges_clean"], "features_raw": val["feature_count"],
        "components_raw": val["connected_components"], "avg_features": f"{val['mean_features_per_user']:.0f}",
        "dups": val["duplicate_edges"], "loops": val["self_loops"], "isolated": val["isolated_nodes"],
        "matches": "yes" if val["matches_documented_counts"] else "no", "nofeat": val["users_without_features"],
        "g_comm": int(m["greedy_community_count"]), "g_mod": f"{m['greedy_modularity']:.3f}", "modularity": f"{o['modularity']:.3f}",
        "largest": f(o["largest_community"]["members"]), "largest_id": o["largest_community"]["community_id"],
        "top3": f"{top3:.0%}", "cross": f"{o['inter_community_edge_share']:.1%}", "c1_members": f(c1["member_count"]),
        "sim_a": int(sim.community_a), "sim_b": int(sim.community_b), "sim_cos": f"{sim.cosine_similarity:.2f}",
        "sim_jac": f"{sim.jaccard_similarity:.2f}", "sim_cross": f(int(sim.cross_edges)), "within": f"{comms.within_cosine.mean():.2f}",
        "hub_id": hub_id, "hub_deg": hub["degree"], "hub_ext": hub["external_degree"], "hub_nc": hub["neighbour_communities"],
        "hub_pc": f"{hub['participation_coefficient']:.2f}", "br_id": br["user_id"], "br_deg": br["degree"],
        "br_ext": br["external_degree"], "br_nc": br["neighbour_communities"], "br_pc": f"{br['participation_coefficient']:.2f}",
        "mod_range": f"{m['stability_modularity_min']:.3f} to {m['stability_modularity_max']:.3f}",
        "nmi": f"{m['stability_pairwise_nmi_mean']:.2f}", "ari": f"{m['stability_pairwise_ari_mean']:.2f}",
        "agree": f"{m['louvain_vs_greedy_nmi']:.2f}", "nq": nq or "n/a", "tests": test_line,
        "shots": "".join(shot_tag(ROOT / "docs" / "screenshots" / fn, alt) for fn, alt in SHOTS if (ROOT / "docs" / "screenshots" / fn).exists()),
    }
    html = (ROOT / "scripts" / "presentation_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/{}", json.dumps(data))
    for k, v in tokens.items():
        html = html.replace("{{" + k + "}}", str(v))
    left = re.findall(r"\{\{\w+\}\}", html)
    if left:
        raise SystemExit(f"Unreplaced placeholders: {sorted(set(left))}")
    out = ROOT / "docs" / "Social_DNA_Presentation.html"
    out.write_text(html, encoding="utf-8")
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
