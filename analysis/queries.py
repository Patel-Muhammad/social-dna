"""Verified backend functions used by the dashboard and the AI analyst.

Every function returns a plain JSON-serialisable dict built from stored calculations.
The LLM never computes these numbers; it only explains them.
"""
from __future__ import annotations

import pandas as pd

from analysis import influence_analysis
from core.graph_builder import NetworkData
from database.repositories import AnalysisRepository


class InvalidCommunityError(ValueError):
    pass


class InvalidUserError(ValueError):
    pass


def feature_label(feature_id: int) -> str:
    """The dataset encodes artists as anonymous IDs; there is no reliable name mapping."""
    return f"artist feature #{int(feature_id)}"


def _clean(value):
    if isinstance(value, float):
        return None if pd.isna(value) else float(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


class AnalysisService:
    def __init__(self, repo: AnalysisRepository, run_id: int, network: NetworkData | None = None):
        self.repo, self.run_id, self.network = repo, run_id, network
        self.run = repo.get_run(run_id)
        self.communities = repo.communities(run_id).set_index("community_number", drop=False).rename_axis(None)
        self.interests = repo.interests(run_id)
        self.similarity = repo.similarity(run_id)
        self.users = repo.user_metrics(run_id).set_index("user_id")
        self.metrics = repo.network_metrics(run_id)
        self.metric_notes = repo.network_metric_notes(run_id)
        self.total_users = int(self.communities.member_count.sum())
        members = repo.members(run_id)
        self.partition: dict[int, int] = dict(zip(members.user_id.astype(int), members.community.astype(int)))

    # --- validation ---------------------------------------------------------------------
    def community_ids(self) -> list[int]:
        return [int(c) for c in self.communities.index]

    def validate_community(self, community_id) -> int:
        try:
            cid = int(community_id)
        except (TypeError, ValueError):
            raise InvalidCommunityError(f"'{community_id}' is not a valid community number.") from None
        if cid not in self.communities.index:
            ids = self.community_ids()
            raise InvalidCommunityError(f"Community {cid} does not exist. Valid communities are {ids[0]} to {ids[-1]}.")
        return cid

    # --- building blocks ----------------------------------------------------------------
    def _features(self, cid: int, kind: str, n: int = 10) -> list[dict]:
        df = self.interests[(self.interests.community == cid) & (self.interests.kind == kind)].head(n)
        out = _records(df[["rank", "feature_id", "frequency", "prevalence", "network_prevalence", "lift",
                           "distinctiveness_score"]])
        for r in out:
            r["label"] = feature_label(r["feature_id"])
        return out

    def _community_row(self, cid: int) -> dict:
        r = self.communities.loc[cid]
        return {
            "community_id": cid, "member_count": int(r.member_count),
            "percentage_of_network": float(r.percentage_of_network),
            "internal_edges": int(r.internal_edges), "external_edges": int(r.external_edges),
            "internal_density": float(r.internal_density), "avg_internal_degree": float(r.avg_internal_degree),
            "avg_degree": float(r.avg_degree), "conductance": float(r.conductance),
            "within_community_cosine": float(r.within_cosine), "within_community_jaccard": float(r.within_jaccard),
            "cosine_to_network": float(r.cosine_to_network),
            "avg_features_per_user": float(r.avg_features_per_user),
        }

    def _similar_to(self, cid: int, n: int = 5) -> list[dict]:
        s = self.similarity
        a = s[s.community_a == cid].rename(columns={"community_b": "other"}).drop(columns="community_a")
        b = s[s.community_b == cid].rename(columns={"community_a": "other"}).drop(columns="community_b")
        both = pd.concat([a, b]).sort_values("cosine_similarity", ascending=False).head(n)
        return _records(both)

    def pair_similarity(self, a: int, b: int) -> dict:
        lo, hi = min(a, b), max(a, b)
        row = self.similarity[(self.similarity.community_a == lo) & (self.similarity.community_b == hi)]
        if row.empty:
            raise InvalidCommunityError("Choose two different communities.")
        r = row.iloc[0]
        return {"cosine_similarity": float(r.cosine_similarity), "jaccard_similarity": float(r.jaccard_similarity),
                "cross_edges": int(r.cross_edges)}

    # --- public API (names mirror the specification) --------------------------------------
    def list_communities(self) -> dict:
        df = self.communities[["community_number", "member_count", "percentage_of_network",
                               "internal_edges", "external_edges"]]
        return {"kind": "communities_table", "community_count": len(df),
                "communities": _records(df.rename(columns={"community_number": "community_id"}))}

    def get_largest_community(self) -> dict:
        cid = int(self.communities.member_count.idxmax())
        res = self.get_community_details(cid)
        res["note"] = "largest community by member count"
        return res

    def get_network_overview(self) -> dict:
        m = self.metrics
        sizes = self.communities.member_count
        return {
            "kind": "overview",
            "run": {"algorithm": self.run["algorithm"], "seed": self.run["random_seed"],
                    "resolution": self.run["resolution"], "run_id": self.run_id},
            "users": int(m["users"]), "friendships": int(m["edges"]),
            "communities": int(self.run["community_count"]), "modularity": float(self.run["modularity"]),
            "largest_community": {"community_id": int(sizes.idxmax()), "members": int(sizes.max())},
            "smallest_community": {"community_id": int(sizes.idxmin()), "members": int(sizes.min())},
            "average_community_size": float(sizes.mean()),
            "density": m["density"], "average_degree": m["average_degree"],
            "average_clustering": m["average_clustering"], "transitivity": m["transitivity"],
            "connected_components": int(m["connected_components"]),
            "inter_community_edge_share": m.get("inter_community_edge_share"),
        }

    def get_community_details(self, community_id: int) -> dict:
        cid = self.validate_community(community_id)
        return {"kind": "community", **self._community_row(cid),
                "common_features": self._features(cid, "common", 10),
                "distinctive_features": self._features(cid, "distinctive", 10),
                "most_similar_communities": self._similar_to(cid, 3),
                "interest_note": "Artists are anonymous feature IDs in this dataset; no names are available."}

    def get_common_interests(self, community_id: int) -> dict:
        cid = self.validate_community(community_id)
        return {"kind": "interests", "community_id": cid, "member_count": int(self.communities.loc[cid].member_count),
                "common_features": self._features(cid, "common", 15),
                "distinctive_features": self._features(cid, "distinctive", 15),
                "interest_note": "Artists are anonymous feature IDs in this dataset; no names are available."}

    def get_community_similarity(self, id_1: int, id_2: int) -> dict:
        a, b = self.validate_community(id_1), self.validate_community(id_2)
        if a == b:
            raise InvalidCommunityError("Choose two different communities.")
        return {"kind": "similarity", "community_a": a, "community_b": b, **self.pair_similarity(a, b),
                "cosine_note": "Cosine on TF-IDF community profiles (0-1).",
                "jaccard_note": "Jaccard on each community's top-30 most common artist features (0-1)."}

    def compare_communities(self, id_1: int, id_2: int) -> dict:
        a, b = self.validate_community(id_1), self.validate_community(id_2)
        if a == b:
            raise InvalidCommunityError("Choose two different communities.")
        fa, fb = self._features(a, "common", 30), self._features(b, "common", 30)
        ids_a, ids_b = {f["feature_id"] for f in fa}, {f["feature_id"] for f in fb}
        shared = sorted(ids_a & ids_b)
        by_a, by_b = {f["feature_id"]: f for f in fa}, {f["feature_id"]: f for f in fb}
        return {
            "kind": "comparison", "community_a": self._community_row(a), "community_b": self._community_row(b),
            **{k: v for k, v in self.pair_similarity(a, b).items()},
            "shared_common_features": [{"feature_id": f, "label": feature_label(f),
                                        "prevalence_a": by_a[f]["prevalence"], "prevalence_b": by_b[f]["prevalence"]}
                                       for f in shared[:15]],
            "shared_common_feature_count": len(shared),
            "only_a_common_features": [by_a[f] for f in sorted(ids_a - ids_b, key=lambda x: by_a[x]["rank"])[:8]],
            "only_b_common_features": [by_b[f] for f in sorted(ids_b - ids_a, key=lambda x: by_b[x]["rank"])[:8]],
            "distinctive_a": self._features(a, "distinctive", 5), "distinctive_b": self._features(b, "distinctive", 5),
        }

    def get_most_similar_communities(self, n: int = 5) -> dict:
        top = self.similarity.sort_values("cosine_similarity", ascending=False).head(max(1, min(int(n), 20)))
        return {"kind": "similar_pairs", "pairs": _records(top)}

    def get_influential_users(self, n: int = 10) -> dict:
        n = max(1, min(int(n), 50))
        out = {"kind": "users", "role": "influential",
               "explanation": "Influence = connections (degree), reach (closeness), and PageRank (being "
                              "connected to well-connected users). IDs are anonymised dataset identifiers.",
               "betweenness_note": self.metric_notes.get("note_betweenness_centrality")}
        for col in ("degree", "betweenness_centrality", "closeness_centrality", "pagerank"):
            top = influence_analysis.top_users(self.users.reset_index(), col, n)
            out[f"top_by_{col}"] = _records(top[["user_id", "community", col, "degree"]
                                                if col != "degree" else ["user_id", "community", "degree"]])
        return out

    def get_bridge_users(self, n: int = 10) -> dict:
        n = max(1, min(int(n), 50))
        sel = influence_analysis.select_bridge_users(self.users, n)
        ranks = {c: self.users[c].rank(ascending=False, method="min")
                 for c in ("degree", "betweenness_centrality", "pagerank")}
        rows = []
        for uid, r in sel.iterrows():
            rows.append({"user_id": int(uid), "community": int(r.community), "degree": int(r.degree),
                         "external_degree": int(r.external_degree), "neighbour_communities": int(r.neighbour_communities),
                         "participation_coefficient": float(r.participation_coefficient),
                         "betweenness_centrality": float(r.betweenness_centrality),
                         "explanation": influence_analysis.explain_user(
                             int(uid), r, int(r.community), int(ranks["degree"][uid]),
                             int(ranks["betweenness_centrality"][uid]), int(ranks["pagerank"][uid]), len(self.users))})
        return {"kind": "users", "role": "bridge", "users": rows,
                "definition": "Bridge users: degree >= 5 and friends in >= 3 communities, ranked by betweenness. "
                              "A hub has many friends; a bridge links different communities.",
                "betweenness_note": self.metric_notes.get("note_betweenness_centrality")}

    def get_user_details(self, user_id: int) -> dict:
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            raise InvalidUserError(f"'{user_id}' is not a valid user ID.") from None
        if uid not in self.users.index:
            raise InvalidUserError(f"User {uid} is not in the dataset.")
        r = self.users.loc[uid]
        return {"kind": "user", "user_id": uid, "community": int(r.community), "degree": int(r.degree),
                "external_degree": int(r.external_degree), "neighbour_communities": int(r.neighbour_communities),
                "participation_coefficient": float(r.participation_coefficient),
                "degree_centrality": float(r.degree_centrality), "betweenness_centrality": float(r.betweenness_centrality),
                "closeness_centrality": float(r.closeness_centrality), "pagerank": float(r.pagerank),
                "liked_artist_features": (len(self.network.user_features(uid)) if self.network else None)}

    def explain_louvain(self) -> dict:
        m = self.metrics
        return {"kind": "louvain", "algorithm": "Louvain", "seed": self.run["random_seed"],
                "resolution": self.run["resolution"], "communities_found": int(self.run["community_count"]),
                "modularity": float(self.run["modularity"]),
                "steps": ["Start with every user in their own community.",
                          "Move each user to the neighbouring community that raises modularity the most.",
                          "Repeat until no single move improves modularity.",
                          "Collapse each community into one super-node and repeat on the smaller graph.",
                          "Stop when modularity can no longer be improved."],
                "stability_nmi_mean": m.get("stability_pairwise_nmi_mean"),
                "greedy_communities": m.get("greedy_community_count"), "greedy_modularity": m.get("greedy_modularity"),
                "louvain_vs_greedy_nmi": m.get("louvain_vs_greedy_nmi")}

    def generate_network_report(self) -> dict:
        return {"kind": "report", "overview": self.get_network_overview(),
                "largest_communities": self.list_communities()["communities"][:5],
                "most_similar_communities": self.get_most_similar_communities(3)["pairs"],
                "top_influential": self.get_influential_users(5), "top_bridges": self.get_bridge_users(5)["users"],
                "algorithms": self.explain_louvain()}
