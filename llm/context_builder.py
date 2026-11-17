"""Turns backend results into compact verified context for Gemini, and into a deterministic
plain-text answer used when Gemini is unavailable."""
from __future__ import annotations

import json

from analysis.queries import feature_label


def _trim(obj, max_list: int = 12):
    """Cap long lists so the prompt stays small; keeps real values untouched."""
    if isinstance(obj, dict):
        return {k: _trim(v, max_list) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_trim(v, max_list) for v in obj[:max_list]]
    if isinstance(obj, float):
        return round(obj, 5)
    return obj


def build_context(result: dict, max_list: int = 12) -> str:
    return json.dumps(_trim(result, max_list), indent=1)


def _feat_list(items: list[dict], n: int = 5) -> str:
    return ", ".join(f"{feature_label(f['feature_id'])} ({f['prevalence']:.0%} of members)" for f in items[:n]) or "none"


ANON_NOTE = " (Artists are anonymous feature IDs in this dataset; real artist names are not available.)"


def plain_answer(result: dict) -> str:
    """Deterministic, number-exact summary. Used as fallback and for evaluation baselines."""
    text = _plain(result)
    return text + ANON_NOTE if result.get("kind") in ("community", "interests", "comparison") else text


def _plain(result: dict) -> str:
    k = result.get("kind")
    if k == "overview":
        return (f"The network has {result['users']:,} users and {result['friendships']:,} friendships. "
                f"Louvain found {result['communities']} communities (modularity {result['modularity']:.3f}). "
                f"The largest is community {result['largest_community']['community_id']} with "
                f"{result['largest_community']['members']:,} members; the smallest is community "
                f"{result['smallest_community']['community_id']} with {result['smallest_community']['members']:,}. "
                f"Average clustering is {result['average_clustering']:.3f}; average degree {result['average_degree']:.2f}.")
    if k == "communities_table":
        rows = "; ".join(f"community {c['community_id']}: {c['member_count']:,} members" for c in result["communities"])
        return f"There are {result['community_count']} communities. {rows}."
    if k == "community":
        return (f"Community {result['community_id']} has {result['member_count']:,} members "
                f"({result['percentage_of_network']:.1f}% of the network), {result['internal_edges']:,} internal and "
                f"{result['external_edges']:,} external connections. Most common: {_feat_list(result['common_features'])}. "
                f"Most distinctive: {_feat_list(result['distinctive_features'])}.")
    if k == "interests":
        return (f"Community {result['community_id']} ({result['member_count']:,} members). Most common interests: "
                f"{_feat_list(result['common_features'])}. Most distinctive: {_feat_list(result['distinctive_features'])}.")
    if k == "similarity":
        return (f"Communities {result['community_a']} and {result['community_b']}: cosine similarity "
                f"{result['cosine_similarity']:.3f}, Jaccard similarity {result['jaccard_similarity']:.3f}, "
                f"{result['cross_edges']:,} friendships between them.")
    if k == "comparison":
        a, b = result["community_a"], result["community_b"]
        return (f"Community {a['community_id']} has {a['member_count']:,} members and community {b['community_id']} "
                f"has {b['member_count']:,}. Cosine similarity {result['cosine_similarity']:.3f}, Jaccard "
                f"{result['jaccard_similarity']:.3f}; they share {result['shared_common_feature_count']} of their top-30 "
                f"common artist features. Internal edges: {a['internal_edges']:,} vs {b['internal_edges']:,}.")
    if k == "similar_pairs":
        rows = "; ".join(f"communities {p['community_a']} & {p['community_b']} (cosine {p['cosine_similarity']:.3f})"
                         for p in result["pairs"])
        return f"Most similar community pairs: {rows}."
    if k == "users" and result["role"] == "bridge":
        rows = " ".join(f"- {u['explanation']}" for u in result["users"][:5])
        return f"Users who connect different communities (bridges):\n{rows}"
    if k == "users":
        rows = "; ".join(f"user {u['user_id']} ({u['degree']} friends)" for u in result["top_by_degree"][:5])
        pr = "; ".join(f"user {u['user_id']}" for u in result["top_by_pagerank"][:5])
        return f"Most connected: {rows}. Top PageRank: {pr}."
    if k == "user":
        return (f"User {result['user_id']} is in community {result['community']} with {result['degree']} friends "
                f"({result['external_degree']} outside their community, across {result['neighbour_communities']} communities).")
    if k == "louvain":
        return (f"Louvain (seed {result['seed']}) found {result['communities_found']} communities with modularity "
                f"{result['modularity']:.3f}. Steps: " + " ".join(f"{i + 1}) {s}" for i, s in enumerate(result["steps"])))
    if k == "report":
        o = result["overview"]
        return plain_answer(o) + " " + plain_answer({"kind": "similar_pairs", "pairs": result["most_similar_communities"]})
    return json.dumps(_trim(result))
