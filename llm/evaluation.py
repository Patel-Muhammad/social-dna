"""Fixed-question evaluation of the AI analyst against REAL backend results.

Expected values are computed from the stored analysis at run time (never hard-coded), so the
benchmark stays valid for any seed/dataset. Scoring procedure (documented in docs/evaluation.md):

  routing accuracy        correct function AND parameters chosen
  numerical correctness   share of required key numbers (from the verified result) found in the answer
                          (precision-aware: 0.8155 satisfies 0.81550; focused per question where the question asks for one fact)
  unsupported-claim rate  share of numbers in answers that do not appear in the verified result
  missing-info handling   unsupported / invalid questions are refused or corrected, with no invented data
  factual consistency     per-answer: 0.5 * numerical correctness + 0.5 * (1 - unsupported-claim share);
                          for unsupported/invalid questions: 1 if handled correctly else 0. Mean over questions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from analysis.queries import AnalysisService
from llm.query_router import Analyst

STRUCTURAL_NUMBERS = {1, 2, 3, 4, 5, 10, 30}   # "top 5", "top-30", step numbers: not data claims


@dataclass
class Q:
    text: str
    kind: str                                  # 'supported' | 'unsupported' | 'invalid'
    function: str | None = None
    params: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    must_mention: list[str] = field(default_factory=list)
    facts: str | None = None            # which key facts this question is really asking for (see FOCUSED)


def question_set(service: AnalysisService) -> list[Q]:
    ids = service.community_ids()
    c = lambda i: ids[min(i, len(ids)) - 1]   # noqa: E731  (community number i, clamped for small datasets)
    top_user = int(service.users["degree"].idxmax())
    follow = [{"role": "user", "text": f"Explain community {c(3)}", "params": {"community_id": c(3)}},
              {"role": "assistant", "text": "...", "params": {"community_id": c(3)}}]
    return [
        Q("Discover all the communities.", "supported", "list_communities"),
        Q("How many communities are there?", "supported", "list_communities", facts="count"),
        Q("Which is the largest community?", "supported", "get_largest_community", facts="largest"),
        Q(f"Explain community {c(5)}.", "supported", "get_community_details", {"community_id": c(5)}),
        Q(f"Explain community {c(1)}.", "supported", "get_community_details", {"community_id": c(1)}),
        Q(f"Is community {c(9)} tightly connected?", "supported", "get_community_details", {"community_id": c(9)}),
        Q(f"What interests are common in community {c(3)}?", "supported", "get_common_interests", {"community_id": c(3)}),
        Q(f"What do members of community {c(7)} like?", "supported", "get_common_interests", {"community_id": c(7)}),
        Q(f"What are the distinctive interests of community {c(4)}?", "supported", "get_common_interests", {"community_id": c(4)}, facts="distinctive"),
        Q(f"Compare community {c(2)} and community {c(4)}.", "supported", "compare_communities", {"id_1": c(2), "id_2": c(4)}),
        Q(f"Compare community {c(1)} with community {c(6)}.", "supported", "compare_communities", {"id_1": c(1), "id_2": c(6)}),
        Q(f"How similar are community {c(2)} and community {c(3)}?", "supported", "get_community_similarity",
          {"id_1": c(2), "id_2": c(3)}),
        Q("Which communities are most similar?", "supported", "get_most_similar_communities"),
        Q("Find users who connect different communities.", "supported", "get_bridge_users"),
        Q("Who are the most influential users?", "supported", "get_influential_users"),
        Q("Who has the highest PageRank?", "supported", "get_influential_users"),
        Q("Give me a report on the entire network.", "supported", "generate_network_report"),
        Q("Explain how Louvain discovered these communities.", "supported", "explain_louvain"),
        Q("Give me a network overview.", "supported", "get_network_overview"),
        Q("What is the modularity of the network?", "supported", "get_network_overview", facts="modularity"),
        Q(f"Tell me about user {top_user}.", "supported", "get_user_details", {"user_id": top_user}),
        Q("What interests does it share?", "supported", "get_common_interests", {"community_id": c(3)}, history=follow),
        Q(f"Which artist names do members of community {c(3)} like?", "supported", "get_common_interests",
          {"community_id": c(3)}, must_mention=["anonym"]),
        Q("What is the weather in Tokyo today?", "unsupported"),
        Q(f"Which country is community {c(2)} from?", "unsupported"),
        Q(f"Who is user {top_user} in real life?", "unsupported"),
        Q("What will the network look like next year?", "unsupported"),
        Q(f"Explain community {ids[-1] + 100}.", "invalid", must_mention=["does not exist"]),
        Q(f"Compare community {c(2)} and community {c(2)}.", "invalid", must_mention=["different"]),
    ]


# --- number matching -------------------------------------------------------------------------
NUM = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)")


def extract_numbers(text: str) -> list[tuple[float, int]]:
    """(value, printed decimals) for every number in the text; '27,806' -> (27806, 0)."""
    out = []
    for m in NUM.finditer(text):
        raw = m.group(1).rstrip(".").replace(",", "")
        if not raw:
            continue
        dec = len(raw.split(".")[1]) if "." in raw else 0
        out.append((float(raw), dec))
    return out


def flatten_numbers(obj) -> set[float]:
    found: set[float] = set()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float)):
        found.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        found.add(float(len(obj)))
        for v in obj:
            found |= flatten_numbers(v)
    elif isinstance(obj, str):
        found |= {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", obj) if x.strip(".,")}
    return found


def matches(x: float, dec: int, value: float) -> bool:
    tol = 0.5 * 10 ** (-dec) + 1e-9
    return abs(x - value) <= tol or abs(x - 100 * value) <= tol   # also allow the percentage form


def supported(x: float, dec: int, allowed: set[float]) -> bool:
    return x in STRUCTURAL_NUMBERS or any(matches(x, dec, a) for a in allowed)


FOCUSED = {   # question-specific key facts: only what the question actually asks for
    "count": lambda r: [r["community_count"]],
    "largest": lambda r: [r["community_id"], r["member_count"]],
    "modularity": lambda r: [r["modularity"]],
    "distinctive": lambda r: [r["community_id"], r["distinctive_features"][0]["feature_id"]],
}


def required_facts(result: dict, focus: str | None = None) -> list[float]:
    """Key numbers an answer about this result must contain (taken from the verified result)."""
    if focus:
        return FOCUSED[focus](result)
    k = result["kind"]
    if k == "overview":
        return [result["users"], result["friendships"], result["communities"], result["modularity"]]
    if k == "communities_table":
        return [result["community_count"], max(c["member_count"] for c in result["communities"])]
    if k == "community":
        return [result["community_id"], result["member_count"], result["internal_edges"], result["external_edges"]]
    if k == "interests":
        return [result["community_id"], result["member_count"], result["common_features"][0]["feature_id"]]
    if k == "similarity":
        return [result["cosine_similarity"], result["jaccard_similarity"], result["cross_edges"]]
    if k == "comparison":
        return [result["community_a"]["member_count"], result["community_b"]["member_count"],
                result["cosine_similarity"], result["jaccard_similarity"]]
    if k == "similar_pairs":
        p = result["pairs"][0]
        return [p["community_a"], p["community_b"], p["cosine_similarity"]]
    if k == "users" and result["role"] == "bridge":
        return [result["users"][0]["user_id"], result["users"][0]["degree"]]
    if k == "users":
        return [result["top_by_degree"][0]["user_id"], result["top_by_pagerank"][0]["user_id"]]
    if k == "user":
        return [result["user_id"], result["degree"], result["community"]]
    if k == "louvain":
        return [result["communities_found"], result["modularity"]]
    if k == "report":
        o = result["overview"]
        return [o["users"], o["friendships"], o["communities"]]
    return []


REFUSAL_MARKERS = ("cannot answer", "can't answer", "does not contain", "not available", "do not have",
                   "don't have", "no data", "unable", "isn't available", "not provided", "does not include")


def score_response(q: Q, resp, service: AnalysisService) -> dict:
    row = {"question": q.text, "kind": q.kind, "expected_function": q.function, "got_function": resp.function,
           "used_llm": resp.used_llm, "answer": resp.text, "notice": resp.notice,
           "got_params": resp.params, "error": resp.error, "had_result": resp.result is not None}
    text = resp.text.lower()
    if q.kind == "supported":
        route_ok = resp.function == q.function and all(resp.params.get(k) == v for k, v in q.params.items())
        row["routing_correct"] = route_ok
        expected = getattr(service, q.function)(**q.params)           # recomputed independently of the answer
        facts = required_facts(expected, q.facts)
        nums = extract_numbers(resp.text)
        hit = [any(matches(x, d, f) for x, d in nums) for f in facts]
        row["numerical_correctness"] = sum(hit) / len(facts) if facts else 1.0
        allowed = flatten_numbers(expected)
        claims = [(x, d) for x, d in nums]
        bad = [x for x, d in claims if not supported(x, d, allowed)]
        row["claims"], row["unsupported_claims"] = len(claims), len(bad)
        row["unsupported_values"] = bad[:8]
        mention_ok = all(m in text for m in q.must_mention)
        row["missing_info_ok"] = mention_ok if q.must_mention else None
        unsup_share = len(bad) / len(claims) if claims else 0.0
        row["factual_consistency"] = (0.5 * row["numerical_correctness"] + 0.5 * (1 - unsup_share)) if route_ok else 0.0
        if q.must_mention and not mention_ok:
            row["factual_consistency"] *= 0.5
    else:
        refused = resp.error == "unsupported" or any(m in text for m in REFUSAL_MARKERS)
        if q.kind == "invalid":
            handled = resp.error == "invalid_parameter" and all(m in text for m in q.must_mention)
        else:
            handled = refused and resp.result is None
        row.update(routing_correct=handled, numerical_correctness=None, claims=0, unsupported_claims=0,
                   missing_info_ok=handled, factual_consistency=1.0 if handled else 0.0)
    return row


def summarise(rows: list[dict]) -> dict:
    sup = [r for r in rows if r["kind"] == "supported"]
    miss = [r for r in rows if r["kind"] != "supported"]
    claims = sum(r["claims"] for r in rows)
    bad = sum(r["unsupported_claims"] for r in rows)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")   # noqa: E731
    return {
        "questions": len(rows), "supported_questions": len(sup), "missing_info_questions": len(miss),
        "routing_accuracy": mean([r["routing_correct"] for r in sup]),
        "numerical_correctness": mean([r["numerical_correctness"] for r in sup]),
        "unsupported_claim_rate": bad / claims if claims else 0.0,
        "total_claims": claims, "unsupported_claims": bad,
        "missing_info_handling": mean([r["missing_info_ok"] for r in miss]),
        "factual_consistency": mean([r["factual_consistency"] for r in rows]),
        "answers_written_by_gemini": sum(1 for r in rows if r["used_llm"]),
    }


def run_evaluation(service: AnalysisService, analyst: Analyst, on_row=None) -> dict:
    rows = []
    for q in question_set(service):
        resp = analyst.answer(q.text, q.history)
        rows.append(score_response(q, resp, service))
        if on_row:
            on_row(rows[-1])
    return {"summary": summarise(rows), "rows": rows}
