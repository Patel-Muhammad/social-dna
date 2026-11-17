"""Natural-language question -> validated backend function call -> verified answer.

Routing: Gemini function calling when available, otherwise a deterministic keyword router.
Answering: Gemini explains the verified result; if Gemini is down, a deterministic summary is
shown with a clear notice. Unsupported questions get an honest refusal (never a guess).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from analysis.queries import AnalysisService, InvalidCommunityError, InvalidUserError
from llm import context_builder, prompt_templates as pt
from llm.gemini_client import GeminiClient, GeminiUnavailableError

# name -> (description, {param: {type, description, required}})
FUNCTIONS: dict[str, tuple[str, dict]] = {
    "list_communities": ("List/discover all detected communities with sizes.", {}),
    "get_largest_community": ("Details of the largest community.", {}),
    "get_network_overview": ("Network-level statistics: users, friendships, communities, modularity.", {}),
    "get_community_details": ("Profile of one community: size, connections, common and distinctive interests.",
                              {"community_id": {"type": "integer", "description": "Community number", "required": True}}),
    "get_common_interests": ("Common and distinctive artist features in one community.",
                             {"community_id": {"type": "integer", "description": "Community number", "required": True}}),
    "compare_communities": ("Compare two communities (size, interests, similarity, connectivity).",
                            {"id_1": {"type": "integer", "description": "First community", "required": True},
                             "id_2": {"type": "integer", "description": "Second community", "required": True}}),
    "get_community_similarity": ("Cosine and Jaccard similarity between two communities.",
                                 {"id_1": {"type": "integer", "description": "First community", "required": True},
                                  "id_2": {"type": "integer", "description": "Second community", "required": True}}),
    "get_most_similar_communities": ("Which pairs of communities are most similar.", {}),
    "get_influential_users": ("Most connected / central / PageRank-influential users.", {}),
    "get_bridge_users": ("Users who connect different communities.", {}),
    "get_user_details": ("Metrics of one user by dataset user ID.",
                         {"user_id": {"type": "integer", "description": "Dataset user ID", "required": True}}),
    "generate_network_report": ("Full analytical report on the entire network.", {}),
    "explain_louvain": ("Explain how Louvain discovered the communities and how reliable it is.", {}),
    "unsupported_question": ("The dataset cannot answer this question.",
                            {"reason": {"type": "string", "description": "Why it is unsupported", "required": False}}),
}

SUPPORTED_HELP = ("I can answer questions about the communities, their common interests (as anonymous artist "
                  "feature IDs), similarities, influential users, bridge users and how the communities were "
                  "found. The dataset has no artist names, real identities or other external data.")


class UnsupportedQuestion(Exception):
    pass


@dataclass
class RoutedCall:
    name: str
    params: dict
    source: str  # 'gemini' | 'rules'


@dataclass
class AnalystResponse:
    text: str
    function: str | None = None
    params: dict = field(default_factory=dict)
    result: dict | None = None
    used_llm: bool = False
    route_source: str | None = None
    notice: str | None = None       # non-fatal problem shown to the user (e.g. Gemini down)
    error: str | None = None        # question could not be answered


def declarations() -> list[dict]:
    return [{"name": n, "description": d, "parameters": p} for n, (d, p) in FUNCTIONS.items()]


# --- deterministic router ---------------------------------------------------------------
# Topics the dataset genuinely cannot answer (no names, genres, real identities or external data).
UNSUPPORTED_PATTERNS = (r"\bweather\b", r"\bcountr(y|ies)\b", r"\bnationalit", r"\bgenres?\b", r"real life",
                        r"real name", r"\bidentity\b", r"\bwho is user\b", r"\bemail\b", r"\bgender\b", r"\bage\b",
                        r"\bprice\b", r"\bstock\b", r"next (year|month)", r"\bpredict")


def rule_route(question: str, last_ids: list[int] | None = None) -> RoutedCall | None:
    q = question.lower()
    if any(re.search(p, q) for p in UNSUPPORTED_PATTERNS):
        return None
    ids = [int(x) for x in re.findall(r"(?:communit(?:y|ies)\s*#?|group\s*#?|cluster\s*#?)\s*(\d+)", q)]
    ids += [int(x) for x in re.findall(r"\band\s+(\d+)\b|\bvs\.?\s*(\d+)\b|\bwith\s+(\d+)\b", q) for x in x if x]
    ids = list(dict.fromkeys(ids))
    nums = [int(x) for x in re.findall(r"\b\d+\b", q)]
    user = re.search(r"\buser\s*#?\s*(\d+)", q)
    if user:
        return RoutedCall("get_user_details", {"user_id": int(user.group(1))}, "rules")
    if re.search(r"\b(louvain|how .*(discover|detect|found|find).*communit|how .*algorithm)", q):
        return RoutedCall("explain_louvain", {}, "rules")
    if re.search(r"\b(report|summary of the (entire|whole) network|entire network|whole network)", q):
        return RoutedCall("generate_network_report", {}, "rules")
    if re.search(r"\b(bridge|bridges|connect(s|ing)? (different|between|multiple)|between communities)", q):
        return RoutedCall("get_bridge_users", {}, "rules")
    if re.search(r"\b(influential|influence|pagerank|most connected|central|hub)", q):
        return RoutedCall("get_influential_users", {}, "rules")
    if re.search(r"\bmost similar\b|\bsimilar communities\b|\bwhich communities are .*similar", q):
        return RoutedCall("get_most_similar_communities", {}, "rules")
    if "compar" in q or "differ" in q:
        pool = ids if len(ids) >= 2 else (nums if len(nums) >= 2 else ids)
        if len(pool) == 1 and last_ids:
            pool = [last_ids[-1], pool[0]]
        if len(pool) >= 2:
            return RoutedCall("compare_communities", {"id_1": pool[0], "id_2": pool[1]}, "rules")
    if re.search(r"\bsimilar(ity)?\b", q) and (len(ids) >= 2 or len(nums) >= 2):
        pool = ids if len(ids) >= 2 else nums
        return RoutedCall("get_community_similarity", {"id_1": pool[0], "id_2": pool[1]}, "rules")
    if re.search(r"\blargest\b|\bbiggest\b", q) and "communit" in q:
        return RoutedCall("get_largest_community", {}, "rules")
    cid = ids[0] if ids else (nums[0] if nums and "communit" in q else None)
    if cid is None and last_ids and re.search(r"\b(it|that|this|same)\b", q):
        cid = last_ids[-1]
    if cid is not None and re.search(r"interest|common|like|artist|taste|listen", q):
        return RoutedCall("get_common_interests", {"community_id": cid}, "rules")
    if cid is not None:
        return RoutedCall("get_community_details", {"community_id": cid}, "rules")
    if re.search(r"\b(discover|list|all|how many)\b.*communit|\bcommunities\b.*\b(list|all)\b|^communities\??$", q):
        return RoutedCall("list_communities", {}, "rules")
    if re.search(r"\b(overview|network|statistics|stats|modularity|how many users|density|clustering)", q):
        return RoutedCall("get_network_overview", {}, "rules")
    return None


class Analyst:
    def __init__(self, service: AnalysisService, gemini: GeminiClient | None = None):
        self.service = service
        self.gemini = gemini or GeminiClient()

    # --- routing --------------------------------------------------------------------
    def route(self, question: str, history: list[dict] | None = None) -> tuple[RoutedCall | None, str | None]:
        """Returns (call, notice). call None means unsupported."""
        last_ids = self._last_community_ids(history or [])
        notice = None
        if self.gemini.is_configured:
            hist = "\n".join(f"{m['role']}: {m['text'][:200]}" for m in (history or [])[-6:]) or "(none)"
            try:
                name, args = self.gemini.choose_function(
                    pt.ROUTER_PROMPT.format(history=hist, question=question), declarations())
                if name in FUNCTIONS:
                    if name == "unsupported_question":
                        return None, None
                    return RoutedCall(name, args, "gemini"), None
            except GeminiUnavailableError as exc:
                notice = f"{exc} Using the built-in keyword router instead."
        return rule_route(question, last_ids), notice

    @staticmethod
    def _last_community_ids(history: list[dict]) -> list[int]:
        found: list[int] = []
        for m in history:
            p = m.get("params") or {}
            for k in ("community_id", "id_1", "id_2"):
                if isinstance(p.get(k), int):
                    found.append(p[k])
        return found

    # --- execution ------------------------------------------------------------------
    def execute(self, call: RoutedCall) -> dict:
        if call.name not in FUNCTIONS or call.name == "unsupported_question":
            raise UnsupportedQuestion(call.name)
        spec_params = FUNCTIONS[call.name][1]
        clean = {}
        for key, spec in spec_params.items():
            if key not in call.params:
                if spec.get("required"):
                    raise InvalidCommunityError(f"Please tell me which {key.replace('_', ' ')} you mean.")
                continue
            try:
                clean[key] = int(call.params[key])
            except (TypeError, ValueError):
                raise InvalidCommunityError(f"'{call.params[key]}' is not a valid number.") from None
        return getattr(self.service, call.name)(**clean)

    # --- answering --------------------------------------------------------------------
    def answer(self, question: str, history: list[dict] | None = None) -> AnalystResponse:
        question = (question or "").strip()
        if not question:
            return AnalystResponse(text="Please type a question.", error="empty")
        call, notice = self.route(question, history)
        if call is None:
            return AnalystResponse(text="The available dataset cannot answer that question. " + SUPPORTED_HELP,
                                   notice=notice, error="unsupported")
        try:
            result = self.execute(call)
        except (InvalidCommunityError, InvalidUserError) as exc:
            return AnalystResponse(text=str(exc), function=call.name, params=call.params,
                                   route_source=call.source, notice=notice, error="invalid_parameter")
        base = dict(function=call.name, params=call.params, result=result, route_source=call.source)
        if self.gemini.is_configured:
            try:
                extra = pt.TASK_EXTRA.get(result.get("kind"), "")
                text = self.gemini.generate(
                    pt.ANSWER_PROMPT.format(question=question, context=context_builder.build_context(result),
                                            extra=extra), system=pt.SYSTEM_PROMPT)
                return AnalystResponse(text=text, used_llm=True, notice=notice, **base)
            except GeminiUnavailableError as exc:
                notice = f"{exc} Showing the verified numbers without AI wording."
        else:
            notice = ("Gemini is not configured, so this answer is a plain summary of the verified numbers. "
                      + "Add GEMINI_API_KEY to .env for natural-language explanations.")
        return AnalystResponse(text=context_builder.plain_answer(result), used_llm=False, notice=notice, **base)

    def explain(self, question: str, result: dict) -> str:
        """Gemini explanation of an already-computed result (raises GeminiUnavailableError)."""
        extra = pt.TASK_EXTRA.get(result.get("kind"), "")
        return self.gemini.generate(pt.ANSWER_PROMPT.format(question=question, context=context_builder.build_context(result),
                                                            extra=extra), system=pt.SYSTEM_PROMPT)

    def describe_community(self, community_id: int) -> str:
        """Gemini description of one community (raises GeminiUnavailableError if it cannot)."""
        result = self.service.get_community_details(community_id)
        return self.gemini.generate(pt.DESCRIPTION_PROMPT.format(context=context_builder.build_context(result)),
                                    system=pt.SYSTEM_PROMPT)
