import json
from types import SimpleNamespace

import pytest

from analysis.queries import AnalysisService, InvalidCommunityError, InvalidUserError
from llm import context_builder
from llm.gemini_client import GeminiClient, GeminiUnavailableError
from llm.query_router import Analyst, RoutedCall, rule_route


@pytest.fixture
def service(analysed):
    repo, run_id, net = analysed
    return AnalysisService(repo, run_id, net)


class NoKey(GeminiClient):
    def __init__(self):
        super().__init__(api_key="")


class ApiError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(f"HTTP {code}")


class FakeModels:
    """Stand-in for the SDK's `models` object: used to unit-test error handling only."""
    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, []

    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        out = self.behaviour(model)
        if isinstance(out, Exception):
            raise out
        return out


def fake_client(behaviour):
    return SimpleNamespace(models=FakeModels(behaviour))


# --- routing / validation ------------------------------------------------------------------
@pytest.mark.parametrize("q,fn,params", [
    ("Explain community 5", "get_community_details", {"community_id": 5}),
    ("What interests are common in community 3?", "get_common_interests", {"community_id": 3}),
    ("Compare community 2 and community 4", "compare_communities", {"id_1": 2, "id_2": 4}),
    ("Which communities are most similar?", "get_most_similar_communities", {}),
    ("Find users who connect different communities", "get_bridge_users", {}),
    ("Give me a report on the entire network", "generate_network_report", {}),
    ("Explain how Louvain discovered these communities", "explain_louvain", {}),
    ("Which is the largest community?", "get_largest_community", {}),
    ("Discover all the communities", "list_communities", {}),
    ("Who are the most influential users?", "get_influential_users", {}),
    ("Tell me about user 12", "get_user_details", {"user_id": 12}),
])
def test_rule_router(q, fn, params):
    call = rule_route(q)
    assert call.name == fn and call.params == params


def test_rule_router_followup_uses_history():
    assert rule_route("What interests does it share?", [3]).params == {"community_id": 3}
    assert rule_route("compare with community 2", [3]).params == {"id_1": 3, "id_2": 2}


def test_unsupported_question_is_refused(service):
    r = Analyst(service, NoKey()).answer("What is the weather in Tokyo?")
    assert r.error == "unsupported" and "cannot answer" in r.text and r.result is None


def test_invalid_community_ids(service):
    with pytest.raises(InvalidCommunityError):
        service.get_community_details(999)
    with pytest.raises(InvalidCommunityError):
        service.get_community_details("abc")
    with pytest.raises(InvalidCommunityError):
        service.compare_communities(1, 1)
    r = Analyst(service, NoKey()).answer("Explain community 999")
    assert r.error == "invalid_parameter" and "does not exist" in r.text
    with pytest.raises(InvalidUserError):
        service.get_user_details(123456)


def test_missing_api_key_still_answers_with_verified_numbers(service):
    a = Analyst(service, NoKey())
    assert not a.gemini.is_configured
    r = a.answer("Explain community 1")
    n = service.get_community_details(1)["member_count"]
    assert not r.used_llm and f"{n:,}" in r.text and "GEMINI_API_KEY" in r.notice


def test_all_backend_functions_return_data(service):
    cid = service.community_ids()[0]
    other = service.community_ids()[1]
    for res in (service.list_communities(), service.get_network_overview(), service.get_community_details(cid),
                service.get_common_interests(cid), service.compare_communities(cid, other),
                service.get_community_similarity(cid, other), service.get_most_similar_communities(),
                service.get_influential_users(), service.get_bridge_users(), service.generate_network_report(),
                service.get_largest_community(), service.explain_louvain(), service.get_user_details(0)):
        json.dumps(res)                                         # JSON-serialisable
        assert res["kind"]


def test_overview_numbers_come_from_calculations(service, network):
    o = service.get_network_overview()
    assert o["users"] == 60 and o["friendships"] == network.graph.number_of_edges()
    assert o["communities"] == len(service.community_ids())


# --- context builder --------------------------------------------------------------------------
def test_context_is_valid_json_with_real_values(service):
    res = service.get_community_details(1)
    ctx = json.loads(context_builder.build_context(res))
    assert ctx["community_id"] == 1 and ctx["member_count"] == res["member_count"]
    assert len(ctx["common_features"]) <= 10 and "feature_id" in ctx["common_features"][0]


def test_context_trims_long_lists():
    ctx = json.loads(context_builder.build_context({"items": list(range(100))}, max_list=5))
    assert ctx["items"] == [0, 1, 2, 3, 4]


# --- Gemini client behaviour (SDK stubbed; exercises OUR error handling only) -------------------------
def test_client_success_path():
    c = GeminiClient(api_key="k", client=fake_client(lambda m: SimpleNamespace(text=" hello ")))
    assert c.generate("x") == "hello" and c.last_model_used == c.model


def test_client_invalid_key_fails_fast_with_help():
    fc = fake_client(lambda m: ApiError(403))
    c = GeminiClient(api_key="k", client=fc, backoff_s=0)
    with pytest.raises(GeminiUnavailableError) as e:
        c.generate("x")
    assert "GEMINI_API_KEY" in str(e.value) and len(fc.models.calls) == 1


def test_client_retries_then_falls_back_to_next_model():
    def behaviour(model):
        return ApiError(404) if model == "primary" else SimpleNamespace(text="ok")
    c = GeminiClient(api_key="k", model="primary", client=fake_client(behaviour), backoff_s=0)
    assert c.generate("x") == "ok" and c.last_model_used != "primary"


def test_client_persistent_outage_raises_clear_error():
    c = GeminiClient(api_key="k", client=fake_client(lambda m: ApiError(503)), backoff_s=0, retries=1)
    with pytest.raises(GeminiUnavailableError) as e:
        c.generate("x")
    assert "temporarily unavailable" in str(e.value)


def test_api_failure_keeps_dashboard_answers_working(service):
    c = GeminiClient(api_key="k", client=fake_client(lambda m: ApiError(503)), backoff_s=0, retries=0)
    r = Analyst(service, c).answer("Explain community 1")
    assert not r.used_llm and r.result["kind"] == "community" and "unavailable" in r.notice
    assert f"{service.get_community_details(1)['member_count']:,}" in r.text


def test_llm_answer_uses_verified_context_in_prompt(service):
    seen = {}

    def behaviour(model):
        return SimpleNamespace(text="LLM TEXT")

    class Spy(FakeModels):
        def generate_content(self, model, contents, config=None):
            seen["prompt"] = contents
            return super().generate_content(model, contents, config)

    c = GeminiClient(api_key="k", client=SimpleNamespace(models=Spy(behaviour)))
    a = Analyst(service, c)
    a.route = lambda q, h=None: (RoutedCall("get_community_details", {"community_id": 1}, "gemini"), None)
    r = a.answer("Explain community 1")
    assert r.used_llm and r.text == "LLM TEXT"
    assert str(service.get_community_details(1)["member_count"]) in seen["prompt"]
    assert "VERIFIED DATA" in seen["prompt"]


def test_cooldown_after_outage_skips_further_calls():
    fc = fake_client(lambda m: ApiError(503))
    c = GeminiClient(api_key="k", client=fc, backoff_s=0, retries=0, cooldown_s=60)
    with pytest.raises(GeminiUnavailableError):
        c.generate("x")
    calls = len(fc.models.calls)
    assert c.is_down
    with pytest.raises(GeminiUnavailableError):
        c.generate("y")
    assert len(fc.models.calls) == calls            # no new API calls during cool-down
