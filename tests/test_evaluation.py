"""The evaluation scorer must be able to FAIL: these tests feed it wrong answers."""
import pytest

from analysis.queries import AnalysisService
from llm.evaluation import Q, extract_numbers, matches, score_response, summarise
from llm.query_router import AnalystResponse


@pytest.fixture
def svc(analysed):
    repo, run_id, net = analysed
    return AnalysisService(repo, run_id, net)


def test_extract_numbers():
    assert extract_numbers("27,806 edges, modularity 0.816, 12.5%.") == [(27806.0, 0), (0.816, 3), (12.5, 1)]


def test_match_tolerance_follows_printed_precision():
    assert matches(0.82, 2, 0.8155) and not matches(0.9, 1, 0.8155)
    assert matches(12.0, 0, 0.12) is True            # percentage form accepted


def test_wrong_numbers_are_penalised(svc):
    q = Q("Explain community 1.", "supported", "get_community_details", {"community_id": 1})
    good = svc.get_community_details(1)
    wrong = AnalystResponse(text=f"Community 1 has {good['member_count'] + 777} members and 9999 edges.",
                            function="get_community_details", params={"community_id": 1})
    row = score_response(q, wrong, svc)
    assert row["numerical_correctness"] < 0.5 and row["unsupported_claims"] >= 2 and row["factual_consistency"] < 0.6


def test_correct_numbers_score_full(svc):
    from llm.context_builder import plain_answer
    q = Q("Explain community 1.", "supported", "get_community_details", {"community_id": 1})
    r = AnalystResponse(text=plain_answer(svc.get_community_details(1)), function="get_community_details",
                        params={"community_id": 1})
    row = score_response(q, r, svc)
    assert row["numerical_correctness"] == 1.0 and row["unsupported_claims"] == 0


def test_wrong_routing_scores_zero(svc):
    q = Q("Explain community 1.", "supported", "get_community_details", {"community_id": 1})
    r = AnalystResponse(text="whatever 1", function="get_network_overview", params={})
    assert score_response(q, r, svc)["factual_consistency"] == 0.0


def test_hallucinated_answer_to_unsupported_question_fails(svc):
    q = Q("What is the weather?", "unsupported")
    bad = AnalystResponse(text="It is 21 degrees and sunny.")
    ok = AnalystResponse(text="The available dataset cannot answer that question.", error="unsupported")
    assert score_response(q, bad, svc)["factual_consistency"] == 0.0
    assert score_response(q, ok, svc)["factual_consistency"] == 1.0


def test_summary_aggregates(svc):
    q = Q("What is the weather?", "unsupported")
    rows = [score_response(q, AnalystResponse(text="sunny"), svc),
            score_response(q, AnalystResponse(text="no", error="unsupported"), svc)]
    assert summarise(rows)["missing_info_handling"] == 0.5


def test_precise_modularity_is_not_penalised(svc):
    """Regression: an answer giving 4 decimals (0.8155) must match the true value, not a pre-rounded one."""
    q = Q("What is the modularity of the network?", "supported", "get_network_overview", facts="modularity")
    true = svc.get_network_overview()["modularity"]
    r = AnalystResponse(text=f"The modularity is {true:.4f}.", function="get_network_overview", params={})
    assert score_response(q, r, svc)["numerical_correctness"] == 1.0
    wrong = AnalystResponse(text=f"The modularity is {true + 0.1:.4f}.", function="get_network_overview", params={})
    assert score_response(q, wrong, svc)["numerical_correctness"] == 0.0


def test_focused_question_only_requires_its_own_facts(svc):
    q = Q("How many communities are there?", "supported", "list_communities", facts="count")
    n = len(svc.community_ids())
    r = AnalystResponse(text=f"There are {n} communities.", function="list_communities", params={})
    assert score_response(q, r, svc)["numerical_correctness"] == 1.0


def test_anonymised_spelling_accepted(svc):
    q = Q("Which artist names?", "supported", "get_common_interests", {"community_id": 1}, must_mention=["anonym"])
    r = AnalystResponse(text="Names are unavailable: the data is anonymised.", function="get_common_interests",
                        params={"community_id": 1})
    assert score_response(q, r, svc)["missing_info_ok"] is True
