import pytest

from core import dataset_loader as dl
from core.graph_builder import UnknownUserError, build_graph, load_network, validate_graph
from tests.conftest import write_raw


@pytest.fixture
def dataset(tmp_path):
    raw = write_raw(tmp_path / "raw", [(10, 20), (20, 30), (10, 30), (30, 40)],
                    {10: [5, 6], 20: [6], 30: [5], 40: [7, 8, 9]}, {10: 1, 20: 1, 30: 2, 40: 2})
    return dl.process_dataset(dl.locate_raw_files(raw), tmp_path / "proc")


def test_graph_uses_original_ids_and_is_undirected(dataset):
    g = build_graph(dataset)
    assert set(g.nodes) == {10, 20, 30, 40} and g.number_of_edges() == 4
    assert not g.is_directed() and g.has_edge(30, 10)
    assert g.nodes[40]["n_features"] == 3
    v = validate_graph(g)
    assert v["self_loops"] == 0 and v["isolated_nodes"] == 0 and v["components"] == 1


def test_lookups(dataset, tmp_path):
    net = load_network(dataset, tmp_path / "proc")
    assert net.neighbours(30) == [10, 20, 40]
    assert net.user_features(40) == [7, 8, 9]
    part = {10: 1, 20: 1, 30: 2, 40: 2}
    assert net.community_of(30, part) == 2 and net.members_of(1, part) == [10, 20]
    with pytest.raises(UnknownUserError):
        net.neighbours(999)
    with pytest.raises(UnknownUserError):
        net.community_of(999, part)


def test_graph_cache_roundtrip_and_invalidation(dataset, tmp_path):
    proc = tmp_path / "proc"
    first = load_network(dataset, proc)
    assert (proc / "graph_cache.pkl").exists()
    second = load_network(dataset, proc)
    assert set(first.graph.edges) == set(second.graph.edges)
    (proc / "graph_cache.pkl").write_bytes(b"corrupted")     # unreadable cache must be rebuilt, not crash
    assert load_network(dataset, proc).graph.number_of_edges() == 4
