import sqlite3

import pytest

from database.connection import get_connection, transaction
from database.initialisation import initialise_database
from database.repositories import AnalysisRepository, ChatRepository, DatasetRepository
from database.schema import LATEST_VERSION, current_version

EXPECTED = {"dataset_metadata", "analysis_runs", "communities", "community_members", "community_interests",
            "network_metrics", "user_metrics", "chat_sessions", "chat_messages", "community_metrics",
            "community_similarity", "schema_version"}


def test_initialise_creates_tables_and_is_idempotent(tmp_path):
    p = tmp_path / "a.sqlite"
    assert initialise_database(p) == LATEST_VERSION
    assert initialise_database(p) == LATEST_VERSION            # second call changes nothing
    conn = get_connection(p)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert EXPECTED <= names and current_version(conn) == LATEST_VERSION
    conn.close()


def test_foreign_keys_enforced(db_path):
    with pytest.raises(sqlite3.IntegrityError):
        with transaction(db_path) as conn:
            conn.execute("INSERT INTO communities(analysis_run_id,community_number,member_count,percentage_of_network)"
                         " VALUES (999,1,5,1.0)")


def test_transaction_rolls_back(db_path):
    with pytest.raises(RuntimeError):
        with transaction(db_path) as conn:
            conn.execute("INSERT INTO chat_sessions(title) VALUES ('x')")
            raise RuntimeError("boom")
    assert ChatRepository(db_path).sessions().empty


def test_constraints(db_path):
    with pytest.raises(sqlite3.IntegrityError):
        with transaction(db_path) as conn:
            conn.execute("INSERT INTO analysis_runs(algorithm) VALUES ('kmeans')")


def test_dataset_upsert_is_unique_by_hash(db_path, tmp_path):
    repo = DatasetRepository(db_path)
    kw = dict(name="n", source="s", version="v", users=1, edges=1, feature_count=1, raw_sha256="h",
              processed_path=tmp_path, validation={})
    assert repo.upsert(**kw) == repo.upsert(**kw)
    assert repo.latest()["raw_sha256"] == "h"


def test_results_persist_and_reload(analysed, db_path):
    repo, run_id, net = analysed
    fresh = AnalysisRepository(db_path)                         # a new object = like a new app launch
    run = fresh.get_run(run_id)
    assert run["status"] == "completed" and run["algorithm"] == "louvain" and run["random_seed"] == 7
    comms = fresh.communities(run_id)
    assert comms.member_count.sum() == 60 and len(fresh.members(run_id)) == 60
    assert run["community_count"] == len(comms)
    assert len(fresh.user_metrics(run_id)) == 60
    assert fresh.find_run(run["dataset_id"], "louvain", 7, 1.0)["id"] == run_id
    assert fresh.find_run(run["dataset_id"], "louvain", 8, 1.0) is None
    assert fresh.network_metrics(run_id)["users"] == 60


def test_failed_run_is_marked(db_path):
    repo = AnalysisRepository(db_path)
    rid = repo.create_run(None, "louvain", 1, 1.0)
    repo.mark_failed(rid, "x")
    assert repo.get_run(rid)["status"] == "failed" and repo.find_run(None, "louvain") is None


def test_chat_persistence(db_path):
    chat = ChatRepository(db_path)
    sid = chat.create_session()
    chat.add_message(sid, "user", "hi")
    chat.add_message(sid, "assistant", "hello", {"function": "f"})
    msgs = chat.messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["meta"] == {"function": "f"}
