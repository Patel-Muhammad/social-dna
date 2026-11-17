"""All SQL lives here; UI and analysis code only call these methods."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from database.connection import get_connection, transaction


class Repository:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = db_path

    def _query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        conn = get_connection(self.db_path)
        try:
            return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()

    def _one(self, sql: str, params: tuple = ()):
        conn = get_connection(self.db_path)
        try:
            return conn.execute(sql, params).fetchone()
        finally:
            conn.close()


class DatasetRepository(Repository):
    def upsert(self, *, name, source, version, users, edges, feature_count, raw_sha256, processed_path, validation) -> int:
        with transaction(self.db_path) as conn:
            row = conn.execute("SELECT id FROM dataset_metadata WHERE raw_sha256=?", (raw_sha256,)).fetchone()
            if row:
                return int(row["id"])
            cur = conn.execute(
                """INSERT INTO dataset_metadata(dataset_name,dataset_source,dataset_version,total_users,
                   total_edges,feature_count,raw_sha256,processed_path,validation_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (name, source, version, users, edges, feature_count, raw_sha256, str(processed_path),
                 json.dumps(validation)),
            )
            return int(cur.lastrowid)

    def get_by_hash(self, raw_sha256: str) -> dict | None:
        row = self._one("SELECT * FROM dataset_metadata WHERE raw_sha256=?", (raw_sha256,))
        return dict(row) if row else None

    def latest(self) -> dict | None:
        row = self._one("SELECT * FROM dataset_metadata ORDER BY id DESC LIMIT 1")
        return dict(row) if row else None


class AnalysisRepository(Repository):
    # --- runs ----------------------------------------------------------------------
    def create_run(self, dataset_id, algorithm, seed, resolution) -> int:
        with transaction(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO analysis_runs(dataset_id,algorithm,random_seed,resolution) VALUES (?,?,?,?)",
                (dataset_id, algorithm, seed, resolution),
            )
            return int(cur.lastrowid)

    def mark_failed(self, run_id: int, note: str) -> None:
        with transaction(self.db_path) as conn:
            conn.execute("UPDATE analysis_runs SET status='failed', notes=? WHERE id=?", (note, run_id))

    def find_run(self, dataset_id, algorithm, seed=None, resolution=None) -> dict | None:
        sql = "SELECT * FROM analysis_runs WHERE dataset_id=? AND algorithm=? AND status='completed'"
        params: list = [dataset_id, algorithm]
        if seed is not None:
            sql += " AND random_seed=?"
            params.append(seed)
        if resolution is not None:
            sql += " AND resolution=?"
            params.append(resolution)
        row = self._one(sql + " ORDER BY id DESC LIMIT 1", tuple(params))
        return dict(row) if row else None

    def get_run(self, run_id: int) -> dict | None:
        row = self._one("SELECT * FROM analysis_runs WHERE id=?", (run_id,))
        return dict(row) if row else None

    def list_runs(self, algorithm: str | None = None) -> pd.DataFrame:
        if algorithm:
            return self._query("SELECT * FROM analysis_runs WHERE algorithm=? ORDER BY id DESC", (algorithm,))
        return self._query("SELECT * FROM analysis_runs ORDER BY id DESC")

    # --- bulk saves (one transaction for the whole result set) --------------------------
    def save_partition(self, run_id: int, partition: dict[int, int], total_users: int,
                       modularity: float) -> dict[int, int]:
        """Save communities + members; returns {community_number: community_row_id}."""
        sizes: dict[int, int] = {}
        for c in partition.values():
            sizes[c] = sizes.get(c, 0) + 1
        with transaction(self.db_path) as conn:
            ids: dict[int, int] = {}
            for number, size in sorted(sizes.items()):
                cur = conn.execute(
                    "INSERT INTO communities(analysis_run_id,community_number,member_count,percentage_of_network)"
                    " VALUES (?,?,?,?)",
                    (run_id, number, size, 100.0 * size / total_users),
                )
                ids[number] = int(cur.lastrowid)
            conn.executemany(
                "INSERT INTO community_members(analysis_run_id,community_id,original_user_id) VALUES (?,?,?)",
                [(run_id, ids[c], int(u)) for u, c in partition.items()],
            )
            conn.execute(
                "UPDATE analysis_runs SET modularity=?, community_count=? WHERE id=?",
                (modularity, len(sizes), run_id),
            )
        return ids

    def complete_run(self, run_id: int) -> None:
        with transaction(self.db_path) as conn:
            conn.execute("UPDATE analysis_runs SET status='completed' WHERE id=?", (run_id,))

    def save_interests(self, rows: list[tuple]) -> None:
        with transaction(self.db_path) as conn:
            conn.executemany(
                """INSERT INTO community_interests(analysis_run_id,community_id,feature_id,kind,rank,frequency,
                   prevalence,network_prevalence,lift,distinctiveness_score) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def save_community_metrics(self, rows: list[tuple]) -> None:
        with transaction(self.db_path) as conn:
            conn.executemany(
                """INSERT INTO community_metrics(community_id,analysis_run_id,internal_edges,external_edges,
                   internal_density,avg_internal_degree,avg_degree,conductance,within_cosine,within_jaccard,
                   cosine_to_network,avg_features_per_user) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def save_similarity(self, rows: list[tuple]) -> None:
        with transaction(self.db_path) as conn:
            conn.executemany(
                """INSERT INTO community_similarity(analysis_run_id,community_a,community_b,cosine_similarity,
                   jaccard_similarity,cross_edges) VALUES (?,?,?,?,?,?)""",
                rows,
            )

    def save_network_metrics(self, run_id: int, metrics: dict[str, tuple[float | None, str | None]]) -> None:
        with transaction(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO network_metrics(analysis_run_id,metric_name,metric_value,note) VALUES (?,?,?,?)",
                [(run_id, k, v, note) for k, (v, note) in metrics.items()],
            )

    def save_user_metrics(self, rows: list[tuple]) -> None:
        with transaction(self.db_path) as conn:
            conn.executemany(
                """INSERT INTO user_metrics(analysis_run_id,original_user_id,degree,degree_centrality,
                   betweenness_centrality,closeness_centrality,pagerank,community_id,external_degree,
                   neighbour_communities,participation_coefficient) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def set_description(self, community_id: int, text: str, model: str) -> None:
        with transaction(self.db_path) as conn:
            conn.execute("UPDATE communities SET description=?, description_model=? WHERE id=?",
                         (text, model, community_id))

    # --- reads --------------------------------------------------------------------------
    def communities(self, run_id: int) -> pd.DataFrame:
        return self._query(
            """SELECT c.id AS community_row_id, c.community_number, c.member_count, c.percentage_of_network,
                      c.description, c.description_model, m.internal_edges, m.external_edges,
                      m.internal_density, m.avg_internal_degree, m.avg_degree, m.conductance,
                      m.within_cosine, m.within_jaccard, m.cosine_to_network, m.avg_features_per_user
               FROM communities c LEFT JOIN community_metrics m ON m.community_id=c.id
               WHERE c.analysis_run_id=? ORDER BY c.community_number""", (run_id,))

    def members(self, run_id: int) -> pd.DataFrame:
        return self._query(
            """SELECT m.original_user_id AS user_id, c.community_number AS community
               FROM community_members m JOIN communities c ON c.id=m.community_id
               WHERE m.analysis_run_id=?""", (run_id,))

    def interests(self, run_id: int, kind: str | None = None) -> pd.DataFrame:
        sql = """SELECT c.community_number AS community, i.feature_id, i.kind, i.rank, i.frequency,
                        i.prevalence, i.network_prevalence, i.lift, i.distinctiveness_score
                 FROM community_interests i JOIN communities c ON c.id=i.community_id
                 WHERE i.analysis_run_id=?"""
        params: tuple = (run_id,)
        if kind:
            sql += " AND i.kind=?"
            params += (kind,)
        return self._query(sql + " ORDER BY c.community_number, i.kind, i.rank", params)

    def similarity(self, run_id: int) -> pd.DataFrame:
        return self._query(
            """SELECT ca.community_number AS community_a, cb.community_number AS community_b,
                      s.cosine_similarity, s.jaccard_similarity, s.cross_edges
               FROM community_similarity s
               JOIN communities ca ON ca.id=s.community_a JOIN communities cb ON cb.id=s.community_b
               WHERE s.analysis_run_id=?""", (run_id,))

    def network_metrics(self, run_id: int) -> dict[str, float | None]:
        df = self._query("SELECT metric_name, metric_value FROM network_metrics WHERE analysis_run_id=?", (run_id,))
        return dict(zip(df["metric_name"], df["metric_value"]))

    def network_metric_notes(self, run_id: int) -> dict[str, str | None]:
        df = self._query("SELECT metric_name, note FROM network_metrics WHERE analysis_run_id=?", (run_id,))
        return dict(zip(df["metric_name"], df["note"]))

    def user_metrics(self, run_id: int) -> pd.DataFrame:
        return self._query(
            """SELECT u.original_user_id AS user_id, u.degree, u.degree_centrality, u.betweenness_centrality,
                      u.closeness_centrality, u.pagerank, c.community_number AS community,
                      u.external_degree, u.neighbour_communities, u.participation_coefficient
               FROM user_metrics u JOIN communities c ON c.id=u.community_id
               WHERE u.analysis_run_id=?""", (run_id,))


class ChatRepository(Repository):
    def create_session(self, title: str = "New chat") -> int:
        with transaction(self.db_path) as conn:
            return int(conn.execute("INSERT INTO chat_sessions(title) VALUES (?)", (title,)).lastrowid)

    def rename_session(self, session_id: int, title: str) -> None:
        with transaction(self.db_path) as conn:
            conn.execute("UPDATE chat_sessions SET title=? WHERE id=?", (title[:60], session_id))

    def add_message(self, session_id: int, role: str, message: str, meta: dict | None = None) -> int:
        with transaction(self.db_path) as conn:
            return int(conn.execute(
                "INSERT INTO chat_messages(session_id,role,message,meta_json) VALUES (?,?,?,?)",
                (session_id, role, message, json.dumps(meta) if meta else None)).lastrowid)

    def messages(self, session_id: int) -> list[dict]:
        df = self._query("SELECT role, message, meta_json, created_at FROM chat_messages WHERE session_id=? ORDER BY id",
                         (session_id,))
        out = []
        for r in df.to_dict("records"):
            r["meta"] = json.loads(r.pop("meta_json")) if r.get("meta_json") else None
            out.append(r)
        return out

    def sessions(self) -> pd.DataFrame:
        return self._query("SELECT id, title, created_at FROM chat_sessions ORDER BY id DESC")
